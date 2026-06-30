from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.karaone_recon.losses import grad_reverse
from src.karaone_v9.transport import ConditionalTransportConfig, ConditionalTransportDecoder


@dataclass
class KaraOneV9Config:
    n_channels_eeg: int = 62
    eeg_len: int = 1280
    patch_size: int = 32
    patch_stride: int = 16
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.15
    channel_dropout: float = 0.10
    cond_dim: int = 32
    num_stages: int = 1
    num_subjects: int = 14
    num_labels: int = 11
    semantic_dim: int = 768
    semantic_token_vocab: int = 64
    codec_dim: int = 128
    transport_layers: int = 2
    transport_heads: int = 4


class KaraOneV9NeuralSemanticTransport(nn.Module):
    """Canonical v9 model: EEG tokens -> semantic/prosody -> codec transport."""

    def __init__(self, cfg: KaraOneV9Config):
        super().__init__()
        self.cfg = cfg
        self.patch = nn.Conv1d(
            cfg.n_channels_eeg,
            cfg.d_model,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_stride,
            padding=cfg.patch_size // 2,
        )
        max_steps = _conv_steps(cfg.eeg_len, cfg.patch_size, cfg.patch_stride, cfg.patch_size // 2)
        self.max_steps = int(max_steps)
        self.pos = nn.Parameter(torch.zeros(1, self.max_steps, cfg.d_model))
        self.stage_embedding = nn.Embedding(cfg.num_stages, cfg.cond_dim)
        self.stage_proj = nn.Linear(cfg.cond_dim, cfg.d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.channel_reliability = nn.Sequential(
            nn.LayerNorm(cfg.n_channels_eeg),
            nn.Linear(cfg.n_channels_eeg, cfg.n_channels_eeg),
            nn.Sigmoid(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.content_stream = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model), nn.GELU())
        self.prosody_stream = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model), nn.GELU())
        self.uncertainty_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, 1))

        self.semantic_seq_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.semantic_dim))
        self.semantic_summary_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.semantic_dim))
        self.semantic_token_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.semantic_token_vocab))
        self.prompt_ctc_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.num_labels + 1))
        self.prompt_classifier = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.num_labels))

        self.prosody_active_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, 1))
        self.prosody_energy_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, 1))
        self.prosody_duration_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, 1))
        self.prosody_onset_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, 1))

        self.condition_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model * 2 + 1),
            nn.Linear(cfg.d_model * 2 + 1, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.subject_classifier = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.num_subjects),
        )
        self.pretrain_recon = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model))
        self.transport = ConditionalTransportDecoder(
            ConditionalTransportConfig(
                codec_dim=cfg.codec_dim,
                cond_dim=cfg.d_model,
                hidden_dim=cfg.d_model,
                num_layers=cfg.transport_layers,
                num_heads=cfg.transport_heads,
                dropout=cfg.dropout,
            )
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.stage_embedding.weight, std=0.02)

    def forward(
        self,
        eeg: torch.Tensor,
        stage_idx: torch.Tensor,
        eeg_valid_len: torch.Tensor | None = None,
        *,
        mask_ratio: float = 0.0,
        lambda_subject_adv: float = 0.0,
        codec_seq: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if eeg.ndim != 3:
            raise ValueError(f"eeg must be [B,C,T], got {tuple(eeg.shape)}")
        norm_eeg = self._normalize_eeg(eeg, eeg_valid_len)
        norm_eeg, channel_gate = self._apply_channel_reliability(norm_eeg)
        patch_tokens = self.patch(norm_eeg).transpose(1, 2)
        valid_mask = _patch_valid_mask(
            eeg_valid_len,
            patch_tokens.shape[1],
            eeg.shape[-1],
            batch_size=eeg.shape[0],
            device=eeg.device,
        )
        if self.training and self.cfg.channel_dropout > 0.0:
            keep = (torch.rand(eeg.shape[0], eeg.shape[1], 1, device=eeg.device) > self.cfg.channel_dropout).to(eeg.dtype)
            keep = keep / keep.mean(dim=1, keepdim=True).clamp_min(0.25)
            norm_eeg_dropout = norm_eeg * keep
            patch_tokens = self.patch(norm_eeg_dropout).transpose(1, 2)

        patch_tokens = patch_tokens + self.pos[:, : patch_tokens.shape[1]]
        patch_tokens = patch_tokens + self.stage_proj(self.stage_embedding(stage_idx.long())).unsqueeze(1)
        ssl_target = patch_tokens.detach()
        ssl_mask = torch.zeros(patch_tokens.shape[:2], device=eeg.device, dtype=torch.bool)
        if mask_ratio > 0.0:
            rand = torch.rand(patch_tokens.shape[:2], device=eeg.device)
            ssl_mask = (rand < float(mask_ratio)) & valid_mask
            patch_tokens = torch.where(ssl_mask.unsqueeze(-1), self.mask_token.to(patch_tokens.dtype), patch_tokens)

        encoded = self.encoder(patch_tokens, src_key_padding_mask=~valid_mask)
        encoded = encoded * valid_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = _masked_mean(encoded, valid_mask)
        content = self.content_stream(encoded)
        prosody = self.prosody_stream(encoded)
        uncertainty_logit = self.uncertainty_head(encoded)
        uncertainty = torch.sigmoid(uncertainty_logit)
        condition = self.condition_head(torch.cat([content, prosody, uncertainty], dim=-1))
        out = {
            "eeg_tokens": encoded,
            "content_tokens": content,
            "prosody_tokens": prosody,
            "condition_seq": condition,
            "pooled": pooled,
            "token_valid_mask": valid_mask,
            "patch_tokens_target": ssl_target,
            "patch_recon": self.pretrain_recon(encoded),
            "patch_mask": ssl_mask,
            "channel_gate": channel_gate,
            "pred_semantic_seq": self.semantic_seq_head(content),
            "pred_semantic_summary": self.semantic_summary_head(pooled),
            "semantic_token_logits": self.semantic_token_head(content),
            "prompt_ctc_logits": self.prompt_ctc_head(content),
            "prompt_logits": self.prompt_classifier(pooled),
            "prosody_active_logits": self.prosody_active_head(prosody).squeeze(-1),
            "prosody_energy": self.prosody_energy_head(prosody).squeeze(-1),
            "prosody_duration": torch.sigmoid(self.prosody_duration_head(pooled)).squeeze(-1),
            "prosody_onset": torch.sigmoid(self.prosody_onset_head(pooled)).squeeze(-1),
            "uncertainty": uncertainty.squeeze(-1),
            "subject_logits": self.subject_classifier(grad_reverse(pooled, float(lambda_subject_adv))),
        }
        if codec_seq is not None:
            flow = self.transport.training_loss(codec_seq, condition)
            out.update({f"transport_{key}": value for key, value in flow.items()})
        return out

    @torch.no_grad()
    def generate_codec(self, eeg: torch.Tensor, stage_idx: torch.Tensor, eeg_valid_len: torch.Tensor | None, steps: int, codec_steps: int) -> dict[str, torch.Tensor]:
        was_training = self.training
        self.eval()
        out = self.forward(eeg, stage_idx, eeg_valid_len)
        out["pred_codec_seq"] = self.transport.sample(out["condition_seq"], steps=steps, codec_steps=codec_steps, codec_dim=self.cfg.codec_dim)
        if was_training:
            self.train()
        return out

    def _normalize_eeg(self, eeg: torch.Tensor, eeg_valid_len: torch.Tensor | None) -> torch.Tensor:
        if eeg_valid_len is None:
            mean = eeg.mean(dim=-1, keepdim=True)
            std = eeg.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
            return (eeg - mean) / std
        mask = torch.arange(eeg.shape[-1], device=eeg.device).unsqueeze(0) < eeg_valid_len.long().unsqueeze(1)
        mask_f = mask.unsqueeze(1).to(eeg.dtype)
        denom = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (eeg * mask_f).sum(dim=-1, keepdim=True) / denom
        var = ((eeg - mean) * mask_f).pow(2).sum(dim=-1, keepdim=True) / denom
        return (eeg - mean) / torch.sqrt(var + 1e-4)

    def _apply_channel_reliability(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        channel_stats = torch.log(eeg.var(dim=-1, unbiased=False).clamp_min(1e-6))
        gate = self.channel_reliability(channel_stats)
        return eeg * gate.unsqueeze(-1), gate


def _conv_steps(length: int, kernel: int, stride: int, padding: int) -> int:
    return int((int(length) + 2 * int(padding) - int(kernel)) // int(stride) + 1)


def _patch_valid_mask(valid_len: torch.Tensor | None, steps: int, eeg_len: int, *, batch_size: int, device: torch.device) -> torch.Tensor:
    if valid_len is None:
        return torch.ones(int(batch_size), int(steps), device=device, dtype=torch.bool)
    valid_steps = torch.ceil(valid_len.float() / float(max(eeg_len, 1)) * float(steps)).long().clamp(1, int(steps))
    return torch.arange(int(steps), device=device).unsqueeze(0) < valid_steps.unsqueeze(1)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(x.dtype)
    return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)


def resize_sequence(x: torch.Tensor, steps: int) -> torch.Tensor:
    if x.shape[1] == int(steps):
        return x
    return F.interpolate(x.transpose(1, 2), size=int(steps), mode="linear", align_corners=False).transpose(1, 2)

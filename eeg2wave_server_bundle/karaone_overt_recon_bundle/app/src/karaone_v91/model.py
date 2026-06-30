from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.karaone_recon.losses import grad_reverse
from src.karaone_v91.transport import NeuroSonicCodecFlow, NeuroSonicFlowConfig
from src.karaone_v9.model import _conv_steps, _masked_mean, _patch_valid_mask, resize_sequence


@dataclass
class KaraOneV91Config:
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
    channel_experts: int = 6
    channel_top_k: int = 16
    channel_desc_dim: int = 32
    channel_embedding_dim: int = 16
    moe_temperature: float = 0.7
    shared_dim: int = 128
    domain_dim: int = 64
    transport_layers: int = 2
    transport_heads: int = 4
    heun_steps: int = 32


class ChannelMoEFrontendV91(nn.Module):
    """Sparse channel-selecting MoE for 62-channel EEG."""

    def __init__(self, cfg: KaraOneV91Config):
        super().__init__()
        self.cfg = cfg
        self.num_channels = int(cfg.n_channels_eeg)
        self.num_experts = int(cfg.channel_experts)
        self.top_k = int(cfg.channel_top_k)
        self.temperature = float(cfg.moe_temperature)
        desc_in = 6 + int(cfg.channel_embedding_dim)
        self.channel_embedding = nn.Embedding(self.num_channels, int(cfg.channel_embedding_dim))
        self.descriptor = nn.Sequential(
            nn.LayerNorm(desc_in),
            nn.Linear(desc_in, int(cfg.channel_desc_dim)),
            nn.GELU(),
            nn.Linear(int(cfg.channel_desc_dim), int(cfg.channel_desc_dim)),
            nn.GELU(),
        )
        self.gate_head = nn.Linear(int(cfg.channel_desc_dim), 1)
        self.assign_head = nn.Linear(int(cfg.channel_desc_dim), self.num_experts)
        expert_dim = max(1, int(cfg.d_model) // self.num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, expert_dim, kernel_size=7, padding=3),
                    nn.GELU(),
                    nn.Conv1d(expert_dim, expert_dim, kernel_size=5, padding=2),
                    nn.GELU(),
                )
                for _ in range(self.num_experts)
            ]
        )
        self.out_proj = nn.Conv1d(expert_dim * self.num_experts, int(cfg.d_model), kernel_size=1)

    def forward(self, eeg: torch.Tensor, eeg_valid_len: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if eeg.ndim != 3:
            raise ValueError(f"eeg must be [B,C,T], got {tuple(eeg.shape)}")
        desc = self._descriptors(eeg, eeg_valid_len)
        channel_idx = torch.arange(eeg.shape[1], device=eeg.device)
        embed = self.channel_embedding(channel_idx).unsqueeze(0).expand(eeg.shape[0], -1, -1)
        hidden = self.descriptor(torch.cat([desc, embed], dim=-1))
        logits = self.gate_head(hidden).squeeze(-1)
        raw_gate = torch.sigmoid(logits)
        if self.top_k > 0 and self.top_k < eeg.shape[1]:
            keep_idx = torch.topk(logits, k=self.top_k, dim=1).indices
            mask = torch.zeros_like(raw_gate).scatter_(1, keep_idx, 1.0)
            gate = raw_gate * mask
        else:
            gate = raw_gate
        if self.training and float(self.cfg.channel_dropout) > 0.0:
            keep = (torch.rand_like(gate) > float(self.cfg.channel_dropout)).to(gate.dtype)
            keep = torch.where(keep.sum(dim=1, keepdim=True) > 0, keep, torch.ones_like(keep))
            gate = gate * keep
        assign = torch.softmax(self.assign_head(hidden) / max(self.temperature, 1e-4), dim=-1)
        weighted = assign * gate.unsqueeze(-1)
        denom = weighted.sum(dim=1).clamp_min(1e-4)
        expert_signals = torch.einsum("bct,bce->bet", eeg, weighted) / denom.unsqueeze(-1)
        expert_out = []
        for idx, expert in enumerate(self.experts):
            expert_out.append(expert(expert_signals[:, idx : idx + 1]))
        features = self.out_proj(torch.cat(expert_out, dim=1))
        norm_gate = gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-6)
        entropy = -(norm_gate * torch.log(norm_gate.clamp_min(1e-8))).sum(dim=1) / torch.log(
            torch.tensor(float(eeg.shape[1]), device=eeg.device, dtype=eeg.dtype)
        )
        load = weighted.sum(dim=(0, 1))
        load = load / load.sum().clamp_min(1e-6)
        return features, {
            "channel_gate": gate,
            "channel_gate_logits": logits,
            "channel_assign": assign,
            "channel_load": load,
            "channel_gate_entropy": entropy,
            "channel_sparsity": gate.mean(dim=1),
        }

    def _descriptors(self, eeg: torch.Tensor, eeg_valid_len: torch.Tensor | None) -> torch.Tensor:
        mask = _time_mask(eeg, eeg_valid_len)
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (eeg * mask).sum(dim=-1, keepdim=True) / denom
        centered = (eeg - mean) * mask
        var = centered.pow(2).sum(dim=-1, keepdim=True) / denom
        std = torch.sqrt(var + 1e-5)
        logvar = torch.log(var + 1e-5)
        abs_mean = (eeg.abs() * mask).sum(dim=-1, keepdim=True) / denom
        first_half = mask[..., : eeg.shape[-1] // 2]
        second_half = mask[..., eeg.shape[-1] // 2 :]
        first = (eeg[..., : eeg.shape[-1] // 2] * first_half).sum(dim=-1, keepdim=True) / first_half.sum(dim=-1, keepdim=True).clamp_min(1.0)
        second = (eeg[..., eeg.shape[-1] // 2 :] * second_half).sum(dim=-1, keepdim=True) / second_half.sum(dim=-1, keepdim=True).clamp_min(1.0)
        slope = second - first
        # MPS-friendly spectral proxy: band-limited energy from temporal differences.
        delta = eeg[..., 1:] - eeg[..., :-1]
        delta_mask = mask[..., 1:] * mask[..., :-1]
        diff_energy = (delta.pow(2) * delta_mask).sum(dim=-1, keepdim=True) / delta_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.cat([mean, std, logvar, abs_mean, slope, torch.log(diff_energy + 1e-5)], dim=-1)


class KaraOneV91ClusteredChannelMoEFlow(nn.Module):
    """v9.1 model: EEG Channel-MoE -> semantic/prosody -> codec-space flow."""

    def __init__(self, cfg: KaraOneV91Config):
        super().__init__()
        self.cfg = cfg
        self.channel_moe = ChannelMoEFrontendV91(cfg)
        self.patch = nn.Conv1d(
            cfg.d_model,
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
        self.domain_stream = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.domain_dim), nn.GELU())
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

        self.shared_head = nn.Sequential(nn.LayerNorm(cfg.d_model * 2 + 1), nn.Linear(cfg.d_model * 2 + 1, cfg.shared_dim), nn.GELU())
        self.condition_head = nn.Sequential(
            nn.LayerNorm(cfg.shared_dim),
            nn.Linear(cfg.shared_dim, cfg.d_model),
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
        self.domain_subject_classifier = nn.Sequential(
            nn.LayerNorm(cfg.domain_dim),
            nn.Linear(cfg.domain_dim, cfg.domain_dim),
            nn.GELU(),
            nn.Linear(cfg.domain_dim, cfg.num_subjects),
        )
        self.pretrain_recon = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model))
        self.transport = NeuroSonicCodecFlow(
            NeuroSonicFlowConfig(
                codec_dim=cfg.codec_dim,
                cond_dim=cfg.d_model,
                hidden_dim=cfg.d_model,
                num_layers=cfg.transport_layers,
                num_heads=cfg.transport_heads,
                dropout=cfg.dropout,
                heun_steps=cfg.heun_steps,
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
        teacher_condition_seq: torch.Tensor | None = None,
        scheduled_teacher_ratio: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        norm_eeg = self._normalize_eeg(eeg, eeg_valid_len)
        moe_features, moe_aux = self.channel_moe(norm_eeg, eeg_valid_len)
        patch_tokens = self.patch(moe_features).transpose(1, 2)
        valid_mask = _patch_valid_mask(
            eeg_valid_len,
            patch_tokens.shape[1],
            eeg.shape[-1],
            batch_size=eeg.shape[0],
            device=eeg.device,
        )
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
        domain = self.domain_stream(encoded)
        domain_pooled = _masked_mean(domain, valid_mask)
        uncertainty = torch.sigmoid(self.uncertainty_head(encoded))
        shared = self.shared_head(torch.cat([content, prosody, uncertainty], dim=-1))
        condition = self.condition_head(shared)
        out = {
            "eeg_tokens": encoded,
            "content_tokens": content,
            "prosody_tokens": prosody,
            "domain_tokens": domain,
            "domain_pooled": domain_pooled,
            "shared_tokens": shared,
            "condition_seq": condition,
            "pooled": pooled,
            "token_valid_mask": valid_mask,
            "patch_tokens_target": ssl_target,
            "patch_recon": self.pretrain_recon(encoded),
            "patch_mask": ssl_mask,
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
            "domain_subject_logits": self.domain_subject_classifier(domain_pooled),
        }
        out.update(moe_aux)
        out["content_domain_dot"] = (F.normalize(_masked_mean(content, valid_mask), dim=-1) * F.normalize(self._domain_to_content(domain_pooled), dim=-1)).sum(dim=-1)
        if codec_seq is not None:
            flow = self.transport.training_loss(
                codec_seq,
                condition,
                teacher_condition=teacher_condition_seq,
                teacher_ratio=float(scheduled_teacher_ratio),
            )
            out.update({f"transport_{key}": value for key, value in flow.items()})
        return out

    @torch.no_grad()
    def generate_codec(
        self,
        eeg: torch.Tensor,
        stage_idx: torch.Tensor,
        eeg_valid_len: torch.Tensor | None,
        steps: int | None,
        codec_steps: int,
    ) -> dict[str, torch.Tensor]:
        was_training = self.training
        self.eval()
        out = self.forward(eeg, stage_idx, eeg_valid_len)
        out["pred_codec_seq"] = self.transport.sample_heun(
            out["condition_seq"],
            steps=steps or self.cfg.heun_steps,
            codec_steps=codec_steps,
            codec_dim=self.cfg.codec_dim,
        )
        if was_training:
            self.train()
        return out

    def _domain_to_content(self, domain_pooled: torch.Tensor) -> torch.Tensor:
        if domain_pooled.shape[-1] == self.cfg.d_model:
            return domain_pooled
        return F.pad(domain_pooled, (0, max(0, self.cfg.d_model - domain_pooled.shape[-1])))[:, : self.cfg.d_model]

    def _normalize_eeg(self, eeg: torch.Tensor, eeg_valid_len: torch.Tensor | None) -> torch.Tensor:
        mask = _time_mask(eeg, eeg_valid_len)
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (eeg * mask).sum(dim=-1, keepdim=True) / denom
        var = ((eeg - mean) * mask).pow(2).sum(dim=-1, keepdim=True) / denom
        return (eeg - mean) / torch.sqrt(var + 1e-4)


def _time_mask(eeg: torch.Tensor, eeg_valid_len: torch.Tensor | None) -> torch.Tensor:
    if eeg_valid_len is None:
        return torch.ones_like(eeg)
    mask = torch.arange(eeg.shape[-1], device=eeg.device).unsqueeze(0) < eeg_valid_len.long().unsqueeze(1)
    return mask.unsqueeze(1).to(eeg.dtype)

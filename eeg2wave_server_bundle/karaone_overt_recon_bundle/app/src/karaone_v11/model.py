from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.karaone_recon.losses import grad_reverse
from src.karaone_v9.model import _conv_steps, _masked_mean, _patch_valid_mask, resize_sequence


@dataclass
class KaraOneV11Config:
    n_channels_eeg: int = 62
    eeg_len: int = 1280
    eeg_sample_rate: float = 256.0
    patch_size: int = 32
    patch_stride: int = 16
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.15
    channel_dropout: float = 0.12
    cond_dim: int = 32
    num_stages: int = 1
    num_subjects: int = 14
    num_labels: int = 11
    semantic_dim: int = 768
    semantic_token_vocab: int = 64
    semantic_token_steps: int = 50
    codec_dim: int = 128
    codec_token_vocab: int = 64
    codec_token_steps: int = 75
    channel_experts: int = 6
    channel_top_k: int = 16
    channel_clusters: int = 8
    channel_desc_dim: int = 32
    channel_embedding_dim: int = 16
    channel_cluster_embedding_dim: int = 8
    moe_temperature: float = 0.7
    shared_dim: int = 128
    domain_dim: int = 64
    perceiver_queries: int = 50
    aligner: str = "hybrid"


class ChannelClusterMoEFrontend(nn.Module):
    """Sparse channel/group MoE with train-only channel-cluster ids."""

    def __init__(self, cfg: KaraOneV11Config):
        super().__init__()
        self.cfg = cfg
        self.num_channels = int(cfg.n_channels_eeg)
        self.num_experts = int(cfg.channel_experts)
        self.top_k = int(cfg.channel_top_k)
        self.temperature = float(cfg.moe_temperature)
        desc_in = 11 + int(cfg.channel_embedding_dim) + int(cfg.channel_cluster_embedding_dim)
        self.channel_embedding = nn.Embedding(self.num_channels, int(cfg.channel_embedding_dim))
        self.cluster_embedding = nn.Embedding(max(1, int(cfg.channel_clusters)), int(cfg.channel_cluster_embedding_dim))
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

    def forward(
        self,
        eeg: torch.Tensor,
        eeg_valid_len: torch.Tensor | None = None,
        channel_cluster_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        desc = self._descriptors(eeg, eeg_valid_len)
        channel_idx = torch.arange(eeg.shape[1], device=eeg.device).clamp(max=self.num_channels - 1)
        channel_embed = self.channel_embedding(channel_idx).unsqueeze(0).expand(eeg.shape[0], -1, -1)
        if channel_cluster_id is None:
            channel_cluster_id = torch.zeros(eeg.shape[0], eeg.shape[1], device=eeg.device, dtype=torch.long)
        if channel_cluster_id.ndim == 1:
            channel_cluster_id = channel_cluster_id.unsqueeze(0).expand(eeg.shape[0], -1)
        cluster_embed = self.cluster_embedding(channel_cluster_id.to(eeg.device).long().clamp(min=0, max=self.cluster_embedding.num_embeddings - 1))
        hidden = self.descriptor(torch.cat([desc, channel_embed, cluster_embed], dim=-1))
        logits = self.gate_head(hidden).squeeze(-1)
        raw_gate = torch.sigmoid(logits)
        if self.top_k > 0 and self.top_k < eeg.shape[1]:
            keep = torch.topk(logits, k=self.top_k, dim=1).indices
            mask = torch.zeros_like(raw_gate).scatter_(1, keep, 1.0)
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
        expert_out = [expert(expert_signals[:, idx : idx + 1]) for idx, expert in enumerate(self.experts)]
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
        delta = eeg[..., 1:] - eeg[..., :-1]
        delta_mask = mask[..., 1:] * mask[..., :-1]
        diff_energy = (delta.pow(2) * delta_mask).sum(dim=-1, keepdim=True) / delta_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        band_power = self._band_power(eeg * mask)
        return torch.cat([mean, std, logvar, abs_mean, slope, torch.log(diff_energy + 1e-5), band_power], dim=-1)

    def _band_power(self, eeg: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(eeg.float(), dim=-1).abs().pow(2)
        freqs = torch.fft.rfftfreq(eeg.shape[-1], d=1.0 / max(float(self.cfg.eeg_sample_rate), 1.0)).to(eeg.device)
        bands = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 60.0))
        powers = []
        for low, high in bands:
            band_mask = (freqs >= low) & (freqs < high)
            if bool(band_mask.any()):
                value = spectrum[..., band_mask].mean(dim=-1, keepdim=True)
            else:
                value = spectrum.new_zeros((*spectrum.shape[:-1], 1))
            powers.append(torch.log(value + 1e-5))
        return torch.cat(powers, dim=-1).to(eeg.dtype)


class KaraOneV11TokenGenerator(nn.Module):
    """EEG token encoder -> audio semantic/prosody/codec token generator."""

    def __init__(self, cfg: KaraOneV11Config, codec_codebook: torch.Tensor | None = None):
        super().__init__()
        self.cfg = cfg
        self.channel_moe = ChannelClusterMoEFrontend(cfg)
        self.patch = nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=cfg.patch_size, stride=cfg.patch_stride, padding=cfg.patch_size // 2)
        self.max_steps = int(_conv_steps(cfg.eeg_len, cfg.patch_size, cfg.patch_stride, cfg.patch_size // 2))
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
        self.semantic_token_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.semantic_token_vocab))
        self.semantic_embed_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.semantic_dim))
        self.semantic_summary_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.semantic_dim))
        self.codec_token_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.codec_token_vocab))
        self.prompt_ctc_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.num_labels + 1))
        self.prompt_classifier = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.num_labels))
        self.prosody_active_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, 1))
        self.prosody_energy_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, 1))
        self.prosody_duration_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, 1))
        self.prosody_onset_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, 1))
        self.shared_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.shared_dim), nn.GELU())
        self.subject_classifier = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.num_subjects))
        self.domain_subject_classifier = nn.Sequential(nn.LayerNorm(cfg.domain_dim), nn.Linear(cfg.domain_dim, cfg.domain_dim), nn.GELU(), nn.Linear(cfg.domain_dim, cfg.num_subjects))
        self.pretrain_recon = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model))
        self.perceiver_queries = nn.Parameter(torch.randn(1, cfg.perceiver_queries, cfg.d_model) * 0.02)
        self.perceiver = nn.MultiheadAttention(cfg.d_model, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
        if codec_codebook is None:
            codec_codebook = torch.zeros(cfg.codec_token_vocab, cfg.codec_dim)
        if codec_codebook.shape != (cfg.codec_token_vocab, cfg.codec_dim):
            padded = torch.zeros(cfg.codec_token_vocab, cfg.codec_dim)
            rows = min(codec_codebook.shape[0], padded.shape[0])
            cols = min(codec_codebook.shape[1], padded.shape[1])
            padded[:rows, :cols] = codec_codebook[:rows, :cols]
            codec_codebook = padded
        self.register_buffer("codec_codebook", codec_codebook.float())
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
        channel_cluster_id: torch.Tensor | None = None,
        mask_ratio: float = 0.0,
        lambda_subject_adv: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        norm_eeg = self._normalize_eeg(eeg, eeg_valid_len)
        moe_features, moe_aux = self.channel_moe(norm_eeg, eeg_valid_len, channel_cluster_id=channel_cluster_id)
        patch_tokens = self.patch(moe_features).transpose(1, 2)
        valid_mask = _patch_valid_mask(eeg_valid_len, patch_tokens.shape[1], eeg.shape[-1], batch_size=eeg.shape[0], device=eeg.device)
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
        perceiver_q = self.perceiver_queries.expand(eeg.shape[0], -1, -1)
        perceiver_tokens, _ = self.perceiver(perceiver_q, content, content, key_padding_mask=~valid_mask)
        semantic_tokens_for_head = resize_sequence(content, self.cfg.semantic_token_steps)
        codec_tokens_for_head = resize_sequence(perceiver_tokens, self.cfg.codec_token_steps)
        semantic_token_logits = self.semantic_token_head(semantic_tokens_for_head)
        perceiver_semantic_logits = self.semantic_token_head(perceiver_tokens)
        codec_logits = self.codec_token_head(codec_tokens_for_head)
        codec_probs = torch.softmax(codec_logits, dim=-1)
        pred_codec_seq = torch.einsum("btk,kd->btd", codec_probs, self.codec_codebook.to(codec_probs.dtype))
        out = {
            "eeg_tokens": encoded,
            "content_tokens": content,
            "prosody_tokens": prosody,
            "perceiver_tokens": perceiver_tokens,
            "pooled": pooled,
            "token_valid_mask": valid_mask,
            "patch_tokens_target": ssl_target,
            "patch_recon": self.pretrain_recon(encoded),
            "patch_mask": ssl_mask,
            "pred_semantic_seq": self.semantic_embed_head(resize_sequence(content, self.cfg.semantic_token_steps)),
            "pred_semantic_summary": self.semantic_summary_head(pooled),
            "semantic_token_logits": semantic_token_logits,
            "semantic_token_logits_perceiver": perceiver_semantic_logits,
            "codec_token_logits": codec_logits,
            "pred_codec_seq": pred_codec_seq,
            "prompt_ctc_logits": self.prompt_ctc_head(content),
            "prompt_logits": self.prompt_classifier(pooled),
            "prosody_active_logits": self.prosody_active_head(prosody).squeeze(-1),
            "prosody_energy": self.prosody_energy_head(prosody).squeeze(-1),
            "prosody_duration": torch.sigmoid(self.prosody_duration_head(pooled)).squeeze(-1),
            "prosody_onset": torch.sigmoid(self.prosody_onset_head(pooled)).squeeze(-1),
            "shared_tokens": self.shared_head(content),
            "domain_tokens": domain,
            "domain_pooled": domain_pooled,
            "subject_logits": self.subject_classifier(grad_reverse(pooled, float(lambda_subject_adv))),
            "domain_subject_logits": self.domain_subject_classifier(domain_pooled),
        }
        out.update(moe_aux)
        out["content_domain_dot"] = (F.normalize(pooled, dim=-1) * F.normalize(self._domain_to_content(domain_pooled), dim=-1)).sum(dim=-1)
        return out

    @torch.no_grad()
    def generate_codec_tokens(
        self,
        eeg: torch.Tensor,
        stage_idx: torch.Tensor,
        eeg_valid_len: torch.Tensor | None,
        *,
        channel_cluster_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        was_training = self.training
        self.eval()
        out = self.forward(eeg, stage_idx, eeg_valid_len, channel_cluster_id=channel_cluster_id)
        out["pred_codec_token_ids"] = out["codec_token_logits"].argmax(dim=-1)
        out["pred_codec_seq"] = self.codec_codebook[out["pred_codec_token_ids"].clamp(0, self.codec_codebook.shape[0] - 1)]
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


__all__ = ["ChannelClusterMoEFrontend", "KaraOneV11Config", "KaraOneV11TokenGenerator"]

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _channel_descriptors(eeg: torch.Tensor) -> torch.Tensor:
    mean = eeg.mean(dim=-1)
    std = eeg.std(dim=-1, unbiased=False)
    logvar = torch.log(torch.var(eeg, dim=-1, unbiased=False).clamp_min(1e-6))
    abs_mean = eeg.abs().mean(dim=-1)
    if eeg.shape[-1] > 1:
        diff = eeg[..., 1:] - eeg[..., :-1]
        diff_energy = diff.square().mean(dim=-1)
        slope = (eeg[..., -1] - eeg[..., 0]) / float(eeg.shape[-1])
    else:
        diff_energy = torch.zeros_like(mean)
        slope = torch.zeros_like(mean)
    return torch.stack([mean, std, logvar, abs_mean, diff_energy, slope], dim=-1).nan_to_num()


def _topk_softmax(logits: torch.Tensor, top_k: int, dim: int = -1) -> torch.Tensor:
    if top_k >= logits.shape[dim]:
        return F.softmax(logits, dim=dim)
    vals, idx = logits.topk(k=max(int(top_k), 1), dim=dim)
    masked = torch.full_like(logits, -1e9)
    masked.scatter_(dim, idx, vals)
    return F.softmax(masked, dim=dim)


class ChannelMoEFrontend(nn.Module):
    def __init__(
        self,
        n_channels: int,
        d_model: int,
        cond_dim: int,
        num_experts: int = 4,
        channel_top_k: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.num_experts = int(num_experts)
        self.channel_top_k = max(1, min(int(channel_top_k), self.n_channels))
        router_dim = 6 + int(cond_dim)
        hidden = max(32, d_model // 2)
        self.channel_router = nn.Sequential(
            nn.LayerNorm(router_dim),
            nn.Linear(router_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.expert_router = nn.Sequential(
            nn.LayerNorm(router_dim),
            nn.Linear(router_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.num_experts),
        )
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(self.n_channels, d_model, kernel_size=7, padding=3, bias=False),
                nn.GroupNorm(num_groups=min(8, d_model), num_channels=d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(self.num_experts)
        )
        self.out_norm = nn.GroupNorm(num_groups=min(8, d_model), num_channels=d_model)

    def forward(self, eeg: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        desc = _channel_descriptors(eeg)
        cond_by_channel = cond.unsqueeze(1).expand(eeg.shape[0], eeg.shape[1], cond.shape[-1])
        router_in = torch.cat([desc, cond_by_channel], dim=-1)
        channel_logits = self.channel_router(router_in).squeeze(-1)
        channel_gate = _topk_softmax(channel_logits, self.channel_top_k, dim=-1).unsqueeze(-1)
        expert_route = F.softmax(self.expert_router(router_in), dim=-1)
        mixed = None
        for expert_idx, expert in enumerate(self.experts):
            weight = channel_gate * expert_route[..., expert_idx : expert_idx + 1]
            expert_out = expert(eeg * weight)
            mixed = expert_out if mixed is None else mixed + expert_out
        assert mixed is not None
        usage = expert_route.mean(dim=(0, 1))
        uniform = usage.new_full(usage.shape, 1.0 / max(self.num_experts, 1))
        entropy = -(channel_gate.squeeze(-1) * channel_gate.squeeze(-1).clamp_min(1e-8).log()).sum(dim=-1).mean()
        entropy = entropy / torch.log(channel_gate.new_tensor(float(self.n_channels))).clamp_min(1e-8)
        aux = {
            "channel_gate": channel_gate.squeeze(-1).detach(),
            "channel_gate_top_channels": channel_gate.squeeze(-1).detach().topk(k=self.channel_top_k, dim=-1).indices,
            "moe_load_balance": ((usage - uniform) ** 2).sum() * self.num_experts,
            "moe_channel_sparsity": channel_gate.mean(),
            "moe_route_entropy": entropy,
            "moe_channel_gate_mean": channel_gate.mean().detach(),
            "moe_usage_min": usage.min().detach(),
            "moe_usage_max": usage.max().detach(),
        }
        return self.out_norm(mixed), aux


@dataclass
class FEISV3ModelConfig:
    n_channels_eeg: int = 14
    d_model: int = 160
    num_heads: int = 4
    num_layers: int = 2
    ff_mult: int = 4
    dropout: float = 0.15
    semantic_vocab: int = 64
    codec_vocab: int = 128
    semantic_steps: int = 64
    codec_steps: int = 64
    num_labels: int = 16
    num_stages: int = 5
    channel_clusters: int = 4
    audio_variant_clusters: int = 32
    channel_moe: bool = True
    channel_top_k: int = 6
    channel_experts: int = 4
    use_stage_token: bool = True
    use_subject_id_in_forward: bool = False
    num_subjects_for_adversary: int = 21


class FEISV3TokenGenerator(nn.Module):
    """Tokenized FEIS EEG-to-speech model.

    Forward intentionally accepts no subject_id, speaker_id, or audio-source
    identity input. Subject metadata is reserved for dataset sampling, losses,
    and leakage audits.
    """

    def __init__(self, cfg: FEISV3ModelConfig):
        super().__init__()
        if cfg.use_subject_id_in_forward:
            raise ValueError("FEIS v3 forbids subject_id/speaker_id as model forward inputs")
        self.cfg = cfg
        cond_dim = cfg.d_model
        self.stage_embed = nn.Embedding(cfg.num_stages, cond_dim)
        self.cluster_embed = nn.Embedding(max(cfg.channel_clusters, 1), cond_dim)
        if cfg.channel_moe:
            self.frontend = ChannelMoEFrontend(
                n_channels=cfg.n_channels_eeg,
                d_model=cfg.d_model,
                cond_dim=cond_dim,
                num_experts=cfg.channel_experts,
                channel_top_k=cfg.channel_top_k,
                dropout=cfg.dropout,
            )
        else:
            self.frontend = nn.Sequential(
                nn.Conv1d(cfg.n_channels_eeg, cfg.d_model, kernel_size=7, padding=3),
                nn.GroupNorm(num_groups=min(8, cfg.d_model), num_channels=cfg.d_model),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.d_model * cfg.ff_mult,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.semantic_pos = nn.Parameter(torch.randn(cfg.semantic_steps, cfg.d_model) * 0.02)
        self.codec_pos = nn.Parameter(torch.randn(cfg.codec_steps, cfg.d_model) * 0.02)
        self.content_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model), nn.GELU())
        self.semantic_head = nn.Linear(cfg.d_model, cfg.semantic_vocab)
        self.ctc_head = nn.Linear(cfg.d_model, cfg.semantic_vocab + 1)
        self.perceiver_head = nn.Linear(cfg.d_model, cfg.semantic_vocab)
        self.prompt_head = nn.Linear(cfg.d_model, cfg.num_labels)
        self.clip_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.semantic_vocab))
        self.prosody_active_head = nn.Linear(cfg.d_model, 1)
        self.prosody_duration_head = nn.Linear(cfg.d_model, 1)
        self.prosody_energy_head = nn.Linear(cfg.d_model, 1)
        self.prosody_onset_head = nn.Linear(cfg.d_model, 1)
        self.variant_embed_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model))
        self.audio_variant_head = nn.Linear(cfg.d_model, cfg.audio_variant_clusters)
        self.codec_condition = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model), nn.GELU())
        self.codec_head = nn.Linear(cfg.d_model, cfg.codec_vocab)
        self.subject_adv_head = nn.Linear(cfg.d_model, max(cfg.num_subjects_for_adversary, 1))

    def _pool_tokens(self, x: torch.Tensor, steps: int, pos: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool1d(x.transpose(1, 2), int(steps)).transpose(1, 2)
        return pooled + pos.unsqueeze(0)

    def forward(
        self,
        eeg: torch.Tensor,
        stage_idx: torch.Tensor | None = None,
        eeg_valid_len: torch.Tensor | None = None,
        channel_cluster_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del eeg_valid_len
        batch = eeg.shape[0]
        device = eeg.device
        if stage_idx is None:
            stage_idx = torch.zeros(batch, dtype=torch.long, device=device)
        if channel_cluster_id is None:
            channel_cluster_id = torch.zeros(batch, dtype=torch.long, device=device)
        stage_idx = stage_idx.long().clamp(0, self.cfg.num_stages - 1)
        channel_cluster_id = channel_cluster_id.long().clamp(0, max(self.cfg.channel_clusters - 1, 0))
        cond = self.cluster_embed(channel_cluster_id)
        if self.cfg.use_stage_token:
            cond = cond + self.stage_embed(stage_idx)
        adapted = self.frontend(eeg, cond) if self.cfg.channel_moe else self.frontend(eeg)
        if isinstance(adapted, tuple):
            x, aux = adapted
        else:
            x, aux = adapted, {}
        tokens = x.transpose(1, 2) + cond.unsqueeze(1)
        encoded = self.encoder(tokens)
        semantic_tokens = self.encoder(self._pool_tokens(encoded, self.cfg.semantic_steps, self.semantic_pos))
        codec_tokens = self.encoder(self._pool_tokens(encoded, self.cfg.codec_steps, self.codec_pos))
        pooled = encoded.mean(dim=1)
        content = self.content_head(pooled)
        variant = self.variant_embed_head(pooled)
        out = {
            "semantic_logits": self.semantic_head(semantic_tokens),
            "ctc_logits": self.ctc_head(semantic_tokens),
            "perceiver_logits": self.perceiver_head(semantic_tokens),
            "prompt_logits": self.prompt_head(content),
            "content_embed": content,
            "content_clip": self.clip_head(content),
            "prosody_active_logits": self.prosody_active_head(semantic_tokens).squeeze(-1),
            "prosody_duration": F.softplus(self.prosody_duration_head(semantic_tokens).squeeze(-1)),
            "prosody_energy": self.prosody_energy_head(semantic_tokens).squeeze(-1),
            "prosody_onset_logits": self.prosody_onset_head(semantic_tokens).squeeze(-1),
            "variant_embed": variant,
            "audio_variant_logits": self.audio_variant_head(variant),
            "codec_conditioning": self.codec_condition(variant + content),
            "codec_logits": self.codec_head(codec_tokens),
            "subject_adv_logits": self.subject_adv_head(content),
            "eeg_tokens": encoded,
        }
        out.update(aux)
        return out

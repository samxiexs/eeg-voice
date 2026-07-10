from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask.unsqueeze(-1).to(values.dtype)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1).to(values.dtype)


def _token_mask(valid_len: torch.Tensor, token_steps: int, raw_steps: int) -> torch.Tensor:
    positions = torch.arange(token_steps, device=valid_len.device).unsqueeze(0)
    lengths = torch.ceil(valid_len.to(torch.float32) * token_steps / float(raw_steps)).long().clamp(1, token_steps)
    return positions < lengths.unsqueeze(1)


def _normalise_eeg(eeg: torch.Tensor, valid_len: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(eeg.shape[-1], device=eeg.device).view(1, 1, -1)
    mask = positions < valid_len.view(-1, 1, 1)
    denom = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    mean = (eeg * mask).sum(dim=-1, keepdim=True) / denom
    var = ((eeg - mean).pow(2) * mask).sum(dim=-1, keepdim=True) / denom
    return ((eeg - mean) / torch.sqrt(var + 1e-5)) * mask.to(eeg.dtype)


class RawEEGEncoder(nn.Module):
    def __init__(self, channels: int = 62, d_model: int = 192, layers: int = 4, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv1d(channels, d_model, kernel_size=32, stride=16, padding=16),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(d_model, heads, d_model * 4, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, eeg: torch.Tensor, valid_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eeg = _normalise_eeg(eeg, valid_len)
        tokens = self.patch(eeg).transpose(1, 2)
        mask = _token_mask(valid_len, tokens.shape[1], eeg.shape[-1])
        tokens = self.transformer(tokens, src_key_padding_mask=~mask)
        tokens = self.norm(tokens) * mask.unsqueeze(-1).to(tokens.dtype)
        return tokens, _masked_mean(tokens, mask), mask


class TopographicEEGEncoder(nn.Module):
    """Consumes `[band, time, height, width]`, not rendered image files."""

    def __init__(self, bands: int = 5, d_model: int = 192, dropout: float = 0.1):
        super().__init__()
        hidden = max(32, d_model // 2)
        self.features = nn.Sequential(
            nn.Conv3d(bands, hidden, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(hidden, d_model, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, topography: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.features(topography)
        tokens = values.mean(dim=(-1, -2)).transpose(1, 2)
        tokens = self.norm(tokens)
        return tokens, tokens.mean(dim=1)


@dataclass(frozen=True)
class EEG0711Config:
    channels: int = 62
    d_model: int = 192
    layers: int = 4
    heads: int = 4
    dropout: float = 0.1
    embed_dim: int = 256
    semantic_steps: int = 50
    semantic_vocab: int = 64


class EEG0711Encoder(nn.Module):
    """Raw/time-frequency-topography fusion without any subject-id input."""

    def __init__(self, cfg: EEG0711Config = EEG0711Config()):
        super().__init__()
        self.cfg = cfg
        self.raw_encoder = RawEEGEncoder(cfg.channels, cfg.d_model, cfg.layers, cfg.heads, cfg.dropout)
        self.topo_encoder = TopographicEEGEncoder(5, cfg.d_model, cfg.dropout)
        self.cross_attention = nn.MultiheadAttention(cfg.d_model, cfg.heads, dropout=cfg.dropout, batch_first=True)
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.heads, cfg.d_model * 4, dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, num_layers=2)
        self.raw_projection = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.embed_dim))
        self.topo_projection = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.embed_dim))
        self.eeg_projection = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.embed_dim))
        self.token_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.semantic_vocab))
        self.time_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, 3))

    def forward(self, eeg: torch.Tensor, valid_len: torch.Tensor, topography: torch.Tensor) -> dict[str, torch.Tensor]:
        raw_tokens, raw_global, raw_mask = self.raw_encoder(eeg, valid_len)
        topo_tokens, topo_global = self.topo_encoder(topography)
        attended, _ = self.cross_attention(raw_tokens, topo_tokens, topo_tokens, key_padding_mask=None)
        tokens = self.fusion(raw_tokens + attended, src_key_padding_mask=~raw_mask)
        tokens = tokens * raw_mask.unsqueeze(-1).to(tokens.dtype)
        pooled = _masked_mean(tokens, raw_mask)
        semantic_tokens = F.interpolate(tokens.transpose(1, 2), size=self.cfg.semantic_steps, mode="linear", align_corners=False).transpose(1, 2)
        raw = self.time_head(pooled)
        onset = torch.sigmoid(raw[:, 0]) * 2.0
        duration = 0.05 + torch.sigmoid(raw[:, 1]) * (2.0 - 0.05)
        duration = torch.minimum(duration, (2.0 - onset).clamp_min(0.05))
        return {
            "raw_embed": self.raw_projection(raw_global),
            "topo_embed": self.topo_projection(topo_global),
            "eeg_embed": self.eeg_projection(pooled),
            "tokens": tokens,
            "token_logits": self.token_head(semantic_tokens),
            "pred_onset_sec": onset,
            "pred_duration_sec": duration,
            "pred_active_logit": raw[:, 2],
        }


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    scale = math.log(10000.0) / max(half - 1, 1)
    freqs = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype) * -scale)
    angles = t[:, None] * freqs[None, :]
    values = torch.cat([angles.sin(), angles.cos()], dim=-1)
    return F.pad(values, (0, dim - values.shape[1]))


class ConditionalFlowDecoder(nn.Module):
    """Continuous EnCodec-latent probability-flow velocity decoder."""

    def __init__(self, latent_dim: int = 128, eeg_dim: int = 192, d_model: int = 256, heads: int = 4, layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.input_projection = nn.Linear(latent_dim, d_model)
        self.eeg_projection = nn.Linear(eeg_dim, d_model)
        self.time_projection = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.t_projection = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerDecoderLayer(d_model, heads, d_model * 4, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.output_projection = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, latent_dim))

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, eeg_tokens: torch.Tensor, onset: torch.Tensor, duration: torch.Tensor, active_logit: torch.Tensor) -> torch.Tensor:
        query = self.input_projection(z_t)
        time_features = torch.stack([onset, duration, torch.sigmoid(active_logit)], dim=-1)
        condition = self.time_projection(time_features) + self.t_projection(sinusoidal_time_embedding(t, query.shape[-1]))
        query = query + condition.unsqueeze(1)
        memory = self.eeg_projection(eeg_tokens)
        decoded = self.decoder(query, memory)
        return self.output_projection(decoded)

    @torch.no_grad()
    def sample(self, eeg_tokens: torch.Tensor, onset: torch.Tensor, duration: torch.Tensor, active_logit: torch.Tensor, steps: int = 24) -> torch.Tensor:
        batch = eeg_tokens.shape[0]
        z = torch.randn(batch, 150, self.latent_dim, device=eeg_tokens.device, dtype=eeg_tokens.dtype)
        step_size = 1.0 / float(steps)
        for idx in range(steps):
            t = torch.full((batch,), (idx + 0.5) * step_size, device=z.device, dtype=z.dtype)
            z = z + step_size * self(z, t, eeg_tokens, onset, duration, active_logit)
        return z

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConditionalTransportConfig:
    codec_dim: int
    cond_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1


class ConditionalTransportDecoder(nn.Module):
    """Conditional flow-matching decoder in factorized codec latent space.

    This is the v9 replacement for direct Mel regression / Griffin-Lim as a
    trainable generation path.  It learns a velocity field from Gaussian noise
    to codec latents conditioned on EEG-predicted semantic/prosody tokens.
    """

    def __init__(self, cfg: ConditionalTransportConfig):
        super().__init__()
        self.cfg = cfg
        self.codec_in = nn.Linear(cfg.codec_dim, cfg.hidden_dim)
        self.cond_in = nn.Linear(cfg.cond_dim, cfg.hidden_dim)
        self.time_in = nn.Sequential(
            nn.Linear(1, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.net = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.codec_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 3:
            raise ValueError(f"x_t must be [B,T,D], got {tuple(x_t.shape)}")
        if t.ndim == 1:
            t = t[:, None]
        cond = _resize_sequence(condition, x_t.shape[1])
        h = self.codec_in(x_t) + self.cond_in(cond) + self.time_in(t.float()).unsqueeze(1)
        h = self.net(h)
        return self.out(h)

    def training_loss(self, target_codec: torch.Tensor, condition: torch.Tensor, noise: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(target_codec)
        t = torch.rand(target_codec.shape[0], device=target_codec.device, dtype=target_codec.dtype).clamp(1e-4, 1.0 - 1e-4)
        x_t = (1.0 - t[:, None, None]) * noise + t[:, None, None] * target_codec
        target_velocity = target_codec - noise
        pred_velocity = self.forward(x_t, t, condition)
        return {
            "flow_loss": F.mse_loss(pred_velocity, target_velocity),
            "pred_velocity": pred_velocity,
            "x_t": x_t,
            "t": t,
        }

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, steps: int, codec_steps: int, codec_dim: int | None = None) -> torch.Tensor:
        steps = max(1, int(steps))
        codec_dim = int(codec_dim or self.cfg.codec_dim)
        x = torch.randn(condition.shape[0], int(codec_steps), codec_dim, device=condition.device, dtype=condition.dtype)
        dt = 1.0 / float(steps)
        for step in range(steps):
            t = torch.full((condition.shape[0],), float(step) / float(steps), device=condition.device, dtype=condition.dtype)
            velocity = self.forward(x, t, condition)
            x = x + dt * velocity
        return x


def _resize_sequence(x: torch.Tensor, steps: int) -> torch.Tensor:
    if x.shape[1] == int(steps):
        return x
    return F.interpolate(x.transpose(1, 2), size=int(steps), mode="linear", align_corners=False).transpose(1, 2)

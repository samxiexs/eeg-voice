from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RefinerConfig:
    target_dim: int = 128
    hidden_dim: int = 256
    num_blocks: int = 4
    kernel_size: int = 5
    dropout: float = 0.1


class ResidualDenoisingRefiner(nn.Module):
    """Second-stage latent residual refiner.

    The base model predicts a normalized EnCodec latent. This module receives a
    noisy version of that prediction and learns a residual correction toward the
    target latent. It is deliberately small and stable; it can later be replaced
    by a fuller diffusion schedule without changing the bundle interfaces.
    """

    def __init__(self, cfg: RefinerConfig):
        super().__init__()
        self.cfg = cfg
        padding = (int(cfg.kernel_size) - 1) // 2
        self.in_proj = nn.Conv1d(cfg.target_dim + 1, cfg.hidden_dim, kernel_size=1)
        blocks = []
        for _ in range(int(cfg.num_blocks)):
            blocks.extend(
                [
                    nn.GroupNorm(num_groups=min(8, cfg.hidden_dim), num_channels=cfg.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=cfg.kernel_size, padding=padding),
                ]
            )
        self.blocks = nn.Sequential(*blocks)
        self.out_proj = nn.Conv1d(cfg.hidden_dim, cfg.target_dim, kernel_size=1)

    def forward(self, base_latent: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        # base_latent: [B,T,D], noise_level: [B]
        b, t, _ = base_latent.shape
        level = noise_level.float().view(b, 1, 1).expand(-1, t, 1)
        x = torch.cat([base_latent, level], dim=-1).transpose(1, 2)
        h = self.in_proj(x)
        h = h + self.blocks(h)
        residual = self.out_proj(F.gelu(h)).transpose(1, 2)
        return base_latent + residual


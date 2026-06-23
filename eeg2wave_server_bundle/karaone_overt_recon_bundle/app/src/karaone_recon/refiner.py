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
    """Second-stage latent residual refiner. NOTE: this is *not* a diffusion model.

    The base model predicts a normalized EnCodec latent. This module receives a
    lightly noised version of that prediction (the scalar `noise_level` is only a
    small augmentation, sampled in [0, noise_std]) and learns a single-step
    residual correction toward the target latent, trained with MSE + cosine. At
    inference `noise_level=0` and it runs one forward pass, i.e. it is a learned
    deterministic post-filter, not a generative sampler.

    Why it is not diffusion (and must not be described as such): a real diffusion
    model would (1) define a forward process that interpolates the *target* toward
    pure noise over a schedule, (2) train a network to reverse it (predict noise /
    score / x0 across all noise levels), and (3) generate by *iterative* sampling
    from noise. That models the conditional distribution p(latent | EEG) and lets
    you draw samples instead of regressing to the conditional mean.

    Important caveat: because both stages here regress to a single deterministic
    target with MSE, they are mean-seeking. For audio this yields an over-smoothed
    "average voice" (see synth's `mean_latent` baseline). Escaping that blur is the
    actual reason to adopt diffusion / discrete-token autoregression — but only a
    *real* schedule with iterative sampling buys that; this refiner does not.
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


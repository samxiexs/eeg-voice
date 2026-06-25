from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.direct_eeg2speech.diffusion import LatentDiffusion


@dataclass
class FEISAcousticDiffusionConfig:
    target_dim: int = 80
    target_steps: int = 64
    cond_dim: int = 192
    d_model: int = 192
    num_steps: int = 200
    sample_steps: int = 24
    eval_steps: int = 8
    num_layers: int = 2
    num_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    mode: str = "diffusion"  # diffusion (DDPM+DDIM) | flow (conditional flow matching)


def build_feis_acoustic_diffusion(cfg: FEISAcousticDiffusionConfig) -> LatentDiffusion:
    return LatentDiffusion(
        latent_dim=cfg.target_dim,
        cond_dim=cfg.cond_dim,
        d_model=cfg.d_model,
        num_steps=cfg.num_steps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        ff_mult=cfg.ff_mult,
        dropout=cfg.dropout,
        mode=cfg.mode,
    )


class FEISDiffusionInference(nn.Module):
    """Inference wrapper that exposes the same forward contract as regression."""

    def __init__(
        self,
        base_model: nn.Module,
        diffusion: LatentDiffusion,
        *,
        target_steps: int,
        target_dim: int,
        sample_steps: int,
    ):
        super().__init__()
        self.base_model = base_model
        self.diffusion = diffusion
        self.target_steps = int(target_steps)
        self.target_dim = int(target_dim)
        self.sample_steps = int(sample_steps)

    def forward(self, eeg: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.base_model(eeg)
        sampled = self.diffusion.sample_ddim(
            (eeg.shape[0], self.target_steps, self.target_dim),
            out["eeg_tokens"],
            coarse_latent=out["pred_mel"],
            sample_steps=self.sample_steps,
        )
        merged = dict(out)
        merged["pred_mel"] = sampled
        return merged

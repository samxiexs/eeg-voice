"""Conditional diffusion refinement for native SpeechT5 mel features.

This module is intentionally audio-only.  It refines a coarse mel predicted by
the separately gated MFCC renderer and never receives EEG, subject, stimulus,
or split identifiers.  Consequently the same checkpoint can be applied to the
correct EEG output and every counterfactual control without changing the EEG
evaluation contract.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _time_embedding(timestep: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timestep.float().unsqueeze(1) * frequencies.unsqueeze(0)
    value = torch.cat((angles.sin(), angles.cos()), dim=1)
    return torch.nn.functional.pad(value, (0, dimension - value.shape[1]))


class _ResidualBlock(nn.Module):
    def __init__(self, hidden_dimension: int, time_dimension: int, dropout: float,
                 conditioning_dimension: int = 0):
        super().__init__()
        groups = min(8, hidden_dimension)
        while hidden_dimension % groups:
            groups -= 1
        self.normalization = nn.GroupNorm(groups, hidden_dimension)
        self.depthwise = nn.Conv1d(hidden_dimension, hidden_dimension, 5, padding=2,
                                   groups=hidden_dimension)
        self.pointwise = nn.Conv1d(hidden_dimension, hidden_dimension, 1)
        self.time = nn.Linear(time_dimension, hidden_dimension)
        self.context = (nn.Linear(conditioning_dimension, hidden_dimension)
                        if conditioning_dimension > 0 else None)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor, time: torch.Tensor,
                mask: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        update = torch.nn.functional.silu(self.normalization(value))
        update = self.depthwise(update)
        update = update + self.time(time).unsqueeze(-1)
        if self.context is not None:
            if context is None:
                raise ValueError("EEG-conditioned diffusion requires context")
            update = update + self.context(context).unsqueeze(-1)
        update = self.pointwise(self.dropout(torch.nn.functional.silu(update)))
        return (value + update) * mask.unsqueeze(1).to(value.dtype)


class ConditionalMelDiffusion(nn.Module):
    """Small conditional DDPM trained to predict noise on normalized mel.

    Sampling uses deterministic DDIM (eta=0).  Padding is masked after every
    residual block and every diffusion update, so ragged native-duration audio
    cannot exchange information through batch padding.
    """

    def __init__(self, mel_bins: int = 80, hidden_dimension: int = 128,
                 layers: int = 6, dropout: float = 0.05, timesteps: int = 100,
                 beta_start: float = 1e-4, beta_end: float = 2e-2,
                 conditioning_dimension: int = 0):
        super().__init__()
        if timesteps < 2:
            raise ValueError("diffusion timesteps must be at least two")
        self.mel_bins = int(mel_bins)
        self.hidden_dimension = int(hidden_dimension)
        self.timesteps = int(timesteps)
        self.conditioning_dimension = int(conditioning_dimension)
        if self.conditioning_dimension < 0:
            raise ValueError("conditioning_dimension must be non-negative")
        self.input = nn.Conv1d(self.mel_bins * 2, hidden_dimension, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension * 2), nn.SiLU(),
            nn.Linear(hidden_dimension * 2, hidden_dimension),
        )
        self.blocks = nn.ModuleList([
            _ResidualBlock(hidden_dimension, hidden_dimension, dropout, self.conditioning_dimension)
            for _ in range(int(layers))
        ])
        self.output = nn.Sequential(nn.SiLU(), nn.Conv1d(hidden_dimension, self.mel_bins, 3, padding=1))
        betas = torch.linspace(float(beta_start), float(beta_end), self.timesteps)
        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_cumulative", cumulative)

    def forward(self, noisy: torch.Tensor, condition: torch.Tensor,
                timestep: torch.Tensor, mask: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        if noisy.shape != condition.shape or noisy.ndim != 3 or noisy.shape[1] != self.mel_bins:
            raise ValueError(f"diffusion mel tensors must share [B,{self.mel_bins},T]")
        if mask.shape != (noisy.shape[0], noisy.shape[2]):
            raise ValueError("diffusion mask must be [B,T]")
        if timestep.shape != (noisy.shape[0],):
            raise ValueError("diffusion timestep must be [B]")
        if self.conditioning_dimension:
            if context is None or context.shape != (noisy.shape[0], self.conditioning_dimension):
                raise ValueError("diffusion context must be [B, conditioning_dimension]")
        elif context is not None:
            raise ValueError("audio-only diffusion does not accept EEG context")
        support = mask.unsqueeze(1).to(noisy.dtype)
        state = self.input(torch.cat((noisy * support, condition * support), dim=1)) * support
        time = self.time_mlp(_time_embedding(timestep, self.hidden_dimension))
        for block in self.blocks:
            state = block(state, time, mask, context)
        return self.output(state) * support

    def denoising_loss(self, clean: torch.Tensor, condition: torch.Tensor,
                       mask: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        timestep = torch.randint(self.timesteps, (clean.shape[0],), device=clean.device)
        noise = torch.randn_like(clean) * mask.unsqueeze(1).to(clean.dtype)
        cumulative = self.alpha_cumulative[timestep].view(-1, 1, 1)
        noisy = cumulative.sqrt() * clean + (1.0 - cumulative).sqrt() * noise
        prediction = self(noisy, condition, timestep, mask, context)
        error = (prediction - noise).square().mean(1)
        return (error * mask).sum() / mask.sum().clamp_min(1)

    @torch.no_grad()
    def refine(self, condition: torch.Tensor, mask: torch.Tensor, *, steps: int = 20,
               noise: torch.Tensor | None = None, context: torch.Tensor | None = None) -> torch.Tensor:
        if steps < 1 or steps > self.timesteps:
            raise ValueError("sampling steps must be in [1, timesteps]")
        support = mask.unsqueeze(1).to(condition.dtype)
        if noise is None:
            noise = torch.randn_like(condition)
        if noise.shape != condition.shape:
            raise ValueError("diffusion sampling noise must match condition")
        schedule = torch.linspace(self.timesteps - 1, 0, steps, device=condition.device).round().long()
        schedule = torch.unique_consecutive(schedule)
        first_alpha = self.alpha_cumulative[schedule[0]]
        value = (first_alpha.sqrt() * condition + (1.0 - first_alpha).sqrt() * noise) * support
        for position, current in enumerate(schedule):
            timestep = torch.full((condition.shape[0],), int(current), device=condition.device, dtype=torch.long)
            alpha = self.alpha_cumulative[current]
            prediction = self(value, condition, timestep, mask, context)
            clean = ((value - (1.0 - alpha).sqrt() * prediction) / alpha.sqrt()).clamp(-8.0, 8.0)
            if position + 1 == len(schedule):
                value = clean
            else:
                next_alpha = self.alpha_cumulative[schedule[position + 1]]
                value = next_alpha.sqrt() * clean + (1.0 - next_alpha).sqrt() * prediction
            value = value * support
        return value


def normalize_mel(value: torch.Tensor, mean: torch.Tensor,
                  scale: torch.Tensor) -> torch.Tensor:
    return (value - mean.view(1, -1, 1)) / scale.view(1, -1, 1).clamp_min(1e-5)


def denormalize_mel(value: torch.Tensor, mean: torch.Tensor,
                    scale: torch.Tensor) -> torch.Tensor:
    return value * scale.view(1, -1, 1) + mean.view(1, -1, 1)

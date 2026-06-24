from __future__ import annotations

"""Conditional latent diffusion for KaraOne EEG -> EnCodec latent.

This is a *real* diffusion model (unlike the single-step `refiner.py`): it learns
p(latent | EEG) with an epsilon-prediction objective over a cosine noise schedule
and generates by iterative DDIM sampling. Because inference samples from noise and
denoises, the output keeps full variance and does NOT collapse to the conditional
mean (the failure mode of the MSE regression model). See DIFFUSION_PLAN.md.

It reuses `SpatialTemporalEEGEncoder` to produce a per-frame EEG conditioning
sequence aligned to the 150 latent frames, and is fully subject-agnostic (no
subject-id input), consistent with the rest of the bundle.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import SpatialTemporalEEGEncoder


@dataclass
class DiffusionConfig:
    latent_dim: int = 128          # EnCodec latent channels (target dim)
    target_steps: int = 150        # latent frames
    n_channels_eeg: int = 62
    d_model: int = 256             # EEG encoder width
    cond_ch: int = 128             # projected per-frame EEG conditioning channels
    hidden: int = 256              # denoiser width
    num_blocks: int = 6            # denoiser residual blocks
    kernel_size: int = 5
    time_dim: int = 128
    dropout: float = 0.1
    num_channel_experts: int = 1   # reuse the channel-MoE front-end if > 1
    encoder_blocks: int = 6
    timesteps: int = 1000
    schedule: str = "cosine"
    x0_clip: float = 8.0           # clamp predicted x0 during sampling (z-scored latent is ~unit scale)


def make_beta_schedule(timesteps: int, kind: str = "cosine") -> torch.Tensor:
    """Return betas [T]. Cosine schedule from Nichol & Dhariwal 2021."""
    if kind == "linear":
        return torch.linspace(1e-4, 0.02, timesteps, dtype=torch.float64).float()
    # cosine
    steps = timesteps + 1
    s = 0.008
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-8, 0.999).float()


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embedding, [B] -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device).float() / max(half, 1))
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class FiLMResBlock(nn.Module):
    """Conv residual block over time, FiLM-modulated by the timestep embedding."""

    def __init__(self, hidden: int, time_hidden: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.norm1 = nn.GroupNorm(num_groups=min(8, hidden), num_channels=hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(num_groups=min(8, hidden), num_channels=hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size, padding=padding)
        self.to_scale_shift = nn.Linear(time_hidden, hidden * 2)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.conv1(h)))
        scale, shift = self.to_scale_shift(t_emb).unsqueeze(-1).chunk(2, dim=1)
        x = x * (1 + scale) + shift
        x = self.dropout(x)
        x = self.act(self.norm2(self.conv2(x)))
        return h + x


class EEGLatentDiffusion(nn.Module):
    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.eeg_encoder = SpatialTemporalEEGEncoder(
            in_channels=cfg.n_channels_eeg,
            d_model=cfg.d_model,
            cond_dim=cfg.time_dim,  # unused stage cond; we feed a zero/learned bias below
            target_steps=cfg.target_steps,
            num_blocks=cfg.encoder_blocks,
            kernel_size=cfg.kernel_size,
            channel_dropout=0.15,
            dropout=cfg.dropout,
            num_channel_experts=cfg.num_channel_experts,
        )
        # The encoder's FiLM expects a cond vector; we use a single learned bias
        # (task-only, subject-agnostic). Shape [1, time_dim].
        self.enc_cond_bias = nn.Parameter(torch.zeros(1, cfg.time_dim))
        self.cond_proj = nn.Conv1d(cfg.d_model, cfg.cond_ch, kernel_size=1)

        in_ch = cfg.latent_dim + cfg.cond_ch
        self.in_proj = nn.Conv1d(in_ch, cfg.hidden, kernel_size=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_dim, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, cfg.hidden),
        )
        self.blocks = nn.ModuleList(
            [FiLMResBlock(cfg.hidden, cfg.hidden, cfg.kernel_size, cfg.dropout) for _ in range(cfg.num_blocks)]
        )
        self.out_norm = nn.GroupNorm(num_groups=min(8, cfg.hidden), num_channels=cfg.hidden)
        # Default conv init (not zero-init): we want gradients reaching the EEG
        # encoder from step 1 so the conditioning is learned promptly. Sampling
        # stability when alpha_cumprod -> 0 is handled by clamping x0 in sample().
        self.out_proj = nn.Conv1d(cfg.hidden, cfg.latent_dim, kernel_size=1)

        betas = make_beta_schedule(cfg.timesteps, cfg.schedule)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_acp", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_acp", torch.sqrt(1.0 - alphas_cumprod))

    # -- conditioning -------------------------------------------------------
    def encode_cond(self, eeg: torch.Tensor, eeg_valid_len: torch.Tensor | None = None) -> torch.Tensor:
        """EEG -> per-frame conditioning [B, cond_ch, T]. (eeg_valid_len reserved; the
        encoder already adaptive-pools to T; kept for signature parity.)"""
        cond_vec = self.enc_cond_bias.expand(eeg.shape[0], -1)
        enc, _ = self.eeg_encoder(eeg, cond_vec)  # [B, d_model, T]
        return self.cond_proj(enc)  # [B, cond_ch, T]

    # -- denoiser -----------------------------------------------------------
    def denoise(self, x_t: torch.Tensor, t: torch.Tensor, cond_seq: torch.Tensor) -> torch.Tensor:
        """Predict epsilon. x_t: [B,T,latent]; t: [B]; cond_seq: [B,cond_ch,T]."""
        x = x_t.transpose(1, 2)  # [B, latent, T]
        h = self.in_proj(torch.cat([x, cond_seq], dim=1))
        t_emb = self.time_mlp(timestep_embedding(t, self.cfg.time_dim))
        for block in self.blocks:
            h = block(h, t_emb)
        eps = self.out_proj(F.gelu(self.out_norm(h)))
        return eps.transpose(1, 2)  # [B, T, latent]

    # -- training -----------------------------------------------------------
    def loss(self, x0: torch.Tensor, eeg: torch.Tensor, eeg_valid_len: torch.Tensor | None = None) -> torch.Tensor:
        b = x0.shape[0]
        t = torch.randint(0, self.cfg.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        sqrt_acp = self.sqrt_acp[t].view(b, 1, 1)
        sqrt_omacp = self.sqrt_one_minus_acp[t].view(b, 1, 1)
        x_t = sqrt_acp * x0 + sqrt_omacp * noise
        cond_seq = self.encode_cond(eeg, eeg_valid_len)
        eps_hat = self.denoise(x_t, t, cond_seq)
        return F.mse_loss(eps_hat, noise)

    # -- sampling (DDIM, deterministic eta=0) -------------------------------
    @torch.no_grad()
    def sample(
        self,
        eeg: torch.Tensor,
        eeg_valid_len: torch.Tensor | None = None,
        steps: int = 50,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        b = eeg.shape[0]
        device = eeg.device
        cond_seq = self.encode_cond(eeg, eeg_valid_len)
        shape = (b, self.cfg.target_steps, self.cfg.latent_dim)
        x_t = torch.randn(shape, device=device, generator=generator)
        ts = torch.linspace(self.cfg.timesteps - 1, 0, steps, device=device).round().long()
        for i in range(steps):
            t = ts[i]
            t_batch = torch.full((b,), int(t), device=device, dtype=torch.long)
            eps = self.denoise(x_t, t_batch, cond_seq)
            acp_t = self.alphas_cumprod[t]
            x0_hat = (x_t - torch.sqrt(1 - acp_t) * eps) / torch.sqrt(acp_t.clamp_min(1e-8))
            x0_hat = x0_hat.clamp(-self.cfg.x0_clip, self.cfg.x0_clip)  # stabilize when acp_t -> 0
            if i < steps - 1:
                acp_prev = self.alphas_cumprod[ts[i + 1]]
                x_t = torch.sqrt(acp_prev) * x0_hat + torch.sqrt(1 - acp_prev) * eps
            else:
                x_t = x0_hat
        return x_t  # x0_hat [B, T, latent]

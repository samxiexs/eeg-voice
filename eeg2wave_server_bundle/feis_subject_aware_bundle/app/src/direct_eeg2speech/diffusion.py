from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _extract(buffer: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    values = buffer.gather(0, timesteps.long())
    return values.view(-1, *([1] * (target.ndim - 1)))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = timesteps.device
        if half == 0:
            return timesteps.float().unsqueeze(-1)
        scale = math.log(10000.0) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -scale)
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class LatentDenoiser(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        d_model: int,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        time_dim: int = 128,
    ):
        super().__init__()
        self.noisy_in = nn.Linear(latent_dim, d_model)
        self.cond_in = nn.Linear(cond_dim, d_model)
        self.coarse_in = nn.Linear(latent_dim, d_model)
        self.time_in = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.net = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, latent_dim),
        )

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        cond_seq: torch.Tensor,
        coarse_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cond_seq.shape[1] != noisy_latent.shape[1]:
            cond_seq = F.interpolate(
                cond_seq.transpose(1, 2),
                size=noisy_latent.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        x = self.noisy_in(noisy_latent) + self.cond_in(cond_seq)
        if coarse_latent is not None:
            x = x + self.coarse_in(coarse_latent)
        x = x + self.time_in(timesteps).unsqueeze(1)
        return self.out(self.net(x))


class LatentDiffusion(nn.Module):
    """DDPM training and DDIM sampling in normalized EnCodec latent space."""

    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        d_model: int,
        num_steps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        time_dim: int = 128,
        mode: str = "diffusion",
    ):
        super().__init__()
        if num_steps < 2:
            raise ValueError(f"num_steps must be >= 2, got {num_steps}")
        self.num_steps = int(num_steps)
        # "diffusion" = DDPM(eps)+DDIM; "flow" = conditional flow matching (NeuroSonic/
        # Voicebox): same denoiser reused as a velocity field on the straight path.
        self.mode = str(mode)
        betas = torch.linspace(float(beta_start), float(beta_end), self.num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.denoiser = LatentDenoiser(
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_mult=ff_mult,
            dropout=dropout,
            time_dim=time_dim,
        )

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            _extract(self.sqrt_alpha_bars, timesteps, x0) * x0
            + _extract(self.sqrt_one_minus_alpha_bars, timesteps, x0) * noise
        )

    def training_losses(
        self,
        x0: torch.Tensor,
        cond_seq: torch.Tensor,
        coarse_latent: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.mode == "flow":
            return self._flow_losses(x0, cond_seq, coarse_latent=coarse_latent)
        bsz = x0.shape[0]
        timesteps = torch.randint(0, self.num_steps, (bsz,), device=x0.device)
        noise = torch.randn_like(x0)
        noisy = self.q_sample(x0, timesteps, noise)
        pred_noise = self.denoiser(noisy, timesteps, cond_seq, coarse_latent=coarse_latent)
        eps_mse = F.mse_loss(pred_noise, noise)
        x0_pred = self.predict_x0(noisy, timesteps, pred_noise)
        x0_mse = F.smooth_l1_loss(x0_pred, x0)
        return {
            "diffusion_loss": eps_mse,
            "diffusion_eps_mse": eps_mse.detach(),
            "diffusion_x0_mse": x0_mse.detach(),
            "diffusion_t_mean": timesteps.float().mean().detach(),
        }

    # -- conditional flow matching (NeuroSonic/Voicebox) --------------------
    # Straight path x_t = (1-t)*noise + t*x0, constant target velocity u = x0 - noise.
    # Reuses the SAME denoiser as the velocity field; same return keys as the diffusion
    # branch so the training loop / logging are unchanged.
    def _flow_losses(
        self,
        x0: torch.Tensor,
        cond_seq: torch.Tensor,
        coarse_latent: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz = x0.shape[0]
        t = torch.rand(bsz, device=x0.device)
        noise = torch.randn_like(x0)
        t_ = t.view(bsz, 1, 1)
        x_t = (1.0 - t_) * noise + t_ * x0
        target_v = x0 - noise
        t_in = t * float(self.num_steps - 1)  # scale into the sinusoidal embedding range
        v_hat = self.denoiser(x_t, t_in, cond_seq, coarse_latent=coarse_latent)
        v_mse = F.mse_loss(v_hat, target_v)
        x0_pred = x_t + (1.0 - t_) * v_hat  # along the straight path x0 = x_t + (1-t)*v
        x0_mse = F.smooth_l1_loss(x0_pred, x0)
        return {
            "diffusion_loss": v_mse,
            "diffusion_eps_mse": v_mse.detach(),
            "diffusion_x0_mse": x0_mse.detach(),
            "diffusion_t_mean": t.mean().detach() * float(self.num_steps - 1),
        }

    @torch.no_grad()
    def _sample_flow(
        self,
        shape: tuple[int, int, int],
        cond_seq: torch.Tensor,
        coarse_latent: torch.Tensor | None = None,
        sample_steps: int = 16,
    ) -> torch.Tensor:
        steps = int(max(1, sample_steps))
        device = cond_seq.device
        x = torch.randn(shape, device=device, dtype=cond_seq.dtype)
        dt = 1.0 / steps
        for i in range(steps):
            t_cont = torch.full((shape[0],), i * dt, device=device)
            v = self.denoiser(x, t_cont * float(self.num_steps - 1), cond_seq, coarse_latent=coarse_latent)
            x = x + v * dt
        return x

    def predict_x0(self, xt: torch.Tensor, timesteps: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        return (
            xt - _extract(self.sqrt_one_minus_alpha_bars, timesteps, xt) * pred_noise
        ) / _extract(self.sqrt_alpha_bars, timesteps, xt).clamp_min(1e-6)

    @torch.no_grad()
    def sample_ddim(
        self,
        shape: tuple[int, int, int],
        cond_seq: torch.Tensor,
        coarse_latent: torch.Tensor | None = None,
        sample_steps: int = 24,
    ) -> torch.Tensor:
        if self.mode == "flow":
            return self._sample_flow(shape, cond_seq, coarse_latent=coarse_latent, sample_steps=sample_steps)
        steps = int(max(1, min(sample_steps, self.num_steps)))
        device = cond_seq.device
        schedule = torch.linspace(self.num_steps - 1, 0, steps, device=device).round().long()
        xt = torch.randn(shape, device=device, dtype=cond_seq.dtype)
        for idx, timestep in enumerate(schedule):
            t = torch.full((shape[0],), int(timestep.item()), device=device, dtype=torch.long)
            eps = self.denoiser(xt, t, cond_seq, coarse_latent=coarse_latent)
            x0 = self.predict_x0(xt, t, eps)
            if idx == len(schedule) - 1:
                xt = x0
                continue
            next_t = torch.full((shape[0],), int(schedule[idx + 1].item()), device=device, dtype=torch.long)
            xt = (
                _extract(self.sqrt_alpha_bars, next_t, xt) * x0
                + _extract(self.sqrt_one_minus_alpha_bars, next_t, xt) * eps
            )
        return xt

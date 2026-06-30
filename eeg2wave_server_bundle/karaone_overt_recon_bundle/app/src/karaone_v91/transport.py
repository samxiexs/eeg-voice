from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NeuroSonicFlowConfig:
    codec_dim: int
    cond_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    heun_steps: int = 32


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 2))

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.mod(time_emb).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimeConditionedGatedBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.adaln_attn = AdaLayerNorm(dim)
        self.attn_norm = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.adaln_ff = AdaLayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.gate = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate(time_emb))
        h = self.adaln_attn(x, time_emb)
        h = self.attn_norm(h)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gates[:, 0].view(-1, 1, 1) * self.dropout(attn_out)
        h = self.adaln_ff(x, time_emb)
        x = x + gates[:, 1].view(-1, 1, 1) * self.dropout(self.ff(h))
        return x


class NeuroSonicCodecFlow(nn.Module):
    """Speech-specific conditional flow in codec latent space.

    The condition is the shared EEG/audio semantic-prosody token stream.  Audio
    priors are handled as temporal codec latents with active-envelope and
    boundary continuity losses rather than image-style independent patches.
    """

    def __init__(self, cfg: NeuroSonicFlowConfig):
        super().__init__()
        self.cfg = cfg
        self.codec_in = nn.Linear(cfg.codec_dim, cfg.hidden_dim)
        self.cond_in = nn.Linear(cfg.cond_dim, cfg.hidden_dim)
        self.time_in = nn.Sequential(
            nn.Linear(1, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [TimeConditionedGatedBlock(cfg.hidden_dim, cfg.num_heads, cfg.dropout) for _ in range(cfg.num_layers)]
        )
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
        cond = resize_sequence(condition, x_t.shape[1])
        time_emb = self.time_in(t.float())
        h = self.codec_in(x_t) + self.cond_in(cond) + time_emb.unsqueeze(1)
        for block in self.blocks:
            h = block(h, time_emb)
        return self.out(h)

    def training_loss(
        self,
        target_codec: torch.Tensor,
        condition: torch.Tensor,
        *,
        teacher_condition: torch.Tensor | None = None,
        teacher_ratio: float = 0.0,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if teacher_condition is not None and float(teacher_ratio) > 0.0:
            ratio = min(max(float(teacher_ratio), 0.0), 1.0)
            condition = (1.0 - ratio) * resize_sequence(condition, target_codec.shape[1]) + ratio * resize_sequence(
                teacher_condition, target_codec.shape[1]
            )
        if noise is None:
            noise = torch.randn_like(target_codec)
        t = torch.rand(target_codec.shape[0], device=target_codec.device, dtype=target_codec.dtype).clamp(1e-4, 1.0 - 1e-4)
        x_t = (1.0 - t[:, None, None]) * noise + t[:, None, None] * target_codec
        target_velocity = target_codec - noise
        pred_velocity = self.forward(x_t, t, condition)
        x1_pred = x_t + (1.0 - t[:, None, None]) * pred_velocity
        return {
            "flow_loss": F.mse_loss(pred_velocity, target_velocity),
            "codec_consistency_loss": F.smooth_l1_loss(x1_pred, target_codec),
            "boundary_continuity_loss": boundary_continuity_loss(x1_pred),
            "pred_velocity": pred_velocity,
            "x_t": x_t,
            "t": t,
        }

    @torch.no_grad()
    def sample_heun(self, condition: torch.Tensor, *, steps: int | None = None, codec_steps: int, codec_dim: int | None = None) -> torch.Tensor:
        steps = max(1, int(steps or self.cfg.heun_steps))
        codec_dim = int(codec_dim or self.cfg.codec_dim)
        x = torch.randn(condition.shape[0], int(codec_steps), codec_dim, device=condition.device, dtype=condition.dtype)
        dt = 1.0 / float(steps)
        for step in range(steps):
            t0 = torch.full((condition.shape[0],), float(step) / float(steps), device=condition.device, dtype=condition.dtype)
            t1 = torch.full((condition.shape[0],), float(step + 1) / float(steps), device=condition.device, dtype=condition.dtype)
            v0 = self.forward(x, t0, condition)
            x_euler = x + dt * v0
            v1 = self.forward(x_euler, t1, condition)
            x = x + 0.5 * dt * (v0 + v1)
        return x


def boundary_continuity_loss(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] < 3:
        return x.new_tensor(0.0)
    delta = x[:, 1:] - x[:, :-1]
    return delta.abs().mean()


def resize_sequence(x: torch.Tensor, steps: int) -> torch.Tensor:
    if x.shape[1] == int(steps):
        return x
    return F.interpolate(x.transpose(1, 2), size=int(steps), mode="linear", align_corners=False).transpose(1, 2)

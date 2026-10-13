"""One conditional flow: linear noise-to-Mel path and explicit ODE sampling."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def sinusoidal(value, width):
    frequencies = torch.exp(-math.log(10000) * torch.arange(width // 2, device=value.device, dtype=torch.float32) / (width // 2))
    phase = value.float()[..., None] * frequencies
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


class FlowBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(width, elementwise_affine=False) for _ in range(3)])
        self.self_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.modulation = nn.Sequential(nn.Linear(width, width // 2), nn.SiLU(), nn.Linear(width // 2, 9 * width))
        nn.init.zeros_(self.modulation[-1].weight); nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x, condition, time):
        shift1, scale1, gate1, shift2, scale2, gate2, shift3, scale3, gate3 = self.modulation(F.silu(time)).chunk(9, dim=-1)
        def mod(h, norm, shift, scale): return norm(h) * (1 + scale[:, None]) + shift[:, None]
        h = mod(x, self.norms[0], shift1, scale1)
        x = x + gate1[:, None] * self.self_attention(h, h, h, need_weights=False)[0]
        h = mod(x, self.norms[1], shift2, scale2)
        x = x + gate2[:, None] * self.cross_attention(h, condition, condition, need_weights=False)[0]
        h = mod(x, self.norms[2], shift3, scale3)
        return x + gate3[:, None] * self.ff(h)


class ConditionalFlow(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        f = cfg["flow"]
        self.width, self.mel_bins = f["width"], cfg["audio"]["mel_bins"]
        self.mel_hop_seconds = cfg["audio"]["hop_length"] / cfg["audio"]["sample_rate"]
        self.teacher_first_center = (400 - 1) / 2 / 16000
        self.mel_input = nn.Linear(self.mel_bins, self.width)
        self.condition_input = nn.Linear(cfg["audio"]["teacher_dim"], self.width)
        self.time = nn.Sequential(nn.Linear(self.width, self.width), nn.SiLU(), nn.Linear(self.width, self.width))
        self.blocks = nn.ModuleList([FlowBlock(self.width, f["heads"]) for _ in range(f["layers"])])
        self.norm = nn.LayerNorm(self.width)
        self.output = nn.Linear(self.width, self.mel_bins)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, mel, tau, condition):
        if tau.ndim == 0: tau = tau.expand(len(mel))
        x = self.mel_input(mel.transpose(1, 2))
        mel_position = torch.arange(x.shape[1], device=x.device) * self.mel_hop_seconds * 50
        condition_position = torch.arange(condition.shape[1], device=x.device) + self.teacher_first_center * 50
        x = x + sinusoidal(mel_position, self.width)[None].to(x.dtype)
        cond = self.condition_input(condition) + sinusoidal(condition_position, self.width)[None].to(x.dtype)
        time = self.time(sinusoidal(tau * 1000, self.width))
        for block in self.blocks: x = block(x, cond, time)
        return self.output(self.norm(x)).transpose(1, 2)

    def matching_loss(self, target, condition, null_fraction=.1):
        noise = torch.randn_like(target)
        tau = torch.rand(len(target), device=target.device)
        interpolated = (1 - tau[:, None, None]) * noise + tau[:, None, None] * target
        if null_fraction:
            keep = (torch.rand(len(target), device=target.device) >= null_fraction).to(condition.dtype)
            condition = condition * keep[:, None, None]
        return F.mse_loss(self(interpolated, tau, condition), target - noise)

    def rollout(self, noise, condition, steps=32, method="heun", checkpoint_grad=False):
        if steps < 1 or method not in {"heun", "euler"}: raise ValueError("Invalid integration protocol")
        x, dt = noise.float(), 1.0 / steps
        for step in range(steps):
            tau = torch.full((len(x),), step * dt, device=x.device)
            v = checkpoint(self.forward, x, tau, condition, use_reentrant=False) if checkpoint_grad else self(x, tau, condition)
            candidate = x + dt * v
            if method == "heun":
                next_tau = torch.full_like(tau, (step + 1) * dt)
                next_v = checkpoint(self.forward, candidate, next_tau, condition, use_reentrant=False) if checkpoint_grad else self(candidate, next_tau, condition)
                x = x + dt * (v + next_v) / 2
            else: x = candidate
        return x

    @torch.no_grad()
    def sample(self, condition, frames=376, seed=0, steps=32):
        generator = torch.Generator(device=condition.device).manual_seed(seed)
        noise = torch.randn((len(condition), self.mel_bins, frames), generator=generator, device=condition.device)
        return self.rollout(noise, condition, steps=steps, method="heun")

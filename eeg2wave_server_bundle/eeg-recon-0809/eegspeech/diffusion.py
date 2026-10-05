"""Conditional diffusion over log-mel spectrograms: v-prediction, cosine schedule, DDIM sampling with
classifier-free guidance (ported from this project's earlier generative decoder).

The network is a stack of adaLN-zero residual blocks (dilated depthwise convolution, self-attention
in every other block, MLP) over mel frames.  The condition is one vector per example (a point of
the CLIP speech space: the EEG prediction or a speech embedding of the target).  It is embedded,
added to the timestep embedding that modulates every block (class conditioning as in DiT) and also
added to every input frame, so the content is present from the first layer.  A learned null
condition, used for a fraction of the training examples, gives classifier-free guidance.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal(timestep, dimension):
    half = dimension // 2
    freqs = torch.exp(-math.log(10000.) * torch.arange(half, device=timestep.device).float() / half)
    angles = timestep.float()[:, None] * freqs[None]
    return torch.cat([angles.sin(), angles.cos()], 1)


class Block(nn.Module):
    """adaLN-zero residual block: dilated depthwise conv, optional self-attention, MLP."""

    def __init__(self, hidden, dilation, heads, attention, dropout):
        super().__init__()
        self.norm1, self.film1 = nn.LayerNorm(hidden, elementwise_affine=False), nn.Linear(hidden, 3 * hidden)
        self.depthwise = nn.Conv1d(hidden, hidden, 5, padding=2 * dilation, dilation=dilation, groups=hidden)
        self.pointwise = nn.Conv1d(hidden, hidden, 1)
        self.attention = attention
        if attention:
            self.norm2, self.film2 = nn.LayerNorm(hidden, elementwise_affine=False), nn.Linear(hidden, 3 * hidden)
            self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm3, self.film3 = nn.LayerNorm(hidden, elementwise_affine=False), nn.Linear(hidden, 3 * hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * hidden, hidden))
        for film in (self.film1, self.film3) + ((self.film2,) if attention else ()):
            nn.init.zeros_(film.weight); nn.init.zeros_(film.bias)

    @staticmethod
    def modulate(norm, film, h, e):
        scale, shift, gate = film(e)[:, None, :].chunk(3, -1)
        return norm(h) * (1 + scale) + shift, gate

    def forward(self, h, e):
        u, gate = self.modulate(self.norm1, self.film1, h, e)
        h = h + gate * self.pointwise(F.gelu(self.depthwise(u.transpose(1, 2)))).transpose(1, 2)
        if self.attention:
            u, gate = self.modulate(self.norm2, self.film2, h, e)
            h = h + gate * self.attn(u, u, u, need_weights=False)[0]
        u, gate = self.modulate(self.norm3, self.film3, h, e)
        return h + gate * self.mlp(u)


class MelDiffusion(nn.Module):
    """p(mel | condition vector) over normalised (bins, frames) log-mels."""

    def __init__(self, condition_dim, bins=80, frames=96, hidden=192, blocks=8, heads=4, dropout=.1, timesteps=1000):
        super().__init__()
        self.config = dict(condition_dim=condition_dim, bins=bins, frames=frames, hidden=hidden, blocks=blocks,
                           heads=heads, dropout=dropout, timesteps=timesteps)
        self.bins, self.frames, self.hidden, self.timesteps = bins, frames, hidden, timesteps
        self.input = nn.Conv1d(bins, hidden, 3, padding=1)
        self.position = nn.Parameter(torch.randn(1, frames, hidden) * .02)
        self.time = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.SiLU(), nn.Linear(2 * hidden, hidden))
        self.condition = nn.Sequential(nn.Linear(condition_dim, 2 * hidden), nn.SiLU(), nn.Linear(2 * hidden, hidden))
        self.null = nn.Parameter(torch.zeros(hidden))
        self.inject = nn.Linear(hidden, hidden)                  # condition added to every input frame
        nn.init.zeros_(self.inject.weight); nn.init.zeros_(self.inject.bias)
        dilations = (1, 2, 4, 8)
        self.blocks = nn.ModuleList([Block(hidden, dilations[i % 4], heads, i % 2 == 1, dropout) for i in range(blocks)])
        self.output_norm = nn.LayerNorm(hidden)
        self.output = nn.Conv1d(hidden, bins, 3, padding=1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        steps = torch.arange(timesteps + 1).double() / timesteps
        alpha_bar = torch.cos((steps + .008) / 1.008 * math.pi / 2) ** 2
        self.register_buffer('alpha_bar', (alpha_bar / alpha_bar[0]).clamp(1e-5, 1.).float())   # index t in [0, T]

    def embed(self, c, null):
        """Condition embedding; rows with ``null`` True get the learned null embedding."""
        e = self.condition(c)
        return torch.where(null[:, None], self.null.expand_as(e), e)

    def forward(self, x, t, e):
        """x (B, bins, frames) noisy normalised mel, t (B,) step in [1, T], e (B, hidden) condition embedding."""
        h = self.input(x).transpose(1, 2) + self.position + self.inject(e)[:, None, :]
        e = e + self.time(sinusoidal(t, self.hidden))
        for block in self.blocks:
            h = block(h, e)
        return self.output(self.output_norm(h).transpose(1, 2))

    def loss(self, x0, c, null):
        t = torch.randint(1, self.timesteps + 1, (len(x0),), device=x0.device)
        ab = self.alpha_bar[t][:, None, None]
        noise = torch.randn_like(x0)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        v = ab.sqrt() * noise - (1 - ab).sqrt() * x0
        return F.mse_loss(self(xt, t, self.embed(c, null)), v)

    @torch.no_grad()
    def sample(self, c, *, null=None, steps=50, guidance=2., generator=None, clamp=(-3., 4.)):
        """DDIM (eta = 0) with classifier-free guidance on v; ``null`` rows are sampled unconditionally."""
        batch, device = c.shape[0], c.device
        null = torch.zeros(batch, dtype=torch.bool, device=device) if null is None else null
        e_c = self.embed(c, null)
        e_u = self.null.expand(batch, -1)
        x = torch.randn(batch, self.bins, self.frames, generator=generator).to(device)
        schedule = torch.linspace(self.timesteps, 0, steps + 1).round().long().tolist()
        for t, t_next in zip(schedule[:-1], schedule[1:]):
            tt = torch.full((batch,), t, device=device)
            if guidance == 1.:
                v = self(x, tt, e_c)
            else:
                v_c, v_u = self(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([e_c, e_u])).chunk(2)
                v = v_u + guidance * (v_c - v_u)
            ab, ab_next = self.alpha_bar[t], self.alpha_bar[t_next]
            x0 = (ab.sqrt() * x - (1 - ab).sqrt() * v).clamp(*clamp)
            eps = (1 - ab).sqrt() * x + ab.sqrt() * v
            x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
        return x


class EMA:
    """Exponential moving average of a model's weights (the copy used for sampling)."""

    def __init__(self, model, decay=.999):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.model.parameters(), model.parameters()):
            s.lerp_(p.detach(), 1 - self.decay)
        for s, b in zip(self.model.buffers(), model.buffers()):
            s.copy_(b)


class MelScaler:
    """Global affine normalisation of log-mels floored at ``floor`` (log10 units)."""

    def __init__(self, mean, scale, floor=-7.):
        self.mean, self.scale, self.floor = float(mean), float(scale), float(floor)

    @classmethod
    def fit(cls, mel, floor=-7.):
        value = torch.as_tensor(mel).clamp_min(floor)
        return cls(value.mean(), value.std(), floor)

    def encode(self, mel):
        return (torch.as_tensor(mel).clamp_min(self.floor) - self.mean) / self.scale

    def decode(self, x, silence=-10., quiet=-6.5):
        """Back to log10 mel; frames whose mean stays near the floor become digital silence for the vocoder."""
        mel = (x * self.scale + self.mean).clamp_min(self.floor)
        return torch.where(mel.mean(-2, keepdim=True) < quiet, torch.full_like(mel, silence), mel)

    def state(self):
        return dict(mean=self.mean, scale=self.scale, floor=self.floor)

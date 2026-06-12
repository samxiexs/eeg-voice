"""Spatial-temporal EEG encoder with FiLM subject conditioning (v3).

Design (see NEW_DESIGN_eeg2speech_v3.md, sec 3.1/3.2):

    EEG [B, C, L]
      -> spatial mixing conv (1x1 over electrodes)
      -> N residual dilated temporal blocks, each FiLM-modulated by a
         per-subject embedding (enables cross-subject pooling)
      -> adaptive pool to `target_steps` frames (aligned to EnCodec 75 Hz)
      -> [B, d_model, target_steps]

The FiLM conditioning is the key change vs the old alignment model, which
only concatenated a subject vector in a side head. Here the subject embedding
modulates the trunk, so a single pooled model can specialise per subject while
sharing 20x more data.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """Feature-wise linear modulation: h <- gamma(s) * h + beta(s)."""

    def __init__(self, cond_dim: int, num_features: int):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, num_features)
        self.to_beta = nn.Linear(cond_dim, num_features)
        # Start as identity modulation so an untrained subject vector does no harm.
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # h: [B, C, T]; cond: [B, cond_dim]
        gamma = self.to_gamma(cond).unsqueeze(-1)
        beta = self.to_beta(cond).unsqueeze(-1)
        return gamma * h + beta


class ResidualTemporalBlock(nn.Module):
    """Dilated 1D conv block with optional stride for downsampling + FiLM."""

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        kernel_size: int = 5,
        dilation: int = 1,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.norm1 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.film = FiLM(cond_dim, channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        # Match temporal length of the residual path when stride > 1.
        self.down = (
            nn.Conv1d(channels, channels, kernel_size=1, stride=stride) if stride > 1 else nn.Identity()
        )

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = self.down(h)
        x = self.act(self.norm1(self.conv1(h)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        x = self.film(x, cond)
        # Align lengths defensively (padding can differ by 1 with even kernels).
        if x.shape[-1] != residual.shape[-1]:
            x = F.interpolate(x, size=residual.shape[-1], mode="linear", align_corners=False)
        return self.act(x + residual)


def _maybe_channel_dropout(eeg: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Drop whole electrodes during training as a robustness regulariser."""
    if not training or p <= 0.0:
        return eeg
    b, c, _ = eeg.shape
    keep = (torch.rand(b, c, 1, device=eeg.device) > p).float()
    keep = torch.where(keep.sum(dim=1, keepdim=True) > 0, keep, torch.ones_like(keep))
    return eeg * keep


class SpatialAdapter(nn.Module):
    """Per-dataset front end: electrodes -> d_model via 1x1 conv + FiLM.

    Each dataset (14ch FEIS, 62ch KaraOne, ...) owns one of these; the trunk
    that follows is shared, which is what places every dataset in the same
    representation space.
    """

    def __init__(self, in_channels: int, d_model: int, cond_dim: int, channel_dropout: float = 0.1):
        super().__init__()
        self.channel_dropout_p = float(channel_dropout)
        self.spatial = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=1),
            nn.GroupNorm(num_groups=min(8, d_model), num_channels=d_model),
            nn.GELU(),
        )
        self.spatial_film = FiLM(cond_dim, d_model)

    def forward(self, eeg: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = _maybe_channel_dropout(eeg, self.channel_dropout_p, self.training)
        x = self.spatial(x)
        return self.spatial_film(x, cond)


class TemporalTrunk(nn.Module):
    """Shared temporal encoder operating on d_model channels."""

    def __init__(
        self,
        d_model: int = 256,
        cond_dim: int = 64,
        target_steps: int = 75,
        num_blocks: int = 5,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.target_steps = int(target_steps)
        strides = [2, 2, 2, 2] + [1] * max(0, num_blocks - 4)
        dilations = [1, 1, 2, 4] + [8] * max(0, num_blocks - 4)
        self.blocks = nn.ModuleList(
            ResidualTemporalBlock(
                channels=d_model,
                cond_dim=cond_dim,
                kernel_size=kernel_size,
                dilation=dilations[i],
                stride=strides[i],
                dropout=dropout,
            )
            for i in range(num_blocks)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, cond)
        if x.shape[-1] != self.target_steps:
            x = F.adaptive_avg_pool1d(x, self.target_steps)
        return x


class SpatialTemporalEEGEncoder(nn.Module):
    """Single-dataset encoder = SpatialAdapter + TemporalTrunk (v3 compatible)."""

    def __init__(
        self,
        in_channels: int = 14,
        d_model: int = 256,
        cond_dim: int = 64,
        target_steps: int = 75,
        num_blocks: int = 5,
        kernel_size: int = 5,
        channel_dropout: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.adapter = SpatialAdapter(in_channels, d_model, cond_dim, channel_dropout)
        self.trunk = TemporalTrunk(d_model, cond_dim, target_steps, num_blocks, kernel_size, dropout)

    def forward(self, eeg: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """eeg: [B, C, L]; cond: [B, cond_dim] -> [B, d_model, target_steps]."""
        return self.trunk(self.adapter(eeg, cond), cond)

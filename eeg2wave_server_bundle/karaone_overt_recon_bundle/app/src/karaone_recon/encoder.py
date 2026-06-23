from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, num_features: int):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, num_features)
        self.to_beta = nn.Linear(cond_dim, num_features)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(cond).unsqueeze(-1)
        beta = self.to_beta(cond).unsqueeze(-1)
        return gamma * h + beta


class ResidualTemporalBlock(nn.Module):
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
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.film = FiLM(cond_dim, channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.down = nn.Conv1d(channels, channels, kernel_size=1, stride=stride) if stride > 1 else nn.Identity()

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = self.down(h)
        x = self.act(self.norm1(self.conv1(h)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        x = self.film(x, cond)
        if x.shape[-1] != residual.shape[-1]:
            x = F.interpolate(x, size=residual.shape[-1], mode="linear", align_corners=False)
        return self.act(x + residual)


def _channel_dropout(eeg: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    if not training or p <= 0.0:
        return eeg
    batch, channels, _ = eeg.shape
    keep = (torch.rand(batch, channels, 1, device=eeg.device) > p).float()
    keep = torch.where(keep.sum(dim=1, keepdim=True) > 0, keep, torch.ones_like(keep))
    return eeg * keep


class SpatialTemporalEEGEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 62,
        d_model: int = 256,
        cond_dim: int = 64,
        target_steps: int = 150,
        num_blocks: int = 6,
        kernel_size: int = 5,
        channel_dropout: float = 0.15,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.channel_dropout_p = float(channel_dropout)
        self.spatial = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=1),
            nn.GroupNorm(num_groups=min(8, d_model), num_channels=d_model),
            nn.GELU(),
        )
        self.spatial_film = FiLM(cond_dim, d_model)
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
        self.target_steps = int(target_steps)

    def forward(self, eeg: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = _channel_dropout(eeg, self.channel_dropout_p, self.training)
        x = self.spatial(x)
        x = self.spatial_film(x, cond)
        for block in self.blocks:
            x = block(x, cond)
        if x.shape[-1] != self.target_steps:
            x = F.adaptive_avg_pool1d(x, self.target_steps)
        return x


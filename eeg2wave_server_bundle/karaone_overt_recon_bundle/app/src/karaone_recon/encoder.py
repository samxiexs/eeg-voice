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


class ChannelMoEFrontend(nn.Module):
    """Channel-selecting + channel-clustering MoE front-end for raw EEG.

    Motivation: EEG has many channels and not all are informative, and several
    channels carry redundant/correlated signal. This module replaces a plain
    ``Conv1d(C -> d_model, kernel=1)`` spatial mixer with a mixture-of-experts
    that operates *on the channel axis*, where channel identity still exists
    (before any spatial collapse):

    * **Selection (gate).** A per-channel, input-dependent gate ``g in [0,1]``
      learns which channels are useful for the current trial. It generalizes the
      random ``_channel_dropout`` into a learned, data-driven filter.
    * **Clustering (soft assignment).** Each channel gets a small descriptor from
      its temporal statistics; channels are softly assigned to ``E`` expert
      clusters by similarity to learned expert prototypes, so channels carrying
      similar signal land in the same expert.
    * **Routing (experts).** Each expert mixes only its (gated, assigned) cluster
      of channels into a ``d_model // E`` sub-embedding; expert outputs are
      concatenated to form the ``d_model`` spatial embedding.

    ``forward`` returns the spatial embedding plus an ``aux`` dict (gate,
    assignment, and a load-balance term) for regularization and analysis.
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        num_experts: int = 4,
        desc_dim: int = 16,
        cluster_temp: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % num_experts != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_experts ({num_experts})")
        self.in_channels = int(in_channels)
        self.num_experts = int(num_experts)
        self.cluster_temp = float(cluster_temp)
        per_expert = d_model // num_experts

        # Per-channel temporal descriptor: depthwise conv keeps channels separate,
        # then we summarize each channel by (mean, std) over time -> [B, C, 2].
        self.desc_conv = nn.Conv1d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.gate_mlp = nn.Sequential(nn.Linear(2, desc_dim), nn.GELU(), nn.Linear(desc_dim, 1))
        self.channel_bias = nn.Parameter(torch.zeros(in_channels))  # learned static channel importance
        self.channel_embed = nn.Sequential(nn.Linear(2, desc_dim), nn.GELU(), nn.Linear(desc_dim, desc_dim))
        self.expert_query = nn.Parameter(torch.randn(num_experts, desc_dim) * 0.02)
        self.experts = nn.ModuleList(
            [nn.Conv1d(in_channels, per_expert, kernel_size=1) for _ in range(num_experts)]
        )
        self.proj = nn.Sequential(
            nn.GroupNorm(num_groups=min(8, d_model), num_channels=d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, eeg: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # eeg: [B, C, T]
        desc = self.desc_conv(eeg)
        stats = torch.stack([desc.mean(dim=-1), desc.std(dim=-1)], dim=-1)  # [B, C, 2]

        gate = torch.sigmoid(self.gate_mlp(stats).squeeze(-1) + self.channel_bias)  # [B, C]

        emb = F.normalize(self.channel_embed(stats), dim=-1)  # [B, C, desc_dim]
        query = F.normalize(self.expert_query, dim=-1)  # [E, desc_dim]
        assign = torch.softmax((emb @ query.t()) / max(self.cluster_temp, 1e-4), dim=-1)  # [B, C, E]

        weight = gate.unsqueeze(-1) * assign  # [B, C, E]
        expert_outputs = [expert(eeg * weight[:, :, e : e + 1]) for e, expert in enumerate(self.experts)]
        h = self.proj(torch.cat(expert_outputs, dim=1))  # [B, d_model, T]

        # Load balance: discourage dead experts so clusters stay distinct.
        importance = assign.mean(dim=(0, 1))  # [E]
        uniform = torch.full_like(importance, 1.0 / self.num_experts)
        balance = F.mse_loss(importance, uniform)
        return h, {"channel_gate": gate, "channel_assign": assign, "channel_balance": balance}


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
        num_channel_experts: int = 1,
    ):
        super().__init__()
        self.channel_dropout_p = float(channel_dropout)
        self.num_channel_experts = int(num_channel_experts)
        if self.num_channel_experts > 1:
            # Channel-selecting + channel-clustering MoE spatial front-end.
            self.spatial = ChannelMoEFrontend(
                in_channels=in_channels,
                d_model=d_model,
                num_experts=self.num_channel_experts,
                dropout=dropout,
            )
        else:
            # Plain spatial mixer (baseline): collapses channels in one 1x1 conv.
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

    def forward(
        self, eeg: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = _channel_dropout(eeg, self.channel_dropout_p, self.training)
        if self.num_channel_experts > 1:
            x, aux = self.spatial(x)
        else:
            x, aux = self.spatial(x), {}
        x = self.spatial_film(x, cond)
        for block in self.blocks:
            x = block(x, cond)
        if x.shape[-1] != self.target_steps:
            x = F.adaptive_avg_pool1d(x, self.target_steps)
        return x, aux


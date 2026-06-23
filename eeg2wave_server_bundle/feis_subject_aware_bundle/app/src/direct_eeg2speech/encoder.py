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


def _maybe_channel_dropout(eeg: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    if not training or p <= 0.0:
        return eeg
    b, c, _ = eeg.shape
    keep = (torch.rand(b, c, 1, device=eeg.device) > p).float()
    keep = torch.where(keep.sum(dim=1, keepdim=True) > 0, keep, torch.ones_like(keep))
    return eeg * keep


def _channel_stats(eeg: torch.Tensor) -> torch.Tensor:
    if eeg.ndim != 3:
        raise ValueError(f"Expected EEG tensor [B, C, T], got {tuple(eeg.shape)}")
    mean = eeg.mean(dim=-1)
    std = eeg.std(dim=-1, unbiased=False)
    abs_mean = eeg.abs().mean(dim=-1)
    if eeg.shape[-1] > 1:
        diff = eeg[..., 1:] - eeg[..., :-1]
        diff_std = diff.std(dim=-1, unbiased=False)
    else:
        diff_std = torch.zeros_like(std)
    stats = torch.stack([mean, std, abs_mean, diff_std], dim=-1)
    return torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)


def _topk_softmax(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k >= logits.shape[-1]:
        return F.softmax(logits, dim=-1)
    vals, idx = logits.topk(k=max(int(top_k), 1), dim=-1)
    masked = torch.full_like(logits, -1e9)
    masked.scatter_(-1, idx, vals)
    return F.softmax(masked, dim=-1)


def _cluster_cohesion_loss(eeg: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
    if eeg.shape[1] < 2 or eeg.shape[-1] < 2:
        return eeg.new_tensor(0.0)
    centered = eeg - eeg.mean(dim=-1, keepdim=True)
    normed = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    similarity = torch.relu(torch.bmm(normed, normed.transpose(1, 2)))
    coassign = torch.bmm(route, route.transpose(1, 2)).clamp(0.0, 1.0)
    mask = ~torch.eye(eeg.shape[1], dtype=torch.bool, device=eeg.device)
    weighted_miss = ((1.0 - coassign) * similarity)[:, mask]
    denom = similarity[:, mask].sum().clamp_min(1e-6)
    return weighted_miss.sum() / denom


class SpatialAdapter(nn.Module):
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


class ChannelClusterMoEAdapter(nn.Module):
    """Route EEG channels through learnable spatial experts before temporal encoding."""

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        cond_dim: int,
        channel_dropout: float = 0.1,
        num_experts: int = 4,
        top_k: int = 2,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        self.channel_dropout_p = float(channel_dropout)
        self.num_experts = int(num_experts)
        self.top_k = max(1, min(int(top_k), self.num_experts))
        router_dim = 4 + int(cond_dim)
        hidden = max(32, min(128, d_model // 2))
        self.router = nn.Sequential(
            nn.LayerNorm(router_dim),
            nn.Linear(router_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.num_experts),
        )
        self.channel_gate = nn.Sequential(
            nn.LayerNorm(router_dim),
            nn.Linear(router_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(in_channels, d_model, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=min(8, d_model), num_channels=d_model),
                nn.GELU(),
            )
            for _ in range(self.num_experts)
        )
        self.spatial_film = FiLM(cond_dim, d_model)

    def forward(self, eeg: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = _maybe_channel_dropout(eeg, self.channel_dropout_p, self.training)
        b, c, _ = x.shape
        stats = _channel_stats(x)
        cond_by_channel = cond.unsqueeze(1).expand(b, c, cond.shape[-1])
        router_input = torch.cat([stats, cond_by_channel], dim=-1)
        route_logits = self.router(router_input)
        route = _topk_softmax(route_logits, self.top_k)
        channel_gate = torch.sigmoid(self.channel_gate(router_input))

        mixed = None
        for expert_idx, expert in enumerate(self.experts):
            weight = route[..., expert_idx : expert_idx + 1] * channel_gate
            expert_out = expert(x * weight)
            mixed = expert_out if mixed is None else mixed + expert_out
        assert mixed is not None
        out = self.spatial_film(mixed, cond)

        usage = route.mean(dim=(0, 1))
        uniform = route.new_full((self.num_experts,), 1.0 / self.num_experts)
        route_entropy = -(route * route.clamp_min(1e-8).log()).sum(dim=-1).mean()
        route_entropy = route_entropy / torch.log(route.new_tensor(float(self.num_experts))).clamp_min(1e-6)
        aux = {
            "moe_load_balance": ((usage - uniform) ** 2).sum() * self.num_experts,
            "moe_channel_sparsity": channel_gate.mean(),
            "moe_route_entropy": route_entropy,
            "moe_cluster_cohesion": _cluster_cohesion_loss(x, route),
            "moe_channel_gate_mean": channel_gate.mean().detach(),
            "moe_usage_min": usage.min().detach(),
            "moe_usage_max": usage.max().detach(),
            "moe_active_channels": (channel_gate.squeeze(-1) > 0.5).float().sum(dim=1).mean().detach(),
        }
        return out, aux


class TemporalTrunk(nn.Module):
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
        use_channel_moe: bool = False,
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
    ):
        super().__init__()
        if use_channel_moe:
            self.adapter = ChannelClusterMoEAdapter(
                in_channels,
                d_model,
                cond_dim,
                channel_dropout=channel_dropout,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
            )
        else:
            self.adapter = SpatialAdapter(in_channels, d_model, cond_dim, channel_dropout)
        self.trunk = TemporalTrunk(d_model, cond_dim, target_steps, num_blocks, kernel_size, dropout)

    def forward(self, eeg: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        adapted = self.adapter(eeg, cond)
        if isinstance(adapted, tuple):
            x, aux = adapted
        else:
            x, aux = adapted, {}
        return self.trunk(x, cond), aux

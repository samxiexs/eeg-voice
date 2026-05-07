"""Reusable BrainOmni-style blocks for EEG voice tokenization.

The blocks keep the same ideas as BrainOmni's tokenizer stack:

- sensor position/type embedding
- SEANet-like temporal encoder
- latent neural queries with backward solution
- residual vector quantization
- forward solution decoder

The implementation stays dependency-light and uses only PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm used by BrainOmni-style attention and feed-forward blocks."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class FeedForward(nn.Module):
    """Transformer feed-forward block."""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttentionBlock(nn.Module):
    """Pre-norm self-attention block used as a temporal context mixer."""

    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm_attn = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_ff = RMSNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm_attn(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        return x + self.ff(self.norm_ff(x))


class SensorEmbedding(nn.Module):
    """Embed electrode coordinates and channel type.

    `sensor_pos` may contain 3-D coordinates or a 6-D position+direction vector.
    Shorter inputs are zero-padded; longer inputs are truncated.
    """

    def __init__(self, dim: int, dropout: float, pos_dim: int = 6, n_sensor_types: int = 3):
        super().__init__()
        self.pos_dim = pos_dim
        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_dim, dim // 2),
            nn.SELU(),
            nn.Linear(dim // 2, dim),
        )
        self.type_embedding = nn.Embedding(n_sensor_types, dim)
        self.aggregate_mlp = FeedForward(dim, dropout=0.0)
        self.norm = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def _prepare_pos(self, sensor_pos: torch.Tensor) -> torch.Tensor:
        sensor_pos = sensor_pos.float()
        if sensor_pos.shape[-1] < self.pos_dim:
            sensor_pos = F.pad(sensor_pos, (0, self.pos_dim - sensor_pos.shape[-1]))
        elif sensor_pos.shape[-1] > self.pos_dim:
            sensor_pos = sensor_pos[..., : self.pos_dim]
        return sensor_pos

    def forward(
        self,
        sensor_pos: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        sensor_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sensor_type is None:
            if channel_mask is None:
                sensor_type = torch.ones(sensor_pos.shape[:2], dtype=torch.long, device=sensor_pos.device)
            else:
                sensor_type = channel_mask.long()
        x = self.pos_mlp(self._prepare_pos(sensor_pos)) + self.type_embedding(sensor_type.long())
        x = x + self.aggregate_mlp(x)
        return self.dropout(self.norm(x))


class SEANetResidualBlock(nn.Module):
    """Small SEANet-style residual block with dilated temporal convolutions."""

    def __init__(self, dim: int, dilation: int, dropout: float):
        super().__init__()
        hidden = max(dim // 2, 8)
        self.block = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(dim, hidden, kernel_size=3, dilation=dilation, padding=dilation),
            nn.GroupNorm(1, hidden),
            nn.ELU(),
            nn.Conv1d(hidden, dim, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SEANetDownsampleBlock(nn.Module):
    """Residual stack followed by strided convolution."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, residual_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.residual = nn.Sequential(
            *[SEANetResidualBlock(out_channels, dilation=2**idx, dropout=dropout) for idx in range(residual_layers)]
        )
        self.down = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=2 * stride, stride=stride, padding=max(stride // 2, 0)),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.residual(x)
        return self.down(x)


class TemporalEncoder(nn.Module):
    """SEANet-like temporal encoder applied to each channel/window.

    Input:
        eeg_windows: `[B, C, N, L]`
    Output:
        channel_features: `[B, C, N, W, D]`
    """

    def __init__(
        self,
        dim: int,
        hidden: int,
        downsample_rates: tuple[int, ...],
        dropout: float,
        residual_layers: int = 2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        cur = hidden
        for stride in downsample_rates:
            layers.append(SEANetDownsampleBlock(in_ch, cur, stride, residual_layers, dropout))
            in_ch = cur
            cur = min(dim, cur * 2)
        layers.append(nn.Sequential(nn.ELU(), nn.Conv1d(in_ch, dim, kernel_size=7, padding=3)))
        self.net = nn.Sequential(*layers)

    def forward(self, eeg_windows: torch.Tensor) -> torch.Tensor:
        batch, channels, windows, length = eeg_windows.shape
        x = eeg_windows.reshape(batch * channels * windows, 1, length)
        x = self.net(x)
        return x.reshape(batch, channels, windows, x.shape[-2], x.shape[-1]).permute(0, 1, 2, 4, 3)


class BackwardSolution(nn.Module):
    """Map channel-level sensor activity into latent neural queries."""

    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        head_dim = dim // self.n_heads
        return x.reshape(batch, tokens, self.n_heads, head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, tokens, head_dim = x.shape
        return x.transpose(1, 2).reshape(batch, tokens, heads * head_dim)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        q = self._split_heads(queries)
        k = self._split_heads(keys)
        v = self._split_heads(self.v_proj(values))
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.out_proj(self._merge_heads(out))


class ForwardSolution(nn.Module):
    """Map latent neural queries back to channel-level sensor features."""

    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        head_dim = dim // self.n_heads
        return x.reshape(batch, tokens, self.n_heads, head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, tokens, head_dim = x.shape
        return x.transpose(1, 2).reshape(batch, tokens, heads * head_dim)

    def forward(self, sensor_embedding: torch.Tensor, neural_tokens: torch.Tensor) -> torch.Tensor:
        kv = self.kv_proj(neural_tokens)
        keys, values = torch.chunk(kv, chunks=2, dim=-1)
        q = self._split_heads(sensor_embedding)
        k = self._split_heads(keys)
        v = self._split_heads(values)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.out_proj(self._merge_heads(out))


class LatentQueryAggregator(nn.Module):
    """Compress channel features into latent neural queries.

    Input:
        channel_features: `[B, C, N, W, D]`
        sensor_embedding: `[B, C, D]`
    Output:
        z: `[B, Q, N*W, D]`
    """

    def __init__(
        self,
        dim: int,
        latent_queries: int,
        n_heads: int,
        dropout: float,
        temporal_layers: int = 2,
    ):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.randn(latent_queries, dim) * 0.02)
        self.k_proj = nn.Linear(dim, dim)
        self.backward = BackwardSolution(dim=dim, n_heads=n_heads, dropout=dropout)
        self.query_norm = RMSNorm(dim)
        self.temporal_mixer = nn.Sequential(
            *[SelfAttentionBlock(dim, n_heads=n_heads, dropout=dropout) for _ in range(temporal_layers)]
        )

    def forward(
        self,
        channel_features: torch.Tensor,
        sensor_embedding: torch.Tensor,
        channel_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, channels, n_windows, latent_steps, dim = channel_features.shape
        total_steps = n_windows * latent_steps
        sensor = sensor_embedding[:, :, None, None, :].expand(batch, channels, n_windows, latent_steps, dim)
        x = (channel_features + sensor).permute(0, 2, 3, 1, 4).reshape(batch * total_steps, channels, dim)
        values = channel_features.permute(0, 2, 3, 1, 4).reshape(batch * total_steps, channels, dim)
        keys = self.k_proj(x)

        if channel_mask is not None:
            mask = channel_mask[:, None, :].expand(batch, total_steps, channels).reshape(batch * total_steps, channels)
            keys = keys.masked_fill((~mask.bool()).unsqueeze(-1), 0.0)
            values = values.masked_fill((~mask.bool()).unsqueeze(-1), 0.0)

        queries = self.latent_queries[None, :, :].expand(batch * total_steps, -1, -1)
        z = self.backward(queries, keys, values)
        z = self.query_norm(z)
        z = z.reshape(batch, total_steps, z.shape[1], dim).permute(0, 2, 1, 3)
        z = z.reshape(batch * z.shape[1], total_steps, dim)
        z = self.temporal_mixer(z)
        return z.reshape(batch, -1, total_steps, dim)


class ResidualVectorQuantizer(nn.Module):
    """Residual vector quantizer with optional codebook projection."""

    def __init__(self, dim: int, codebook_size: int, num_quantizers: int, codebook_dim: int | None = None):
        super().__init__()
        self.dim = dim
        self.codebook_dim = codebook_dim or dim
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.input_proj = nn.Linear(dim, self.codebook_dim) if self.codebook_dim != dim else nn.Identity()
        self.output_proj = nn.Linear(self.codebook_dim, dim) if self.codebook_dim != dim else nn.Identity()
        self.codebooks = nn.Parameter(
            torch.randn(num_quantizers, codebook_size, self.codebook_dim) / self.codebook_dim**0.5
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_code = F.normalize(self.input_proj(z), p=2.0, dim=-1)
        residual = z_code
        quantized_total = torch.zeros_like(z_code)
        all_indices = []
        commit_loss = z.new_tensor(0.0)
        flat_shape = z_code.shape[:-1]
        for idx in range(self.num_quantizers):
            codebook = F.normalize(self.codebooks[idx], p=2.0, dim=-1)
            flat = residual.reshape(-1, self.codebook_dim)
            distances = (
                flat.pow(2).sum(dim=1, keepdim=True)
                - 2 * flat @ codebook.T
                + codebook.pow(2).sum(dim=1).unsqueeze(0)
            )
            indices = torch.argmin(distances, dim=-1)
            codes = F.embedding(indices, codebook).reshape_as(residual)
            quantized_total = quantized_total + codes
            commit_loss = commit_loss + F.mse_loss(residual, codes.detach()) + 0.25 * F.mse_loss(
                codes, residual.detach()
            )
            residual = residual - codes.detach()
            all_indices.append(indices.reshape(*flat_shape))
        quantized_code = z_code + (quantized_total - z_code).detach()
        quantized = z + (self.output_proj(quantized_code) - z).detach()
        tokens = torch.stack(all_indices, dim=-1)
        return quantized, tokens, commit_loss / self.num_quantizers


class TemporalDecoder(nn.Module):
    """Reconstruct full-channel EEG from latent query tokens."""

    def __init__(self, dim: int, hidden: int, downsample_rates: tuple[int, ...], n_heads: int, dropout: float):
        super().__init__()
        self.forward_solution = ForwardSolution(dim=dim, n_heads=n_heads, dropout=dropout)
        layers: list[nn.Module] = []
        rates = tuple(reversed(downsample_rates))
        in_ch = dim
        cur = max(hidden, dim // 2)
        for stride in rates:
            layers.extend(
                [
                    nn.ELU(),
                    nn.ConvTranspose1d(in_ch, cur, kernel_size=2 * stride, stride=stride, padding=max(stride // 2, 0)),
                    nn.GroupNorm(1, cur),
                    SEANetResidualBlock(cur, dilation=1, dropout=dropout),
                ]
            )
            in_ch = cur
            cur = max(hidden, cur // 2)
        layers.extend([nn.ELU(), nn.Conv1d(in_ch, 1, kernel_size=7, padding=3)])
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, sensor_embedding: torch.Tensor, output_samples: int) -> torch.Tensor:
        batch, queries, steps, dim = z.shape
        channels = sensor_embedding.shape[1]
        neural = z.permute(0, 2, 1, 3).reshape(batch * steps, queries, dim)
        sensor = sensor_embedding[:, None, :, :].expand(batch, steps, channels, dim).reshape(batch * steps, channels, dim)
        channel_features = self.forward_solution(sensor, neural)
        channel_features = channel_features.reshape(batch, steps, channels, dim).permute(0, 2, 3, 1)
        x = channel_features.reshape(batch * channels, dim, steps)
        x = self.net(x).reshape(batch, channels, -1)
        if x.shape[-1] > output_samples:
            x = x[..., :output_samples]
        elif x.shape[-1] < output_samples:
            x = F.pad(x, (0, output_samples - x.shape[-1]))
        return x

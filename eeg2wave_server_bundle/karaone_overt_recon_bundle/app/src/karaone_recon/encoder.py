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


def _instance_norm(eeg: torch.Tensor, valid_len: torch.Tensor | None) -> torch.Tensor:
    """Per-trial, per-channel instance normalization (RevIN-style) over the valid
    (non-padded) time span.

    No subject id is used: this is the "no-ID statistical alignment" half of the
    cross-subject domain adaptation. EEG is already globally z-scored, but each
    trial/session still carries its own offset+scale drift; removing it per trial
    shrinks the cross-subject gap without ever looking at who the subject is.
    (True CORAL would also align second-order cross-channel statistics to a target
    domain, which needs transductive access to held-out EEG — out of scope here.)"""
    b, c, t = eeg.shape
    if valid_len is None:
        mean = eeg.mean(dim=-1, keepdim=True)
        var = eeg.var(dim=-1, keepdim=True, unbiased=False)
        return (eeg - mean) / torch.sqrt(var + 1e-5)
    idx = torch.arange(t, device=eeg.device).view(1, 1, t)
    mask = (idx < valid_len.view(b, 1, 1).clamp(min=1)).to(eeg.dtype)
    denom = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    mean = (eeg * mask).sum(dim=-1, keepdim=True) / denom
    var = (((eeg - mean) ** 2) * mask).sum(dim=-1, keepdim=True) / denom
    normed = (eeg - mean) / torch.sqrt(var + 1e-5)
    return normed * mask  # keep the zero-padded tail at zero (matches dataset padding)


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


def _key_padding_mask(valid_len: torch.Tensor | None, in_len: int, out_len: int, device: torch.device) -> torch.Tensor | None:
    if valid_len is None:
        return None
    frac = (valid_len.float() / float(max(in_len, 1))).clamp(min=1.0 / out_len, max=1.0)
    valid_frames = (frac * out_len).ceil().clamp(min=1.0)
    idx = torch.arange(out_len, device=device).unsqueeze(0)
    return idx >= valid_frames.unsqueeze(1)


class ConformerEncoderBlock(nn.Module):
    """Lightweight Conformer block for EEG frame sequences.

    This keeps the implementation small: feed-forward -> MHSA -> depthwise temporal
    convolution -> feed-forward, with residual connections and pre-norm.
    """

    def __init__(self, d_model: int, heads: int, dropout: float, kernel_size: int = 7):
        super().__init__()
        hidden = d_model * 4
        self.ff1_norm = nn.LayerNorm(d_model)
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.conv_norm = nn.LayerNorm(d_model)
        padding = (int(kernel_size) - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding, groups=d_model),
            nn.GroupNorm(num_groups=min(8, d_model), num_channels=d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.ff2_norm = nn.LayerNorm(d_model)
        self.ff2 = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + 0.5 * self.ff1(self.ff1_norm(x))
        attn_in = self.attn_norm(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        conv_in = self.conv_norm(x).transpose(1, 2)
        x = x + self.conv(conv_in).transpose(1, 2)
        x = x + 0.5 * self.ff2(self.ff2_norm(x))
        return self.out_norm(x)


class TransformerEEGEncoder(nn.Module):
    """Channel-aware EEG encoder with temporal patching + Transformer/Conformer trunk."""

    def __init__(
        self,
        in_channels: int = 62,
        d_model: int = 256,
        cond_dim: int = 64,
        target_steps: int = 150,
        kernel_size: int = 5,
        channel_dropout: float = 0.15,
        dropout: float = 0.15,
        num_channel_experts: int = 1,
        instance_norm: bool = False,
        encoder_kind: str = "transformer",
        transformer_layers: int = 4,
        transformer_heads: int = 4,
        patch_stride: int = 4,
    ):
        super().__init__()
        self.channel_dropout_p = float(channel_dropout)
        self.instance_norm = bool(instance_norm)
        self.num_channel_experts = int(num_channel_experts)
        self.target_steps = int(target_steps)
        self.encoder_kind = str(encoder_kind)
        if self.num_channel_experts > 1:
            self.spatial = ChannelMoEFrontend(
                in_channels=in_channels,
                d_model=d_model,
                num_experts=self.num_channel_experts,
                dropout=dropout,
            )
        else:
            self.spatial = nn.Sequential(
                nn.Conv1d(in_channels, d_model, kernel_size=1),
                nn.GroupNorm(num_groups=min(8, d_model), num_channels=d_model),
                nn.GELU(),
            )
        self.spatial_film = FiLM(cond_dim, d_model)
        p = max(1, int(patch_stride))
        k = max(3, int(kernel_size))
        self.patch = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=k, stride=p, padding=k // 2),
            nn.GroupNorm(num_groups=min(8, d_model), num_channels=d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos = nn.Parameter(torch.randn(1, self.target_steps, d_model) * 0.02)
        heads = max(1, int(transformer_heads))
        if d_model % heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by transformer_heads ({heads})")
        layers = max(1, int(transformer_layers))
        if self.encoder_kind == "conformer":
            self.blocks = nn.ModuleList(
                [ConformerEncoderBlock(d_model, heads, dropout=dropout, kernel_size=max(3, k)) for _ in range(layers)]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=heads,
                        dim_feedforward=d_model * 4,
                        dropout=dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(layers)
                ]
            )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self, eeg: torch.Tensor, cond: torch.Tensor, valid_len: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        in_len = int(eeg.shape[-1])
        if self.instance_norm:
            eeg = _instance_norm(eeg, valid_len)
        x = _channel_dropout(eeg, self.channel_dropout_p, self.training)
        if self.num_channel_experts > 1:
            x, aux = self.spatial(x)
        else:
            x, aux = self.spatial(x), {}
        x = self.spatial_film(x, cond)
        x = self.patch(x)
        if x.shape[-1] != self.target_steps:
            x = F.adaptive_avg_pool1d(x, self.target_steps)
        seq = x.transpose(1, 2) + self.pos[:, : self.target_steps]
        key_mask = _key_padding_mask(valid_len, in_len, self.target_steps, seq.device)
        for block in self.blocks:
            if isinstance(block, ConformerEncoderBlock):
                seq = block(seq, key_mask)
            else:
                seq = block(seq, src_key_padding_mask=key_mask)
        seq = self.out_norm(seq)
        if key_mask is not None:
            seq = seq.masked_fill(key_mask.unsqueeze(-1), 0.0)
        return seq.transpose(1, 2), aux


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
        instance_norm: bool = False,
    ):
        super().__init__()
        self.channel_dropout_p = float(channel_dropout)
        self.instance_norm = bool(instance_norm)
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
        self, eeg: torch.Tensor, cond: torch.Tensor, valid_len: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.instance_norm:
            eeg = _instance_norm(eeg, valid_len)
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

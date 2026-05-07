"""BrainOmni-style EEG tokenizer v0.

The tokenizer itself is intentionally thin. The explainable model blocks live
in `modules.py`, matching the reference-code style where core blocks are
defined separately and assembled by the top-level model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .losses import tokenizer_reconstruction_loss
from .modules import (
    LatentQueryAggregator,
    ResidualVectorQuantizer,
    SensorEmbedding,
    TemporalDecoder,
    TemporalEncoder,
)


@dataclass
class BrainStyleEEGTokenizerConfig:
    sample_rate: int = 250
    window_sec: float = 2.0
    dim: int = 256
    latent_queries: int = 32
    codebook_dim: int = 128
    codebook_size: int = 1024
    num_quantizers: int = 8
    encoder_channels: int = 96
    downsample_rates: tuple[int, ...] = (2, 2, 2, 2)
    n_heads: int = 8
    dropout: float = 0.1
    sensor_pos_dim: int = 6
    n_sensor_types: int = 3
    mask_ratio: float = 0.25
    noise_std: float = 0.05
    encoder_residual_layers: int = 2
    temporal_layers: int = 2

    @property
    def window_samples(self) -> int:
        return int(round(self.sample_rate * self.window_sec))


class BrainStyleEEGTokenizerV0(nn.Module):
    """Assemble sensor-aware encoder, latent queries, RVQ, and decoder."""

    def __init__(self, config: BrainStyleEEGTokenizerConfig | None = None):
        super().__init__()
        self.config = config or BrainStyleEEGTokenizerConfig()
        cfg = self.config
        self.sensor_embedding = SensorEmbedding(cfg.dim, cfg.dropout, cfg.sensor_pos_dim, cfg.n_sensor_types)
        self.encoder = TemporalEncoder(
            cfg.dim,
            cfg.encoder_channels,
            cfg.downsample_rates,
            cfg.dropout,
            residual_layers=cfg.encoder_residual_layers,
        )
        self.aggregator = LatentQueryAggregator(
            cfg.dim,
            cfg.latent_queries,
            cfg.n_heads,
            cfg.dropout,
            temporal_layers=cfg.temporal_layers,
        )
        self.quantizer = ResidualVectorQuantizer(
            cfg.dim,
            cfg.codebook_size,
            cfg.num_quantizers,
            codebook_dim=cfg.codebook_dim,
        )
        self.decoder = TemporalDecoder(cfg.dim, cfg.encoder_channels, cfg.downsample_rates, cfg.n_heads, cfg.dropout)

    @staticmethod
    def normalize_eeg(eeg: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        eeg = eeg.float()
        return (eeg - eeg.mean(dim=-1, keepdim=True)) / (eeg.std(dim=-1, keepdim=True) + eps)

    def unfold_windows(self, eeg: torch.Tensor, overlap_ratio: float = 0.0) -> tuple[torch.Tensor, int]:
        """Pad and split EEG into BrainOmni-style windows."""
        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError("overlap_ratio must be in [0, 1)")
        original_samples = eeg.shape[-1]
        win = self.config.window_samples
        step = max(1, int(round(win * (1.0 - overlap_ratio))))
        if eeg.shape[-1] < win:
            eeg = torch.nn.functional.pad(eeg, (0, win - eeg.shape[-1]))
        remainder = (eeg.shape[-1] - win) % step
        if remainder:
            eeg = torch.nn.functional.pad(eeg, (0, step - remainder))
        windows = eeg.unfold(dimension=-1, size=win, step=step)
        return windows, original_samples

    def apply_channel_masking(
        self, eeg_windows: torch.Tensor, channel_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Randomly hide channels during tokenizer pretraining."""
        if channel_mask is None:
            channel_mask = torch.ones(eeg_windows.shape[:2], dtype=torch.bool, device=eeg_windows.device)
        effective_mask = channel_mask.clone().bool()
        if self.training and self.config.mask_ratio > 0:
            batch, channels = effective_mask.shape
            random_keep = torch.rand(batch, channels, device=eeg_windows.device) > self.config.mask_ratio
            min_keep = random_keep.sum(dim=1) == 0
            if min_keep.any():
                random_keep[min_keep, 0] = True
            effective_mask = effective_mask & random_keep
        eeg_windows = eeg_windows * effective_mask[:, :, None, None].type_as(eeg_windows)
        return eeg_windows, effective_mask

    def add_noise(self, eeg_windows: torch.Tensor) -> torch.Tensor:
        if self.training and self.config.noise_std > 0:
            return eeg_windows + torch.randn_like(eeg_windows) * self.config.noise_std
        return eeg_windows

    def encode_continuous(
        self,
        eeg: torch.Tensor,
        sensor_pos: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        sensor_type: torch.Tensor | None = None,
        overlap_ratio: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Return normalized target, sensor embedding, and continuous latent z."""
        target = self.normalize_eeg(eeg)
        eeg_windows, original_samples = self.unfold_windows(target, overlap_ratio=overlap_ratio)
        eeg_windows, effective_mask = self.apply_channel_masking(eeg_windows, channel_mask)
        eeg_windows = self.add_noise(eeg_windows)
        sensor = self.sensor_embedding(sensor_pos, channel_mask, sensor_type=sensor_type)
        channel_features = self.encoder(eeg_windows)
        z = self.aggregator(channel_features, sensor, effective_mask)
        return target, sensor, z, original_samples

    def quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize continuous latent z with RVQ."""
        return self.quantizer(z)

    def reconstruct(self, z_q: torch.Tensor, sensor_embedding: torch.Tensor, output_samples: int) -> torch.Tensor:
        """Decode quantized latent tokens back to EEG channels."""
        return self.decoder(z_q, sensor_embedding, output_samples=output_samples)

    def forward(
        self,
        eeg: torch.Tensor,
        sensor_pos: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        sensor_type: torch.Tensor | None = None,
        overlap_ratio: float = 0.0,
        compute_loss: bool = True,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        target, sensor, z, original_samples = self.encode_continuous(
            eeg,
            sensor_pos,
            channel_mask,
            sensor_type,
            overlap_ratio=overlap_ratio,
        )
        z_q, tokens, commitment_loss = self.quantize(z)
        x_rec = self.reconstruct(z_q, sensor, output_samples=original_samples)
        out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "z": z,
            "z_q": z_q,
            "tokens": tokens,
            "x_rec": x_rec,
            "target": target,
            "commitment_loss": commitment_loss,
        }
        if compute_loss:
            out["losses"] = tokenizer_reconstruction_loss(x_rec, target, commitment_loss)
        return out

    @torch.no_grad()
    def tokenize(
        self,
        eeg: torch.Tensor,
        sensor_pos: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        sensor_type: torch.Tensor | None = None,
        overlap_ratio: float = 0.0,
    ) -> torch.Tensor:
        return self.forward(
            eeg,
            sensor_pos,
            channel_mask,
            sensor_type=sensor_type,
            overlap_ratio=overlap_ratio,
            compute_loss=False,
        )["tokens"]

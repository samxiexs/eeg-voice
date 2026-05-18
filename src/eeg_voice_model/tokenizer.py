"""Grouped EEG tokenizer for EEGVoiceTokenV1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import token_usage_metrics
from .modules import (
    LatentQueryAggregator,
    ResidualVectorQuantizer,
    SensorEmbedding,
    TemporalDecoder,
    TemporalEncoder,
)


def default_quantizer_groups() -> dict[str, tuple[int, ...]]:
    return {
        "base": (0, 1),
        "content": (2, 3),
        "prosody": (4,),
        "voice": (5, 6),
        "residual": (7,),
    }


@dataclass
class EEGVoiceV1Config:
    """Configuration for the English-first EEG voice token foundation model."""

    sample_rate: int = 250
    window_sec: float = 2.0
    dim: int = 256
    latent_queries: int = 32
    codebook_dim: int = 128
    codebook_size: int = 1024
    num_quantizers: int = 8
    quantizer_groups: Mapping[str, tuple[int, ...]] = field(default_factory=default_quantizer_groups)
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
    q7_full_recon_weight: float = 0.25
    q7_group_dropout: float = 0.5
    retrieval_queue_size: int = 4096
    retrieval_queue_negatives: int = 256
    retrieval_temperature: float = 0.07
    audio_embedding_dim: int = 11
    projection_dim: int = 256
    content_classes: int = 64
    phoneme_classes: int = 48
    pitch_dim: int = 1
    prosody_dim: int = 2
    timbre_dim: int = 8
    style_classes: int = 5
    mode_labels: tuple[str, ...] = ("heard", "imagined", "inner", "overt", "visualized_control")
    dataset_adapter_count: int = 128

    @property
    def window_samples(self) -> int:
        return int(round(self.sample_rate * self.window_sec))

    def validate(self) -> None:
        seen = sorted(idx for values in self.quantizer_groups.values() for idx in values)
        expected = list(range(self.num_quantizers))
        if seen != expected:
            raise ValueError(f"quantizer_groups must cover quantizers {expected}, got {seen}")
        if "residual" not in self.quantizer_groups:
            raise ValueError("quantizer_groups must define a residual group for q7")


@dataclass
class GroupedRVQOutput:
    z: torch.Tensor
    z_q: torch.Tensor
    tokens: torch.Tensor
    group_latents: dict[str, torch.Tensor]
    group_tokens: dict[str, torch.Tensor]
    group_names: tuple[str, ...]
    commitment_loss: torch.Tensor
    token_metrics: dict[str, torch.Tensor]


class GroupedResidualVectorQuantizer(nn.Module):
    """Residual vector quantizer that exposes interpretable quantizer groups."""

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        num_quantizers: int,
        quantizer_groups: Mapping[str, tuple[int, ...]],
        codebook_dim: int | None = None,
    ):
        super().__init__()
        self.dim = int(dim)
        self.codebook_dim = int(codebook_dim or dim)
        self.codebook_size = int(codebook_size)
        self.num_quantizers = int(num_quantizers)
        self.quantizer_groups = {name: tuple(indices) for name, indices in quantizer_groups.items()}
        self.group_names = tuple(self.quantizer_groups.keys())
        self.input_proj = nn.Linear(dim, self.codebook_dim, bias=False) if self.codebook_dim != dim else nn.Identity()
        self.output_proj = nn.Linear(self.codebook_dim, dim, bias=False) if self.codebook_dim != dim else nn.Identity()
        self.codebooks = nn.Parameter(
            torch.randn(num_quantizers, codebook_size, self.codebook_dim) / self.codebook_dim**0.5
        )

    def _group_for_quantizer(self, index: int) -> str:
        for name, indices in self.quantizer_groups.items():
            if index in indices:
                return name
        raise KeyError(index)

    def forward(self, z: torch.Tensor) -> GroupedRVQOutput:
        z_code = F.normalize(self.input_proj(z), p=2.0, dim=-1)
        residual = z_code
        quantized_total = torch.zeros_like(z_code)
        group_code = {name: torch.zeros_like(z_code) for name in self.group_names}
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
            group_code[self._group_for_quantizer(idx)] = group_code[self._group_for_quantizer(idx)] + codes
            commit_loss = commit_loss + F.mse_loss(residual, codes.detach()) + 0.25 * F.mse_loss(
                codes, residual.detach()
            )
            residual = residual - codes.detach()
            all_indices.append(indices.reshape(*flat_shape))

        tokens = torch.stack(all_indices, dim=-1)
        total_projected = self.output_proj(quantized_total)
        z_q = total_projected + (z - z.detach())
        group_latents = {name: self.output_proj(values) for name, values in group_code.items()}
        group_tokens = {
            name: tokens[..., torch.tensor(indices, device=tokens.device, dtype=torch.long)]
            for name, indices in self.quantizer_groups.items()
        }
        metrics = token_usage_metrics(tokens, self.codebook_size)
        return GroupedRVQOutput(
            z=z,
            z_q=z_q,
            tokens=tokens,
            group_latents=group_latents,
            group_tokens=group_tokens,
            group_names=self.group_names,
            commitment_loss=commit_loss / self.num_quantizers,
            token_metrics={f"token_{key}": value for key, value in metrics.items()},
        )


class EEGVoiceTokenizerV1(nn.Module):
    """Sensor-aware EEG encoder with hierarchical grouped RVQ tokens."""

    def __init__(self, config: EEGVoiceV1Config | None = None):
        super().__init__()
        self.config = config or EEGVoiceV1Config()
        self.config.validate()
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
        self.quantizer = GroupedResidualVectorQuantizer(
            cfg.dim,
            cfg.codebook_size,
            cfg.num_quantizers,
            cfg.quantizer_groups,
            codebook_dim=cfg.codebook_dim,
        )
        self.aligned_decoder = TemporalDecoder(cfg.dim, cfg.encoder_channels, cfg.downsample_rates, cfg.n_heads, cfg.dropout)
        self.full_decoder = TemporalDecoder(cfg.dim, cfg.encoder_channels, cfg.downsample_rates, cfg.n_heads, cfg.dropout)

    @staticmethod
    def normalize_eeg(eeg: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        eeg = eeg.float()
        return (eeg - eeg.mean(dim=-1, keepdim=True)) / (eeg.std(dim=-1, keepdim=True) + eps)

    def unfold_windows(self, eeg: torch.Tensor, overlap_ratio: float = 0.0) -> tuple[torch.Tensor, int]:
        if not 0.0 <= overlap_ratio < 1.0:
            raise ValueError("overlap_ratio must be in [0, 1)")
        original_samples = eeg.shape[-1]
        win = self.config.window_samples
        step = max(1, int(round(win * (1.0 - overlap_ratio))))
        if eeg.shape[-1] < win:
            eeg = F.pad(eeg, (0, win - eeg.shape[-1]))
        remainder = (eeg.shape[-1] - win) % step
        if remainder:
            eeg = F.pad(eeg, (0, step - remainder))
        return eeg.unfold(dimension=-1, size=win, step=step), original_samples

    def apply_channel_masking(
        self, eeg_windows: torch.Tensor, channel_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if channel_mask is None:
            channel_mask = torch.ones(eeg_windows.shape[:2], dtype=torch.bool, device=eeg_windows.device)
        effective_mask = channel_mask.clone().bool()
        if self.training and self.config.mask_ratio > 0:
            batch, channels = effective_mask.shape
            random_keep = torch.rand(batch, channels, device=eeg_windows.device) > self.config.mask_ratio
            empty = random_keep.sum(dim=1) == 0
            if empty.any():
                random_keep[empty, 0] = True
            effective_mask = effective_mask & random_keep
        return eeg_windows * effective_mask[:, :, None, None].type_as(eeg_windows), effective_mask

    def add_noise(self, eeg_windows: torch.Tensor) -> torch.Tensor:
        if self.training and self.config.noise_std > 0:
            return eeg_windows + torch.randn_like(eeg_windows) * self.config.noise_std
        return eeg_windows

    def encode(
        self,
        eeg: torch.Tensor,
        sensor_pos: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        sensor_type: torch.Tensor | None = None,
        overlap_ratio: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        target = self.normalize_eeg(eeg)
        eeg_windows, original_samples = self.unfold_windows(target, overlap_ratio=overlap_ratio)
        eeg_windows, effective_mask = self.apply_channel_masking(eeg_windows, channel_mask)
        eeg_windows = self.add_noise(eeg_windows)
        sensor = self.sensor_embedding(sensor_pos, channel_mask, sensor_type=sensor_type)
        channel_features = self.encoder(eeg_windows)
        z = self.aggregator(channel_features, sensor, effective_mask)
        return target, sensor, z, original_samples

    def group_latent(
        self,
        rvq: GroupedRVQOutput,
        groups: tuple[str, ...],
        residual_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = []
        for name in groups:
            if name == "residual" and residual_override is not None:
                values.append(residual_override)
            else:
                values.append(rvq.group_latents[name])
        routed = torch.stack(values, dim=0).sum(dim=0)
        return routed + (rvq.z - rvq.z.detach())

    def dropout_residual(self, residual: torch.Tensor) -> torch.Tensor:
        if not self.training or self.config.q7_group_dropout <= 0:
            return residual
        keep_prob = 1.0 - self.config.q7_group_dropout
        if keep_prob <= 0:
            return torch.zeros_like(residual)
        mask = torch.rand(residual.shape[0], 1, 1, 1, device=residual.device) < keep_prob
        return residual * mask.type_as(residual) / keep_prob

    def forward(
        self,
        eeg: torch.Tensor,
        sensor_pos: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        sensor_type: torch.Tensor | None = None,
        overlap_ratio: float = 0.0,
    ) -> dict[str, torch.Tensor | GroupedRVQOutput | dict[str, torch.Tensor]]:
        target, sensor, z, original_samples = self.encode(
            eeg,
            sensor_pos,
            channel_mask=channel_mask,
            sensor_type=sensor_type,
            overlap_ratio=overlap_ratio,
        )
        rvq = self.quantizer(z)
        aligned_groups = ("base", "content", "prosody", "voice")
        aligned_z = self.group_latent(rvq, aligned_groups)
        dropped_residual = self.dropout_residual(rvq.group_latents["residual"])
        full_z = self.group_latent(rvq, (*aligned_groups, "residual"), residual_override=dropped_residual)
        recon_aligned = self.aligned_decoder(aligned_z, sensor, output_samples=original_samples)
        recon_full = self.full_decoder(full_z, sensor, output_samples=original_samples)
        recon_full_no_q7 = self.full_decoder(aligned_z, sensor, output_samples=original_samples)
        return {
            "target": target,
            "sensor_embedding": sensor,
            "rvq": rvq,
            "z": z,
            "z_q": rvq.z_q,
            "tokens": rvq.tokens,
            "group_tokens": rvq.group_tokens,
            "group_latents": rvq.group_latents,
            "recon_aligned": recon_aligned,
            "recon_full": recon_full,
            "recon_full_no_q7": recon_full_no_q7,
            "q7_dropout_ablation": torch.mean(torch.abs(recon_full.detach() - recon_full_no_q7.detach())),
        }

    @torch.no_grad()
    def tokenize(
        self,
        eeg: torch.Tensor,
        sensor_pos: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        sensor_type: torch.Tensor | None = None,
        overlap_ratio: float = 0.0,
    ) -> torch.Tensor:
        return self.forward(eeg, sensor_pos, channel_mask, sensor_type, overlap_ratio=overlap_ratio)["tokens"]

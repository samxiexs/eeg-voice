from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


_KARA_APP = Path(__file__).resolve().parents[4] / "karaone_overt_recon_bundle" / "app"
if str(_KARA_APP) not in sys.path:
    sys.path.insert(0, str(_KARA_APP))

from src.karaone_0715.model import (  # noqa: E402
    AudioCodeAutoencoder,
    AudioCodeModelConfig,
    grad_reverse,
    random_code_mask,
)


@dataclass(frozen=True)
class EEGModelConfig:
    channels: int = 14
    eeg_len: int = 768
    d_model: int = 192
    condition_steps: int = 50
    code_steps: int = 150
    global_labels: int = 30
    label_dims: tuple[int, ...] = (16, 11, 3)
    num_train_subjects: int = 38
    transformer_layers: int = 3
    heads: int = 6
    dropout: float = 0.15
    temporal_kernels: tuple[int, ...] = (15, 31, 63)
    stem_stride: int = 4


def _transformer_encoder(d_model: int, heads: int, layers: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=heads,
        dim_feedforward=d_model * 4,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers)


class _SingleStreamStem(nn.Module):
    """0715 multiscale stem for already CAR-normalised 14-channel EEG."""

    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        branch_channels = cfg.d_model // len(cfg.temporal_kernels)
        if branch_channels * len(cfg.temporal_kernels) != cfg.d_model:
            raise ValueError("d_model must be divisible by the number of temporal kernels")
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(cfg.channels, branch_channels, kernel_size=kernel, stride=cfg.stem_stride, padding=kernel // 2, bias=False),
                    nn.GroupNorm(max(1, branch_channels // 16), branch_channels),
                    nn.GELU(),
                )
                for kernel in cfg.temporal_kernels
            ]
        )
        self.mix = nn.Sequential(
            nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=1, bias=False),
            nn.GroupNorm(max(1, cfg.d_model // 16), cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.mix(torch.cat([branch(eeg) for branch in self.branches], dim=1))


class EEGConditionEncoder(nn.Module):
    """Imagined-EEG encoder aligned to the frozen 0715 audio condition space."""

    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        self.cfg = cfg
        self.stem = _SingleStreamStem(cfg)
        stem_steps = math.floor((cfg.eeg_len - 1) / cfg.stem_stride) + 1
        self.position = nn.Parameter(torch.zeros(1, stem_steps, cfg.d_model))
        self.encoder = _transformer_encoder(cfg.d_model, cfg.heads, cfg.transformer_layers, cfg.dropout)
        self.queries = nn.Parameter(torch.zeros(1, cfg.condition_steps, cfg.d_model))
        self.query_attention = nn.MultiheadAttention(cfg.d_model, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.condition_refiner = _transformer_encoder(cfg.d_model, cfg.heads, 2, cfg.dropout)
        self.condition_norm = nn.LayerNorm(cfg.d_model)
        self.label_heads = nn.ModuleDict({str(index): nn.Linear(cfg.d_model, classes) for index, classes in enumerate(cfg.label_dims)})
        self.envelope_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, 1))
        self.timing_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, 2))
        self.subject_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.num_train_subjects),
        )
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.queries, std=0.02)

    def augment(self, eeg: torch.Tensor, *, channel_dropout: float, time_mask_ratio: float, noise_std: float) -> torch.Tensor:
        output = eeg
        if channel_dropout > 0:
            keep = (torch.rand(eeg.shape[0], eeg.shape[1], 1, device=eeg.device) > float(channel_dropout)).to(eeg.dtype)
            output = output * keep / keep.mean(dim=1, keepdim=True).clamp_min(0.25)
        if time_mask_ratio > 0:
            width = max(1, int(round(eeg.shape[-1] * float(time_mask_ratio))))
            for item in range(eeg.shape[0]):
                start = int(torch.randint(0, max(1, eeg.shape[-1] - width + 1), (1,), device=eeg.device).item())
                output[item, :, start : start + width] = 0
        if noise_std > 0:
            output = output + float(noise_std) * torch.randn_like(output)
        return output

    def forward(
        self,
        eeg: torch.Tensor,
        valid_len: torch.Tensor,
        dataset_idx: torch.Tensor,
        *,
        subject_adversary_strength: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if eeg.ndim != 3 or eeg.shape[1] != self.cfg.channels:
            raise ValueError(f"eeg must be [B,{self.cfg.channels},T], got {tuple(eeg.shape)}")
        if dataset_idx.ndim != 1 or len(dataset_idx) != len(eeg):
            raise ValueError("dataset_idx must be [B]")
        stem = self.stem(eeg).transpose(1, 2)
        stem_steps = stem.shape[1]
        valid_steps = torch.ceil(valid_len.float() / float(self.cfg.stem_stride)).long().clamp(1, stem_steps)
        valid_mask = torch.arange(stem_steps, device=eeg.device).unsqueeze(0) < valid_steps.unsqueeze(1)
        encoded = self.encoder(stem + self.position[:, :stem_steps], src_key_padding_mask=~valid_mask)
        queries = self.queries.expand(eeg.shape[0], -1, -1)
        condition, _ = self.query_attention(queries, encoded, encoded, key_padding_mask=~valid_mask, need_weights=False)
        condition = self.condition_norm(self.condition_refiner(condition))
        pooled = condition.mean(dim=1)
        global_logits = pooled.new_full((len(eeg), self.cfg.global_labels), -1.0e4)
        for dataset_number, classes in enumerate(self.cfg.label_dims):
            selected = dataset_idx == dataset_number
            if selected.any():
                start = sum(self.cfg.label_dims[:dataset_number])
                global_logits[selected, start : start + classes] = self.label_heads[str(dataset_number)](pooled[selected])
        envelope = self.envelope_head(condition).squeeze(-1)
        envelope = F.interpolate(envelope.unsqueeze(1), size=self.cfg.code_steps, mode="linear", align_corners=False).squeeze(1)
        timing = torch.sigmoid(self.timing_head(pooled))
        return {
            "condition": condition,
            "pooled": pooled,
            "label_logits": global_logits,
            "envelope_logits": envelope,
            "onset": timing[:, 0],
            "duration": timing[:, 1],
            "subject_logits": self.subject_head(grad_reverse(pooled, subject_adversary_strength)),
            "eeg_tokens": encoded,
            "eeg_valid_mask": valid_mask,
        }


__all__ = [
    "AudioCodeAutoencoder",
    "AudioCodeModelConfig",
    "EEGConditionEncoder",
    "EEGModelConfig",
    "random_code_mask",
]

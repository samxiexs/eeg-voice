"""Physical-time EEG → frozen speech space → native SpeechT5 mel.

This module deliberately has no access to audio IDs, target durations or text.
Legacy MFCC models remain in model.py for reproducibility.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .model import JointEEGContentModel

CONTRACT = "aligned_speech_v1"
SAMPLE_RATE = 16000
WAVE_SAMPLES = 64000
EEG_SAMPLES = 1178
EEG_RATE = 256
EEG_START = -0.25
LAGS_MS = (0, 100, 200, 300, 400)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(path: Path) -> str:
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"empty model directory: {path}")
    return hashlib.sha256(json.dumps([(str(p.relative_to(path)), sha256(p)) for p in files]).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def fixed_wave(wave: np.ndarray, *, allow_crop: bool = False) -> np.ndarray:
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
        raise ValueError("expected finite nonempty mono waveform")
    if len(wave) > WAVE_SAMPLES and not allow_crop:
        raise ValueError("stimulus exceeds four seconds; do not silently truncate")
    return np.pad(wave[:WAVE_SAMPLES], (0, max(0, WAVE_SAMPLES - len(wave))))


def convolution_times(samples: int, kernels: list[int], strides: list[int], rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Exact centers of an unpadded HuBERT convolutional feature extractor."""
    if len(kernels) != len(strides) or not kernels:
        raise ValueError("invalid convolution contract")
    jump, receptive, count = 1, 1, samples
    for kernel, stride in zip(kernels, strides):
        receptive += (kernel - 1) * jump
        jump *= stride
        count = (count - kernel) // stride + 1
    if count <= 0:
        raise ValueError("waveform shorter than teacher receptive field")
    return ((receptive - 1) / 2 + torch.arange(count).float() * jump) / rate


def sample_at(value: torch.Tensor, source_times: torch.Tensor, query_times: torch.Tensor) -> torch.Tensor:
    """B,T,D interpolation in seconds, with fixed endpoint extension.

    Endpoint extension handles vocoder boundaries and the final ~20 ms at
    400 ms neural lag. It never depends on an individual stimulus duration.
    """
    if value.ndim != 3 or len(source_times) != value.shape[1] or len(source_times) < 2:
        raise ValueError("sequence/time-axis mismatch")
    if not bool((source_times[1:] > source_times[:-1]).all()):
        raise ValueError("time coordinates must increase")
    query = query_times.to(value).clamp(source_times[0], source_times[-1])
    right = torch.searchsorted(source_times.contiguous(), query.contiguous()).clamp(1, len(source_times) - 1)
    left = right - 1
    weight = (query - source_times[left]) / (source_times[right] - source_times[left])
    return value[:, left] * (1 - weight[None, :, None]) + value[:, right] * weight[None, :, None]


class SpeechNormalizer(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("scale", torch.ones(dimension))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean) / self.scale.clamp_min(1e-4)


class AcousticDecoder(nn.Module):
    def __init__(self, speech_times: torch.Tensor, mel_times: torch.Tensor,
                 speech_dimension: int = 768, hidden: int = 256, layers: int = 6):
        super().__init__()
        self.register_buffer("speech_times", speech_times.clone())
        self.register_buffer("mel_times", mel_times.clone())
        self.normalizer = SpeechNormalizer(speech_dimension)
        self.input = nn.Conv1d(speech_dimension, hidden, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv1d(hidden, hidden, 3, padding=2 ** (i % 4), dilation=2 ** (i % 4)),
                          nn.GELU(), nn.Conv1d(hidden, hidden, 1)) for i in range(layers)
        ])
        self.output = nn.Conv1d(hidden, 80, 1)

    def forward(self, aligned_sequence: torch.Tensor) -> torch.Tensor:
        value = self.input(self.normalizer(aligned_sequence).transpose(1, 2))
        for block in self.blocks:
            value = value + block(value)
        value = sample_at(value.transpose(1, 2), self.speech_times, self.mel_times)
        return self.output(value.transpose(1, 2))


@dataclass
class AlignedState:
    aligned_sequence: torch.Tensor
    sequence_times_s: torch.Tensor
    native_mel: torch.Tensor
    mel_times_s: torch.Tensor
    global_embedding: torch.Tensor


class AlignedEEGModel(nn.Module):
    def __init__(self, decoder: AcousticDecoder, *, dimension: int = 192, heads: int = 6,
                 layers: int = 4, local_layers: int = 2, token_steps: int = 236, lag_ms: int = 0):
        super().__init__()
        if lag_ms not in LAGS_MS:
            raise ValueError("lag must be one of the registered validation candidates")
        self.lag_ms = lag_ms
        self.decoder = decoder
        self.decoder.requires_grad_(False)
        self.backbone = JointEEGContentModel(dimension=dimension, heads=heads, layers=layers,
                                            local_layers=local_layers, token_steps=token_steps,
                                            dropout=0.0, zero_centered=False)
        # Unused legacy output heads cannot influence this experiment.
        for name in ("global_head", "mfcc_head", "audio_projection", "phoneme_head", "duration_head", "activity_head"):
            getattr(self.backbone, name).requires_grad_(False)
        self.backbone.clip_logit_scale.requires_grad_(False)
        self.speech_head = nn.Linear(dimension, decoder.normalizer.mean.numel())
        # F.interpolate(..., align_corners=False) uses these source sample centers.
        self.register_buffer("eeg_token_times", EEG_START +
                             ((torch.arange(token_steps).float() + .5) * EEG_SAMPLES / token_steps - .5) / EEG_RATE)

    def forward(self, eeg: torch.Tensor, channel_xyz: torch.Tensor,
                channel_mask: torch.Tensor, time_mask: torch.Tensor) -> AlignedState:
        if eeg.shape[-1] != EEG_SAMPLES or time_mask.shape != (len(eeg), EEG_SAMPLES) or not bool(time_mask.all()):
            raise ValueError("aligned model requires real fixed 1178-sample EEG windows with full time masks")
        if not torch.isfinite(eeg).all() or not torch.isfinite(channel_xyz).all() or not channel_mask.any(1).all():
            raise ValueError("invalid EEG/channel input")
        raw = self.backbone._forward_raw(eeg, channel_xyz, channel_mask, time_mask,
                                         torch.zeros(len(eeg), dtype=torch.long, device=eeg.device))
        sequence = sample_at(raw.local, self.eeg_token_times,
                             self.decoder.speech_times + self.lag_ms / 1000)
        # Regress standardized targets but supply the frozen decoder with its
        # original speech space. Only training-derived statistics are used.
        sequence = self.speech_head(sequence) * self.decoder.normalizer.scale + self.decoder.normalizer.mean
        embedding = F.normalize(self.decoder.normalizer(sequence).mean(1), dim=-1)
        return AlignedState(sequence, self.decoder.speech_times, self.decoder(sequence),
                            self.decoder.mel_times, embedding)


def acoustic_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(prediction, target) + .2 * F.l1_loss(prediction.diff(dim=-1), target.diff(dim=-1))


def alignment_loss(state: AlignedState, teacher: torch.Tensor, mel: torch.Tensor,
                   normalizer: SpeechNormalizer, bank: torch.Tensor, indices: torch.Tensor,
                   *, acoustic: bool) -> tuple[torch.Tensor, dict]:
    target = normalizer(teacher).detach()
    prediction = normalizer(state.aligned_sequence)
    regression = F.smooth_l1_loss(prediction, target)
    delta = F.l1_loss(prediction.diff(dim=1), target.diff(dim=1))
    # The complete immutable content bank supplies negatives even for a
    # one-example physical batch; repeated contents map to the same positive.
    contrastive = F.cross_entropy(state.global_embedding @ F.normalize(bank.detach(), dim=-1).T / .07, indices)
    mel_loss = acoustic_loss(state.native_mel, mel)
    loss = regression + .2 * delta + .1 * contrastive + (mel_loss if acoustic else 0)
    return loss, {"sequence": float(regression.detach()), "contrastive": float(contrastive.detach()),
                  "mel": float(mel_loss.detach())}


def two_way_bootstrap(records: list[dict], key: str, *, repeats: int = 1000, seed: int = 31) -> dict:
    """Crossed subject/content bootstrap; repeated trials are not IID units."""
    subjects = sorted({r["subject"] for r in records})
    contents = sorted({r["content"] for r in records})
    if not records:
        raise ValueError("empty evaluation")
    si, ci = {s: i for i, s in enumerate(subjects)}, {c: i for i, c in enumerate(contents)}
    s = np.array([si[r["subject"]] for r in records]); c = np.array([ci[r["content"]] for r in records])
    values = np.array([r[key] for r in records], dtype=float)
    rng = np.random.default_rng(seed); estimates = []
    for _ in range(repeats):
        sw = np.bincount(rng.integers(len(subjects), size=len(subjects)), minlength=len(subjects))
        cw = np.bincount(rng.integers(len(contents), size=len(contents)), minlength=len(contents))
        weights = sw[s] * cw[c]
        if weights.sum():
            estimates.append(float(np.average(values, weights=weights)))
    return {"mean": float(values.mean()), "ci_low": float(np.quantile(estimates, .025)),
            "ci_high": float(np.quantile(estimates, .975)), "subjects": len(subjects), "contents": len(contents)}

from __future__ import annotations

import csv
import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TimeAnchor:
    onset_sec: float
    duration_sec: float
    center_sec: float
    lag_sec: float
    confidence: float
    active_mask: np.ndarray
    envelope: np.ndarray


def rms_envelope(audio: np.ndarray, *, sample_rate: int = 16000, hop_sec: float = 0.01, win_sec: float = 0.025) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    hop = max(1, int(round(float(sample_rate) * float(hop_sec))))
    win = max(hop, int(round(float(sample_rate) * float(win_sec))))
    if audio.size == 0:
        return np.zeros(1, dtype=np.float32)
    if audio.size < win:
        value = math.sqrt(float(np.mean(audio.astype(np.float64) ** 2)) + 1e-12)
        return np.asarray([value], dtype=np.float32)
    values = []
    for start in range(0, audio.size - win + 1, hop):
        seg = audio[start : start + win].astype(np.float64)
        values.append(math.sqrt(float(np.mean(seg * seg)) + 1e-12))
    return np.asarray(values, dtype=np.float32)


def extract_time_anchor(
    audio: np.ndarray,
    *,
    sample_rate: int = 16000,
    duration_sec: float = 2.0,
    hop_sec: float = 0.01,
    min_active_sec: float = 0.12,
    merge_gap_sec: float = 0.08,
    threshold_mad: float = 1.5,
    threshold_peak_ratio: float = 0.15,
) -> TimeAnchor:
    env = rms_envelope(audio, sample_rate=sample_rate, hop_sec=hop_sec)
    if env.size == 0 or float(env.max(initial=0.0)) <= 1e-9:
        active = np.zeros(max(1, int(round(duration_sec / hop_sec))), dtype=np.float32)
        return TimeAnchor(0.0, 0.0, 0.0, 0.0, 0.0, active, active.copy())
    median = float(np.median(env))
    mad = float(np.median(np.abs(env - median))) + 1e-8
    threshold = max(median + float(threshold_mad) * mad, float(threshold_peak_ratio) * float(env.max()))
    active = (env >= threshold).astype(np.float32)
    active = _merge_short_gaps(active, max_gap=max(1, int(round(float(merge_gap_sec) / float(hop_sec)))))
    active = _keep_long_regions(active, min_len=max(1, int(round(float(min_active_sec) / float(hop_sec)))))
    if not bool(active.any()):
        peak = int(np.argmax(env))
        half = max(1, int(round(float(min_active_sec) / float(hop_sec))) // 2)
        start = max(0, peak - half)
        end = min(active.shape[0], peak + half + 1)
        active[start:end] = 1.0
    indices = np.flatnonzero(active > 0)
    start_idx = int(indices[0])
    end_idx = int(indices[-1]) + 1
    onset = float(start_idx * hop_sec)
    dur = float(max(float(min_active_sec), (end_idx - start_idx) * hop_sec))
    dur = min(dur, max(0.0, float(duration_sec) - onset))
    center = float(onset + 0.5 * dur)
    conf = float(np.clip((env[indices].mean() - median) / (env.max() + 1e-8), 0.0, 1.0)) if indices.size else 0.0
    target_steps = max(1, int(round(float(duration_sec) / float(hop_sec))))
    active = _resize_1d(active, target_steps)
    env = _resize_1d(env, target_steps)
    return TimeAnchor(onset, dur, center, 0.0, conf, active.astype(np.float32), env.astype(np.float32))


def best_lag_corr(
    a: np.ndarray,
    b: np.ndarray,
    *,
    sample_rate: float,
    max_lag_sec: float,
    min_overlap_sec: float = 0.25,
    stride: int = 1,
) -> tuple[float, float, int]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    max_lag = int(round(float(max_lag_sec) * float(sample_rate)))
    min_overlap = int(round(float(min_overlap_sec) * float(sample_rate)))
    best = (-1.0e9, 0, 0)
    for lag in range(-max_lag, max_lag + 1, max(1, int(stride))):
        aa, bb = overlap_by_lag(a, b, lag)
        n = min(aa.shape[0], bb.shape[0])
        if n < min_overlap:
            continue
        value = pearson_corr(aa[:n], bb[:n])
        if np.isfinite(value) and value > best[0]:
            best = (float(value), int(lag), int(n))
    return float(best[0]), float(best[1] / max(float(sample_rate), 1e-8)), int(best[2])


def overlap_by_lag(a: np.ndarray, b: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag >= 0:
        aa = a[int(lag) :]
        bb = b[: aa.shape[0]]
    else:
        bb = b[int(-lag) :]
        aa = a[: bb.shape[0]]
    n = min(aa.shape[0], bb.shape[0])
    return aa[:n], bb[:n]


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(int(np.asarray(a).shape[0]), int(np.asarray(b).shape[0]))
    if n < 3:
        return float("nan")
    aa = np.asarray(a[:n], dtype=np.float64)
    bb = np.asarray(b[:n], dtype=np.float64)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = math.sqrt(float(np.sum(aa * aa) * np.sum(bb * bb)))
    return float(np.sum(aa * bb) / denom) if denom > 1e-12 else float("nan")


def shift_audio(audio: np.ndarray, lag_sec: float, *, sample_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    shift = int(round(float(lag_sec) * int(sample_rate)))
    if shift == 0:
        return audio.copy()
    out = np.zeros_like(audio)
    if shift > 0:
        out[shift:] = audio[: audio.shape[0] - shift]
    else:
        out[: audio.shape[0] + shift] = audio[-shift:]
    return out.astype(np.float32)


def read_wav_mono(path: str | Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as handle:
        sr = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width == 1:
        values = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        values = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        values = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width {width}: {path}")
    if channels > 1:
        values = values.reshape(-1, channels).mean(axis=1)
    return int(sr), values.astype(np.float32)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def coarse_log_spectral_frames(audio: np.ndarray, *, sample_rate: int, hop_sec: float = 0.01, win_sec: float = 0.025, n_bins: int = 24) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    hop = max(1, int(round(float(sample_rate) * float(hop_sec))))
    win = max(hop, int(round(float(sample_rate) * float(win_sec))))
    if audio.shape[0] < win:
        audio = np.pad(audio, (0, win - audio.shape[0]))
    windows = []
    freqs = np.fft.rfftfreq(win, d=1.0 / max(float(sample_rate), 1.0))
    edges = np.linspace(0.0, min(float(sample_rate) / 2.0, 8000.0), n_bins + 1)
    for start in range(0, audio.shape[0] - win + 1, hop):
        frame = audio[start : start + win].astype(np.float64) * np.hanning(win)
        spec = np.abs(np.fft.rfft(frame)) ** 2
        vals = []
        for left, right in zip(edges[:-1], edges[1:]):
            mask = (freqs >= left) & (freqs < right)
            vals.append(float(np.log(spec[mask].mean() + 1e-8)) if bool(mask.any()) else -18.0)
        windows.append(vals)
    return np.asarray(windows, dtype=np.float32)


def _merge_short_gaps(active: np.ndarray, *, max_gap: int) -> np.ndarray:
    active = active.astype(np.float32).copy()
    idx = np.flatnonzero(active > 0)
    if idx.size <= 1:
        return active
    for left, right in zip(idx[:-1], idx[1:]):
        if 1 < right - left <= max_gap + 1:
            active[left:right + 1] = 1.0
    return active


def _keep_long_regions(active: np.ndarray, *, min_len: int) -> np.ndarray:
    active = active.astype(np.float32).copy()
    out = np.zeros_like(active)
    start = None
    for idx, value in enumerate(np.r_[active, 0.0]):
        if value > 0 and start is None:
            start = idx
        elif value <= 0 and start is not None:
            if idx - start >= int(min_len):
                out[start:idx] = 1.0
            start = None
    return out


def _resize_1d(values: np.ndarray, steps: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] == steps:
        return values
    if values.shape[0] <= 1:
        return np.full(steps, float(values[0]) if values.size else 0.0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, values.shape[0])
    x_new = np.linspace(0.0, 1.0, int(steps))
    return np.interp(x_new, x_old, values).astype(np.float32)


__all__ = [
    "TimeAnchor",
    "best_lag_corr",
    "coarse_log_spectral_frames",
    "extract_time_anchor",
    "overlap_by_lag",
    "pearson_corr",
    "read_wav_mono",
    "rms_envelope",
    "shift_audio",
    "write_csv",
]

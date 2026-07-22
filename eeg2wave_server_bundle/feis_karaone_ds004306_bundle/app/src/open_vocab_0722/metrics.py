from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.signal import stft


def _unit_rms(value: np.ndarray) -> np.ndarray:
    audio = np.asarray(value, dtype=np.float64).reshape(-1)
    rms = math.sqrt(float(np.mean(audio * audio)) + 1e-12)
    return audio / rms if rms > 1e-8 else np.zeros_like(audio)


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    size = min(len(first), len(second))
    if size < 2:
        return 0.0
    raw_x = np.asarray(first[:size], dtype=np.float64)
    raw_y = np.asarray(second[:size], dtype=np.float64)
    # Pearson correlation is undefined for a constant envelope.  Identical
    # constant signals are nevertheless a perfect structural match.
    if np.allclose(raw_x, raw_y, atol=1e-10, rtol=1e-8):
        return 1.0
    x = raw_x - float(np.mean(raw_x))
    y = raw_y - float(np.mean(raw_y))
    denominator = math.sqrt(float(np.sum(x * x) * np.sum(y * y)))
    return float(np.sum(x * y) / denominator) if denominator > 1e-12 else 0.0


def rms_envelope(audio: np.ndarray, sample_rate: int, window_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float64).reshape(-1)
    window = max(1, round(sample_rate * window_ms / 1000.0))
    hop = max(1, round(sample_rate * hop_ms / 1000.0))
    if len(value) < window:
        value = np.pad(value, (0, window - len(value)))
    return np.asarray([math.sqrt(float(np.mean(value[start : start + window] ** 2)) + 1e-12) for start in range(0, len(value) - window + 1, hop)])


def best_lag_envelope_correlation(reference: np.ndarray, candidate: np.ndarray, sample_rate: int, max_lag_ms: float = 250.0) -> tuple[float, float]:
    first = rms_envelope(reference, sample_rate)
    second = rms_envelope(candidate, sample_rate)
    max_steps = round(max_lag_ms / 10.0)
    scores: list[tuple[float, int]] = []
    for lag in range(-max_steps, max_steps + 1):
        if lag < 0:
            x, y = first[-lag:], second[:lag]
        elif lag > 0:
            x, y = first[:-lag], second[lag:]
        else:
            x, y = first, second
        scores.append((_correlation(x, y), lag))
    score, lag = max(scores, key=lambda value: value[0])
    return float(score), float(lag * 10.0)


def soft_dtw_distance(first: np.ndarray, second: np.ndarray, gamma: float = 0.05) -> float:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    x = x / max(float(np.max(x)), 1e-8)
    y = y / max(float(np.max(y)), 1e-8)
    cost = (x[:, None] - y[None, :]) ** 2
    table = np.full((len(x) + 1, len(y) + 1), np.inf, dtype=np.float64)
    table[0, 0] = 0.0
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            previous = np.asarray([table[i - 1, j], table[i, j - 1], table[i - 1, j - 1]])
            minimum = float(np.min(previous))
            softmin = minimum - gamma * math.log(float(np.exp(-(previous - minimum) / gamma).sum()))
            table[i, j] = cost[i - 1, j - 1] + softmin
    return float(table[-1, -1] / max(len(x) + len(y), 1))


def modulation_correlation(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> float:
    first = np.log1p(np.abs(np.fft.rfft(rms_envelope(reference, sample_rate))))
    second = np.log1p(np.abs(np.fft.rfft(rms_envelope(candidate, sample_rate))))
    return _correlation(first, second)


def _hz_to_mel(value: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(value) / 700.0)


def _mel_to_hz(value: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(value) / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, bins: int = 80) -> np.ndarray:
    frequencies = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    points = _mel_to_hz(np.linspace(_hz_to_mel(20.0), _hz_to_mel(sample_rate / 2.0), bins + 2))
    bank = np.zeros((bins, len(frequencies)), dtype=np.float64)
    for index in range(bins):
        left, center, right = points[index : index + 3]
        bank[index] = np.maximum(0.0, np.minimum((frequencies - left) / max(center - left, 1e-8), (right - frequencies) / max(right - center, 1e-8)))
    return bank


def log_mel(audio: np.ndarray, sample_rate: int, bins: int = 80) -> np.ndarray:
    value = _unit_rms(audio)
    n_fft = max(128, round(sample_rate * 0.025))
    _, _, spectrum = stft(value, fs=sample_rate, nperseg=n_fft, noverlap=max(0, n_fft - round(sample_rate * 0.010)), nfft=n_fft, boundary=None, padded=False)
    power = np.abs(spectrum) ** 2
    return 10.0 * np.log10(mel_filterbank(sample_rate, n_fft, bins) @ power + 1e-10)


def log_mel_mae(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> float:
    first, second = log_mel(reference, sample_rate), log_mel(candidate, sample_rate)
    frames = min(first.shape[1], second.shape[1])
    return float(np.mean(np.abs(first[:, :frames] - second[:, :frames])))


def multi_resolution_stft_distance(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> float:
    first, second = _unit_rms(reference), _unit_rms(candidate)
    values = []
    for window_ms in (20.0, 50.0, 100.0):
        window = max(16, round(sample_rate * window_ms / 1000.0))
        _, _, x = stft(first, fs=sample_rate, nperseg=window, noverlap=window // 2, boundary=None)
        _, _, y = stft(second, fs=sample_rate, nperseg=window, noverlap=window // 2, boundary=None)
        frames = min(x.shape[1], y.shape[1])
        x, y = np.abs(x[:, :frames]), np.abs(y[:, :frames])
        values.append(float(np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-8)))
    return float(np.mean(values))


def reconstruction_metrics(reference: np.ndarray, candidate: np.ndarray, sample_rate: int, *, max_lag_ms: float = 250.0) -> dict[str, float]:
    size = min(len(reference), len(candidate))
    reference = np.asarray(reference[:size], dtype=np.float32)
    candidate = np.asarray(candidate[:size], dtype=np.float32)
    envelope = rms_envelope(reference, sample_rate)
    candidate_envelope = rms_envelope(candidate, sample_rate)
    lag_corr, lag = best_lag_envelope_correlation(reference, candidate, sample_rate, max_lag_ms)
    reference_zero = reference.astype(np.float64) - float(np.mean(reference))
    candidate_zero = candidate.astype(np.float64) - float(np.mean(candidate))
    projection = float(np.dot(candidate_zero, reference_zero) / max(np.dot(reference_zero, reference_zero), 1e-12)) * reference_zero
    residual = candidate_zero - projection
    si_sdr = 10.0 * math.log10(max(float(np.dot(projection, projection)), 1e-12) / max(float(np.dot(residual, residual)), 1e-12))
    return {
        "waveform_correlation": _correlation(reference, candidate),
        "si_sdr_db": si_sdr,
        "envelope_correlation": _correlation(envelope, candidate_envelope),
        "lag_envelope_correlation": lag_corr,
        "envelope_best_lag_ms": lag,
        "soft_dtw_envelope_distance": soft_dtw_distance(envelope, candidate_envelope),
        "modulation_correlation": modulation_correlation(reference, candidate, sample_rate),
        "log_mel_mae_db": log_mel_mae(reference, candidate, sample_rate),
        "multi_resolution_stft_distance": multi_resolution_stft_distance(reference, candidate, sample_rate),
    }


def summarize(records: Iterable[dict[str, float]]) -> dict[str, dict[str, float]]:
    rows = list(records)
    keys = sorted({key for row in rows for key in row})
    output: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows if key in row and np.isfinite(row[key])])
        output[key] = {
            "mean": float(np.mean(values)) if len(values) else float("nan"),
            "median": float(np.median(values)) if len(values) else float("nan"),
            "p05": float(np.percentile(values, 5)) if len(values) else float("nan"),
            "min": float(np.min(values)) if len(values) else float("nan"),
        }
    return output


__all__ = ["reconstruction_metrics", "rms_envelope", "summarize"]

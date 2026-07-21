from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Protocol

import numpy as np
from scipy.signal import correlate, correlation_lags, stft


CACHE_SCHEMA_VERSION = "combined-0715-cache-v2"
REQUIRED_CACHE_FIELDS = frozenset(
    {
        "version",
        "cache_schema_version",
        "keys",
        "datasets",
        "labels",
        "audio_relpaths",
        "audio_valid_samples",
        "encodec_codes",
        "encodec_scale",
        "encodec_scale_valid",
        "audio_envelope",
        "onset",
        "duration",
        "code_valid_steps",
        "fit_split",
        "audio_sample_rate",
        "codec_sample_rate",
        "codec_duration_sec",
        "codec_bandwidth",
    }
)
METRIC_NAMES = (
    # Fine waveform fidelity.  These deliberately remain strict and are not
    # used as the sole criterion for coarse morphology reconstruction.
    "waveform_correlation",
    "lag_aligned_waveform_correlation_abs",
    "waveform_best_lag_ms",
    "si_sdr_db",
    # Coarse temporal morphology: what is visually preserved in a waveform
    # plot after the carrier/phase detail has been compressed.
    "envelope_correlation",
    "lag_aligned_envelope_correlation",
    "envelope_best_lag_ms",
    "envelope_overlap",
    "activity_iou",
    "onset_error_ms",
    "offset_error_ms",
    "short_time_rms_correlation_20ms",
    "short_time_rms_correlation_50ms",
    "short_time_rms_correlation_100ms",
    "short_time_rms_correlation_mean",
    "structure_score",
    # Spectral fidelity at one legacy and three morphology-relevant scales.
    "log_spectrogram_mae_db",
    "multiscale_log_spectrogram_mae_db",
)

METRIC_DIRECTIONS = {
    name: ("lower" if name in {
        "waveform_best_lag_ms",
        "envelope_best_lag_ms",
        "onset_error_ms",
        "offset_error_ms",
        "log_spectrogram_mae_db",
        "multiscale_log_spectrogram_mae_db",
    } else "higher")
    for name in METRIC_NAMES
}


class _Context(Protocol):
    rows: Iterable[dict[str, str]]

    def split_for(self, row: dict[str, str]) -> str: ...


def _field_names(raw: Any) -> set[str]:
    if hasattr(raw, "files"):
        return set(raw.files)
    if isinstance(raw, Mapping):
        return set(raw)
    raise TypeError("cache must be an np.load result or a mapping of numpy arrays")


def scalar_text(value: np.ndarray | Any) -> str:
    array = np.asarray(value)
    if array.size != 1:
        return ""
    return str(array.reshape(-1)[0])


def validate_cache_arrays(
    raw: Any,
    *,
    codebooks: int = 8,
    code_steps: int = 150,
    vocab_size: int = 1024,
    audio_sample_rate: int = 16000,
    duration_sec: float = 2.0,
) -> dict[str, bool]:
    """Return strict booleans only; observed metadata belongs outside checks."""

    fields = _field_names(raw)
    required_present = REQUIRED_CACHE_FIELDS <= fields
    checks: dict[str, bool] = {"required_fields_present": bool(required_present)}
    if not required_present:
        checks.update(
            {
                "schema_version_valid": False,
                "codes_shape_valid": False,
                "first_dimension_consistent": False,
                "code_dtype_valid": False,
                "code_range_valid": False,
                "numeric_arrays_finite": False,
                "valid_samples_range": False,
                "valid_steps_range": False,
                "audio_sample_rate_valid": False,
                "duration_valid": False,
                "keys_unique": False,
                "paths_nonempty": False,
                "scale_valid_shape": False,
                "scale_shape_valid": False,
                "envelope_shape_valid": False,
                "metadata_vector_shapes_valid": False,
                "boolean_vector_dtypes_valid": False,
                "codec_metadata_valid": False,
            }
        )
        return checks

    keys = np.asarray(raw["keys"]).astype(str)
    n = int(len(keys))
    codes = np.asarray(raw["encodec_codes"])
    scales = np.asarray(raw["encodec_scale"])
    scale_valid = np.asarray(raw["encodec_scale_valid"])
    envelopes = np.asarray(raw["audio_envelope"])
    onset = np.asarray(raw["onset"])
    duration = np.asarray(raw["duration"])
    valid_samples = np.asarray(raw["audio_valid_samples"])
    valid_steps = np.asarray(raw["code_valid_steps"])
    expected_audio_samples = int(round(float(audio_sample_rate) * float(duration_sec)))
    first_dimensional = (
        "datasets",
        "labels",
        "audio_relpaths",
        "audio_valid_samples",
        "encodec_codes",
        "encodec_scale",
        "encodec_scale_valid",
        "audio_envelope",
        "onset",
        "duration",
        "code_valid_steps",
        "fit_split",
    )
    first_dimension_consistent = all(
        np.asarray(raw[name]).ndim >= 1 and len(np.asarray(raw[name])) == n for name in first_dimensional
    )
    numeric = (codes, scales, envelopes, onset, duration)
    checks.update(
        {
            "schema_version_valid": bool(
                scalar_text(raw["version"]) == CACHE_SCHEMA_VERSION
                and scalar_text(raw["cache_schema_version"]) == CACHE_SCHEMA_VERSION
            ),
            "codes_shape_valid": bool(codes.shape == (n, int(codebooks), int(code_steps))),
            "first_dimension_consistent": bool(first_dimension_consistent),
            "code_dtype_valid": bool(np.issubdtype(codes.dtype, np.integer)),
            "code_range_valid": bool(codes.size > 0 and codes.min() >= 0 and codes.max() < int(vocab_size)),
            "numeric_arrays_finite": bool(all(np.isfinite(value).all() for value in numeric)),
            "valid_samples_range": bool(
                valid_samples.shape == (n,)
                and np.issubdtype(valid_samples.dtype, np.integer)
                and valid_samples.size > 0
                and valid_samples.min() >= 1
                and valid_samples.max() <= expected_audio_samples
            ),
            "valid_steps_range": bool(
                valid_steps.shape == (n,)
                and np.issubdtype(valid_steps.dtype, np.integer)
                and valid_steps.size > 0
                and valid_steps.min() >= 1
                and valid_steps.max() <= int(code_steps)
            ),
            "audio_sample_rate_valid": bool(
                np.asarray(raw["audio_sample_rate"]).size == 1
                and int(np.asarray(raw["audio_sample_rate"]).reshape(-1)[0]) == int(audio_sample_rate)
                and np.asarray(raw["codec_sample_rate"]).size == 1
                and int(np.asarray(raw["codec_sample_rate"]).reshape(-1)[0]) > 0
            ),
            "duration_valid": bool(
                np.asarray(raw["codec_duration_sec"]).size == 1
                and np.isclose(float(np.asarray(raw["codec_duration_sec"]).reshape(-1)[0]), float(duration_sec))
            ),
            "keys_unique": bool(n > 0 and len(set(keys.tolist())) == n and all(keys)),
            "paths_nonempty": bool(all(np.asarray(raw["audio_relpaths"]).astype(str).tolist())),
            "scale_valid_shape": bool(scale_valid.shape == (n,)),
            "scale_shape_valid": bool(scales.shape == (n, 1)),
            "envelope_shape_valid": bool(envelopes.shape == (n, int(code_steps))),
            "metadata_vector_shapes_valid": bool(
                all(
                    np.asarray(raw[name]).shape == (n,)
                    for name in ("datasets", "labels", "audio_relpaths", "onset", "duration", "fit_split")
                )
            ),
            "boolean_vector_dtypes_valid": bool(
                np.issubdtype(scale_valid.dtype, np.bool_)
                and np.issubdtype(np.asarray(raw["fit_split"]).dtype, np.bool_)
            ),
            "codec_metadata_valid": bool(
                np.asarray(raw["codec_bandwidth"]).size == 1
                and np.isfinite(float(np.asarray(raw["codec_bandwidth"]).reshape(-1)[0]))
                and float(np.asarray(raw["codec_bandwidth"]).reshape(-1)[0]) > 0.0
            ),
        }
    )
    return checks


def select_stratified_validation_audio(
    context: _Context,
    available_keys: Iterable[str],
    *,
    max_per_label: int = 4,
    seed: int = 15,
) -> list[dict[str, str]]:
    """Select unique locked-validation keys by dataset and label."""

    if int(max_per_label) < 1:
        raise ValueError("max_per_label must be at least 1")
    available = {str(key) for key in available_keys}
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in context.rows:
        if context.split_for(row) != "validation":
            continue
        key = str(row["audio_key"])
        if key in available:
            grouped[(str(row["dataset"]), str(row["label"]))].setdefault(key, dict(row))

    rng = np.random.default_rng(int(seed))
    selected: list[dict[str, str]] = []
    for group in sorted(grouped):
        candidates = [grouped[group][key] for key in sorted(grouped[group])]
        if len(candidates) > int(max_per_label):
            indices = np.sort(rng.choice(len(candidates), size=int(max_per_label), replace=False))
            candidates = [candidates[int(index)] for index in indices]
        selected.extend(candidates)
    return selected


def _match_length(audio: np.ndarray, length: int) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float64).reshape(-1)
    if len(value) >= int(length):
        return value[: int(length)]
    return np.pad(value, (0, int(length) - len(value)))


def _unit_rms(audio: np.ndarray) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float64)
    rms = float(np.sqrt(np.mean(np.square(value), dtype=np.float64) + 1e-12))
    return value / max(rms, 1e-8)


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    n = min(len(first), len(second))
    if n < 2:
        return 0.0
    first = first[:n]
    second = second[:n]
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    if denominator <= 1e-12:
        return 1.0 if np.allclose(first, second, atol=1e-10, rtol=1e-7) else 0.0
    return float(np.clip((first_centered @ second_centered) / denominator, -1.0, 1.0))


def short_time_rms(audio: np.ndarray, sample_rate: int, *, window_ms: float, hop_ms: float = 10.0) -> np.ndarray:
    """Return a finite frame-RMS envelope, including short/silent signals."""

    value = np.asarray(audio, dtype=np.float64).reshape(-1)
    window = max(1, int(round(float(window_ms) * int(sample_rate) / 1000.0)))
    hop = max(1, int(round(float(hop_ms) * int(sample_rate) / 1000.0)))
    if len(value) < window:
        value = np.pad(value, (0, window - len(value)))
    starts = np.arange(0, max(len(value) - window + 1, 1), hop, dtype=np.int64)
    if starts[-1] != len(value) - window:
        starts = np.append(starts, len(value) - window)
    cumulative = np.concatenate(([0.0], np.cumsum(np.square(value), dtype=np.float64)))
    frame_energy = (cumulative[starts + window] - cumulative[starts]) / float(window)
    return np.sqrt(np.maximum(frame_energy, 0.0) + 1e-12).astype(np.float64)


def _best_lag_correlation(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    max_lag_samples: int,
    use_absolute: bool,
) -> tuple[float, int]:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    n = min(len(reference), len(candidate))
    reference = reference[:n] - reference[:n].mean()
    candidate = candidate[:n] - candidate[:n].mean()
    if n < 2 or np.linalg.norm(reference) <= 1e-12 or np.linalg.norm(candidate) <= 1e-12:
        return (_safe_correlation(reference, candidate), 0)
    cross = correlate(candidate, reference, mode="full", method="fft")
    lags = correlation_lags(len(candidate), len(reference), mode="full")
    selected = np.abs(lags) <= max(0, int(max_lag_samples))
    scores = np.abs(cross[selected]) if use_absolute else cross[selected]
    lag = int(lags[selected][int(np.argmax(scores))])
    if lag >= 0:
        aligned_reference = reference[: n - lag]
        aligned_candidate = candidate[lag:]
    else:
        aligned_reference = reference[-lag:]
        aligned_candidate = candidate[: n + lag]
    correlation = _safe_correlation(aligned_reference, aligned_candidate)
    return (abs(correlation) if use_absolute else correlation, lag)


def _normalized_overlap(first: np.ndarray, second: np.ndarray) -> float:
    first = np.maximum(np.asarray(first, dtype=np.float64), 0.0)
    second = np.maximum(np.asarray(second, dtype=np.float64), 0.0)
    first_peak = float(first.max(initial=0.0))
    second_peak = float(second.max(initial=0.0))
    if first_peak <= 1e-10 and second_peak <= 1e-10:
        return 1.0
    if first_peak <= 1e-10 or second_peak <= 1e-10:
        return 0.0
    first = first / first_peak
    second = second / second_peak
    denominator = float(np.maximum(first, second).sum())
    return float(np.minimum(first, second).sum() / max(denominator, 1e-12))


def _activity_metrics(reference: np.ndarray, candidate: np.ndarray, hop_ms: float) -> tuple[float, float, float]:
    def active(value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        peak = float(value.max(initial=0.0))
        return value >= max(peak * 0.10, 1e-8) if peak > 1e-8 else np.zeros(len(value), dtype=bool)

    ref_active = active(reference)
    candidate_active = active(candidate)
    union = ref_active | candidate_active
    activity_iou = float((ref_active & candidate_active).sum() / union.sum()) if union.any() else 1.0
    duration_ms = max(len(reference), len(candidate)) * float(hop_ms)

    def bounds(mask: np.ndarray) -> tuple[int, int] | None:
        indices = np.flatnonzero(mask)
        return (int(indices[0]), int(indices[-1])) if len(indices) else None

    ref_bounds = bounds(ref_active)
    candidate_bounds = bounds(candidate_active)
    if ref_bounds is None and candidate_bounds is None:
        return activity_iou, 0.0, 0.0
    if ref_bounds is None or candidate_bounds is None:
        return activity_iou, duration_ms, duration_ms
    return (
        activity_iou,
        abs(candidate_bounds[0] - ref_bounds[0]) * float(hop_ms),
        abs(candidate_bounds[1] - ref_bounds[1]) * float(hop_ms),
    )


def _log_spectrogram_mae(reference: np.ndarray, candidate: np.ndarray, sample_rate: int, window_ms: float) -> float:
    n = min(len(reference), len(candidate))
    nperseg = min(n, max(16, int(round(float(window_ms) * int(sample_rate) / 1000.0))))
    noverlap = min(nperseg - 1, int(round(0.75 * nperseg)))
    _, _, ref_spec = stft(reference[:n], fs=int(sample_rate), nperseg=nperseg, noverlap=noverlap, boundary=None)
    _, _, candidate_spec = stft(candidate[:n], fs=int(sample_rate), nperseg=nperseg, noverlap=noverlap, boundary=None)
    frames = min(ref_spec.shape[1], candidate_spec.shape[1])
    ref_db = 20.0 * np.log10(np.maximum(np.abs(ref_spec[:, :frames]), 1e-4))
    candidate_db = 20.0 * np.log10(np.maximum(np.abs(candidate_spec[:, :frames]), 1e-4))
    return float(np.mean(np.abs(ref_db - candidate_db)))


def waveform_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    *,
    valid_samples: int | None = None,
) -> dict[str, float]:
    """Compute strict fidelity plus phase-robust temporal morphology metrics."""

    n = min(len(np.asarray(reference).reshape(-1)), len(np.asarray(candidate).reshape(-1)))
    if valid_samples is not None:
        n = min(n, int(valid_samples))
    if n < 2:
        raise ValueError("At least two valid audio samples are required")
    reference = _match_length(reference, n)
    candidate = _match_length(candidate, n)
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Audio contains NaN or Inf")

    ref_centered = reference - reference.mean()
    candidate_centered = candidate - candidate.mean()
    correlation = _safe_correlation(ref_centered, candidate_centered)
    lag_correlation, waveform_lag = _best_lag_correlation(
        reference,
        candidate,
        max_lag_samples=int(round(0.100 * int(sample_rate))),
        use_absolute=True,
    )
    projection = float((candidate @ reference) / (reference @ reference + 1e-12))
    target = projection * reference
    residual = candidate - target
    si_sdr = float(10.0 * np.log10((target @ target + 1e-12) / (residual @ residual + 1e-12)))

    ref_normalized = _unit_rms(reference)
    candidate_normalized = _unit_rms(candidate)
    # Preserve the original 512-sample definition for checkpoint/report
    # comparability; the new multi-scale metric is explicitly time based.
    legacy_spectral_mae = _log_spectrogram_mae(
        ref_normalized,
        candidate_normalized,
        sample_rate,
        512.0 * 1000.0 / float(sample_rate),
    )
    multiscale_spectral_mae = float(
        np.mean(
            [
                _log_spectrogram_mae(ref_normalized, candidate_normalized, sample_rate, window_ms)
                for window_ms in (20.0, 50.0, 100.0)
            ]
        )
    )

    envelopes = {
        window_ms: (
            short_time_rms(reference, sample_rate, window_ms=window_ms),
            short_time_rms(candidate, sample_rate, window_ms=window_ms),
        )
        for window_ms in (20.0, 25.0, 50.0, 100.0)
    }
    envelope_reference, envelope_candidate = envelopes[25.0]
    envelope_correlation = _safe_correlation(envelope_reference, envelope_candidate)
    lag_envelope_correlation, envelope_lag_frames = _best_lag_correlation(
        envelope_reference,
        envelope_candidate,
        max_lag_samples=10,  # 10 frames x 10-ms hop = +/-100 ms
        use_absolute=False,
    )
    rms_correlations = {
        window_ms: _safe_correlation(*envelopes[window_ms])
        for window_ms in (20.0, 50.0, 100.0)
    }
    envelope_overlap = _normalized_overlap(envelope_reference, envelope_candidate)
    activity_iou, onset_error_ms, offset_error_ms = _activity_metrics(
        envelope_reference, envelope_candidate, 10.0
    )
    structure_score = float(
        0.45 * np.clip(envelope_correlation, 0.0, 1.0)
        + 0.30 * envelope_overlap
        + 0.25 * activity_iou
    )
    return {
        "waveform_correlation": correlation,
        "lag_aligned_waveform_correlation_abs": float(lag_correlation),
        "waveform_best_lag_ms": float(abs(waveform_lag) * 1000.0 / int(sample_rate)),
        "si_sdr_db": si_sdr,
        "envelope_correlation": envelope_correlation,
        "lag_aligned_envelope_correlation": float(lag_envelope_correlation),
        "envelope_best_lag_ms": float(abs(envelope_lag_frames) * 10.0),
        "envelope_overlap": envelope_overlap,
        "activity_iou": activity_iou,
        "onset_error_ms": onset_error_ms,
        "offset_error_ms": offset_error_ms,
        "short_time_rms_correlation_20ms": rms_correlations[20.0],
        "short_time_rms_correlation_50ms": rms_correlations[50.0],
        "short_time_rms_correlation_100ms": rms_correlations[100.0],
        "short_time_rms_correlation_mean": float(np.mean(list(rms_correlations.values()))),
        "structure_score": structure_score,
        "log_spectrogram_mae_db": legacy_spectral_mae,
        "multiscale_log_spectrogram_mae_db": multiscale_spectral_mae,
    }


def decode_cached_sample(codec: Any, codes: np.ndarray, scale: np.ndarray, scale_valid: bool) -> np.ndarray:
    """Decode exact cached codes while respecting EnCodec's optional scale."""

    return np.asarray(codec.decode(codes, scale=scale if bool(scale_valid) else None), dtype=np.float32)


def summarise_metric_records(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    rows = list(records)
    if not rows:
        return {}
    summary: dict[str, dict[str, float]] = {}
    for name in METRIC_NAMES:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        summary[name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)),
            "min": float(np.min(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
    return summary

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Protocol

import numpy as np
from scipy.signal import stft


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
METRIC_NAMES = ("waveform_correlation", "si_sdr_db", "log_spectrogram_mae_db")


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


def waveform_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    *,
    valid_samples: int | None = None,
) -> dict[str, float]:
    """Compute correlation, SI-SDR and RMS-normalized spectral MAE."""

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
    correlation = float(
        (ref_centered @ candidate_centered)
        / (np.linalg.norm(ref_centered) * np.linalg.norm(candidate_centered) + 1e-12)
    )
    projection = float((candidate @ reference) / (reference @ reference + 1e-12))
    target = projection * reference
    residual = candidate - target
    si_sdr = float(10.0 * np.log10((target @ target + 1e-12) / (residual @ residual + 1e-12)))

    ref_normalized = _unit_rms(reference)
    candidate_normalized = _unit_rms(candidate)
    nperseg = min(512, n)
    noverlap = min(384, max(nperseg - 1, 0))
    _, _, ref_spec = stft(ref_normalized, fs=int(sample_rate), nperseg=nperseg, noverlap=noverlap, boundary=None)
    _, _, candidate_spec = stft(candidate_normalized, fs=int(sample_rate), nperseg=nperseg, noverlap=noverlap, boundary=None)
    frames = min(ref_spec.shape[1], candidate_spec.shape[1])
    # A fixed -80 dB floor after unit-RMS normalization prevents inaudible
    # codec noise in near-zero bins from dominating the spectrogram metric.
    ref_db = 20.0 * np.log10(np.maximum(np.abs(ref_spec[:, :frames]), 1e-4))
    candidate_db = 20.0 * np.log10(np.maximum(np.abs(candidate_spec[:, :frames]), 1e-4))
    return {
        "waveform_correlation": correlation,
        "si_sdr_db": si_sdr,
        "log_spectrogram_mae_db": float(np.mean(np.abs(ref_db - candidate_db))),
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
        }
    return summary

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.combined_0715.audio_eval import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    decode_cached_sample,
    select_stratified_validation_audio,
    validate_cache_arrays,
    waveform_metrics,
)
from src.combined_0715.data import AudioCodeBank  # noqa: E402


def cache_payload(*, codebooks: int = 8) -> dict[str, np.ndarray]:
    n, steps = 2, 150
    return {
        "version": np.asarray(CACHE_SCHEMA_VERSION),
        "cache_schema_version": np.asarray(CACHE_SCHEMA_VERSION),
        "keys": np.asarray(["key-a", "key-b"]),
        "datasets": np.asarray(["feis", "karaone"]),
        "labels": np.asarray(["a", "b"]),
        "audio_relpaths": np.asarray(["audio/a.wav", "audio/b.wav"]),
        "audio_valid_samples": np.asarray([16000, 32000], dtype=np.int64),
        "encodec_codes": np.zeros((n, codebooks, steps), dtype=np.int16),
        "encodec_scale": np.ones((n, 1), dtype=np.float32),
        "encodec_scale_valid": np.asarray([True, False]),
        "audio_envelope": np.zeros((n, steps), dtype=np.float32),
        "onset": np.zeros(n, dtype=np.float32),
        "duration": np.ones(n, dtype=np.float32),
        "code_valid_steps": np.asarray([75, 150], dtype=np.int64),
        "fit_split": np.asarray([True, False]),
        "audio_sample_rate": np.asarray(16000, dtype=np.int32),
        "codec_sample_rate": np.asarray(24000, dtype=np.int32),
        "codec_duration_sec": np.asarray(2.0, dtype=np.float32),
        "codec_bandwidth": np.asarray(6.0, dtype=np.float32),
    }


def test_cache_shape_check_is_a_strict_boolean() -> None:
    valid = validate_cache_arrays(cache_payload())
    assert all(valid.values())

    invalid = validate_cache_arrays(cache_payload(codebooks=7))
    assert invalid["codes_shape_valid"] is False
    assert not all(invalid.values())
    assert all(isinstance(value, bool) for value in invalid.values())


def test_runtime_code_bank_enforces_the_same_v2_contract(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.npz"
    np.savez_compressed(valid_path, **cache_payload())
    assert len(AudioCodeBank(valid_path).keys) == 2

    invalid_path = tmp_path / "invalid.npz"
    np.savez_compressed(invalid_path, **cache_payload(codebooks=7))
    with pytest.raises(ValueError, match="codes_shape_valid"):
        AudioCodeBank(invalid_path)


def test_identical_waveform_has_near_perfect_roundtrip_metrics() -> None:
    time = np.arange(16000, dtype=np.float64) / 16000.0
    reference = np.sin(2.0 * np.pi * 220.0 * time).astype(np.float32)
    metrics = waveform_metrics(reference, 2.0 * reference, 16000)
    assert metrics["waveform_correlation"] > 0.9999
    assert metrics["si_sdr_db"] > 80.0
    assert metrics["log_spectrogram_mae_db"] < 1.0e-4


def test_decode_cached_sample_uses_scale_only_when_valid() -> None:
    class DummyCodec:
        def __init__(self) -> None:
            self.scales: list[np.ndarray | None] = []

        def decode(self, codes: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
            self.scales.append(scale)
            return np.asarray(codes[0], dtype=np.float32)

    codec = DummyCodec()
    codes = np.zeros((8, 150), dtype=np.int16)
    scale = np.asarray([0.25], dtype=np.float32)
    decode_cached_sample(codec, codes, scale, True)
    decode_cached_sample(codec, codes, scale, False)
    assert np.array_equal(codec.scales[0], scale)
    assert codec.scales[1] is None


def test_validation_selection_is_stratified_deterministic_and_excludes_test() -> None:
    class Context:
        def __init__(self) -> None:
            self.rows = []
            for split in ("validation", "test"):
                for index in range(6):
                    self.rows.append(
                        {
                            "split": split,
                            "dataset": "feis",
                            "label": "word",
                            "audio_key": f"{split}-{index}",
                        }
                    )
            self.rows.append(
                {"split": "validation", "dataset": "karaone", "label": "phoneme", "audio_key": "kara-val"}
            )

        def split_for(self, row: dict[str, str]) -> str:
            return row["split"]

    context = Context()
    available = [row["audio_key"] for row in context.rows]
    first = select_stratified_validation_audio(context, available, max_per_label=4, seed=19)
    second = select_stratified_validation_audio(context, available, max_per_label=4, seed=19)
    assert [row["audio_key"] for row in first] == [row["audio_key"] for row in second]
    assert len([row for row in first if row["dataset"] == "feis"]) == 4
    assert len([row for row in first if row["dataset"] == "karaone"]) == 1
    assert all(row["split"] == "validation" for row in first)

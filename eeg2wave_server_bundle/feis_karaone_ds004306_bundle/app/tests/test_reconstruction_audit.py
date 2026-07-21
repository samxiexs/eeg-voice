from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


APP_DIR = Path(__file__).resolve().parents[1]
SCRIPT = APP_DIR / "scripts" / "audit_combined_0715_reconstruction.py"
sys.path.insert(0, str(APP_DIR))
SPEC = importlib.util.spec_from_file_location("combined_reconstruction_audit", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["combined_reconstruction_audit"] = MODULE
SPEC.loader.exec_module(MODULE)


def _sample(eeg: float, shuffled: float, zero: float, dataset: float, label: float) -> dict:
    return {
        mode: {
            "envelope_correlation": value,
            "activity_iou": value,
            "log_spectrogram_mae_db": 1.0 - value,
        }
        for mode, value in {
            "codec_oracle": 0.95,
            "eeg_conditioned": eeg,
            "label_only": label,
            "zero_eeg": zero,
            "shuffled_eeg": shuffled,
            "dataset_only": dataset,
        }.items()
    }


def _evaluate(samples: list[dict]) -> dict:
    return MODULE.evaluate_dataset_metrics(
        samples,
        structure_thresholds={"envelope_correlation": 0.30, "activity_iou": 0.30},
        paired_median_delta_min=0.03,
        paired_win_rate_min=0.55,
    )


def test_temporal_metrics_reward_matching_coarse_energy_structure() -> None:
    sample_rate = 1000
    reference = np.zeros(2000, dtype=np.float64)
    reference[600:1200] = np.sin(np.linspace(0, 80 * np.pi, 600))
    candidate = np.zeros_like(reference)
    candidate[600:1200] = np.sin(np.linspace(0, 35 * np.pi, 600) + 0.8)
    metrics = MODULE.temporal_structure_metrics(reference, candidate, sample_rate)
    assert metrics["envelope_correlation"] > 0.85
    assert metrics["activity_iou"] > 0.85


def test_structure_and_all_three_eeg_specific_controls_must_pass() -> None:
    samples = [_sample(0.72, 0.20, 0.10, 0.15, 0.75) for _ in range(12)]
    result = _evaluate(samples)
    assert result["structure_reconstruction_passed"] is True
    assert result["eeg_specific_reconstruction_passed"] is True
    envelope = result["metrics"]["envelope_correlation"]
    assert set(envelope["paired_negative_controls"]) == set(MODULE.NEGATIVE_CONTROLS)
    assert "label_only" not in envelope["paired_negative_controls"]
    assert envelope["mode_summaries"]["label_only"]["median"] == pytest.approx(0.75)


def test_visual_overlap_without_advantage_over_dataset_only_does_not_pass_eeg_specificity() -> None:
    samples = [_sample(0.82, 0.30, 0.20, 0.84, 0.85) for _ in range(12)]
    result = _evaluate(samples)
    assert result["structure_reconstruction_passed"] is True
    assert result["eeg_specific_reconstruction_passed"] is False
    comparison = result["metrics"]["activity_iou"]["paired_negative_controls"]["dataset_only"]
    assert comparison["median_delta"] < 0.0
    assert comparison["win_rate"] == 0.0


def test_lower_is_better_metrics_reverse_the_paired_delta_direction() -> None:
    summary = MODULE.paired_summary(
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        direction="lower_is_better",
    )
    assert summary["median_delta"] == pytest.approx(3.0)
    assert summary["win_rate"] == pytest.approx(1.0)
    assert summary["delta_definition"] == "control_minus_eeg"


def test_win_rate_cannot_be_replaced_by_a_large_median_gain() -> None:
    samples = []
    for index in range(20):
        eeg = 0.90 if index < 10 else 0.29
        samples.append(_sample(eeg, 0.30, 0.20, 0.10, 0.50))
    result = _evaluate(samples)
    comparison = result["metrics"]["envelope_correlation"]["paired_negative_controls"]["shuffled_eeg"]
    assert comparison["median_delta"] >= 0.03
    assert comparison["win_rate"] == pytest.approx(0.50)
    assert comparison["passed"] is False

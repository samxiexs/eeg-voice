from __future__ import annotations

import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import numpy as np

from scripts.train_combined_0715 import SameLabelPairBatchSampler, multi_scale_vector_correlation


def _rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label in ("a", "b"):
        for subject in ("s1", "s2", "s3"):
            rows.append(
                {
                    "label": label,
                    "subject_group_id": subject,
                    "audio_key": f"{label}-{subject}",
                }
            )
    return rows


def test_same_label_pair_sampler_always_activates_different_audio_pairs() -> None:
    rows = _rows()
    sampler = SameLabelPairBatchSampler(rows, batch_size=4, seed=21)
    for batch in sampler:
        assert len(batch) == 4
        for first, second in zip(batch[::2], batch[1::2]):
            assert rows[first]["label"] == rows[second]["label"]
            assert rows[first]["audio_key"] != rows[second]["audio_key"]


def test_same_label_pair_sampler_is_reproducible_from_seed() -> None:
    rows = _rows()
    first = list(SameLabelPairBatchSampler(rows, batch_size=4, seed=21))
    second = list(SameLabelPairBatchSampler(rows, batch_size=4, seed=21))
    assert first == second


def test_multiscale_validation_score_rewards_matching_energy_structure() -> None:
    target = np.asarray([0.0, 0.0, 0.2, 0.9, 1.0, 0.3, 0.0, 0.0])
    matching = target * 0.8 + 0.01
    shifted = np.roll(target, 3)
    assert multi_scale_vector_correlation(matching, target) > multi_scale_vector_correlation(
        shifted, target
    )

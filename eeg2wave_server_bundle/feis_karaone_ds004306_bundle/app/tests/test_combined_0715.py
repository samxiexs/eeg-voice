from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from src.combined_0715.model import EEGConditionEncoder, EEGModelConfig  # noqa: E402


def test_eeg_model_shape_and_dataset_heads() -> None:
    model = EEGConditionEncoder(EEGModelConfig())
    output = model(torch.randn(2, 14, 768), torch.tensor([768, 700]), torch.tensor([0, 2]))
    assert output["condition"].shape == (2, 50, 192)
    assert output["label_logits"].shape == (2, 30)
    assert output["subject_logits"].shape[1] == 38


def test_valid_baseline_tail_is_ignored() -> None:
    script = APP_DIR.parent / "scripts" / "preprocess_combined_eeg.py"
    spec = importlib.util.spec_from_file_location("combined_preprocess", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["combined_preprocess"] = module
    spec.loader.exec_module(module)
    signal = np.zeros((1, 14, 4), dtype=np.float32)
    baseline = np.zeros((1, 14, 6), dtype=np.float32)
    for channel in range(14):
        baseline[0, channel, :3] = (channel + 1) * np.asarray([-1.0, 0.0, 1.0])
    changed_tail = baseline.copy()
    changed_tail[0, :, 3:] = (np.arange(14, dtype=np.float32)[:, None] + 1.0) * 1.0e6
    original = module.robust_baseline_normalise(signal, baseline, np.asarray([3]))
    modified = module.robust_baseline_normalise(signal, changed_tail, np.asarray([3]))
    assert np.isfinite(original).all()
    np.testing.assert_allclose(original, modified, rtol=0.0, atol=0.0)


def test_sample_key_is_recording_scoped() -> None:
    manifest = APP_DIR.parent / "eeg_output" / "manifests" / "unified_trials.csv"
    if not manifest.is_file():
        return
    import csv

    with manifest.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keys = [row["sample_key"] for row in rows]
    assert len(keys) == len(set(keys))
    assert all(row["sample_key"].startswith(row["subject_recording_id"] + ":") for row in rows)

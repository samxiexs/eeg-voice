from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


APP_DIR = Path(__file__).resolve().parents[1]
BUNDLE_DIR = APP_DIR.parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SIGNAL = _load_module("combined_signal_probe_test", APP_DIR / "scripts" / "diagnose_combined_0715_signal.py")
PREPROCESS = _load_module("combined_preprocess_qc_test", BUNDLE_DIR / "scripts" / "preprocess_combined_eeg.py")


def test_signal_features_preserve_channel_information_and_ignore_padding() -> None:
    samples = 768
    valid = 512
    time = np.arange(samples, dtype=np.float32) / 256.0
    base = np.zeros((14, samples), dtype=np.float32)
    base[0, :valid] = np.sin(2.0 * np.pi * 10.0 * time[:valid])

    other_channel = base.copy()
    other_channel[1, :valid] = 0.75 * np.sin(2.0 * np.pi * 20.0 * time[:valid])
    base_features = SIGNAL.extract_signal_features(base, valid)
    channel_features = SIGNAL.extract_signal_features(other_channel, valid)
    assert base_features.shape == (141,)
    assert not np.allclose(base_features, channel_features)

    changed_padding = base.copy()
    changed_padding[:, valid:] = 1.0e6
    padding_features = SIGNAL.extract_signal_features(changed_padding, valid)
    np.testing.assert_allclose(base_features, padding_features, rtol=0.0, atol=0.0)


def _write_synthetic_output(root: Path, *, mutation: str | None = None) -> Path:
    subjects = ("feis:01", "feis:02")
    manifest_rows = []
    for subject_number, group in enumerate(subjects, start=1):
        eeg = np.zeros((1, 14, 1280), dtype=np.float32)
        eeg[:, :, :768] = 0.1
        valid_lengths = np.asarray([768], dtype=np.int32)
        channels = np.asarray(PREPROCESS.COMMON_CHANNELS)
        if mutation == "nan" and subject_number == 1:
            eeg[0, 0, 0] = np.nan
        if mutation == "channels" and subject_number == 1:
            channels = channels[::-1]
        if mutation == "valid_length" and subject_number == 1:
            valid_lengths[0] = 1281
        relative = Path("subjects") / "feis" / f"{subject_number:02d}.npz"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            eeg=eeg,
            valid_lengths=valid_lengths,
            trial_indices=np.asarray([0], dtype=np.int32),
            labels=np.asarray(["label"]),
            channel_names=channels,
            eeg_sfreq_hz=np.asarray([256], dtype=np.int32),
            dataset=np.asarray(["feis"]),
        )
        manifest_rows.append({
            "dataset": "feis",
            "subject_group_id": group,
            "subject_recording_id": group,
            "trial_index": "0",
            "sample_key": f"{group}:0",
            "label": "label",
            "eeg_relpath": str(relative),
            "eeg_row": "0",
            "eeg_valid_samples": str(int(valid_lengths[0])),
            "eeg_sfreq_hz": "256",
        })
    manifest = root / "manifests" / "unified_trials.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    qc = root / "qc" / "channel_qc.csv"
    qc.parent.mkdir(parents=True, exist_ok=True)
    qc.write_text("dataset,channel\nfeis,F3\n", encoding="utf-8")

    split = root / "locked.yaml"
    validation = '["feis:02"]'
    if mutation == "split_overlap":
        validation = '["feis:01", "feis:02"]'
    split.write_text(
        "version: test-split\n"
        "unit: subject_group_id\n"
        "datasets:\n"
        "  feis:\n"
        '    train: ["feis:01"]\n'
        f"    validation: {validation}\n"
        "    test: []\n",
        encoding="utf-8",
    )
    return split


def test_preprocessing_qc_accepts_valid_output(tmp_path: Path) -> None:
    split = _write_synthetic_output(tmp_path)
    report = PREPROCESS.verify_preprocessed_outputs(tmp_path, split)
    assert report["status"] == "passed"
    assert report["summary"]["finite_output"] is True
    assert (tmp_path / "qc" / "eeg_verification.jsonl").is_file()


@pytest.mark.parametrize(
    ("mutation", "expected_check"),
    (
        ("nan", "finite_output"),
        ("channels", "npz_integrity"),
        ("valid_length", "npz_integrity"),
        ("split_overlap", "locked_split"),
    ),
)
def test_preprocessing_qc_rejects_critical_failures(
    tmp_path: Path,
    mutation: str,
    expected_check: str,
) -> None:
    split = _write_synthetic_output(tmp_path, mutation=mutation)
    report = PREPROCESS.verify_preprocessed_outputs(tmp_path, split)
    assert report["status"] == "failed"
    assert report["checks"][expected_check]["passed"] is False

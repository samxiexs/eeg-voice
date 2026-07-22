from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch


APP_DIR = Path(__file__).resolve().parents[1]
SCRIPT = APP_DIR / "scripts" / "select_combined_0721_checkpoint.py"
sys.path.insert(0, str(APP_DIR))
SPEC = importlib.util.spec_from_file_location("combined_checkpoint_selection", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["combined_checkpoint_selection"] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_candidate(tmp_path: Path, name: str, eeg: float, controls: float) -> tuple[Path, Path]:
    lineage = {
        "schema_version": "combined-0715-lineage-v1",
        "config_sha256": "config",
        "split_sha256": "split",
        "split_version": "v1",
        "manifest_sha256": "manifest",
        "preprocessing_sha256": "eeg",
        "cache_sha256": "cache",
        "cache_version": "combined-0715-cache-v2",
    }
    audio_sha = "audio"
    checkpoint = tmp_path / f"{name}.pt"
    torch.save(
        {
            "checkpoint_schema_version": MODULE.CHECKPOINT_SCHEMA_VERSION,
            "phase": "eeg",
            "epoch": 4 if name == "early" else 40,
            "lineage": lineage,
            "dependencies": {"audio_checkpoint_sha256": audio_sha},
            "state_dict": {},
        },
        checkpoint,
    )
    rows = []
    for index in range(12):
        mode_metrics = {}
        for mode, value in {
            "eeg_conditioned": eeg,
            "shuffled_eeg": controls,
            "zero_eeg": controls,
            "dataset_only": controls,
        }.items():
            mode_metrics[mode] = {
                "structure_score": value,
                "envelope_correlation": value,
                "activity_iou": value,
                "short_time_rms_correlation_mean": value,
            }
        rows.append({"sample_key": f"sample-{index}", "mode_metrics": mode_metrics})
    manifest = {
        "version": MODULE.EXPECTED_MANIFEST_VERSION,
        "dataset": "karaone",
        "split": "validation",
        "test_accessed": False,
        "n_generated": len(rows),
        "eeg_checkpoint_sha256": MODULE.file_sha256(checkpoint),
        "audio_checkpoint_sha256": audio_sha,
        "lineage": lineage,
        "files": rows,
    }
    manifest_path = tmp_path / f"{name}.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return checkpoint, manifest_path


def _evaluate(name: str, checkpoint: Path, manifest: Path) -> dict:
    return MODULE.evaluate_candidate(
        name,
        checkpoint,
        manifest,
        minimum_samples=12,
        envelope_correlation_min=0.30,
        activity_iou_min=0.30,
        paired_median_delta_min=0.03,
        paired_win_rate_min=0.55,
    )


def test_decoded_selection_rewards_eeg_specific_gain_not_generic_overlap(tmp_path: Path) -> None:
    generic_checkpoint, generic_manifest = _write_candidate(tmp_path, "generic", eeg=0.90, controls=0.89)
    specific_checkpoint, specific_manifest = _write_candidate(tmp_path, "specific", eeg=0.70, controls=0.20)
    generic = _evaluate("generic", generic_checkpoint, generic_manifest)
    specific = _evaluate("specific", specific_checkpoint, specific_manifest)
    assert generic["absolute_structure_passed"] is True
    assert generic["eeg_specific_controls_passed"] is False
    assert specific["passed"] is True
    assert specific["selection_score"] > generic["selection_score"]


def test_selection_refuses_locked_test_manifest(tmp_path: Path) -> None:
    checkpoint, manifest_path = _write_candidate(tmp_path, "test", eeg=0.70, controls=0.20)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split"] = "test"
    manifest["test_accessed"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        _evaluate("test", checkpoint, manifest_path)
    except ValueError as error:
        assert "validation" in str(error)
    else:
        raise AssertionError("locked test manifest was accepted")

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from statistics import median
from typing import Any

import torch


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.combined_0715.lineage import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    comparable_lineage,
    file_sha256,
)


EXPECTED_MANIFEST_VERSION = "combined-0715-synthesis-v3"
CONTROL_MODES = ("shuffled_eeg", "zero_eeg", "dataset_only")
GATE_METRICS = ("envelope_correlation", "activity_iou")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select an EEG checkpoint from actual decoded KaraOne validation reconstructions. "
            "The locked test split is never accepted or read."
        )
    )
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=3,
        metavar=("NAME", "CHECKPOINT", "SYNTHESIS_MANIFEST"),
        required=True,
        help="Repeat for each candidate checkpoint and its KaraOne validation synthesis manifest.",
    )
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--minimum-samples", type=int, default=12)
    parser.add_argument("--envelope-correlation-min", type=float, default=0.30)
    parser.add_argument("--activity-iou-min", type=float, default=0.30)
    parser.add_argument("--paired-median-delta-min", type=float, default=0.03)
    parser.add_argument("--paired-win-rate-min", type=float, default=0.55)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def finite_metric(row: dict[str, Any], mode: str, metric: str, source: str) -> float:
    modes = row.get("mode_metrics")
    require(isinstance(modes, dict), f"{source} has no mode_metrics object")
    values = modes.get(mode)
    require(isinstance(values, dict), f"{source} has no {mode} metrics")
    value = values.get(metric)
    require(isinstance(value, (int, float)), f"{source} has no numeric {mode}.{metric}")
    number = float(value)
    require(math.isfinite(number), f"{source} has non-finite {mode}.{metric}")
    return number


def paired_summary(rows: list[dict[str, Any]], metric: str, control: str) -> dict[str, float]:
    deltas = [
        finite_metric(row, "eeg_conditioned", metric, str(row.get("sample_key")))
        - finite_metric(row, control, metric, str(row.get("sample_key")))
        for row in rows
    ]
    return {
        "median_delta": float(median(deltas)),
        "mean_delta": float(sum(deltas) / len(deltas)),
        "win_rate": float(sum(value > 0.0 for value in deltas) / len(deltas)),
    }


def evaluate_candidate(
    name: str,
    checkpoint_path: Path,
    manifest_path: Path,
    *,
    minimum_samples: int,
    envelope_correlation_min: float,
    activity_iou_min: float,
    paired_median_delta_min: float,
    paired_win_rate_min: float,
) -> dict[str, Any]:
    require(checkpoint_path.is_file(), f"Missing checkpoint for {name}: {checkpoint_path}")
    require(manifest_path.is_file(), f"Missing synthesis manifest for {name}: {manifest_path}")
    checkpoint_sha = file_sha256(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(isinstance(payload, dict), f"Checkpoint is not a mapping: {checkpoint_path}")
    require(
        payload.get("checkpoint_schema_version") == CHECKPOINT_SCHEMA_VERSION,
        f"{checkpoint_path} uses an incompatible checkpoint schema",
    )
    require(payload.get("phase") == "eeg", f"{checkpoint_path} is not an EEG checkpoint")

    manifest = read_json(manifest_path)
    require(manifest.get("version") == EXPECTED_MANIFEST_VERSION, f"Unsupported manifest: {manifest_path}")
    require(manifest.get("dataset") == "karaone", f"Selection must use KaraOne, not {manifest.get('dataset')!r}")
    require(manifest.get("split") == "validation", "Checkpoint selection must use validation only")
    require(manifest.get("test_accessed") is False, "Refusing a manifest that accessed locked test")
    require(
        manifest.get("eeg_checkpoint_sha256") == checkpoint_sha,
        f"Manifest/checkpoint SHA mismatch for {name}",
    )
    require(
        comparable_lineage(payload.get("lineage") or {}) == comparable_lineage(manifest.get("lineage") or {}),
        f"Manifest/checkpoint lineage mismatch for {name}",
    )
    dependencies = payload.get("dependencies") or {}
    require(
        dependencies.get("audio_checkpoint_sha256") == manifest.get("audio_checkpoint_sha256"),
        f"Audio dependency mismatch for {name}",
    )
    rows = manifest.get("files")
    require(isinstance(rows, list), f"Manifest files must be a list: {manifest_path}")
    require(len(rows) >= minimum_samples, f"{name} has {len(rows)} samples; need at least {minimum_samples}")
    require(int(manifest.get("n_generated", -1)) == len(rows), f"n_generated mismatch for {name}")
    sample_keys = [str(row.get("sample_key")) for row in rows]
    require(len(sample_keys) == len(set(sample_keys)), f"Duplicate sample keys in {name}")

    absolute = {
        metric: float(median([
            finite_metric(row, "eeg_conditioned", metric, str(row.get("sample_key"))) for row in rows
        ]))
        for metric in ("structure_score", *GATE_METRICS, "short_time_rms_correlation_mean")
    }
    paired = {
        control: {
            metric: paired_summary(rows, metric, control)
            for metric in ("structure_score", *GATE_METRICS)
        }
        for control in CONTROL_MODES
    }
    absolute_passed = bool(
        absolute["envelope_correlation"] >= envelope_correlation_min
        and absolute["activity_iou"] >= activity_iou_min
    )
    paired_passed = bool(all(
        paired[control][metric]["median_delta"] >= paired_median_delta_min
        and paired[control][metric]["win_rate"] >= paired_win_rate_min
        for control in CONTROL_MODES
        for metric in GATE_METRICS
    ))

    # Absolute structure prevents selecting pure noise. The larger weights on
    # shuffled/zero/dataset-only gains prevent a category template or generic
    # decoder prior from being mistaken for EEG-specific reconstruction.
    selection_score = float(
        absolute["structure_score"]
        + 1.50 * paired["shuffled_eeg"]["structure_score"]["median_delta"]
        + 0.50 * paired["zero_eeg"]["structure_score"]["median_delta"]
        + 0.50 * paired["dataset_only"]["structure_score"]["median_delta"]
        + 0.25 * (paired["shuffled_eeg"]["structure_score"]["win_rate"] - 0.50)
        + 0.10 * (paired["zero_eeg"]["structure_score"]["win_rate"] - 0.50)
        + 0.10 * (paired["dataset_only"]["structure_score"]["win_rate"] - 0.50)
    )
    return {
        "name": name,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "n_samples": len(rows),
        "audio_checkpoint_sha256": manifest["audio_checkpoint_sha256"],
        "lineage": manifest["lineage"],
        "exploratory_synthesis": bool(manifest.get("exploratory")),
        "absolute_medians": absolute,
        "paired_controls": paired,
        "absolute_structure_passed": absolute_passed,
        "eeg_specific_controls_passed": paired_passed,
        "passed": bool(absolute_passed and paired_passed),
        "selection_score": selection_score,
    }


def main() -> None:
    args = parse_args()
    require(args.minimum_samples >= 12, "--minimum-samples cannot be lower than 12")
    names = [values[0] for values in args.candidate]
    require(len(names) == len(set(names)), "Candidate names must be unique")
    candidates = [
        evaluate_candidate(
            name,
            Path(checkpoint),
            Path(manifest),
            minimum_samples=args.minimum_samples,
            envelope_correlation_min=args.envelope_correlation_min,
            activity_iou_min=args.activity_iou_min,
            paired_median_delta_min=args.paired_median_delta_min,
            paired_win_rate_min=args.paired_win_rate_min,
        )
        for name, checkpoint, manifest in args.candidate
    ]
    baseline = candidates[0]
    for candidate in candidates[1:]:
        require(candidate["lineage"] == baseline["lineage"], "Candidate lineage mismatch")
        require(
            candidate["audio_checkpoint_sha256"] == baseline["audio_checkpoint_sha256"],
            "Candidates use different audio checkpoints",
        )

    selected = max(
        candidates,
        key=lambda item: (bool(item["passed"]), float(item["selection_score"]), int(item["checkpoint_epoch"])),
    )
    output_checkpoint = Path(args.output_checkpoint)
    output_report = Path(args.output_report)
    require(output_checkpoint.resolve() not in {Path(item["checkpoint"]).resolve() for item in candidates},
            "Output checkpoint must not overwrite a candidate")
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected["checkpoint"], output_checkpoint)
    selected_output_sha = file_sha256(output_checkpoint)
    require(selected_output_sha == selected["checkpoint_sha256"], "Selected checkpoint copy failed SHA verification")

    report = {
        "version": "combined-0721-decoded-checkpoint-selection-v1",
        "phase": "decoded_validation_checkpoint_selection",
        "dataset": "karaone",
        "split": "validation",
        "test_accessed": False,
        "passed": bool(selected["passed"]),
        "selection_policy": (
            "Prefer candidates passing absolute structure and EEG-vs-shuffled/zero/dataset-only paired controls; "
            "otherwise choose the highest EEG-specific decoded-validation score and retain passed=false."
        ),
        "thresholds": {
            "minimum_samples": int(args.minimum_samples),
            "envelope_correlation_median_min": float(args.envelope_correlation_min),
            "activity_iou_median_min": float(args.activity_iou_min),
            "paired_median_delta_min": float(args.paired_median_delta_min),
            "paired_win_rate_min": float(args.paired_win_rate_min),
        },
        "selected_name": selected["name"],
        "selected_source_checkpoint": selected["checkpoint"],
        "selected_checkpoint": str(output_checkpoint.resolve()),
        "selected_checkpoint_sha256": selected_output_sha,
        "selected_checkpoint_epoch": selected["checkpoint_epoch"],
        "candidates": candidates,
    }
    output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

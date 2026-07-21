from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.io import wavfile
from tqdm.auto import tqdm


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.combined_0715.audio_eval import METRIC_DIRECTIONS as AUDIO_METRIC_DIRECTIONS  # noqa: E402
from src.combined_0715.lineage import file_sha256  # noqa: E402


DATASETS = ("feis", "karaone", "ds004306")
SYNTHESIS_VERSIONS = {
    "combined-0715-synthesis-v2",
    "combined-0715-synthesis-v3",
}
OUTPUT_KINDS = (
    "reference",
    "codec_oracle",
    "eeg_conditioned",
    "label_only",
    "zero_eeg",
    "shuffled_eeg",
    "dataset_only",
)
GENERATED_MODES = OUTPUT_KINDS[1:]
NEGATIVE_CONTROLS = ("shuffled_eeg", "zero_eeg", "dataset_only")
STRUCTURE_METRICS = ("envelope_correlation", "activity_iou")
METRIC_DIRECTIONS = {
    **{
        name: ("higher_is_better" if direction == "higher" else "lower_is_better")
        for name, direction in AUDIO_METRIC_DIRECTIONS.items()
    },
    "q0_accuracy": "higher_is_better",
    "q1_accuracy": "higher_is_better",
}
DATASET_SCOPES = {
    "feis": "subject_label_canonical",
    "karaone": "trial_level_overt",
    "ds004306": "category_candidate",
}
EXPECTED_PAIRING_CONFIDENCE = {
    # v2 uses the source-facing names; v3 may use the claim-scope names.
    "feis": {"feis_subject_label", "subject_label_canonical"},
    "karaone": {"karaone_same_trial_overt", "trial_level_overt"},
    "ds004306": {"weak_category_level", "category_candidate"},
}
LINEAGE_KEYS = (
    "schema_version",
    "config_sha256",
    "split_sha256",
    "split_version",
    "manifest_sha256",
    "preprocessing_sha256",
    "cache_sha256",
    "cache_version",
)
REPORT_VERSION = "combined-0715-reconstruction-validation-v1"
GATE_VERSION = "combined-0715-reconstruction-gate-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit validation reconstructions using coarse temporal structure and "
            "paired EEG-specific negative controls."
        )
    )
    parser.add_argument(
        "--synthesis-root",
        required=True,
        help="Root containing <dataset>/validation/synthesis_manifest.json.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Combined artifact root. Defaults to the parent of --synthesis-root; "
            "the report is written under eeg/metrics/."
        ),
    )
    parser.add_argument("--validation-report", default=None)
    parser.add_argument("--validation-gate", default=None)
    parser.add_argument("--minimum-samples", type=int, default=12)
    parser.add_argument("--envelope-correlation-min", type=float, default=0.30)
    parser.add_argument("--activity-iou-min", type=float, default=0.30)
    parser.add_argument("--paired-median-delta-min", type=float, default=0.03)
    parser.add_argument("--paired-win-rate-min", type=float, default=0.55)
    parser.add_argument("--envelope-window-ms", type=float, default=25.0)
    parser.add_argument("--activity-threshold-fraction", type=float, default=0.10)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when the reconstruction gate does not pass.",
    )
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Required synthesis manifest is missing: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read synthesis manifest {path}: {error}") from error
    _require(isinstance(value, dict), f"Synthesis manifest must be a JSON object: {path}")
    return value


def _canonical_lineage(lineage: Any, source: str) -> dict[str, Any]:
    _require(isinstance(lineage, dict), f"{source} has no lineage object")
    missing = [key for key in LINEAGE_KEYS if lineage.get(key) in (None, "")]
    _require(not missing, f"{source} has incomplete lineage fields: {missing}")
    return {key: lineage[key] for key in LINEAGE_KEYS}


def _safe_file(validation_root: Path, relative: Any, *, source: str) -> Path:
    _require(isinstance(relative, str) and bool(relative), f"{source} has an empty file path")
    path = (validation_root / relative).resolve()
    try:
        path.relative_to(validation_root.resolve())
    except ValueError as error:
        raise ValueError(f"{source} escapes the validation output directory: {relative}") from error
    _require(path.is_file(), f"{source} is missing: {path}")
    _require(path.stat().st_size > 44, f"{source} is empty or not a complete WAV file: {path}")
    return path


def validate_manifest(
    path: Path,
    dataset: str,
    *,
    minimum_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _read_json(path)
    source = f"{dataset} manifest {path}"
    _require(manifest.get("version") in SYNTHESIS_VERSIONS, f"{source} has unsupported version {manifest.get('version')!r}")
    _require(manifest.get("phase") == "synthesis_controls", f"{source} phase is not synthesis_controls")
    _require(manifest.get("dataset") == dataset, f"{source} dataset field does not match {dataset!r}")
    _require(manifest.get("split") == "validation", f"{source} must describe validation, not {manifest.get('split')!r}")
    _require(manifest.get("test_accessed") is False, f"{source} must have test_accessed=false")
    _require(_is_sha256(manifest.get("audio_checkpoint_sha256")), f"{source} has an invalid audio checkpoint SHA256")
    _require(_is_sha256(manifest.get("eeg_checkpoint_sha256")), f"{source} has an invalid EEG checkpoint SHA256")
    _canonical_lineage(manifest.get("lineage"), source)

    declared_kinds = manifest.get("output_kinds")
    _require(isinstance(declared_kinds, list), f"{source} has no output_kinds list")
    _require(set(declared_kinds) == set(OUTPUT_KINDS), f"{source} must contain exactly the seven reconstruction output kinds")
    rows = manifest.get("files")
    _require(isinstance(rows, list), f"{source} has no files list")
    declared_n = manifest.get("n_generated")
    _require(isinstance(declared_n, int) and not isinstance(declared_n, bool), f"{source} n_generated must be an integer")
    _require(declared_n == len(rows), f"{source} n_generated={declared_n} but files has {len(rows)} rows")
    _require(declared_n >= int(minimum_samples), f"{source} has only {declared_n} samples; at least {minimum_samples} are required")

    expected_pairings = EXPECTED_PAIRING_CONFIDENCE[dataset]
    validation_root = path.parent
    sample_keys: set[str] = set()
    validated_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        row_source = f"{source} files[{index}]"
        _require(isinstance(row, dict), f"{row_source} must be an object")
        sample_key = row.get("sample_key")
        _require(isinstance(sample_key, str) and bool(sample_key), f"{row_source} has no sample_key")
        _require(sample_key not in sample_keys, f"{source} contains duplicate sample_key {sample_key!r}")
        sample_keys.add(sample_key)
        _require(
            row.get("pairing_confidence") in expected_pairings,
            f"{row_source} pairing confidence {row.get('pairing_confidence')!r} is not one of {sorted(expected_pairings)!r}",
        )
        if dataset == "karaone":
            _require(row.get("trial_level_claim_allowed") is True, f"{row_source} is not trial-level overt paired")
        else:
            _require(row.get("trial_level_claim_allowed") is False, f"{row_source} must not claim trial-level reconstruction")

        files = row.get("files")
        _require(isinstance(files, dict), f"{row_source} has no files object")
        _require(set(files) == set(OUTPUT_KINDS), f"{row_source} does not contain all seven output files")
        resolved = {
            mode: _safe_file(validation_root, files[mode], source=f"{row_source}.{mode}")
            for mode in OUTPUT_KINDS
        }
        metrics = row.get("mode_metrics")
        _require(isinstance(metrics, dict), f"{row_source} has no mode_metrics object")
        _require(set(GENERATED_MODES).issubset(metrics), f"{row_source} is missing one or more generated mode metrics")
        if manifest.get("version") == "combined-0715-synthesis-v3":
            for mode in GENERATED_MODES:
                _require(
                    all(metric in metrics[mode] for metric in STRUCTURE_METRICS),
                    f"{row_source}.{mode} is missing v3 temporal structure metrics",
                )
        validated_rows.append({**row, "_resolved_files": resolved})

    manifest_trial_claim = manifest.get("trial_level_claim_allowed")
    if dataset == "karaone":
        _require(manifest_trial_claim is True, f"{source} must be marked trial_level_claim_allowed=true")
    else:
        _require(manifest_trial_claim is False, f"{source} must be marked trial_level_claim_allowed=false")
    if dataset == "ds004306":
        _require(manifest.get("ds004306_trial_level_claim_allowed") is False, f"{source} must explicitly forbid ds004306 trial-level claims")
    return manifest, validated_rows


def _wav_float(path: Path) -> tuple[int, np.ndarray]:
    try:
        sample_rate, audio = wavfile.read(path)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read WAV file {path}: {error}") from error
    value = np.asarray(audio)
    if value.ndim == 2:
        value = np.mean(value.astype(np.float64), axis=1)
    _require(value.ndim == 1 and value.size > 0, f"WAV file must contain a non-empty mono-compatible waveform: {path}")
    if np.issubdtype(value.dtype, np.integer):
        info = np.iinfo(value.dtype)
        if np.issubdtype(value.dtype, np.unsignedinteger):
            midpoint = (float(info.max) + 1.0) / 2.0
            value = (value.astype(np.float64) - midpoint) / midpoint
        else:
            scale = float(max(abs(info.min), abs(info.max)))
            value = value.astype(np.float64) / scale
    else:
        value = value.astype(np.float64)
    _require(np.all(np.isfinite(value)), f"WAV file contains NaN/Inf: {path}")
    return int(sample_rate), value


def _rms_envelope(audio: np.ndarray, sample_rate: int, window_ms: float) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float64).reshape(-1)
    value = value - float(np.mean(value))
    window = max(1, int(round(float(sample_rate) * float(window_ms) / 1000.0)))
    energy = uniform_filter1d(np.square(value), size=window, mode="constant", cval=0.0)
    return np.sqrt(np.maximum(energy, 0.0))


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = np.asarray(second, dtype=np.float64).reshape(-1)
    _require(x.shape == y.shape and x.size > 1, "Correlation inputs must have equal non-trivial shape")
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= 1e-15:
        return 1.0 if np.allclose(x, y, atol=1e-12, rtol=0.0) else 0.0
    return float(np.clip(np.dot(x, y) / denominator, -1.0, 1.0))


def temporal_structure_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    *,
    window_ms: float = 25.0,
    activity_threshold_fraction: float = 0.10,
) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    _require(reference.shape == candidate.shape, "Reference and reconstruction WAV lengths differ")
    ref_envelope = _rms_envelope(reference, sample_rate, window_ms)
    cand_envelope = _rms_envelope(candidate, sample_rate, window_ms)
    envelope_correlation = _correlation(ref_envelope, cand_envelope)

    fraction = float(activity_threshold_fraction)
    _require(0.0 < fraction < 1.0, "activity_threshold_fraction must be between 0 and 1")
    ref_peak = float(np.max(ref_envelope))
    cand_peak = float(np.max(cand_envelope))
    ref_active = ref_envelope >= max(ref_peak * fraction, 1e-12)
    cand_active = cand_envelope >= max(cand_peak * fraction, 1e-12)
    union = int(np.count_nonzero(ref_active | cand_active))
    intersection = int(np.count_nonzero(ref_active & cand_active))
    activity_iou = float(intersection / union) if union else 1.0
    return {
        "envelope_correlation": envelope_correlation,
        "activity_iou": activity_iou,
    }


def _finite(values: Iterable[Any]) -> np.ndarray:
    converted: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            converted.append(number)
    return np.asarray(converted, dtype=np.float64)


def distribution_summary(values: Iterable[Any]) -> dict[str, Any]:
    array = _finite(values)
    _require(array.size > 0, "Cannot summarise an empty/non-finite metric")
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def paired_summary(
    eeg_values: Sequence[Any],
    control_values: Sequence[Any],
    *,
    direction: str,
) -> dict[str, Any]:
    _require(len(eeg_values) == len(control_values), "Paired metric vectors must have equal length")
    pairs: list[tuple[float, float]] = []
    for eeg, control in zip(eeg_values, control_values):
        try:
            eeg_value, control_value = float(eeg), float(control)
        except (TypeError, ValueError):
            continue
        if math.isfinite(eeg_value) and math.isfinite(control_value):
            pairs.append((eeg_value, control_value))
    _require(bool(pairs), "No finite paired metric values are available")
    if direction == "higher_is_better":
        deltas = np.asarray([eeg - control for eeg, control in pairs], dtype=np.float64)
    elif direction == "lower_is_better":
        deltas = np.asarray([control - eeg for eeg, control in pairs], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported metric direction {direction!r}")
    return {
        "n": int(len(deltas)),
        "median_delta": float(np.median(deltas)),
        "mean_delta": float(np.mean(deltas)),
        "win_rate": float(np.mean(deltas > 0.0)),
        "tie_rate": float(np.mean(deltas == 0.0)),
        "delta_definition": "eeg_minus_control" if direction == "higher_is_better" else "control_minus_eeg",
    }


def evaluate_dataset_metrics(
    sample_metrics: Sequence[Mapping[str, Mapping[str, float]]],
    *,
    structure_thresholds: Mapping[str, float],
    paired_median_delta_min: float,
    paired_win_rate_min: float,
) -> dict[str, Any]:
    _require(bool(sample_metrics), "At least one sample is required for reconstruction evaluation")
    metric_names = set(STRUCTURE_METRICS)
    for sample in sample_metrics:
        for values in sample.values():
            metric_names.update(values)

    metric_reports: dict[str, Any] = {}
    for metric in sorted(metric_names):
        direction = METRIC_DIRECTIONS.get(metric)
        if direction is None:
            continue
        by_mode: dict[str, list[Any]] = {}
        for mode in GENERATED_MODES:
            values = [sample.get(mode, {}).get(metric) for sample in sample_metrics]
            finite_values = _finite(values)
            if finite_values.size:
                # Retain the trial positions so paired comparisons cannot be
                # silently misaligned when an optional auxiliary metric is
                # absent for only some rows.
                by_mode[mode] = values
        if "eeg_conditioned" not in by_mode:
            continue
        mode_summaries = {
            mode: distribution_summary(values)
            for mode, values in by_mode.items()
        }
        comparisons: dict[str, Any] = {}
        for control in NEGATIVE_CONTROLS:
            if control not in by_mode:
                continue
            paired = paired_summary(
                by_mode["eeg_conditioned"],
                by_mode[control],
                direction=direction,
            )
            if metric in STRUCTURE_METRICS:
                paired["median_delta_passed"] = bool(
                    paired["median_delta"] >= float(paired_median_delta_min)
                )
                paired["win_rate_passed"] = bool(
                    paired["win_rate"] >= float(paired_win_rate_min)
                )
                paired["passed"] = bool(
                    paired["median_delta_passed"] and paired["win_rate_passed"]
                )
            else:
                paired["passed"] = None
            comparisons[control] = paired
        structure_threshold = structure_thresholds.get(metric)
        structure_passed = (
            bool(mode_summaries["eeg_conditioned"]["median"] >= float(structure_threshold))
            if structure_threshold is not None
            else None
        )
        negative_control_passed = (
            bool(set(comparisons) == set(NEGATIVE_CONTROLS) and all(value["passed"] for value in comparisons.values()))
            if metric in STRUCTURE_METRICS
            else None
        )
        metric_reports[metric] = {
            "direction": direction,
            "mode_summaries": mode_summaries,
            "label_only_role": "reported_only_not_a_negative_control",
            "paired_negative_controls": comparisons,
            "structure_threshold": structure_threshold,
            "structure_passed": structure_passed,
            "eeg_specific_passed": negative_control_passed,
        }

    missing_structure = [metric for metric in STRUCTURE_METRICS if metric not in metric_reports]
    _require(not missing_structure, f"Missing required temporal structure metrics: {missing_structure}")
    structure_passed = all(metric_reports[metric]["structure_passed"] for metric in STRUCTURE_METRICS)
    eeg_specific_passed = all(metric_reports[metric]["eeg_specific_passed"] for metric in STRUCTURE_METRICS)
    return {
        "structure_reconstruction_passed": bool(structure_passed),
        "eeg_specific_reconstruction_passed": bool(eeg_specific_passed),
        "metrics": metric_reports,
    }


def _sample_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    description: str,
    envelope_window_ms: float,
    activity_threshold_fraction: float,
) -> list[dict[str, dict[str, float]]]:
    result: list[dict[str, dict[str, float]]] = []
    for row in tqdm(rows, desc=description, unit="sample", dynamic_ncols=True):
        paths = row["_resolved_files"]
        modes: dict[str, dict[str, float]] = {}
        existing = row["mode_metrics"]
        has_stored_structure = all(
            isinstance(existing.get(mode), dict)
            and all(metric in existing[mode] for metric in STRUCTURE_METRICS)
            for mode in GENERATED_MODES
        )
        if not has_stored_structure:
            reference_rate, reference = _wav_float(paths["reference"])
        for mode in GENERATED_MODES:
            stored = existing.get(mode)
            _require(isinstance(stored, dict), f"Sample {row['sample_key']} has no metrics for {mode}")
            auxiliary = {
                metric: float(value)
                for metric, value in stored.items()
                if metric in METRIC_DIRECTIONS and isinstance(value, (int, float)) and math.isfinite(float(value))
            }
            if has_stored_structure:
                modes[mode] = auxiliary
            else:
                candidate_rate, candidate = _wav_float(paths[mode])
                _require(candidate_rate == reference_rate, f"Sample {row['sample_key']} has inconsistent WAV sample rates")
                _require(candidate.shape == reference.shape, f"Sample {row['sample_key']} has inconsistent WAV lengths")
                structure = temporal_structure_metrics(
                    reference,
                    candidate,
                    reference_rate,
                    window_ms=envelope_window_ms,
                    activity_threshold_fraction=activity_threshold_fraction,
                )
                modes[mode] = {**auxiliary, **structure}
        result.append(modes)
    return result


def build_report(
    synthesis_root: Path,
    *,
    minimum_samples: int = 12,
    envelope_correlation_min: float = 0.30,
    activity_iou_min: float = 0.30,
    paired_median_delta_min: float = 0.03,
    paired_win_rate_min: float = 0.55,
    envelope_window_ms: float = 25.0,
    activity_threshold_fraction: float = 0.10,
) -> dict[str, Any]:
    _require(minimum_samples >= 12, "minimum_samples cannot be lower than the locked audit minimum of 12")
    structure_thresholds = {
        "envelope_correlation": float(envelope_correlation_min),
        "activity_iou": float(activity_iou_min),
    }
    manifests: dict[str, dict[str, Any]] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    bindings: dict[str, Any] = {}
    for dataset in DATASETS:
        manifest_path = synthesis_root / dataset / "validation" / "synthesis_manifest.json"
        manifest, dataset_rows = validate_manifest(
            manifest_path,
            dataset,
            minimum_samples=minimum_samples,
        )
        manifests[dataset] = manifest
        rows[dataset] = dataset_rows
        bindings[dataset] = {
            "path": str(manifest_path.resolve()),
            "sha256": file_sha256(manifest_path),
            "version": manifest["version"],
            "n_generated": int(manifest["n_generated"]),
        }

    first = manifests[DATASETS[0]]
    common_lineage = _canonical_lineage(first["lineage"], f"{DATASETS[0]} manifest")
    audio_sha = first["audio_checkpoint_sha256"]
    eeg_sha = first["eeg_checkpoint_sha256"]
    for dataset in DATASETS[1:]:
        manifest = manifests[dataset]
        _require(
            _canonical_lineage(manifest["lineage"], f"{dataset} manifest") == common_lineage,
            f"{dataset} synthesis lineage does not match the other validation manifests",
        )
        _require(manifest["audio_checkpoint_sha256"] == audio_sha, f"{dataset} uses a different audio checkpoint")
        _require(manifest["eeg_checkpoint_sha256"] == eeg_sha, f"{dataset} uses a different EEG checkpoint")

    dataset_reports: dict[str, Any] = {}
    for dataset in DATASETS:
        evaluated = evaluate_dataset_metrics(
            _sample_metrics(
                rows[dataset],
                description=f"[reconstruction audit] {dataset}",
                envelope_window_ms=envelope_window_ms,
                activity_threshold_fraction=activity_threshold_fraction,
            ),
            structure_thresholds=structure_thresholds,
            paired_median_delta_min=paired_median_delta_min,
            paired_win_rate_min=paired_win_rate_min,
        )
        contributes = dataset == "karaone"
        trial_level_overt_verified = bool(
            dataset == "karaone"
            and manifests[dataset].get("trial_level_claim_allowed") is True
            and all(row.get("pairing_confidence") in EXPECTED_PAIRING_CONFIDENCE["karaone"] for row in rows[dataset])
        )
        dataset_passed = bool(
            evaluated["structure_reconstruction_passed"]
            and evaluated["eeg_specific_reconstruction_passed"]
        )
        dataset_reports[dataset] = {
            "scope": DATASET_SCOPES[dataset],
            "n_samples": len(rows[dataset]),
            "n_subject_groups": len({str(row.get("subject_group_id")) for row in rows[dataset]}),
            "trial_level_claim_allowed": trial_level_overt_verified,
            "contributes_to_top_level_pass": contributes,
            **evaluated,
            "passed_within_scope": dataset_passed,
            "top_level_pass_contribution": bool(contributes and trial_level_overt_verified and dataset_passed),
        }

    karaone = dataset_reports["karaone"]
    structure_passed = bool(karaone["trial_level_claim_allowed"] and karaone["structure_reconstruction_passed"])
    eeg_specific_passed = bool(karaone["trial_level_claim_allowed"] and karaone["eeg_specific_reconstruction_passed"])
    passed = bool(structure_passed and eeg_specific_passed)
    reasons: list[str] = []
    if not structure_passed:
        reasons.append("karaone_trial_level_structure_reconstruction_failed")
    if not eeg_specific_passed:
        reasons.append("karaone_eeg_specific_negative_control_criteria_failed")
    return {
        "version": REPORT_VERSION,
        "phase": "reconstruction_validation",
        "split": "validation",
        "test_accessed": False,
        "passed": passed,
        "structure_reconstruction_passed": structure_passed,
        "eeg_specific_reconstruction_passed": eeg_specific_passed,
        "claim_policy": (
            "Only KaraOne same-trial overt pairs may unlock the top-level reconstruction gate. "
            "FEIS is evaluated at subject-label canonical scope and ds004306 at category-candidate scope."
        ),
        "label_only_policy": "reported_only_not_a_negative_control_because_it_uses_same_trial_eeg_label_predictions",
        "generalization_limit": {
            "validation_subject_groups": {
                dataset: dataset_reports[dataset]["n_subject_groups"] for dataset in DATASETS
            },
            "single_subject_validation_datasets": [
                dataset
                for dataset in ("feis", "karaone")
                if dataset_reports[dataset]["n_subject_groups"] == 1
            ],
            "note": (
                "The current locked validation split contains only one subject group for FEIS and "
                "one for KaraOne. Paired trial statistics therefore do not establish cross-subject "
                "generalization."
            ),
        },
        "thresholds": {
            **{f"{metric}_median_min": value for metric, value in structure_thresholds.items()},
            "paired_median_delta_min": float(paired_median_delta_min),
            "paired_win_rate_min": float(paired_win_rate_min),
            "minimum_samples_per_dataset": int(minimum_samples),
        },
        "metric_definitions": {
            "envelope_correlation": f"Pearson correlation of {envelope_window_ms:g}-ms RMS envelopes; higher is better.",
            "activity_iou": (
                "Intersection-over-union of per-waveform active regions after RMS smoothing; "
                f"active means >= {activity_threshold_fraction:g} of that waveform's envelope peak."
            ),
            "paired_delta": "EEG minus control for higher-is-better metrics; control minus EEG for lower-is-better metrics.",
            "paired_win_rate": "Fraction of same-trial pairs with strictly positive direction-normalized delta.",
            "metric_source": (
                "Synthesis-v3 structure metrics are computed on raw in-memory decoded audio before display WAV writing; "
                "v2 compatibility fallback recomputes them from RMS-normalized WAV files."
            ),
        },
        "lineage": first["lineage"],
        "audio_checkpoint_sha256": audio_sha,
        "eeg_checkpoint_sha256": eeg_sha,
        "manifest_bindings": bindings,
        "datasets": dataset_reports,
        "reasons": reasons,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_report_and_gate(
    report: dict[str, Any],
    *,
    report_path: Path,
    gate_path: Path,
) -> dict[str, Any]:
    _write_json(report_path, report)
    gate = {
        "version": GATE_VERSION,
        "phase": "reconstruction_validation_gate",
        "passed": bool(report["passed"]),
        "structure_reconstruction_passed": bool(report["structure_reconstruction_passed"]),
        "eeg_specific_reconstruction_passed": bool(report["eeg_specific_reconstruction_passed"]),
        "reasons": list(report["reasons"]),
        "validation_report": str(report_path.resolve()),
        "validation_report_sha256": file_sha256(report_path),
        "lineage": report["lineage"],
        "audio_checkpoint_sha256": report["audio_checkpoint_sha256"],
        "eeg_checkpoint_sha256": report["eeg_checkpoint_sha256"],
        "manifest_bindings": report["manifest_bindings"],
        "split": "validation",
        "test_accessed": False,
    }
    _write_json(gate_path, gate)
    return gate


def main() -> None:
    args = parse_args()
    synthesis_root = Path(args.synthesis_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else synthesis_root.parent
    report_path = (
        Path(args.validation_report).resolve()
        if args.validation_report
        else output_root / "eeg" / "metrics" / "reconstruction_validation_report.json"
    )
    gate_path = (
        Path(args.validation_gate).resolve()
        if args.validation_gate
        else output_root / "eeg" / "metrics" / "validation_gate.json"
    )
    report = build_report(
        synthesis_root,
        minimum_samples=args.minimum_samples,
        envelope_correlation_min=args.envelope_correlation_min,
        activity_iou_min=args.activity_iou_min,
        paired_median_delta_min=args.paired_median_delta_min,
        paired_win_rate_min=args.paired_win_rate_min,
        envelope_window_ms=args.envelope_window_ms,
        activity_threshold_fraction=args.activity_threshold_fraction,
    )
    gate = write_report_and_gate(report, report_path=report_path, gate_path=gate_path)
    summary = {
        "passed": gate["passed"],
        "structure_reconstruction_passed": gate["structure_reconstruction_passed"],
        "eeg_specific_reconstruction_passed": gate["eeg_specific_reconstruction_passed"],
        "reasons": gate["reasons"],
        "validation_report": str(report_path),
        "validation_gate": str(gate_path),
        "datasets": {
            dataset: {
                "scope": report["datasets"][dataset]["scope"],
                "n_samples": report["datasets"][dataset]["n_samples"],
                "structure_reconstruction_passed": report["datasets"][dataset]["structure_reconstruction_passed"],
                "eeg_specific_reconstruction_passed": report["datasets"][dataset]["eeg_specific_reconstruction_passed"],
            }
            for dataset in DATASETS
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    if args.strict and not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

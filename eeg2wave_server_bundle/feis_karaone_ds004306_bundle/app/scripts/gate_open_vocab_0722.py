#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path: sys.path.insert(0, str(APP))

from src.open_vocab_0722.data import load_context, resolve_config_path  # noqa: E402
from src.open_vocab_0722.lineage import GATE_SCHEMA_VERSION, build_lineage, file_sha256, validate_lineage  # noqa: E402


def mode_score(metrics: dict[str, float], r5: float) -> float:
    return 0.35 * metrics["lag_envelope_correlation"] + 0.25 * metrics["modulation_correlation"] + 0.20 * (1.0 - np.clip(metrics["log_mel_mae_db"] / 12.0, 0.0, 1.0)) + 0.20 * r5


def subject_macro(values: list[tuple[str, float]]) -> float:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for subject, value in values: grouped[subject].append(value)
    return float(np.mean([np.mean(items) for items in grouped.values()])) if grouped else float("nan")


def bootstrap_lower(values: list[tuple[str, float]], samples: int, seed: int = 15) -> float:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for subject, value in values: grouped[subject].append(value)
    subjects = sorted(grouped)
    if not subjects: return float("nan")
    rng = np.random.default_rng(seed); estimates = []
    for _ in range(samples):
        selected = rng.choice(subjects, size=len(subjects), replace=True)
        subject_values = []
        for subject in selected:
            trials = np.asarray(grouped[str(subject)], dtype=float)
            subject_values.append(float(np.mean(rng.choice(trials, size=len(trials), replace=True))))
        estimates.append(float(np.mean(subject_values)))
    return float(np.percentile(estimates, 2.5))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the formal OpenVoice 0722 validation report and locked-test gate")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--synthesis-manifest", type=Path, required=True)
    parser.add_argument("--model-audit", type=Path, default=None)
    parser.add_argument("--generalization", choices=("g1", "g2", "g3"), default="g1")
    parser.add_argument("--seed-summary", type=Path, default=None)
    parser.add_argument("--dense-baseline-report", type=Path, default=None)
    parser.add_argument("--project-audio-only", action="store_true")
    args = parser.parse_args()
    context = load_context(args.config); cfg = context.config
    lineage = build_lineage(context, require_optional_caches=not args.project_audio_only)
    synthesis = json.loads(args.synthesis_manifest.resolve().read_text(encoding="utf-8"))
    if synthesis.get("dataset") != "karaone" or synthesis.get("split") != "validation":
        raise ValueError("Formal generation gate requires KaraOne validation synthesis")
    validate_lineage(synthesis.get("lineage"), lineage, source="synthesis manifest")
    r5 = float(synthesis["retrieval"]["r5"]); chance = float(synthesis["retrieval"]["chance_r5"])
    envelope_gain, mel_gain, composite_gain, main_composite = [], [], [], []
    unavailable_same_label = 0
    for sample in synthesis["samples"]:
        if not sample.get("trial_level_claim_allowed"): continue
        subject = str(sample["subject_group_id"]); metrics = sample["mode_metrics"]
        if not sample.get("same_label_shuffle_available"):
            unavailable_same_label += 1; continue
        main = metrics["eeg_conditioned"]; shuffled = metrics["shuffled_eeg_same_label"]; zero = metrics["zero_eeg"]
        envelope_gain.append((subject, min(main["lag_envelope_correlation"] - shuffled["lag_envelope_correlation"], main["lag_envelope_correlation"] - zero["lag_envelope_correlation"])))
        mel_gain.append((subject, min(shuffled["log_mel_mae_db"] - main["log_mel_mae_db"], zero["log_mel_mae_db"] - main["log_mel_mae_db"])))
        composite_gain.append((subject, mode_score(main, r5) - max(mode_score(shuffled, r5), mode_score(zero, r5))))
        main_composite.append((subject, mode_score(main, r5)))
    model_audit_path = args.model_audit.resolve() if args.model_audit else resolve_config_path(context.config_path, cfg["paths"]["output_root"]) / "evaluation/model_audit.json"
    audit = json.loads(model_audit_path.read_text(encoding="utf-8"))
    validate_lineage(audit.get("lineage"), lineage, source="model audit")
    thresholds = cfg["evaluation"]
    statistics = {
        "envelope_gain_subject_macro": subject_macro(envelope_gain),
        "log_mel_gain_db_subject_macro": subject_macro(mel_gain),
        "composite_gain_subject_macro": subject_macro(composite_gain),
        "eeg_composite_subject_macro": subject_macro(main_composite),
        "composite_gain_bootstrap_95ci_lower": bootstrap_lower(composite_gain, int(thresholds["bootstrap_samples"])),
        "retrieval_r5": r5, "retrieval_chance_r5": chance,
        "same_label_control_unavailable": unavailable_same_label,
    }
    criteria = {
        "full_public_audio_prior_used": not args.project_audio_only,
        "envelope_gain": statistics["envelope_gain_subject_macro"] >= float(thresholds["g1"]["minimum_envelope_gain"]),
        "log_mel_gain": statistics["log_mel_gain_db_subject_macro"] >= float(thresholds["g1"]["minimum_log_mel_gain_db"]),
        "retrieval_r5": r5 >= chance * float(thresholds["g1"]["minimum_retrieval_chance_multiple"]),
        "bootstrap_lower": statistics["composite_gain_bootstrap_95ci_lower"] > float(thresholds["g1"]["bootstrap_lower_bound"]),
        "same_label_control_complete": unavailable_same_label == 0,
        "common14_dataset_probe": float(audit["common14_dataset_balanced_accuracy"]) <= float(thresholds["moe"]["maximum_common14_dataset_balanced_accuracy"]),
        "full_common_consistency": float(audit["full_common_condition_cosine_median"]) >= float(thresholds["moe"]["minimum_full_common_condition_cosine"]),
        "missing25_robustness": float(audit["missing25"]["drop_fraction"]) <= float(thresholds["moe"]["maximum_missing_channel_drop_fraction"]),
        "router_not_collapsed": bool(audit["no_dying_or_collapsed_expert"]),
        "router_not_dataset_specialist": bool(audit["no_single_dataset_specialist"]),
    }
    dense_comparison: dict[str, Any] = {"available": False, "passed": False}
    if args.dense_baseline_report:
        dense = json.loads(args.dense_baseline_report.resolve().read_text(encoding="utf-8"))
        gains = np.asarray(dense.get("moe_minus_dense_composite_by_seed", []), dtype=float)
        dense_comparison = {
            "available": len(gains) == 3,
            "mean_gain": float(np.mean(gains)) if len(gains) else float("nan"),
            "new_subject_not_worse": bool(dense.get("new_subject_not_worse", False)),
            "passed": len(gains) == 3 and float(np.mean(gains)) >= float(thresholds["moe"]["minimum_dense_gain"]) and bool(dense.get("new_subject_not_worse", False)),
        }
    criteria["moe_beats_matched_dense_three_seeds"] = bool(dense_comparison["passed"])
    seed_evidence: dict[str, Any] = {"required": args.generalization in {"g2", "g3"}, "available": False, "passed": args.generalization == "g1"}
    if args.seed_summary:
        seeds = json.loads(args.seed_summary.resolve().read_text(encoding="utf-8"))
        passes = [bool(value) for value in seeds.get("seed_passed", [])]
        seed_evidence = {"required": args.generalization in {"g2", "g3"}, "available": len(passes) == 3, "individual_passes": passes, "pooled_passed": bool(seeds.get("pooled_passed", False)), "passed": len(passes) == 3 and sum(passes) >= 2 and bool(seeds.get("pooled_passed", False))}
    if args.generalization in {"g2", "g3"}: criteria["two_of_three_seeds_and_pooled"] = bool(seed_evidence["passed"])
    reasons = [name for name, passed in criteria.items() if not passed]
    report_path = resolve_config_path(context.config_path, cfg["paths"]["validation_report"])
    report = {
        "schema_version": "openvoice-0722-validation-report-v1", "generalization": args.generalization,
        "split": "validation", "test_accessed": False, "statistics": statistics, "criteria": criteria,
        "dense_comparison": dense_comparison, "seed_evidence": seed_evidence, "passed": not reasons, "reasons": reasons,
        "synthesis_manifest": str(args.synthesis_manifest.resolve()), "synthesis_manifest_sha256": file_sha256(args.synthesis_manifest),
        "model_audit": str(model_audit_path), "model_audit_sha256": file_sha256(model_audit_path),
        "lineage": lineage, "audio_checkpoint_sha256": synthesis["audio_checkpoint_sha256"], "eeg_checkpoint_sha256": synthesis["eeg_checkpoint_sha256"],
        "claim_boundary": "G1 only" if args.generalization == "g1" else f"{args.generalization}; requires three-seed evidence",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True); report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gate_path = resolve_config_path(context.config_path, cfg["paths"]["validation_gate"])
    gate = {
        "schema_version": GATE_SCHEMA_VERSION, "passed": not reasons, "reasons": reasons,
        "validation_report": str(report_path), "validation_report_sha256": file_sha256(report_path),
        "lineage": lineage, "audio_checkpoint_sha256": synthesis["audio_checkpoint_sha256"], "eeg_checkpoint_sha256": synthesis["eeg_checkpoint_sha256"],
        "test_eeg_or_audio_accessed": False,
    }
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "gate": str(gate_path), "passed": not reasons, "reasons": reasons, "statistics": statistics}, indent=2, sort_keys=True))


if __name__ == "__main__": main()

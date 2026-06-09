from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.utils import load_simple_yaml, resolve_bundle_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one consolidated Phase 2 FEIS report.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "alignment_ssl_local.yaml"))
    parser.add_argument("--retrieval-eval", default=None)
    parser.add_argument("--space-summary", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--holdout-subject", default=None)
    parser.add_argument("--stage", default=None)
    parser.add_argument("--ablation-mode", default=None)
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def build_run_name(config: dict, args: argparse.Namespace) -> str:
    protocol = str(config["data"]["protocol"]).upper()
    run_name = f"{protocol.lower()}_{config['data']['stage']}_{config['data'].get('ablation_mode', 'none')}"
    if protocol == "S":
        run_name += f"_subject_{args.subject or config['data'].get('subject_id')}"
    if protocol == "U":
        run_name += f"_holdout_{args.holdout_subject or config['data'].get('holdout_subject_id')}"
    return run_name


def fmt_metric(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def load_json(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def extract_model_block(payload: dict | None) -> dict | None:
    if payload is None:
        return None
    model = payload.get("model")
    if isinstance(model, dict):
        nested = model.get("model")
        if isinstance(nested, dict):
            return nested
        return model
    return None


def choose_recommendation(retrieval_eval: dict | None) -> tuple[str, str]:
    if retrieval_eval is None:
        return (
            "Recommendation unavailable",
            "Retrieval evaluation JSON was not found, so the route recommendation could not be derived from evidence.",
        )
    main_model = extract_model_block(retrieval_eval)
    if main_model is None:
        return (
            "Recommendation unavailable",
            "Retrieval evaluation JSON did not contain a usable model summary block.",
        )
    match_mode = str(main_model["match_mode"])
    top1_key = f"retrieval_top1_{'exact' if match_mode == 'exact' else 'label'}"
    top1 = float(main_model.get(top1_key) or 0.0)
    cosine = float(main_model.get("embedding_cosine") or 0.0)
    oracle = retrieval_eval.get("oracle_ceiling")
    oracle_top1 = None
    if oracle is not None:
        oracle_model = extract_model_block(oracle)
        if oracle_model is not None:
            oracle_top1 = float(oracle_model.get("retrieval_top1_exact") or 0.0)
    if cosine >= 0.85 and top1 <= 0.15:
        return (
            "4. EEG → audio token → codec decoder",
            "Pooled HuBERT appears aligned enough to give high cosine but not discriminative enough for stable retrieval. "
            "The next investment should shift toward richer targets: first test sequence-level HuBERT, then codec-latent/audio-token targets, "
            "rather than spending much more time optimizing pooled-template retrieval.",
        )
    if oracle_top1 is not None and top1 >= 0.40 and oracle_top1 - top1 <= 0.15:
        return (
            "2. EEG → speech embedding → retrieval waveform",
            "Strict retrieval is already competitive and close to the oracle ceiling, so template retrieval is the most robust FEIS-specific path to prioritize now.",
        )
    return (
        "3. EEG → speech embedding → vocoder waveform",
        "Retrieval remains useful as a ceiling and sanity baseline, but the main forward path should move toward a decoder that consumes speech representations directly.",
    )


def route_table() -> str:
    return "\n".join(
        [
            "| Route | Complexity | Expected intelligibility | Expected robustness on FEIS | Research value |",
            "|---|---|---|---|---|",
            "| A. Nearest-template waveform | Low | Moderate if retrieval is label-stable | High | Strong baseline / ceiling, limited novelty |",
            "| B. Speech embedding -> frozen vocoder | Medium | Medium to high if target representation matches the vocoder | Medium | Good balance of realism and tractability |",
            "| C. Speech embedding -> trainable waveform decoder | High | Uncertain under FEIS scale | Low to medium | High upside, highest data risk |",
        ]
    )


def build_report(retrieval_eval: dict | None, space_summary: dict | None) -> str:
    recommendation_title, recommendation_body = choose_recommendation(retrieval_eval)
    lines: list[str] = [
        "# FEIS Phase 2 Report",
        "",
        "## Task 1. Alignment Audit",
        "",
        "- Current serious target path: pooled HuBERT SSL embeddings, not frame-level HuBERT, not WavLM, not codec latent.",
        "- Target cache: `feis_subject_templates_ssl.npz` with `speech_embeddings=(336, 768)` and `prosody_targets=(336, 4)`.",
        "- Alignment model output: one pooled speech embedding per EEG trial plus prosody.",
    ]
    if retrieval_eval is not None:
        main_model = extract_model_block(retrieval_eval)
        if main_model is not None:
            lines.extend(
                [
                    f"- Current evaluated checkpoint embedding cosine: `{fmt_metric(main_model.get('embedding_cosine'))}`.",
                    f"- Current retrieval policy: `{retrieval_eval['model']['retrieval_policy']}` with match mode `{retrieval_eval['model']['match_mode']}`.",
                ]
            )
    lines.extend(
        [
            "",
            "## Task 2. Retrieval Waveform Reconstruction",
            "",
            "- EEG trials are mapped to predicted speech embeddings, ranked against a protocol-aware template bank, and reconstructed by copying the top-1 retrieved waveform.",
            "- Per-trial outputs include top-5 candidates, retrieved template metadata, cosine scores, saved waveform paths, and waveform-space NTA fields.",
            "",
            "## Task 3. Retrieval Benchmark",
            "",
        ]
    )
    if retrieval_eval is None:
        lines.append("- Retrieval evaluation file not found.")
    else:
        main_model = extract_model_block(retrieval_eval)
        main_controls = retrieval_eval["model"]["controls"]
        if main_model is None:
            lines.append("- Retrieval evaluation file was found, but the model summary block was malformed.")
            main_controls = {}
        else:
            main_mode = str(main_model["match_mode"])
            main_top1_key = f"retrieval_top1_{'exact' if main_mode == 'exact' else 'label'}"
            main_top5_key = f"retrieval_top5_{'exact' if main_mode == 'exact' else 'label'}"
            main_nta_key = f"NTA_{'exact' if main_mode == 'exact' else 'label'}"
            lines.extend(
                [
                    f"- Main protocol result: `{main_top1_key}={fmt_metric(main_model.get(main_top1_key))}`, `{main_top5_key}={fmt_metric(main_model.get(main_top5_key))}`, `{main_nta_key}={fmt_metric(main_model.get(main_nta_key))}`.",
                    f"- Target-match availability in bank: `{fmt_metric(main_model.get('target_match_available_rate'))}`.",
                    f"- Mean retrieved-vs-target waveform STFT distance: `{fmt_metric(main_model.get('mean_retrieved_target_stft_distance'))}`.",
                    f"- Random baseline: `{fmt_metric(main_controls['random'].get(main_top1_key))}` / `{fmt_metric(main_controls['random'].get(main_top5_key))}`.",
                    f"- Label-only baseline: `{fmt_metric(main_controls['label_only'].get(main_top1_key))}` / `{fmt_metric(main_controls['label_only'].get(main_top5_key))}`.",
                ]
            )
            for key in sorted(main_controls):
                if key in {"random", "label_only"}:
                    continue
                lines.append(
                    f"- {key}: `{fmt_metric(main_controls[key].get(main_top1_key))}` / `{fmt_metric(main_controls[key].get(main_top5_key))}`."
                )
            if "oracle_ceiling" in retrieval_eval:
                oracle_model = extract_model_block(retrieval_eval["oracle_ceiling"])
                if oracle_model is not None:
                    lines.extend(
                        [
                            "",
                            "### Oracle Ceiling",
                            "",
                            f"- `oracle_exact_top1={fmt_metric(oracle_model.get('retrieval_top1_exact'))}`, `oracle_exact_top5={fmt_metric(oracle_model.get('retrieval_top5_exact'))}`, `oracle_NTA_exact={fmt_metric(oracle_model.get('NTA_exact'))}`.",
                        ]
                    )
    lines.extend(
        [
            "",
            "## Task 4. Speech Space Structure",
            "",
        ]
    )
    if space_summary is None:
        lines.append("- Template-space summary file not found.")
    else:
        lines.extend(
            [
                f"- Dominant structure: `{space_summary['dominant_structure']}`.",
                f"- Subject centroid probe accuracy: `{fmt_metric(space_summary['subject_centroid_probe']['accuracy'])}`.",
                f"- Label centroid probe accuracy: `{fmt_metric(space_summary['label_centroid_probe']['accuracy'])}`.",
                f"- Within-label mean cosine distance: `{fmt_metric(space_summary['pairwise_cosine_distance'].get('within_label', {}).get('mean'))}`.",
                f"- Within-subject mean cosine distance: `{fmt_metric(space_summary['pairwise_cosine_distance'].get('within_subject', {}).get('mean'))}`.",
                f"- Cross-subject same-label mean cosine distance: `{fmt_metric(space_summary.get('cross_subject_same_label_mean'))}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Task 5. Retrieval Ceiling",
            "",
        ]
    )
    if retrieval_eval is None:
        lines.append("- Ceiling analysis unavailable because retrieval metrics were not found.")
    elif "oracle_ceiling" not in retrieval_eval:
        lines.append("- Oracle ceiling was not computed for this run.")
    else:
        main_model = extract_model_block(retrieval_eval)
        oracle_model = extract_model_block(retrieval_eval["oracle_ceiling"])
        if main_model is None or oracle_model is None:
            lines.append("- Oracle ceiling JSON exists, but one of the model summary blocks is malformed.")
        else:
            main_mode = str(main_model["match_mode"])
            main_top1_key = f"retrieval_top1_{'exact' if main_mode == 'exact' else 'label'}"
            lines.extend(
                [
                    f"- Main top-1 vs oracle exact top-1: `{fmt_metric(main_model.get(main_top1_key))}` vs `{fmt_metric(oracle_model.get('retrieval_top1_exact'))}`.",
                    f"- Main NTA vs oracle NTA: `{fmt_metric(main_model.get(f'NTA_{'exact' if main_mode == 'exact' else 'label'}'))}` vs `{fmt_metric(oracle_model.get('NTA_exact'))}`.",
                ]
            )
    lines.extend(
        [
            "",
            "## Task 6. Decoder Route Memo",
            "",
            route_table(),
            "",
            "## Task 7. Recommended Main Direction",
            "",
            f"- Recommended path: **{recommendation_title}**",
            f"- Justification: {recommendation_body}",
        ]
    )
    if retrieval_eval is not None:
        alerts = retrieval_eval["model"].get("alerts", {})
        if alerts.get("high_cosine_poor_retrieval"):
            lines.extend(
                [
                    "",
                    "## Follow-up Trigger",
                    "",
                    f"- {alerts['recommended_followup']}",
                ]
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_simple_yaml(args.config)
    if args.protocol is not None:
        config["data"]["protocol"] = args.protocol
    if args.stage is not None:
        config["data"]["stage"] = args.stage
    if args.ablation_mode is not None:
        config["data"]["ablation_mode"] = args.ablation_mode

    output_root = resolve_bundle_path(config["output"]["root"], BUNDLE_DIR)
    run_name = build_run_name(config, args)
    retrieval_eval_path = (
        Path(args.retrieval_eval)
        if args.retrieval_eval is not None
        else output_root / run_name / "metrics" / f"{args.split}_retrieval_evaluation.json"
    )
    target_cache_path = resolve_bundle_path(config["targets"]["cache_path"], BUNDLE_DIR)
    space_summary_path = (
        Path(args.space_summary)
        if args.space_summary is not None
        else output_root / "template_space" / Path(target_cache_path).stem / "space_summary.json"
    )
    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else output_root / run_name / "metrics" / f"{args.split}_phase2_report.md"
    )

    retrieval_eval = load_json(retrieval_eval_path)
    space_summary = load_json(space_summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report(retrieval_eval=retrieval_eval, space_summary=space_summary), encoding="utf-8")
    print(f"Saved Phase 2 report to {output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.utils import build_protocol_run_name, load_simple_yaml, resolve_bundle_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a consolidated FEIS alignment report.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "alignment_ssl_local.yaml"))
    parser.add_argument("--alignment-eval", default=None)
    parser.add_argument("--sequence-eval", default=None)
    parser.add_argument("--codec-eval", default=None)
    parser.add_argument("--waveform-eval", default=None)
    parser.add_argument("--space-summary", default=None)
    parser.add_argument("--sequence-audio-qc", default=None)
    parser.add_argument("--codec-audio-qc", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--holdout-subject", default=None)
    parser.add_argument("--stage", default=None)
    parser.add_argument("--ablation-mode", default=None)
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def build_run_name(config: dict, args: argparse.Namespace) -> str:
    return build_protocol_run_name(
        config=config,
        protocol=str(config["data"]["protocol"]).upper(),
        stage=str(config["data"]["stage"]),
        ablation_mode=str(config["data"].get("ablation_mode", "none")),
        subject_id=args.subject or config["data"].get("subject_id"),
        holdout_subject_id=args.holdout_subject or config["data"].get("holdout_subject_id"),
    )


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


def load_alignment_summary(path: Path | None) -> dict | None:
    payload = load_json(path)
    if payload is None:
        return None
    return payload


def load_waveform_summary(path: Path | None) -> dict | None:
    return load_json(path)


def _alignment_model_block(payload: dict | None) -> dict | None:
    if payload is None:
        return None
    block = payload.get("model")
    if isinstance(block, dict) and isinstance(block.get("model"), dict):
        return block["model"]
    return None


def _default_audio_qc_path(eval_payload: dict | None) -> Path | None:
    if eval_payload is None:
        return None
    model = eval_payload.get("model", {})
    predictions_path = model.get("predictions_path")
    if not predictions_path:
        return None
    path = Path(str(predictions_path))
    candidates = [path.parent / "audio_qc.json"]
    if len(path.parents) >= 4:
        candidates.append(path.parents[3] / "metrics" / "audio_qc.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _route_row(name: str, payload: dict | None) -> str | None:
    model = _alignment_model_block(payload)
    if model is None:
        return None
    match_mode = str(model.get("match_mode", "exact"))
    top1_key = f"retrieval_top1_{'exact' if match_mode == 'exact' else 'label'}"
    top5_key = f"retrieval_top5_{'exact' if match_mode == 'exact' else 'label'}"
    nta_key = f"NTA_{'exact' if match_mode == 'exact' else 'label'}"
    return (
        f"| {name} | {payload.get('target_kind', 'unknown')} | {payload.get('reconstruction_mode', 'unknown')} | "
        f"{fmt_metric(model.get(top1_key))} | {fmt_metric(model.get(top5_key))} | {fmt_metric(model.get('MRR'))} | "
        f"{fmt_metric(model.get('mean_rank'))} | {fmt_metric(model.get(nta_key))} | "
        f"{fmt_metric(model.get('mean_retrieved_target_stft_distance'))} |"
    )


def _waveform_row(name: str, payload: dict | None) -> str | None:
    if payload is None:
        return None
    return (
        f"| {name} | raw_waveform | direct_decode | N/A | N/A | N/A | N/A | "
        f"{fmt_metric(payload.get('nearest_template_accuracy'))} | {fmt_metric(payload.get('stft_distance'))} |"
    )


def choose_recommendation(
    pooled_eval: dict | None,
    sequence_eval: dict | None,
    codec_eval: dict | None,
) -> tuple[str, str]:
    codec_model = _alignment_model_block(codec_eval)
    if codec_model is not None:
        return (
            "C. EEG -> codec latent -> frozen codec decoder",
            "Codec latents remain the primary forward path because they are decoder-compatible and directly optimize waveform reconstruction instead of stopping at retrieval diagnostics.",
        )
    sequence_model = _alignment_model_block(sequence_eval)
    pooled_model = _alignment_model_block(pooled_eval)
    if pooled_model is not None:
        match_mode = str(pooled_model.get("match_mode", "exact"))
        top1_key = f"retrieval_top1_{'exact' if match_mode == 'exact' else 'label'}"
        cosine = pooled_model.get("embedding_cosine")
        retrieval = pooled_model.get(top1_key)
        if isinstance(cosine, (float, int)) and isinstance(retrieval, (float, int)) and float(cosine) >= 0.85 and float(retrieval) <= 0.15:
            return (
                "B -> C. Sequence HuBERT diagnostics, then codec latent reconstruction",
                "Pooled HuBERT is aligned but not discriminative. Sequence HuBERT should be treated as the diagnostic bridge, while codec latents should absorb the main reconstruction investment.",
            )
    if sequence_model is not None:
        return (
            "B. Sequence HuBERT retrieval diagnostics",
            "Sequence-level HuBERT is the right immediate representation-learning target when pooled HuBERT collapses, but it should stay an evaluation bridge rather than the terminal reconstruction system.",
        )
    return (
        "Recommendation unavailable",
        "No alignment evaluation JSON was found, so the route recommendation could not be derived from evidence.",
    )


def build_report(
    pooled_eval: dict | None,
    sequence_eval: dict | None,
    codec_eval: dict | None,
    waveform_eval: dict | None,
    space_summary: dict | None,
    sequence_audio_qc: dict | None = None,
    codec_audio_qc: dict | None = None,
) -> str:
    recommendation_title, recommendation_body = choose_recommendation(
        pooled_eval=pooled_eval,
        sequence_eval=sequence_eval,
        codec_eval=codec_eval,
    )
    current = codec_eval or sequence_eval or pooled_eval
    current_model = _alignment_model_block(current)
    lines: list[str] = [
        "# FEIS Next-Gen Alignment Report",
        "",
        "## Representation Strategy and Long-Term Goal",
        "",
        "- Sequence-level HuBERT is a representation-learning and diagnostic stage, not the final reconstruction method.",
        "- Retrieval metrics (`Top-1`, `Top-5`, `MRR`, `Mean Rank`, `NTA`) are representation diagnostics rather than terminal success criteria.",
        "- The long-term reconstruction path is `EEG -> speech representation -> audio codec latent -> waveform`.",
        "- Phase 3 and later decisions should prioritize downstream waveform quality over isolated retrieval gains.",
        "",
        "## Current Run",
        "",
    ]
    if current_model is None:
        lines.append("- No alignment evaluation file was found for the current run.")
    else:
        match_mode = str(current_model["match_mode"])
        top1_key = f"retrieval_top1_{'exact' if match_mode == 'exact' else 'label'}"
        top5_key = f"retrieval_top5_{'exact' if match_mode == 'exact' else 'label'}"
        nta_key = f"NTA_{'exact' if match_mode == 'exact' else 'label'}"
        lines.extend(
            [
                f"- target kind: `{current.get('target_kind', 'unknown')}`",
                f"- reconstruction mode: `{current.get('reconstruction_mode', 'unknown')}`",
                f"- `embedding_cosine={fmt_metric(current_model.get('embedding_cosine'))}`",
                f"- `{top1_key}={fmt_metric(current_model.get(top1_key))}`, `{top5_key}={fmt_metric(current_model.get(top5_key))}`",
                f"- `MRR={fmt_metric(current_model.get('MRR'))}`, `mean_rank={fmt_metric(current_model.get('mean_rank'))}`",
                f"- `{nta_key}={fmt_metric(current_model.get(nta_key))}`",
                f"- mean waveform STFT distance to target: `{fmt_metric(current_model.get('mean_retrieved_target_stft_distance'))}`",
                f"- unique top-1 templates: `{fmt_metric(current_model.get('unique_top1_count'))}`, max template share: `{fmt_metric(current_model.get('max_template_share'))}`",
                f"- latent std ratio: `{fmt_metric(current_model.get('pred_target_std_ratio'))}`, frame variance ratio: `{fmt_metric(current_model.get('frame_variance_ratio'))}`",
            ]
        )
        if current.get("model", {}).get("alerts", {}).get("high_cosine_poor_retrieval"):
            lines.extend(
                [
                    "",
                    "## Follow-up Trigger",
                    "",
                    f"- {current['model']['alerts']['recommended_followup']}",
                ]
            )
    lines.extend(
        [
            "",
            "## A/B/C Comparison",
            "",
            "| System | Target | Reconstruction | Top-1 | Top-5 | MRR | Mean Rank | NTA | Mean STFT |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in [
        _waveform_row("A. Raw waveform baseline", waveform_eval),
        _route_row("B. Sequence HuBERT retrieval", sequence_eval),
        _route_row("C. Codec latent reconstruction", codec_eval),
        _route_row("Legacy pooled HuBERT", pooled_eval),
    ]:
        if row is not None:
            lines.append(row)
    lines.extend(
        [
            "",
            "## Waveform QC",
            "",
            "| System | Recon RMS | Target RMS | Recon Centroid Hz | Target Centroid Hz | Unique Top-1 | Max Template Share | Target-Latent Oracle RMS |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for name, qc in [("B. Sequence HuBERT retrieval", sequence_audio_qc), ("C. Codec latent reconstruction", codec_audio_qc)]:
        if qc is None:
            continue
        lines.append(
            f"| {name} | {fmt_metric(qc.get('recon_rms_mean'))} | {fmt_metric(qc.get('target_rms_mean'))} | "
            f"{fmt_metric(qc.get('recon_spectral_centroid_hz_mean'))} | {fmt_metric(qc.get('target_spectral_centroid_hz_mean'))} | "
            f"{fmt_metric(qc.get('unique_top1_count'))} | {fmt_metric(qc.get('max_template_share'))} | "
            f"{fmt_metric(qc.get('target_latent_oracle_rms_mean'))} |"
        )
    lines.extend(
        [
            "",
            "## Speech Space Structure",
            "",
        ]
    )
    if space_summary is None:
        lines.append("- Space summary file not found.")
    else:
        lines.extend(
            [
                f"- dominant structure: `{space_summary.get('dominant_structure')}`",
                f"- target kind: `{space_summary.get('target_kind', 'unknown')}`",
                f"- subject centroid probe accuracy: `{fmt_metric(space_summary['subject_centroid_probe']['accuracy'])}`",
                f"- label centroid probe accuracy: `{fmt_metric(space_summary['label_centroid_probe']['accuracy'])}`",
                f"- within-label mean cosine distance: `{fmt_metric(space_summary['pairwise_cosine_distance'].get('within_label', {}).get('mean'))}`",
                f"- within-subject mean cosine distance: `{fmt_metric(space_summary['pairwise_cosine_distance'].get('within_subject', {}).get('mean'))}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Recommended path: **{recommendation_title}**",
            f"- Justification: {recommendation_body}",
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
    run_root = resolve_bundle_path(config["output"]["root"], BUNDLE_DIR) / build_run_name(config, args)
    split = str(args.split)
    default_alignment_eval = run_root / "metrics" / f"{split}_evaluation.json"
    pooled_eval = load_alignment_summary(resolve_bundle_path(args.alignment_eval, BUNDLE_DIR) if args.alignment_eval else None)
    current_eval = load_alignment_summary(default_alignment_eval)
    if pooled_eval is None and current_eval is not None and current_eval.get("target_kind") == "hubert_pooled":
        pooled_eval = current_eval
    sequence_eval = load_alignment_summary(resolve_bundle_path(args.sequence_eval, BUNDLE_DIR) if args.sequence_eval else None)
    codec_eval = load_alignment_summary(resolve_bundle_path(args.codec_eval, BUNDLE_DIR) if args.codec_eval else None)
    if current_eval is not None:
        if current_eval.get("target_kind") == "hubert_sequence" and sequence_eval is None:
            sequence_eval = current_eval
        if current_eval.get("target_kind") == "encodec_latent" and codec_eval is None:
            codec_eval = current_eval
    waveform_eval = load_waveform_summary(resolve_bundle_path(args.waveform_eval, BUNDLE_DIR) if args.waveform_eval else None)
    if waveform_eval is None:
        waveform_path = run_root / "metrics" / "test_metrics.json"
        waveform_eval = load_waveform_summary(waveform_path if waveform_path.exists() else None)
    space_summary = load_json(
        resolve_bundle_path(args.space_summary, BUNDLE_DIR)
        if args.space_summary
        else resolve_bundle_path(config["output"]["root"], BUNDLE_DIR)
        / "template_space"
        / Path(config["targets"]["cache_path"]).stem
        / "space_summary.json"
    )
    report = build_report(
        pooled_eval=pooled_eval,
        sequence_eval=sequence_eval,
        codec_eval=codec_eval,
        waveform_eval=waveform_eval,
        space_summary=space_summary,
        sequence_audio_qc=load_json(resolve_bundle_path(args.sequence_audio_qc, BUNDLE_DIR) if args.sequence_audio_qc else _default_audio_qc_path(sequence_eval)),
        codec_audio_qc=load_json(resolve_bundle_path(args.codec_audio_qc, BUNDLE_DIR) if args.codec_audio_qc else _default_audio_qc_path(codec_eval)),
    )
    output_path = (
        resolve_bundle_path(args.output_path, BUNDLE_DIR)
        if args.output_path
        else run_root / "metrics" / f"{split}_phase_report.md"
    )
    output_path.write_text(report, encoding="utf-8")
    print(f"Saved FEIS report to {output_path}")


if __name__ == "__main__":
    main()

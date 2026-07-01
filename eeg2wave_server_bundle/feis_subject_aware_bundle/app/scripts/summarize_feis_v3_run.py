"""Write FEIS v3 run summary report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.utils import ensure_dir


def parse_args():
    p = argparse.ArgumentParser(description="Summarize a FEIS v3 run.")
    p.add_argument("--run-dir", required=True)
    return p.parse_args()


def _fmt_bool(value) -> str:
    return "PASS" if bool(value) else "FAIL"


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    latest = json.loads((run_dir / "metrics" / "latest_metrics.json").read_text(encoding="utf-8"))
    report_dir = ensure_dir(run_dir / "reports")
    subject_test = latest.get("subject_holdout", {}).get("subject_test", {})
    subject_val = latest.get("subject_holdout", {}).get("subject_val", {})
    repeat = latest.get("repeat_holdout", {})
    resting = latest.get("resting_negative_control", {})
    lines = [
        "# FEIS v3 Run Summary",
        "",
        f"- run_dir: `{run_dir}`",
        f"- stage: `{latest.get('stage')}`",
        f"- aligner: `{latest.get('aligner')}`",
        f"- train phase: `{latest.get('train_phase', 'joint')}`",
        f"- generated artifact: `{latest.get('generated_artifact', 'generated_codec')}`",
        f"- retrieval output name: `{latest.get('retrieval_name', 'retrieval_diagnostic')}`",
        f"- retrieval diagnostic only: `{latest.get('retrieval_is_diagnostic_only')}`",
        f"- subject_id forward input: `{latest.get('subject_id_forward_input')}`",
        f"- allow negative-stage training: `{latest.get('allow_negative_train', False)}`",
        f"- claim status: **{latest.get('claim_status')}**",
        "",
        "## Required Gates",
        "",
        f"- subject_holdout val alignment: {_fmt_bool(subject_val.get('alignment_gate_pass'))}",
        f"- subject_holdout test alignment: {_fmt_bool(subject_test.get('alignment_gate_pass'))}",
        f"- repeat_holdout reliability: {_fmt_bool(repeat.get('repeat_reliability_gate_pass'))}",
        f"- resting negative control stayed negative: {_fmt_bool(resting.get('resting_negative_control_pass'))}",
        f"- generation gate: {_fmt_bool(latest.get('generation_gate_pass'))}",
        "",
        "## Subject Test Metrics",
        "",
        f"- prompt_acc: `{subject_test.get('prompt_acc')}`",
        f"- semantic_token_top3_gain_over_prior: `{subject_test.get('semantic_token_top3_gain_over_prior')}`",
        f"- token_retrieval_cross_subject_gain: `{subject_test.get('token_retrieval_cross_subject_gain')}`",
        f"- codec_token_top1: `{subject_test.get('codec_token_top1')}`",
        f"- generated_over_zero_codec_margin: `{subject_test.get('generated_over_zero_codec_margin')}`",
        f"- generated_over_shuffled_codec_margin: `{subject_test.get('generated_over_shuffled_codec_margin')}`",
        f"- generated_over_labelprior_codec_margin: `{subject_test.get('generated_over_labelprior_codec_margin')}`",
        "",
        "## Interpretation",
        "",
        "This run implements the FEIS v3 tokenized pipeline and writes diagnostic generated codec wavs. "
        "Do not claim EEG-to-Speech generation success unless the subject-holdout, repeat-holdout, "
        "strict codec-baseline, and resting negative-control gates all pass.",
        "",
    ]
    out = report_dir / "feis_v3_run_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()

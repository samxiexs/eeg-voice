from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write KaraOne v11 run summary report.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--wav-dir", default=None, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = write_summary(args.run_dir, wav_dir=args.wav_dir)
    print(json.dumps({"report": str(out)}, ensure_ascii=False, indent=2))


def write_summary(run_dir: str | Path, *, wav_dir: str | Path | None = None) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    wav_dir = Path(wav_dir).expanduser().resolve() if wav_dir else run_dir / "wavs"
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    history = read_json(run_dir / "metrics" / "history.json", default={})
    latest = read_json(run_dir / "metrics" / "latest_metrics.json", default={})
    synth = read_json(wav_dir / "v11_synthesis_summary.json", default={})
    compare_count = count_csv_rows(wav_dir / "waveform_compare" / "waveform_compare_manifest.csv")
    lines = [
        "# KaraOne v11 Tokenized Neural Speech Generation Summary",
        "",
        f"- run_dir: `{run_dir}`",
        f"- wav_dir: `{wav_dir}`",
        f"- best_epoch: `{history.get('best_epoch', '')}`",
        f"- best_score: `{history.get('best_score', '')}`",
        f"- subject_val_v11_alignment_gate_pass: `{bool(latest.get('subject_val_v11_alignment_gate_pass', False))}`",
        f"- subject_test_v11_alignment_gate_pass: `{bool(latest.get('subject_test_v11_alignment_gate_pass', False))}`",
        f"- subject_val_v11_generation_gate_pass: `{bool(latest.get('subject_val_v11_generation_gate_pass', False))}`",
        f"- subject_test_v11_generation_gate_pass: `{bool(latest.get('subject_test_v11_generation_gate_pass', False))}`",
        f"- wav_pairs: `{int(synth.get('n_pairs', 0) or 0)}`",
        f"- waveform_compare_figures: `{compare_count}`",
        f"- waveform_status: `{synth.get('waveform_status', 'not_generated')}`",
        "",
        "## Gate Metrics",
        "",
        "| metric | subject_val | subject_test |",
        "|---|---:|---:|",
    ]
    for metric in [
        "semantic_token_top3_gain_over_prior",
        "token_retrieval_cross_subject_gain",
        "same_label_cross_subject_gain",
        "prompt_acc",
        "pred_token_entropy",
        "channel_gate_entropy_mean",
        "codec_token_acc",
        "codec_token_top3_acc",
    ]:
        lines.append(f"| `{metric}` | {fmt(latest.get('subject_val_' + metric))} | {fmt(latest.get('subject_test_' + metric))} |")
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "Retrieval wavs are diagnostic baselines only. EEG-to-Speech generation success requires both v11 alignment and generation gates to pass on subject_val with same-sign behavior on subject_test.",
            "",
        ]
    )
    out = reports / "v11_run_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def read_json(path: Path, *, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def fmt(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return ""


if __name__ == "__main__":
    main()

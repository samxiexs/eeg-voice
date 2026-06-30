from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write KaraOne v10 final run summary report.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--wav-dir", default=None, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = write_summary(args.run_dir, wav_dir=args.wav_dir)
    print(json.dumps({"report": str(out)}, ensure_ascii=False, indent=2))


def write_summary(run_dir: str | Path, *, wav_dir: str | Path | None = None) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    wav_dir = Path(wav_dir).expanduser().resolve() if wav_dir is not None else run_dir / "wavs"
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    history_payload = read_json(run_dir / "metrics" / "history.json", default={})
    latest = read_json(run_dir / "metrics" / "latest_metrics.json", default={})
    synth = read_json(wav_dir / "v10_synthesis_summary.json", default={})
    best_epoch = history_payload.get("best_epoch", "")
    best_score = history_payload.get("best_score", "")
    wav_pairs = int(synth.get("n_pairs", 0) or 0)
    compare_manifest = wav_dir / "waveform_compare" / "waveform_compare_manifest.csv"
    compare_count = count_csv_rows(compare_manifest)
    val_gate = bool(latest.get("subject_val_v10_research_gate_pass", False))
    test_gate = bool(latest.get("subject_test_v10_research_gate_pass", False))
    lines = [
        "# KaraOne v10 Final Run Summary",
        "",
        f"- run_dir: `{run_dir}`",
        f"- best_epoch: `{best_epoch}`",
        f"- best_score: `{best_score}`",
        f"- subject_val_v10_research_gate_pass: `{val_gate}`",
        f"- subject_test_v10_research_gate_pass: `{test_gate}`",
        f"- wav_pairs: `{wav_pairs}`",
        f"- waveform_compare_figures: `{compare_count}`",
        f"- waveform_status: `{synth.get('waveform_status', 'not_generated')}`",
        "",
        "## Gate Metrics",
        "",
        "| metric | subject_val | subject_test |",
        "|---|---:|---:|",
    ]
    for metric in [
        "semantic_over_zero_gain",
        "semantic_over_mean_gain",
        "semantic_top3_gain_over_mean",
        "same_label_cross_subject_gain",
        "prompt_acc",
        "pred_std_ratio_median",
        "pred_pairwise_corr_median",
        "channel_gate_entropy_mean",
    ]:
        lines.append(
            f"| `{metric}` | {fmt(latest.get('subject_val_' + metric))} | {fmt(latest.get('subject_test_' + metric))} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- wav_dir: `{wav_dir}`",
            "- `figures/training_curves.png`",
            "- `figures/gate_metrics.png`",
            "- `figures/collapse_metrics.png`",
            "- `figures/channel_gate_top_channels.png`",
            "- `wavs/listening_manifest.csv`",
            "- `wavs/waveform_compare/`",
            "",
            "## Interpretation Rule",
            "",
            "The wav files are diagnostic semantic-retrieval artifacts. They should not be described as successful EEG-to-waveform reconstruction unless the semantic/prosody subject-holdout gates pass and a codec/vocoder decoder is used.",
            "",
        ]
    )
    out = reports / "v10_run_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def read_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


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

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot KaraOne v11 token alignment training curves.")
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = plot_run(args.run_dir)
    print(json.dumps({"figures": [str(path) for path in paths]}, ensure_ascii=False, indent=2))


def plot_run(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir).expanduser().resolve()
    hist_path = run_dir / "metrics" / "history.json"
    if not hist_path.exists():
        raise FileNotFoundError(f"Missing history: {hist_path}")
    payload = json.loads(hist_path.read_text(encoding="utf-8"))
    rows = payload.get("history", [])
    if not rows:
        raise RuntimeError(f"Empty history: {hist_path}")
    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_lines(rows, out_dir / "training_curves.png", "Training Losses", ["train_total", "train_semantic_token_ce", "train_semantic_token_ctc", "train_clip_nce", "train_codec_token_ce"]),
        plot_lines(rows, out_dir / "token_alignment_metrics.png", "Token Alignment Metrics", ["subject_val_semantic_token_top3_gain_over_prior", "subject_test_semantic_token_top3_gain_over_prior", "subject_val_token_retrieval_cross_subject_gain", "subject_test_token_retrieval_cross_subject_gain"]),
        plot_lines(rows, out_dir / "gate_metrics.png", "Gate Metrics", ["subject_val_prompt_acc", "subject_test_prompt_acc", "subject_val_same_label_cross_subject_gain", "subject_test_same_label_cross_subject_gain"]),
        plot_lines(rows, out_dir / "codec_metrics.png", "Codec Token Metrics", ["subject_val_codec_token_acc", "subject_test_codec_token_acc", "subject_val_codec_token_top3_acc", "subject_test_codec_token_top3_acc"]),
        plot_lines(rows, out_dir / "channel_gate_top_channels.png", "Channel Gate / Token Entropy", ["subject_val_channel_gate_entropy_mean", "subject_test_channel_gate_entropy_mean", "subject_val_pred_token_entropy", "subject_test_pred_token_entropy"]),
    ]
    return paths


def plot_lines(rows: list[dict], path: Path, title: str, keys: list[str]) -> Path:
    epochs = [int(row["epoch"]) for row in rows]
    plt.figure(figsize=(10, 5))
    plotted = False
    for key in keys:
        vals = [row.get(key) for row in rows]
        if any(isinstance(v, (int, float)) for v in vals):
            plt.plot(epochs, [float(v) if isinstance(v, (int, float)) else float("nan") for v in vals], label=key)
            plotted = True
    if not plotted:
        plt.plot(epochs, [0.0 for _ in epochs], label="no_numeric_values")
    plt.title(title)
    plt.xlabel("epoch")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


if __name__ == "__main__":
    main()

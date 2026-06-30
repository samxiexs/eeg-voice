from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/karaone-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/karaone-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot KaraOne v10 training, gate, collapse, and channel diagnostics.")
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = plot_run(args.run_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def plot_run(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    history = load_history(run_dir / "metrics" / "history.json")
    fig_dir = ensure_dir(run_dir / "figures")
    outputs: dict[str, Any] = {"run_dir": str(run_dir), "n_epochs": len(history), "figures": {}}
    outputs["figures"]["training_curves"] = str(plot_training_curves(history, fig_dir / "training_curves.png"))
    outputs["figures"]["gate_metrics"] = str(plot_gate_metrics(history, fig_dir / "gate_metrics.png"))
    outputs["figures"]["collapse_metrics"] = str(plot_collapse_metrics(history, fig_dir / "collapse_metrics.png"))
    outputs["figures"]["channel_gate_top_channels"] = str(
        plot_channel_top(run_dir / "channel_reports" / "best_subject_val" / "channel_gate_summary.csv", fig_dir / "channel_gate_top_channels.png")
    )
    write_json(run_dir / "figures" / "v10_plot_summary.json", outputs)
    return outputs


def load_history(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("history"), list):
        return list(payload["history"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported history format: {path}")


def plot_training_curves(history: list[dict[str, Any]], path: Path) -> Path:
    epochs = _epochs(history)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _plot(axes[0, 0], epochs, history, ["train_total"], "Total Loss")
    _plot(
        axes[0, 1],
        epochs,
        history,
        ["train_seq_ot", "train_seq_cos", "train_global_nce", "train_cross_subject_semantic_nce"],
        "Semantic Alignment Losses",
    )
    _plot(
        axes[1, 0],
        epochs,
        history,
        ["train_prompt_ce", "train_prompt_balanced_ce", "train_prompt_ctc", "train_semantic_token_ce"],
        "Prompt / Token Losses",
    )
    _plot(
        axes[1, 1],
        epochs,
        history,
        ["train_zero_prior_margin", "train_mean_prior_margin", "train_same_label_prototype_pull", "train_pairwise_decorrelation"],
        "v10 Prior / Prototype Penalties",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_gate_metrics(history: list[dict[str, Any]], path: Path) -> Path:
    epochs = _epochs(history)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _plot(
        axes[0, 0],
        epochs,
        history,
        ["subject_val_semantic_over_zero_gain", "subject_test_semantic_over_zero_gain"],
        "Semantic Over Zero Gain",
        hline=0.01,
    )
    _plot(
        axes[0, 1],
        epochs,
        history,
        ["subject_val_semantic_top3_gain_over_mean", "subject_test_semantic_top3_gain_over_mean"],
        "Top-3 Gain Over Mean",
        hline=0.02,
    )
    _plot(
        axes[1, 0],
        epochs,
        history,
        ["subject_val_same_label_cross_subject_gain", "subject_test_same_label_cross_subject_gain"],
        "Same-Label Cross-Subject Gain",
        hline=0.0,
    )
    _plot(
        axes[1, 1],
        epochs,
        history,
        ["subject_val_prompt_acc", "subject_test_prompt_acc"],
        "Prompt Accuracy",
        hline=0.13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_collapse_metrics(history: list[dict[str, Any]], path: Path) -> Path:
    epochs = _epochs(history)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _plot(
        axes[0, 0],
        epochs,
        history,
        ["subject_val_pred_std_ratio_median", "subject_test_pred_std_ratio_median"],
        "Pred Std Ratio",
    )
    axes[0, 0].axhspan(0.7, 1.5, color="tab:green", alpha=0.12)
    _plot(
        axes[0, 1],
        epochs,
        history,
        ["subject_val_pred_pairwise_corr_median", "subject_test_pred_pairwise_corr_median"],
        "Pairwise Corr Median",
        hline=0.75,
    )
    _plot(
        axes[1, 0],
        epochs,
        history,
        ["subject_val_channel_gate_entropy_mean", "subject_test_channel_gate_entropy_mean"],
        "Channel Gate Entropy",
        hline=0.20,
    )
    _plot(axes[1, 1], epochs, history, ["selection_score"], "v10 Selection Score")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_channel_top(summary_csv: Path, path: Path, top_n: int = 16) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    rows: list[dict[str, str]] = []
    if summary_csv.exists():
        with summary_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    if not rows:
        ax.text(0.5, 0.5, "No channel summary found", ha="center", va="center")
        ax.set_axis_off()
    else:
        rows.sort(key=lambda row: float(row.get("mean_gate", 0.0)), reverse=True)
        top = rows[: int(top_n)]
        names = [row["channel"] for row in top][::-1]
        values = [float(row["mean_gate"]) for row in top][::-1]
        ax.barh(names, values, color="#2f6f73")
        ax.set_title("Top Channel-MoE Gates")
        ax.set_xlabel("Mean gate")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def _epochs(history: list[dict[str, Any]]) -> list[int]:
    return [int(row.get("epoch", idx + 1)) for idx, row in enumerate(history)]


def _values(history: list[dict[str, Any]], key: str) -> list[float]:
    out = []
    for row in history:
        value = row.get(key)
        out.append(float(value) if isinstance(value, (int, float)) else float("nan"))
    return out


def _plot(ax, epochs: list[int], history: list[dict[str, Any]], keys: list[str], title: str, hline: float | None = None) -> None:
    plotted = False
    for key in keys:
        if any(key in row for row in history):
            ax.plot(epochs, _values(history, key), marker="o", markersize=2.5, linewidth=1.2, label=key)
            plotted = True
    if hline is not None:
        ax.axhline(float(hline), color="black", linewidth=1.0, linestyle="--", alpha=0.55)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No metric", ha="center", va="center", transform=ax.transAxes)


if __name__ == "__main__":
    main()

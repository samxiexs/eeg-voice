from __future__ import annotations

"""Training-curve plotting for KaraOne reconstruction runs.

Reads a ``metrics/history.jsonl`` file (one line per epoch with ``train`` and
``val`` dicts) and renders a multi-panel PNG of how the key quantities evolve.
Used both live during training (called each epoch) and as a standalone script.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / server-safe
import matplotlib.pyplot as plt


def _load_history(jsonl_path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(jsonl_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _series(rows: list[dict], section: str, key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for row in rows:
        value = row.get(section, {}).get(key)
        if value is not None:
            xs.append(int(row["epoch"]))
            ys.append(float(value))
    return xs, ys


def _panel(ax, rows, title, specs, ylabel=None, hline=None):
    plotted = False
    for section, key, label in specs:
        xs, ys = _series(rows, section, key)
        if xs:
            ax.plot(xs, ys, marker=".", markersize=3, linewidth=1.3, label=label)
            plotted = True
    if hline is not None:
        ax.axhline(hline, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("epoch", fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=9)


def plot_history(jsonl_path: str | Path, out_png: str | Path, title: str | None = None) -> Path:
    """Render training curves from a history.jsonl file to a PNG. Returns the path."""
    rows = _load_history(jsonl_path)
    out_png = Path(out_png)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # NOTE: train['recon_cos'] is a LOSS (1 - cosine, lower is better);
    # val['pred_recon_cos'] is a SIMILARITY (higher is better). Kept on separate panels.
    _panel(axes[0, 0], rows, "Train total loss", [("train", "total", "total")], ylabel="loss")
    _panel(
        axes[0, 1],
        rows,
        "Val reconstruction cosine (higher=better)",
        [
            ("val", "pred_recon_cos", "pred"),
            ("val", "zeroeeg_recon_cos", "zero-EEG"),
            ("val", "mean_recon_cos", "mean-latent"),
        ],
        ylabel="cosine",
    )
    _panel(
        axes[0, 2],
        rows,
        "Val gain over baselines (THE metric)",
        [
            ("val", "pred_over_zero_cos_gain", "pred - zero"),
            ("val", "pred_over_mean_cos_gain", "pred - mean"),
        ],
        ylabel="cosine gain",
        hline=0.0,
    )
    _panel(
        axes[1, 0],
        rows,
        "Content (phoneme) accuracy",
        [("train", "content_acc", "train"), ("val", "content_acc", "val")],
        ylabel="accuracy",
        hline=1.0 / 11.0,  # chance for 11 classes
    )
    _panel(
        axes[1, 1],
        rows,
        "Alignment / aux losses",
        [
            ("train", "clip_loss", "clip (InfoNCE)"),
            ("train", "supcon", "supcon"),
            ("train", "recon_mse", "recon_mse"),
            ("train", "ctc", "ctc"),
            ("train", "frame_energy", "frame energy"),
            ("train", "voiced_rms", "voiced rms"),
            ("train", "decoder_scale", "decoder scale"),
        ],
        ylabel="loss",
    )
    _panel(
        axes[1, 2],
        rows,
        "Prediction std ratio (1.0=matches target)",
        [("train", "std_ratio", "train"), ("val", "pred_std_ratio_median", "val median")],
        ylabel="std ratio",
        hline=1.0,
    )

    fig.suptitle(title or out_png.parent.parent.name, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def plot_diffusion_history(jsonl_path: str | Path, out_png: str | Path, title: str | None = None) -> Path:
    """Render training curves for the diffusion model. The headline panels are the
    anti-collapse diagnostics: std-ratio (should rise toward 1.0) and pairwise
    correlation of samples (should stay low, vs ~0.94 for the collapsed regression)."""
    rows = _load_history(jsonl_path)
    out_png = Path(out_png)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    _panel(axes[0, 0], rows, "Train diffusion loss (eps MSE)", [("train", "loss", "eps MSE")], ylabel="loss")
    _panel(
        axes[0, 1],
        rows,
        "Val reconstruction cosine (sampled)",
        [
            ("val", "pred_recon_cos", "sample"),
            ("val", "mean_recon_cos", "mean-latent"),
            ("val", "zeroeeg_recon_cos", "zero-EEG"),
        ],
        ylabel="cosine",
    )
    _panel(
        axes[0, 2],
        rows,
        "Val gain over baselines (honest)",
        [
            ("val", "pred_over_mean_cos_gain", "sample - mean"),
            ("val", "pred_over_zero_cos_gain", "sample - zero"),
        ],
        ylabel="cosine gain",
        hline=0.0,
    )
    _panel(
        axes[1, 0],
        rows,
        "Std ratio — ANTI-COLLAPSE (target 1.0)",
        [("val", "pred_std_ratio_median", "sample")],
        ylabel="std ratio",
        hline=1.0,
    )
    _panel(
        axes[1, 1],
        rows,
        "Pairwise corr of samples (lower=diverse)",
        [("val", "pred_pairwise_corr_median", "sample")],
        ylabel="|corr|",
        hline=0.0,
    )
    _panel(
        axes[1, 2],
        rows,
        "Within-subject retrieval top-1",
        [
            ("val", "pred_within_subject_label_top1", "label"),
            ("val", "pred_within_subject_trial_top1", "trial"),
        ],
        ylabel="top-1",
        hline=1.0 / 11.0,
    )
    fig.suptitle(title or out_png.parent.parent.name, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot KaraOne training curves from history.jsonl")
    parser.add_argument("--history", required=True, help="path to metrics/history.jsonl")
    parser.add_argument("--out", default=None, help="output PNG (default: alongside history)")
    args = parser.parse_args()
    history = Path(args.history)
    out = args.out or (history.parent / "training_curves.png")
    path = plot_history(history, out)
    print(f"[plot] wrote {path}")

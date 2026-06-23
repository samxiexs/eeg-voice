"""Make training and waveform figures for EEG-only direct runs."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

BUNDLE_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(BUNDLE_DIR / "../artifacts/matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly


def parse_args():
    p = argparse.ArgumentParser(description="Create training curves and wav comparison figures.")
    p.add_argument("--run-dir", default=None, help="Run directory containing metrics/history.csv.")
    p.add_argument("--wav-dir", default=None, help="Directory containing listening_manifest.csv and wav files.")
    p.add_argument("--max-waveforms", type=int, default=24)
    return p.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def _plot_lines(rows: list[dict[str, str]], x_key: str, y_keys: list[str], title: str, out_path: Path) -> None:
    if not rows:
        return
    x = np.asarray([_float(row, x_key) for row in rows], dtype=np.float64)
    present = [key for key in y_keys if key in rows[0]]
    if not present:
        return
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for key in present:
        y = np.asarray([_float(row, key) for row in rows], dtype=np.float64)
        ax.plot(x, y, linewidth=1.7, marker="o" if len(rows) <= 12 else None, label=key)
    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_training_figures(run_dir: Path) -> list[Path]:
    history = run_dir / "metrics" / "history.csv"
    if not history.exists():
        raise FileNotFoundError(f"Missing training history: {history}")
    rows = _read_csv(history)
    out_dir = run_dir / "figures"
    made = []
    specs = [
        (
            ["train_total", "train_recon_cos", "train_temporal_envelope", "train_diffusion_loss"],
            "Training losses",
            out_dir / "training_losses.png",
        ),
        (
            ["train_content_acc", "val_top1", "val_recon_cos", "val_score"],
            "Validation and selection metrics",
            out_dir / "validation_metrics.png",
        ),
        (
            ["train_moe_gate_mean", "train_moe_usage_min", "train_moe_usage_max", "train_moe_active_channels"],
            "MoE routing diagnostics",
            out_dir / "moe_routing.png",
        ),
        (
            ["train_std_ratio", "train_mean_distance", "val_std_ratio", "val_pred_corr"],
            "Collapse diagnostics",
            out_dir / "collapse_diagnostics.png",
        ),
    ]
    for keys, title, out_path in specs:
        _plot_lines(rows, "epoch", keys, title, out_path)
        if out_path.exists():
            made.append(out_path)
    return made


def _read_wav(path: Path) -> tuple[int, np.ndarray]:
    sr, audio = wavfile.read(str(path))
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / max(float(np.iinfo(audio.dtype).max), 1.0)
    else:
        audio = audio.astype(np.float32)
    return int(sr), audio


def _match_rate(sr_a: int, a: np.ndarray, sr_b: int, b: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    if sr_a == sr_b:
        return sr_a, a, b
    gcd = np.gcd(sr_b, sr_a)
    b = resample_poly(b, sr_a // gcd, sr_b // gcd).astype(np.float32)
    return sr_a, a, b


def _norm(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return audio / max(peak, 1e-8)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)) + 1e-12)


def make_waveform_figures(wav_dir: Path, max_waveforms: int = 24) -> list[Path]:
    manifest = wav_dir / "listening_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing listening manifest: {manifest}")
    rows = _read_csv(manifest)
    grouped: dict[tuple[str, str, str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        unit = row.get("source_unit", "")
        key = (unit, row["sample_key"], row["label"], row["stage"])
        grouped.setdefault(key, {})[row["wav_type"]] = row

    out_dir = wav_dir / "waveform_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    figures = []
    for (source_unit, sample_key, label, stage), items in list(sorted(grouped.items()))[:max_waveforms]:
        if "original_ref" not in items or "pred_scaled" not in items:
            continue
        orig_path = wav_dir / items["original_ref"]["file"]
        pred_path = wav_dir / items["pred_scaled"]["file"]
        sr_o, orig = _read_wav(orig_path)
        sr_p, pred = _read_wav(pred_path)
        sr, orig, pred = _match_rate(sr_o, orig, sr_p, pred)
        n = min(len(orig), len(pred))
        orig, pred = orig[:n], pred[:n]
        t = np.arange(n, dtype=np.float32) / float(sr)
        corr = float(np.corrcoef(_norm(orig), _norm(pred))[0, 1]) if n > 2 else float("nan")
        title_prefix = f"unit={source_unit} | " if source_unit else ""
        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
        fig.suptitle(f"{title_prefix}{sample_key} | label={label} | stage={stage} | original vs generated", fontsize=13)
        axes[0].plot(t, orig, color="#1f77b4", linewidth=0.9)
        axes[0].set_title(f"Original reference waveform | RMS={_rms(orig):.4f}", loc="left", fontsize=10)
        axes[0].set_ylabel("amp")
        axes[0].grid(True, alpha=0.2)
        axes[1].plot(t, pred, color="#d62728", linewidth=0.9)
        axes[1].set_title(f"Generated pred_scaled waveform | RMS={_rms(pred):.4f}", loc="left", fontsize=10)
        axes[1].set_ylabel("amp")
        axes[1].grid(True, alpha=0.2)
        axes[2].plot(t, _norm(orig), color="#1f77b4", linewidth=0.9, label="original normalized")
        axes[2].plot(t, _norm(pred), color="#d62728", linewidth=0.8, alpha=0.78, label="generated normalized")
        axes[2].set_title(f"Peak-normalized overlay | corr={corr:.4f}", loc="left", fontsize=10)
        axes[2].set_ylabel("norm amp")
        axes[2].set_xlabel("time (s)")
        axes[2].grid(True, alpha=0.2)
        axes[2].legend(loc="upper right", frameon=False)
        for ax in axes:
            ax.set_xlim(float(t[0]) if len(t) else 0.0, float(t[-1]) if len(t) else 1.0)
        unit_tag = f"unit{source_unit}_" if source_unit else ""
        out_path = out_dir / f"{unit_tag}{sample_key}_{label}_{stage}_original_vs_pred_scaled_waveform.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        figures.append(out_path)
        summary_rows.append({
            "source_unit": source_unit,
            "sample_key": sample_key,
            "label": label,
            "stage": stage,
            "sample_rate": sr,
            "duration_sec": n / sr,
            "original_rms": _rms(orig),
            "generated_rms": _rms(pred),
            "normalized_corr": corr,
            "png": out_path.name,
            "original_wav": orig_path.name,
            "generated_wav": pred_path.name,
        })

    if summary_rows:
        with (out_dir / "waveform_compare_summary.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        figures.append(out_dir / "waveform_compare_summary.csv")

        cols = 2
        rows_n = int(np.ceil(len(summary_rows) / cols))
        fig, axes = plt.subplots(rows_n, cols, figsize=(14, max(3.0 * rows_n, 4.0)), squeeze=False, constrained_layout=True)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, summary in zip(axes.ravel(), summary_rows):
            sr_o, orig = _read_wav(wav_dir / summary["original_wav"])
            sr_p, pred = _read_wav(wav_dir / summary["generated_wav"])
            sr, orig, pred = _match_rate(sr_o, orig, sr_p, pred)
            n = min(len(orig), len(pred))
            orig, pred = orig[:n], pred[:n]
            t = np.arange(n, dtype=np.float32) / float(sr)
            unit_part = f"unit={summary['source_unit']} " if summary["source_unit"] else ""
            ax.axis("on")
            ax.plot(t, _norm(orig), color="#1f77b4", linewidth=0.65, label="orig")
            ax.plot(t, _norm(pred), color="#d62728", linewidth=0.6, alpha=0.75, label="gen")
            ax.set_title(
                f"{unit_part}label={summary['label']} stage={summary['stage']} corr={float(summary['normalized_corr']):.3f}",
                fontsize=9,
            )
            ax.set_xlim(0, 1.0)
            ax.set_ylim(-1.05, 1.05)
            ax.grid(True, alpha=0.18)
            ax.legend(loc="upper right", frameon=False, fontsize=8)
        contact = out_dir / "original_vs_pred_scaled_contact_sheet.png"
        fig.savefig(contact, dpi=180)
        plt.close(fig)
        figures.append(contact)
    return figures


def main() -> None:
    args = parse_args()
    made: list[Path] = []
    if args.run_dir:
        made.extend(make_training_figures(Path(args.run_dir).resolve()))
    if args.wav_dir:
        made.extend(make_waveform_figures(Path(args.wav_dir).resolve(), max_waveforms=args.max_waveforms))
    if not made:
        raise SystemExit("Nothing to do: pass --run-dir and/or --wav-dir")
    print("[figures]")
    for path in made:
        print(path)


if __name__ == "__main__":
    main()

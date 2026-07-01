from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BUNDLE_DIR = Path(__file__).resolve().parents[1]
REPO_TOOLS_ROOT = BUNDLE_DIR.parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))
if str(REPO_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_TOOLS_ROOT))

from generate_all_waveform_comparisons import pairs_from_manifest, resample_linear  # noqa: E402
from src.karaone_v12.time_anchor import best_lag_corr, coarse_log_spectral_frames, overlap_by_lag, pearson_corr, read_wav_mono, rms_envelope, shift_audio, write_csv  # noqa: E402
from src.utils import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KaraOne wavs with zero-lag, diagnostic best-lag, and predicted-lag metrics.")
    parser.add_argument("--wav-dir", required=True, type=Path)
    parser.add_argument("--original-type", default="original")
    parser.add_argument("--reconstruction-type", default="generated_codec")
    parser.add_argument("--max-lag-sec", type=float, default=1.0)
    parser.add_argument("--write-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_lagaware(args.wav_dir, original_type=args.original_type, reconstruction_type=args.reconstruction_type, max_lag_sec=float(args.max_lag_sec), write_figures=bool(args.write_figures))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def evaluate_lagaware(wav_dir: str | Path, *, original_type: str, reconstruction_type: str, max_lag_sec: float, write_figures: bool) -> dict[str, Any]:
    wav_dir = Path(wav_dir).expanduser().resolve()
    pairs = pairs_from_manifest(wav_dir, original_type, reconstruction_type)
    if not pairs:
        raise RuntimeError(f"No waveform pairs found in {wav_dir} for {original_type} vs {reconstruction_type}")
    trace = load_trace(wav_dir / "generation_trace.csv")
    out_dir = wav_dir / "waveform_compare_lagaware"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair in pairs:
        sr_o, original = read_wav_mono(pair["original_wav"])
        sr_r, recon = read_wav_mono(pair["reconstruction_wav"])
        if sr_r != sr_o:
            recon = np.asarray(resample_linear(recon.tolist(), sr_r, sr_o), dtype=np.float32)
        n = min(original.shape[0], recon.shape[0])
        original = original[:n]
        recon = recon[:n]
        key = str(pair["key"])
        pred_lag = float(trace.get((key, reconstruction_type), trace.get((key, "generated_codec"), {})).get("pred_lag_sec", 0.0) or 0.0)
        zero_wave = pearson_corr(original, recon)
        stride = max(1, int(round(sr_o * 0.001)))
        best_wave, best_lag, best_n = best_lag_corr(original, recon, sample_rate=sr_o, max_lag_sec=max_lag_sec, min_overlap_sec=0.5, stride=stride)
        pred_recon = recon if reconstruction_type.endswith("pred_lag") else shift_audio(recon, pred_lag, sample_rate=sr_o)
        pred_wave = pearson_corr(original, pred_recon)
        env_o = rms_envelope(original, sample_rate=sr_o)
        env_r = rms_envelope(recon, sample_rate=sr_o)
        env_pred = rms_envelope(pred_recon, sample_rate=sr_o)
        zero_env = pearson_corr(env_o, env_r)
        best_env, best_env_lag, _ = best_lag_corr(env_o, env_r, sample_rate=100.0, max_lag_sec=max_lag_sec, min_overlap_sec=0.2, stride=1)
        pred_env = pearson_corr(env_o, env_pred)
        mel_o = coarse_log_spectral_frames(original, sample_rate=sr_o)
        mel_r = coarse_log_spectral_frames(recon, sample_rate=sr_o)
        mel_pred = coarse_log_spectral_frames(pred_recon, sample_rate=sr_o)
        zero_mel = corr_2d(mel_o, mel_r)
        best_mel, best_mel_lag = best_lag_corr_2d(mel_o, mel_r, max_lag_frames=int(round(max_lag_sec * 100.0)))
        pred_mel = corr_2d(mel_o, mel_pred)
        row = {
            "key": key,
            "label": pair.get("label", ""),
            "stage": pair.get("stage", ""),
            "reconstruction_type": reconstruction_type,
            "original_wav": pair["original_wav"],
            "reconstruction_wav": pair["reconstruction_wav"],
            "sample_rate": sr_o,
            "duration_s": n / sr_o if sr_o else 0.0,
            "zero_lag_waveform_corr": zero_wave,
            "best_lag_waveform_corr": best_wave,
            "best_lag_waveform_sec": best_lag,
            "pred_lag_waveform_corr": pred_wave,
            "pred_lag_sec": pred_lag,
            "zero_lag_envelope_corr": zero_env,
            "best_lag_envelope_corr": best_env,
            "best_lag_envelope_sec": best_env_lag,
            "pred_lag_envelope_corr": pred_env,
            "zero_lag_mel_corr": zero_mel,
            "best_lag_mel_corr": best_mel,
            "best_lag_mel_sec": best_mel_lag / 100.0,
            "pred_lag_mel_corr": pred_mel,
            "diagnostic_best_lag_only": True,
            "n_best_lag_overlap": best_n,
        }
        rows.append(row)
    manifest = out_dir / f"lagaware_manifest_{reconstruction_type}.csv"
    write_csv(manifest, rows)
    run_metrics_dir = wav_dir.parent / "metrics"
    if run_metrics_dir.exists() or wav_dir.parent.name:
        write_csv(run_metrics_dir / "lagaware_waveform_metrics.csv", rows)
    summary = summarize(rows, reconstruction_type=reconstruction_type, max_lag_sec=max_lag_sec)
    write_json(out_dir / f"lagaware_summary_{reconstruction_type}.json", summary)
    if write_figures:
        write_summary_figures(rows, wav_dir.parent / "figures", out_dir, reconstruction_type)
    return summary


def load_trace(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(str(row.get("sample_key", "")), str(row.get("wav_type", ""))): row for row in rows}


def corr_2d(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.shape[0], b.shape[0])
    if n < 2:
        return float("nan")
    return pearson_corr(a[:n].reshape(-1), b[:n].reshape(-1))


def best_lag_corr_2d(a: np.ndarray, b: np.ndarray, *, max_lag_frames: int) -> tuple[float, int]:
    best = (-1.0e9, 0)
    for lag in range(-int(max_lag_frames), int(max_lag_frames) + 1):
        aa, bb = overlap_by_lag(a, b, lag)
        if aa.shape[0] < 5 or bb.shape[0] < 5:
            continue
        value = corr_2d(aa, bb)
        if np.isfinite(value) and value > best[0]:
            best = (float(value), int(lag))
    return best


def summarize(rows: list[dict[str, Any]], *, reconstruction_type: str, max_lag_sec: float) -> dict[str, Any]:
    metrics = {}
    for key in [
        "zero_lag_waveform_corr",
        "best_lag_waveform_corr",
        "pred_lag_waveform_corr",
        "zero_lag_envelope_corr",
        "best_lag_envelope_corr",
        "pred_lag_envelope_corr",
        "zero_lag_mel_corr",
        "best_lag_mel_corr",
        "pred_lag_mel_corr",
    ]:
        vals = np.asarray([float(row[key]) for row in rows if np.isfinite(float(row[key]))], dtype=np.float32)
        metrics[f"{key}_mean"] = float(vals.mean()) if vals.size else 0.0
        metrics[f"{key}_median"] = float(np.median(vals)) if vals.size else 0.0
    lags = np.asarray([float(row["best_lag_envelope_sec"]) for row in rows], dtype=np.float32)
    return {
        "event": "karaone_v12_lagaware_wav_eval",
        "reconstruction_type": reconstruction_type,
        "n": len(rows),
        "max_lag_sec": float(max_lag_sec),
        "best_lag_envelope_sec_median": float(np.median(lags)) if lags.size else 0.0,
        "best_lag_envelope_sec_q25": float(np.percentile(lags, 25)) if lags.size else 0.0,
        "best_lag_envelope_sec_q75": float(np.percentile(lags, 75)) if lags.size else 0.0,
        "diagnostic_rule": "best_lag metrics use reference audio and are diagnostic only; predicted_lag metrics use model-predicted lag when available",
        **metrics,
    }


def write_summary_figures(rows: list[dict[str, Any]], fig_dir: Path, out_dir: Path, reconstruction_type: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    names = [
        ("zero_lag_envelope_corr", "best_lag_envelope_corr", "pred_lag_envelope_corr", "Envelope Corr"),
        ("zero_lag_mel_corr", "best_lag_mel_corr", "pred_lag_mel_corr", "Mel-Proxy Corr"),
        ("zero_lag_waveform_corr", "best_lag_waveform_corr", "pred_lag_waveform_corr", "Waveform Corr"),
    ]
    plt.figure(figsize=(10, 5))
    labels, values = [], []
    for zero, best, pred, title in names:
        labels.extend([f"{title}\nzero", f"{title}\nbest", f"{title}\npred"])
        values.extend([mean(rows, zero), mean(rows, best), mean(rows, pred)])
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=30, ha="right", fontsize=8)
    plt.ylabel("mean correlation")
    plt.title(f"Zero vs Best-Lag vs Predicted-Lag ({reconstruction_type})")
    plt.tight_layout()
    for path in [fig_dir / "zero_vs_best_vs_pred_lag_corr.png", out_dir / f"zero_vs_best_vs_pred_lag_corr_{reconstruction_type}.png"]:
        plt.savefig(path, dpi=180)
    plt.close()

    lags = [float(row["best_lag_envelope_sec"]) for row in rows]
    plt.figure(figsize=(8, 4))
    plt.hist(lags, bins=30)
    plt.xlabel("best lag by envelope (sec)")
    plt.ylabel("count")
    plt.title(f"Best-Lag Distribution ({reconstruction_type})")
    plt.tight_layout()
    for path in [fig_dir / "lag_histogram.png", out_dir / f"lag_histogram_{reconstruction_type}.png"]:
        plt.savefig(path, dpi=180)
    plt.close()


def mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
    return float(np.mean(vals)) if vals else 0.0


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/feis-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-feis")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    HAS_MATPLOTLIB = False

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.utils import ensure_dir, resolve_bundle_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit reconstructed FEIS alignment wavs.")
    parser.add_argument("--eval-json", required=True, help="Path to metrics/test_evaluation.json")
    parser.add_argument("--data-root", default=str(BUNDLE_DIR / "../data/feis"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--include-oracle", action="store_true")
    parser.add_argument("--n-mels", type=int, default=40)
    return parser.parse_args()


def _resolve_path(path: str | Path, *bases: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    for base in bases:
        resolved = base / candidate
        if resolved.exists():
            return resolved
    return bases[0] / candidate


def _read_wav(path: Path, target_sr: int | None = None, target_len: int | None = None) -> tuple[int, np.ndarray]:
    sr, audio = wavfile.read(str(path))
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / float(np.iinfo(audio.dtype).max)
    else:
        audio = audio.astype(np.float32)
    if target_sr is not None and int(sr) != int(target_sr):
        gcd = math.gcd(int(sr), int(target_sr))
        audio = resample_poly(audio, target_sr // gcd, int(sr) // gcd).astype(np.float32)
        sr = int(target_sr)
    if target_len is not None:
        if audio.shape[0] >= target_len:
            audio = audio[:target_len]
        else:
            audio = np.pad(audio, (0, target_len - audio.shape[0]))
    return int(sr), audio.astype(np.float32)


def _audio_stats(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    audio = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)) + 1e-12)
    peak = float(np.max(np.abs(audio)) if audio.size else 0.0)
    duration_sec = float(audio.shape[0] / sample_rate)
    if audio.size < 8 or rms < 1e-10:
        return {
            "duration_sec": duration_sec,
            "rms": rms,
            "peak": peak,
            "spectral_centroid_hz": 0.0,
            "low_energy_lt_500hz": 0.0,
            "low_energy_lt_1000hz": 0.0,
            "zero_crossing_rate": 0.0,
            "silence_fraction": 1.0,
            "clipping_fraction": 0.0,
        }
    window = np.hanning(audio.shape[0]).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(audio * window))
    freqs = np.fft.rfftfreq(audio.shape[0], 1.0 / sample_rate)
    power = np.square(spectrum)
    centroid = float(np.sum(freqs * spectrum) / max(float(np.sum(spectrum)), 1e-12))
    frame = 400
    hop = 160
    frame_rms = []
    for start in range(0, max(audio.shape[0] - frame + 1, 1), hop):
        chunk = audio[start : start + frame]
        if chunk.shape[0] < frame:
            chunk = np.pad(chunk, (0, frame - chunk.shape[0]))
        frame_rms.append(float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)) + 1e-12))
    return {
        "duration_sec": duration_sec,
        "rms": rms,
        "peak": peak,
        "spectral_centroid_hz": centroid,
        "low_energy_lt_500hz": float(power[freqs < 500.0].sum() / max(float(power.sum()), 1e-12)),
        "low_energy_lt_1000hz": float(power[freqs < 1000.0].sum() / max(float(power.sum()), 1e-12)),
        "zero_crossing_rate": float(np.mean(np.abs(np.diff(np.signbit(audio))).astype(np.float32))),
        "silence_fraction": float(np.mean(np.asarray(frame_rms) < 0.003)),
        "clipping_fraction": float(np.mean(np.abs(audio) >= 0.98)),
    }


def _mel_filterbank(sample_rate: int, n_fft_bins: int, n_mels: int) -> np.ndarray:
    def hz_to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)

    freqs = np.linspace(0.0, sample_rate / 2.0, n_fft_bins)
    mel_points = np.linspace(hz_to_mel(np.asarray([0.0]))[0], hz_to_mel(np.asarray([sample_rate / 2.0]))[0], n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    filters = np.zeros((n_mels, n_fft_bins), dtype=np.float32)
    for mel_idx in range(n_mels):
        left, center, right = hz_points[mel_idx : mel_idx + 3]
        left_slope = (freqs - left) / max(center - left, 1e-8)
        right_slope = (right - freqs) / max(right - center, 1e-8)
        filters[mel_idx] = np.maximum(0.0, np.minimum(left_slope, right_slope))
    return filters


def _spectral_profiles(waveforms: list[np.ndarray], sample_rate: int, n_mels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not waveforms:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    max_len = max(item.shape[0] for item in waveforms)
    spectra = []
    for audio in waveforms:
        padded = audio if audio.shape[0] == max_len else np.pad(audio, (0, max_len - audio.shape[0]))
        spectra.append(np.square(np.abs(np.fft.rfft(padded * np.hanning(max_len)))))
    mean_power = np.mean(np.stack(spectra, axis=0), axis=0).astype(np.float32)
    freqs = np.fft.rfftfreq(max_len, 1.0 / sample_rate).astype(np.float32)
    mel = _mel_filterbank(sample_rate=sample_rate, n_fft_bins=mean_power.shape[0], n_mels=n_mels) @ mean_power
    return freqs, mean_power, np.log(np.maximum(mel, 1e-10)).astype(np.float32)


def _mean_stats(stats: list[dict[str, float]], prefix: str) -> dict[str, float]:
    if not stats:
        return {}
    keys = sorted(stats[0])
    return {f"{prefix}_{key}_mean": float(np.mean([item[key] for item in stats])) for key in keys}


def _top1_collapse(rows: list[dict[str, object]]) -> dict[str, float | int | None]:
    top1 = [
        str(row.get("retrieved_template_id") or (row.get("topk") or [{}])[0].get("template_id", ""))
        for row in rows
        if row.get("retrieved_template_id") or row.get("topk")
    ]
    if not top1:
        return {
            "unique_top1_count": 0,
            "unique_top1_rate": None,
            "top1_entropy": None,
            "top1_entropy_normalized": None,
            "max_template_share": None,
        }
    _, counts = np.unique(np.asarray(top1), return_counts=True)
    probs = counts.astype(np.float64) / float(len(top1))
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    max_entropy = float(np.log(max(len(counts), 1)))
    return {
        "unique_top1_count": int(len(counts)),
        "unique_top1_rate": float(len(counts) / len(top1)),
        "top1_entropy": entropy,
        "top1_entropy_normalized": float(entropy / max(max_entropy, 1e-12)) if len(counts) > 1 else 0.0,
        "max_template_share": float(np.max(probs)),
    }


def main() -> None:
    args = parse_args()
    eval_json = _resolve_path(args.eval_json, Path.cwd(), BUNDLE_DIR)
    payload = json.loads(eval_json.read_text(encoding="utf-8"))
    output_dir = ensure_dir(args.output_dir or eval_json.parent)
    data_root = resolve_bundle_path(args.data_root, BUNDLE_DIR)

    predictions_path = _resolve_path(payload["model"]["predictions_path"], eval_json.parent, BUNDLE_DIR, Path.cwd())
    rows = json.loads(predictions_path.read_text(encoding="utf-8"))["predictions"]
    recon_stats: list[dict[str, float]] = []
    target_stats: list[dict[str, float]] = []
    target_scale_stats: list[dict[str, float]] = []
    target_latent_stats: list[dict[str, float]] = []
    recon_waveforms: list[np.ndarray] = []
    target_waveforms: list[np.ndarray] = []
    manifest_rows: list[dict[str, object]] = []

    for idx, row in enumerate(rows):
        recon_path = _resolve_path(row["saved_wav_path"], predictions_path.parent, BUNDLE_DIR, Path.cwd())
        sr, recon = _read_wav(recon_path, target_sr=args.sample_rate)
        target_path = _resolve_path(row["audio_path"], data_root, BUNDLE_DIR, Path.cwd())
        _, target = _read_wav(target_path, target_sr=sr, target_len=recon.shape[0])
        recon_stat = _audio_stats(recon, sr)
        target_stat = _audio_stats(target, sr)
        recon_stats.append(recon_stat)
        target_stats.append(target_stat)
        recon_waveforms.append(recon)
        target_waveforms.append(target)

        target_scale_path = row.get("target_scale_oracle_wav_path")
        target_latent_path = row.get("target_latent_oracle_wav_path")
        if target_scale_path:
            _, target_scale = _read_wav(_resolve_path(str(target_scale_path), predictions_path.parent, BUNDLE_DIR), target_sr=sr)
            target_scale_stats.append(_audio_stats(target_scale, sr))
        if target_latent_path:
            _, target_latent = _read_wav(_resolve_path(str(target_latent_path), predictions_path.parent, BUNDLE_DIR), target_sr=sr)
            target_latent_stats.append(_audio_stats(target_latent, sr))

        manifest_rows.append(
            {
                "index": idx,
                "subject_id": row.get("subject_id"),
                "label": row.get("label"),
                "trial_index": row.get("trial_index"),
                "template_id": row.get("template_id"),
                "retrieved_template_id": row.get("retrieved_template_id"),
                "retrieved_label": row.get("retrieved_label"),
                "top1_cosine_similarity": row.get("top1_cosine_similarity"),
                "first_match_rank": row.get("first_match_rank"),
                "saved_wav_path": str(recon_path),
                "target_audio_path": str(target_path),
                "target_scale_oracle_wav_path": target_scale_path,
                "target_latent_oracle_wav_path": target_latent_path,
                "recon_rms": recon_stat["rms"],
                "target_rms": target_stat["rms"],
                "recon_spectral_centroid_hz": recon_stat["spectral_centroid_hz"],
                "target_spectral_centroid_hz": target_stat["spectral_centroid_hz"],
            }
        )

    freqs, recon_power, recon_mel = _spectral_profiles(recon_waveforms, args.sample_rate, args.n_mels)
    _, target_power, target_mel = _spectral_profiles(target_waveforms, args.sample_rate, args.n_mels)
    np.savez_compressed(
        output_dir / "mel_summary.npz",
        recon_log_mel=recon_mel,
        target_log_mel=target_mel,
        freqs=freqs,
        recon_mean_power=recon_power,
        target_mean_power=target_power,
    )
    spectral_plot_path = output_dir / "spectral_mel_summary.png"
    if freqs.size and HAS_MATPLOTLIB and plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(freqs, 10.0 * np.log10(np.maximum(recon_power, 1e-12)), label="recon")
        axes[0].plot(freqs, 10.0 * np.log10(np.maximum(target_power, 1e-12)), label="target")
        axes[0].set_xlim(0, min(args.sample_rate / 2, 5000))
        axes[0].set_title("Mean Spectrum")
        axes[0].set_xlabel("Hz")
        axes[0].legend()
        axes[1].plot(recon_mel, label="recon")
        axes[1].plot(target_mel, label="target")
        axes[1].set_title("Mean Log-Mel")
        axes[1].set_xlabel("Mel bin")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(spectral_plot_path, dpi=160)
        plt.close(fig)

    manifest_path = output_dir / "listening_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else ["index"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "eval_json": str(eval_json),
        "predictions_path": str(predictions_path),
        "num_samples": len(rows),
        "target_kind": payload.get("target_kind"),
        "reconstruction_mode": payload.get("reconstruction_mode"),
        **_top1_collapse(rows),
        **_mean_stats(recon_stats, "recon"),
        **_mean_stats(target_stats, "target"),
        **_mean_stats(target_scale_stats, "target_scale_oracle"),
        **_mean_stats(target_latent_stats, "target_latent_oracle"),
        "listening_manifest_path": str(manifest_path),
        "spectral_mel_summary_path": str(spectral_plot_path) if spectral_plot_path.exists() else None,
        "mel_summary_path": str(output_dir / "mel_summary.npz"),
    }
    write_json(output_dir / "audio_qc.json", summary)
    print(f"Saved audio QC to {output_dir / 'audio_qc.json'}")
    print(f"Saved listening manifest to {manifest_path}")


if __name__ == "__main__":
    main()

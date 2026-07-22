#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "openvoice_0722_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "openvoice_0722_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import sys
APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path: sys.path.insert(0, str(APP))
from src.open_vocab_0722.metrics import log_mel, rms_envelope  # noqa: E402
from src.open_vocab_0722.audio_io import read_wav  # noqa: E402


def unit_rms(value: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(np.asarray(value, dtype=np.float64) ** 2) + 1e-12)
    return np.asarray(value / rms if rms > 1e-8 else value, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot waveform/envelope/log-mel OpenVoice reconstruction controls")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve(); root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest["samples"] if args.limit < 0 else manifest["samples"][: args.limit]
    output = root / "comparison_pairs"; output.mkdir(exist_ok=True)
    modes = [mode for mode in manifest["modes"] if mode != "reference"]
    for index, sample in enumerate(tqdm(samples, desc="[0722 plots]", unit="figure")):
        reference, rate = read_wav(root / sample["files"]["reference"])
        reference = unit_rms(reference); time = np.arange(len(reference)) / rate
        ref_envelope = rms_envelope(reference, rate); envelope_time = np.arange(len(ref_envelope)) * 0.010
        ref_mel = log_mel(reference, rate)
        figure, axes = plt.subplots(len(modes), 3, figsize=(18, max(16, len(modes) * 2.2)), constrained_layout=True)
        for row, mode in enumerate(modes):
            candidate, candidate_rate = read_wav(root / sample["files"][mode])
            if candidate_rate != rate: raise ValueError("Comparison WAV sample rates differ")
            candidate = unit_rms(candidate); metrics = sample["mode_metrics"][mode]
            axes[row, 0].plot(time, reference, color="0.65", linewidth=0.6, label="reference")
            axes[row, 0].plot(time, candidate[: len(time)], color="#2864dc", linewidth=0.6, alpha=0.85, label=mode)
            axes[row, 0].set_title(f"{mode}: env={metrics['lag_envelope_correlation']:.3f}, mel={metrics['log_mel_mae_db']:.2f} dB")
            candidate_envelope = rms_envelope(candidate, rate)
            frames = min(len(ref_envelope), len(candidate_envelope))
            axes[row, 1].plot(envelope_time[:frames], ref_envelope[:frames], color="0.45", label="reference")
            axes[row, 1].plot(envelope_time[:frames], candidate_envelope[:frames], color="#e4572e", label=mode)
            axes[row, 1].set_title("25-ms RMS envelope")
            mel = log_mel(candidate, rate)
            frames = min(ref_mel.shape[1], mel.shape[1])
            paired_mel = np.concatenate((ref_mel[:, :frames], mel[:, :frames]), axis=0)
            axes[row, 2].imshow(paired_mel, origin="lower", aspect="auto", cmap="magma", extent=(0, len(candidate) / rate, 0, 160))
            axes[row, 2].axhline(80, color="white", linewidth=0.8)
            axes[row, 2].set_yticks([40, 120], ["reference", mode])
            axes[row, 2].set_title("reference / reconstruction log-mel")
            for column in range(3): axes[row, column].grid(alpha=0.12)
        axes[0, 0].legend(loc="upper right", fontsize=7); axes[0, 1].legend(loc="upper right", fontsize=7)
        figure.suptitle(f"OpenVoice-EEG 0722 | {sample['sample_key']} | label={sample['label']}\nmain inference is label/dataset/subject-free", fontsize=14)
        figure.savefig(output / f"{index:04d}_{sample['sample_key'].replace(':', '_')}.png", dpi=130)
        plt.close(figure)
    print(json.dumps({"manifest": str(manifest_path), "plots": len(samples), "output": str(output)}, indent=2))


if __name__ == "__main__": main()

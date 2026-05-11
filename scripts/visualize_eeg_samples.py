#!/usr/bin/env python3
"""Generate EEG visualization images for all local sample files.

Outputs one PNG per dataset into its probe_artifacts/ directory.
Supports .npz (derived), .vhdr (BrainVision), and .set (EEGLAB) formats.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SAMPLES_ROOT = Path("data/voice_eeg_dataset_samples")
MAX_CHANNELS = 20
DURATION_SEC = 20.0


def robust_scale(ch: np.ndarray) -> np.ndarray:
    ch = ch - np.nanmedian(ch)
    scale = np.nanpercentile(np.abs(ch), 95)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanstd(ch) or 1.0
    return ch / scale


def plot_eeg(data: np.ndarray, sfreq: float, ch_names: list[str], title: str, out: Path) -> None:
    n_ch = min(MAX_CHANNELS, data.shape[0])
    n_samp = min(data.shape[1], int(DURATION_SEC * sfreq))
    data = data[:n_ch, :n_samp].astype("float64")
    times = np.arange(n_samp) / sfreq

    fig, ax = plt.subplots(figsize=(14, max(5.0, 0.36 * n_ch + 1.8)), constrained_layout=True)
    for i, ch in enumerate(data):
        ax.plot(times, robust_scale(ch) + i * 2.0, linewidth=0.7)
    ax.set_yticks([i * 2.0 for i in range(n_ch)])
    ax.set_yticklabels(ch_names[:n_ch], fontsize=7)
    ax.set_xlabel("Time (s)")
    ax.set_title(title, fontsize=9)
    ax.grid(axis="x", alpha=0.2)
    ax.set_xlim(0, times[-1])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


def from_npz(path: Path, out: Path) -> None:
    z = np.load(path, allow_pickle=True)
    eeg = z["eeg"]
    if eeg.ndim == 3:
        eeg = eeg[0]
    sfreq = float(z["sfreq"])
    ch_names = [str(x) for x in z["ch_names"]]
    plot_eeg(eeg, sfreq, ch_names, f"{path.name} | {z['eeg_kind']} | {sfreq:g} Hz", out)


def from_vhdr(path: Path, out: Path) -> None:
    import mne
    raw = mne.io.read_raw_brainvision(path, preload=True, verbose=False)
    data, _ = raw[:MAX_CHANNELS, : int(DURATION_SEC * raw.info["sfreq"])]
    plot_eeg(data, raw.info["sfreq"], raw.ch_names[:MAX_CHANNELS],
             f"{path.name} | BrainVision | {raw.info['sfreq']:g} Hz", out)


def from_set(path: Path, out: Path) -> None:
    import mne
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    data, _ = raw[:MAX_CHANNELS, : int(DURATION_SEC * raw.info["sfreq"])]
    plot_eeg(data, raw.info["sfreq"], raw.ch_names[:MAX_CHANNELS],
             f"{path.name} | EEGLAB | {raw.info['sfreq']:g} Hz", out)


HANDLERS = {".npz": from_npz, ".vhdr": from_vhdr, ".set": from_set}


def process_dataset(dataset_dir: Path) -> None:
    out_dir = dataset_dir / "probe_artifacts"
    # pick first file of each supported format
    seen_formats: set[str] = set()
    candidates = sorted(dataset_dir.rglob("*"))
    for f in candidates:
        if f.suffix not in HANDLERS or f.suffix in seen_formats:
            continue
        if f.parent.name == "probe_artifacts":
            continue
        out = out_dir / f"eeg_preview{f.suffix}.png"
        if out.exists():
            print(f"  skip (exists): {out.name}")
            seen_formats.add(f.suffix)
            continue
        try:
            HANDLERS[f.suffix](f, out)
            seen_formats.add(f.suffix)
        except Exception as e:
            print(f"  ERROR {f.name}: {e}", file=sys.stderr)


def main() -> None:
    for dataset_dir in sorted(SAMPLES_ROOT.glob("*/*")):
        if not dataset_dir.is_dir():
            continue
        has_local = any(
            f.suffix in HANDLERS
            for f in dataset_dir.rglob("*")
            if f.parent.name != "probe_artifacts"
        )
        if not has_local:
            continue
        print(dataset_dir.relative_to(SAMPLES_ROOT))
        process_dataset(dataset_dir)


if __name__ == "__main__":
    main()

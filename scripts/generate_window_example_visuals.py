#!/usr/bin/env python3
"""Generate explicit trial-window example visuals for FEIS and KaraOne."""

from __future__ import annotations

import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from eeg_dataset_analysis_lib import FEIS_ROOT, KARAONE_ROOT, extract_karaone_subject, load_karaone_epoch_inds


OUT_DIR = Path("outputs/window_examples")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def savefig(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def load_feis_full_df(subject: str = "01") -> pd.DataFrame:
    zip_path = FEIS_ROOT / "scottwellington-FEIS-7e726fd" / "experiments" / subject / "full_eeg.zip"
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("full_eeg.csv") as handle:
            return pd.read_csv(handle)


def make_feis_window_example() -> Path:
    df = load_feis_full_df("01")
    trial_df = df[df["Epoch"].astype(int) == 0].copy()
    label = str(trial_df["Label"].iloc[0])
    channel_names = ["F3", "FC5", "T7", "O1"]

    stage_bounds = []
    grouped = trial_df.groupby("Stage", sort=False)
    for stage, part in grouped:
        stage_bounds.append((stage, float(part["Time:256Hz"].iloc[0]), float(part["Time:256Hz"].iloc[-1])))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, height_ratios=[1, 3])

    stage_colors = {
        "stimuli": "#f6bd60",
        "articulators": "#f28482",
        "thinking": "#84a59d",
        "speaking": "#4d908e",
        "resting": "#577590",
    }
    for stage, start, end in stage_bounds:
        axes[0].axvspan(start, end, color=stage_colors.get(stage, "#cccccc"), alpha=0.8)
        axes[0].text((start + end) / 2, 0.5, stage, ha="center", va="center", fontsize=11, color="black")
        axes[1].axvspan(start, end, color=stage_colors.get(stage, "#cccccc"), alpha=0.08)
    axes[0].set_ylim(0, 1)
    axes[0].set_yticks([])
    axes[0].set_title(f"FEIS example trial: label={label}, epoch=0, subject=01")

    offsets = np.arange(len(channel_names)) * 160.0
    t = trial_df["Time:256Hz"].to_numpy()
    for idx, channel in enumerate(channel_names):
        y = trial_df[channel].to_numpy()
        y = y - np.nanmean(y)
        axes[1].plot(t, y + offsets[idx], label=channel, linewidth=0.9)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Offset EEG amplitude")
    axes[1].legend(loc="upper right", ncol=4)
    axes[1].text(
        0.01,
        0.98,
        "Already segmented in data:\nstimuli -> articulators -> thinking -> speaking -> resting",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    out = OUT_DIR / "feis_window_example.png"
    savefig(out)
    return out


def make_karaone_window_example(subject: str = "MM05") -> Path:
    subject_dir = extract_karaone_subject(subject, root=KARAONE_ROOT)
    raw = mne.io.read_raw_cnt(str(subject_dir / "Acquisition 232 Data.cnt"), preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    labels = (subject_dir / "kinect_data" / "labels.txt").read_text(errors="replace").splitlines()
    short_labels = (subject_dir / "kinect_data" / f"{subject}_p.txt").read_text(errors="replace").splitlines()
    epoch_inds = load_karaone_epoch_inds(subject_dir / "epoch_inds.mat")

    clearing = epoch_inds["clearing_inds"][0]
    thinking = epoch_inds["thinking_inds"][0]
    stimulus_like = epoch_inds["speaking_inds"][0]
    overt_like = epoch_inds["speaking_inds"][1]

    tmin = max(0.0, clearing[0] / sfreq - 1.0)
    tmax = min(raw.n_times / sfreq, overt_like[1] / sfreq + 1.0)
    picks = ["FP1", "FZ", "FC5", "C3", "P7", "O1"]
    segment = raw.copy().pick(picks).crop(tmin=tmin, tmax=tmax).load_data()
    times = segment.times + tmin
    data = segment.get_data() * 1e6

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, height_ratios=[1, 3])
    spans = [
        ("clearing", clearing, "#577590"),
        ("stimulus-like", stimulus_like, "#f6bd60"),
        ("thinking_inds", thinking, "#84a59d"),
        ("overt-like", overt_like, "#f28482"),
    ]
    for label_name, pair, color in spans:
        start = pair[0] / sfreq
        end = pair[1] / sfreq
        axes[0].axvspan(start, end, color=color, alpha=0.85)
        axes[0].text((start + end) / 2, 0.5, label_name, ha="center", va="center", fontsize=10)
        axes[1].axvspan(start, end, color=color, alpha=0.08)
    axes[0].set_ylim(0, 1)
    axes[0].set_yticks([])
    axes[0].set_title(f"KaraOne example trial: label={labels[0]} ({short_labels[0]}), subject={subject}")

    offsets = np.arange(len(picks)) * 120.0
    for idx, channel in enumerate(picks):
        y = data[idx] - np.nanmean(data[idx])
        axes[1].plot(times, y + offsets[idx], label=channel, linewidth=0.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("EEG amplitude (uV, offset)")
    axes[1].legend(loc="upper right", ncol=3)
    axes[1].text(
        0.01,
        0.98,
        "Key point:\nlabels.txt + epoch_inds.mat define the imagined window.\nTrigger channel is not the reliable alignment source here.",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    out = OUT_DIR / "karaone_window_example.png"
    savefig(out)
    return out


def write_readme(feis_path: Path, karaone_path: Path) -> Path:
    text = f"""# Window Example Visuals

## FEIS

- 图文件：`{feis_path.as_posix()}`
- 展示内容：subject `01`、epoch `0` 的完整 trial。
- 上半部分是阶段时间轴：`stimuli -> articulators -> thinking -> speaking -> resting`
- 下半部分是同一个 trial 的 EEG 示例波形。

## KaraOne

- 图文件：`{karaone_path.as_posix()}`
- 展示内容：subject `MM05` 第一个 label 的 trial 示例。
- 上半部分是 `epoch_inds.mat` 提供的关键区间：`clearing / stimulus-like / thinking_inds / overt-like`
- 下半部分是连续 EEG 中对应片段。

这两张图专门用于说明：我们确实可以从数据里拿到“想象/确认窗口”以及这个窗口对应的 EEG。
"""
    readme = OUT_DIR / "README.md"
    readme.write_text(text, encoding="utf-8")
    return readme


def main() -> None:
    ensure_dir(OUT_DIR)
    feis_path = make_feis_window_example()
    karaone_path = make_karaone_window_example()
    readme = write_readme(feis_path, karaone_path)
    print(feis_path.as_posix())
    print(karaone_path.as_posix())
    print(readme.as_posix())


if __name__ == "__main__":
    main()

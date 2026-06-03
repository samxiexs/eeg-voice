#!/usr/bin/env python3
"""Utilities for FEIS and KaraOne analysis artifacts."""

from __future__ import annotations

import io
import json
import os
import tarfile
import wave
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEIS_ROOT = Path("/Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/data/FEIS")
KARAONE_ROOT = Path("/Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/data/KaraOne")

ASSET_ROOT = PROJECT_ROOT / "outputs" / "dataset_analysis_assets"
FEIS_ASSET_DIR = ASSET_ROOT / "feis"
KARAONE_ASSET_DIR = ASSET_ROOT / "karaone"


@dataclass
class ReportBundle:
    dataset_name: str
    summary: dict
    assets: dict[str, str]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def relpath_str(path: Path) -> str:
    return path.as_posix()


def relative_to(base: Path, target: Path) -> str:
    return target.resolve().relative_to(base.resolve()).as_posix() if target.resolve().is_relative_to(base.resolve()) else target.as_posix()


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def savefig(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def list_feis_subject_dirs(root: Path = FEIS_ROOT) -> list[Path]:
    exp_root = root / "scottwellington-FEIS-7e726fd" / "experiments"
    return sorted(path for path in exp_root.iterdir() if path.is_dir())


def read_feis_phase_df(subject_dir: Path, phase: str) -> pd.DataFrame:
    zip_path = subject_dir / f"{phase}.zip"
    member_name = f"{phase}.csv"
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member_name) as handle:
            return pd.read_csv(handle)


def feis_subject_overview(subject_dir: Path) -> dict:
    phase_names = ["stimuli", "thinking", "speaking", "resting", "articulators", "full_eeg"]
    phase_presence = {phase: (subject_dir / f"{phase}.zip").exists() for phase in phase_names}
    thinking = read_feis_phase_df(subject_dir, "thinking")
    epochs = thinking["Epoch"].astype(int)
    label_col = "Label" if "Label" in thinking.columns else ("Event Id" if "Event Id" in thinking.columns else None)
    if label_col is not None:
        labels = thinking.groupby("Epoch")[label_col].first()
        label_values = labels.tolist()
        label_count = int(pd.Series(label_values).nunique(dropna=True))
        label_distribution = Counter(label_values)
    else:
        labels = thinking.groupby("Epoch").size()
        label_values = []
        label_count = 0
        label_distribution = Counter()
    overview = {
        "subject": subject_dir.name,
        "trial_count": int(labels.shape[0]),
        "label_count": label_count,
        "labels": label_values,
        "label_distribution": label_distribution,
        "phase_presence": phase_presence,
        "samples_per_trial_thinking": int(thinking.groupby("Epoch").size().iloc[0]),
        "thinking_duration_sec": float(thinking.groupby("Epoch").size().iloc[0] / 256.0),
        "channel_columns": thinking.columns[2:16].tolist(),
        "label_column": label_col,
    }
    if phase_presence["full_eeg"]:
        full_df = read_feis_phase_df(subject_dir, "full_eeg")
        stage_counts = full_df.groupby("Stage").size().to_dict()
        stage_transitions = (
            full_df.loc[full_df["Stage"].ne(full_df["Stage"].shift(1)), ["Time:256Hz", "Epoch", "Label", "Stage"]]
            .reset_index(drop=True)
            .to_dict(orient="records")
        )
        overview["full_stage_counts"] = {str(k): int(v) for k, v in stage_counts.items()}
        overview["transition_count"] = len(stage_transitions)
        overview["transition_preview"] = stage_transitions[:10]
    return overview


def analyze_feis(root: Path = FEIS_ROOT, asset_dir: Path = FEIS_ASSET_DIR) -> ReportBundle:
    ensure_dir(asset_dir)
    subject_dirs = list_feis_subject_dirs(root)
    subject_summaries = [feis_subject_overview(path) for path in subject_dirs]
    numbered_subjects = [item for item in subject_summaries if item["subject"].isdigit()]
    chinese_subjects = [item for item in subject_summaries if item["subject"].startswith("chinese")]
    irregular_subjects = [item["subject"] for item in subject_summaries if item["trial_count"] != 160]

    representative = next(item for item in subject_summaries if item["subject"] == "01")
    subject01_dir = next(path for path in subject_dirs if path.name == "01")
    full_df = read_feis_phase_df(subject01_dir, "full_eeg")
    channel_names = representative["channel_columns"]
    segment_df = full_df[(full_df["Time:256Hz"] >= 5.0) & (full_df["Time:256Hz"] <= 32.0)].copy()

    overview_rows = [[item["subject"], item["trial_count"], item["label_count"], "yes" if item["phase_presence"]["full_eeg"] else "no"] for item in subject_summaries]

    plt.figure(figsize=(10, 4))
    plt.bar([row[0] for row in overview_rows], [row[1] for row in overview_rows], color=["#b84a62" if row[1] != 160 else "#3a6ea5" for row in overview_rows])
    plt.xticks(rotation=75)
    plt.ylabel("Trial count")
    plt.title("FEIS trial count by subject folder")
    feis_trial_count_plot = asset_dir / "feis_trial_counts.png"
    savefig(feis_trial_count_plot)

    plot_channels = channel_names[:6]
    offsets = np.arange(len(plot_channels)) * 250.0
    plt.figure(figsize=(12, 5))
    stage_colors = {
        "stimuli": "#f6bd60",
        "articulators": "#f28482",
        "thinking": "#84a59d",
        "speaking": "#4d908e",
        "resting": "#577590",
    }
    transitions = representative["transition_preview"] + []
    full_transitions = (
        full_df.loc[full_df["Stage"].ne(full_df["Stage"].shift(1)), ["Time:256Hz", "Stage"]].reset_index(drop=True)
    )
    times = segment_df["Time:256Hz"].to_numpy()
    for idx, channel in enumerate(plot_channels):
        values = segment_df[channel].to_numpy()
        plt.plot(times, values - np.nanmean(values) + offsets[idx], linewidth=0.8, label=channel)
    for i in range(len(full_transitions) - 1):
        start = float(full_transitions.iloc[i]["Time:256Hz"])
        end = float(full_transitions.iloc[i + 1]["Time:256Hz"])
        if end < 5.0 or start > 32.0:
            continue
        stage = str(full_transitions.iloc[i]["Stage"])
        plt.axvspan(max(start, 5.0), min(end, 32.0), color=stage_colors.get(stage, "#cccccc"), alpha=0.08)
    plt.xlabel("Time (s)")
    plt.ylabel("Offset amplitude")
    plt.title("FEIS subject 01 full EEG excerpt with stage shading")
    plt.legend(loc="upper right", ncol=3, fontsize=8)
    feis_waveform_plot = asset_dir / "feis_subject01_waveform.png"
    savefig(feis_waveform_plot)

    label_counts = Counter(representative["labels"])
    plt.figure(figsize=(10, 4))
    plt.bar(label_counts.keys(), label_counts.values(), color="#84a59d")
    plt.xticks(rotation=45)
    plt.ylabel("Trial count")
    plt.title("FEIS subject 01 label distribution")
    feis_label_plot = asset_dir / "feis_subject01_labels.png"
    savefig(feis_label_plot)

    channel_std = full_df[channel_names].std().sort_values()
    plt.figure(figsize=(8, 4))
    plt.bar(channel_std.index, channel_std.values, color="#577590")
    plt.xticks(rotation=45)
    plt.ylabel("Standard deviation")
    plt.title("FEIS subject 01 channel variability")
    feis_channel_plot = asset_dir / "feis_subject01_channel_std.png"
    savefig(feis_channel_plot)

    summary = {
        "dataset_name": "FEIS",
        "root": relpath_str(root),
        "subject_folder_count": len(subject_summaries),
        "numbered_subject_count": len(numbered_subjects),
        "chinese_subject_count": len(chinese_subjects),
        "trial_count_distribution": {item["subject"]: item["trial_count"] for item in subject_summaries},
        "irregular_subjects": irregular_subjects,
        "representative_subject": representative["subject"],
        "representative_trial_count": representative["trial_count"],
        "representative_label_count": representative["label_count"],
        "representative_stage_counts": representative.get("full_stage_counts", {}),
        "samples_per_trial": representative["samples_per_trial_thinking"],
        "trial_duration_sec_per_phase": representative["thinking_duration_sec"],
        "channel_names": channel_names,
        "stage_order_in_full_eeg": ["stimuli", "articulators", "thinking", "speaking", "resting"],
        "overview_rows": overview_rows,
        "data_quality_notes": [
            "每个英文被试大多有 160 个 trial，但 subject 12 只有 112 个 trial，是当前下载包里最明显的不规则个体。",
            "full_eeg.csv 的 Stage / Epoch / Label 列为每个时间点直接给出标签，对齐非常方便。",
            "两个 chinese supplementary 文件夹使用了另一套列名（Channel 1-14, Event Id, Event Date, Event Duration），与英文主集 schema 不完全一致。",
            "当前发布的是派生 CSV，而不是带 trigger 的原始 EEG 流；重新做更细粒度事件切分的自由度有限。",
        ],
        "research_fit": {
            "phoneme_classification": "Yes - strong fit",
            "word_classification": "Weak - labels are mostly phoneme/syllable-level rather than a word vocabulary",
            "speech_decoding": "Moderate for imagined-vs-spoken/phoneme-level decoding, weak for rich linguistic decoding",
            "speech_reconstruction": "Weak foundation only",
        },
    }
    assets = {
        "trial_counts": relpath_str(feis_trial_count_plot),
        "waveform": relpath_str(feis_waveform_plot),
        "labels": relpath_str(feis_label_plot),
        "channel_std": relpath_str(feis_channel_plot),
    }
    return ReportBundle("FEIS", summary, assets)


def extract_karaone_subject(subject: str = "MM05", root: Path = KARAONE_ROOT, extract_root: Path | None = None) -> Path:
    if extract_root is None:
        extract_root = Path("/tmp") / f"karaone_{subject}_extract"
    subject_dir = extract_root / "p" / "spoclab" / "users" / "szhao" / "EEG" / "data" / subject
    required = [
        subject_dir / "Acquisition 232 Data.cnt",
        subject_dir / "epoch_inds.mat",
        subject_dir / "kinect_data" / "labels.txt",
    ]
    if all(path.exists() for path in required):
        return subject_dir
    ensure_dir(extract_root)
    archive = root / f"{subject}.tar.bz2"
    with tarfile.open(archive, "r:bz2") as tf:
        tf.extractall(path=extract_root)
    return subject_dir


def load_karaone_epoch_inds(path: Path) -> dict[str, np.ndarray]:
    mat = loadmat(path)
    out = {}
    for key in ["clearing_inds", "thinking_inds", "speaking_inds"]:
        arr = mat[key].ravel()
        out[key] = np.array([[int(item[0, 0]), int(item[0, 1])] for item in arr], dtype=int)
    return out


def audio_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def analyze_karaone(root: Path = KARAONE_ROOT, asset_dir: Path = KARAONE_ASSET_DIR, representative_subject: str = "MM05") -> ReportBundle:
    ensure_dir(asset_dir)
    archives = sorted(root.glob("*.tar.bz2"))
    archive_rows = [[path.stem.replace(".tar", ""), round(path.stat().st_size / (1024 ** 3), 2)] for path in archives]
    subject_dir = extract_karaone_subject(representative_subject, root=root)
    labels = (subject_dir / "kinect_data" / "labels.txt").read_text(errors="replace").splitlines()
    prompt_text = (subject_dir / "kinect_data" / f"{representative_subject}.txt").read_text(errors="replace").splitlines()
    prompt_short = (subject_dir / "kinect_data" / f"{representative_subject}_p.txt").read_text(errors="replace").splitlines()
    epoch_inds = load_karaone_epoch_inds(subject_dir / "epoch_inds.mat")

    raw = mne.io.read_raw_cnt(str(subject_dir / "Acquisition 232 Data.cnt"), preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    thinking = epoch_inds["thinking_inds"]
    clearing = epoch_inds["clearing_inds"]
    speaking = epoch_inds["speaking_inds"]
    stimulus_like = speaking[0::2]
    overt_like = speaking[1::2]

    first_window = (
        max(0.0, clearing[0, 0] / sfreq - 1.0),
        min(raw.n_times / sfreq, overt_like[0, 1] / sfreq + 1.0),
    )
    plot_channels = raw.ch_names[:8]
    segment = raw.copy().pick(plot_channels).crop(tmin=first_window[0], tmax=first_window[1]).load_data()
    data = segment.get_data()
    times = segment.times + first_window[0]
    plt.figure(figsize=(12, 5))
    offsets = np.arange(len(plot_channels)) * 80.0
    for idx, channel in enumerate(plot_channels):
        plt.plot(times, data[idx] * 1e6 + offsets[idx], linewidth=0.8, label=channel)
    spans = [
        ("clearing", clearing[0], "#577590"),
        ("stimulus_like", stimulus_like[0], "#f6bd60"),
        ("thinking", thinking[0], "#84a59d"),
        ("overt_like", overt_like[0], "#f28482"),
    ]
    for label, pair, color in spans:
        plt.axvspan(pair[0] / sfreq, pair[1] / sfreq, color=color, alpha=0.10, label=label)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV, offset)")
    plt.title(f"KaraOne {representative_subject} first trial excerpt")
    handles, labels_ = plt.gca().get_legend_handles_labels()
    dedup = dict(zip(labels_, handles))
    plt.legend(dedup.values(), dedup.keys(), ncol=4, fontsize=8, loc="upper right")
    karaone_waveform_plot = asset_dir / "karaone_mm05_waveform.png"
    savefig(karaone_waveform_plot)

    label_counts = Counter(labels)
    plt.figure(figsize=(10, 4))
    plt.bar(label_counts.keys(), label_counts.values(), color="#b84a62")
    plt.xticks(rotation=45)
    plt.ylabel("Trial count")
    plt.title(f"KaraOne {representative_subject} label distribution")
    karaone_label_plot = asset_dir / "karaone_mm05_labels.png"
    savefig(karaone_label_plot)

    plt.figure(figsize=(9, 4))
    plt.bar([row[0] for row in archive_rows], [row[1] for row in archive_rows], color="#3a6ea5")
    plt.xticks(rotation=45)
    plt.ylabel("Archive size (GB)")
    plt.title("KaraOne downloaded archive sizes")
    karaone_archive_plot = asset_dir / "karaone_archive_sizes.png"
    savefig(karaone_archive_plot)

    wav_paths = sorted((subject_dir / "kinect_data").glob("*.wav"))
    wav_durations = [audio_duration_seconds(path) for path in wav_paths]
    plt.figure(figsize=(8, 4))
    plt.hist(wav_durations, bins=20, color="#84a59d", edgecolor="black")
    plt.xlabel("Duration (s)")
    plt.ylabel("Count")
    plt.title(f"KaraOne {representative_subject} overt audio duration histogram")
    karaone_audio_plot = asset_dir / "karaone_mm05_audio_durations.png"
    savefig(karaone_audio_plot)

    trigger_data = raw.get_data(picks=["Trigger"])[0]
    unique_trigger = np.unique(trigger_data)

    summary = {
        "dataset_name": "KaraOne",
        "root": relpath_str(root),
        "downloaded_archive_count": len(archives),
        "archive_rows": archive_rows,
        "representative_subject": representative_subject,
        "representative_trial_count": len(labels),
        "representative_unique_labels": sorted(label_counts.keys()),
        "representative_label_counter": dict(label_counts),
        "raw_channel_count_cnt": int(raw.info["nchan"]),
        "raw_duration_sec": float(raw.n_times / sfreq),
        "sampling_rate_hz": sfreq,
        "thinking_interval_count": int(thinking.shape[0]),
        "clearing_interval_count": int(clearing.shape[0]),
        "speaking_interval_count": int(speaking.shape[0]),
        "stimulus_like_interval_count": int(stimulus_like.shape[0]),
        "overt_like_interval_count": int(overt_like.shape[0]),
        "thinking_duration_mean_sec": float(np.mean((thinking[:, 1] - thinking[:, 0] + 1) / sfreq)),
        "clearing_duration_mean_sec": float(np.mean((clearing[:, 1] - clearing[:, 0] + 1) / sfreq)),
        "stimulus_like_duration_mean_sec": float(np.mean((stimulus_like[:, 1] - stimulus_like[:, 0] + 1) / sfreq)),
        "overt_like_duration_mean_sec": float(np.mean((overt_like[:, 1] - overt_like[:, 0] + 1) / sfreq)),
        "audio_file_count": len(wav_paths),
        "audio_duration_mean_sec": float(np.mean(wav_durations)),
        "audio_duration_range_sec": [float(np.min(wav_durations)), float(np.max(wav_durations))],
        "prompt_text_preview": prompt_text[:10],
        "prompt_short_preview": prompt_short[:10],
        "trigger_unique_values": [float(value) for value in unique_trigger[:10]],
        "data_quality_notes": [
            "epoch_inds.mat 是真正可用于 trial 对齐的关键信息源；MNE 读出的 Trigger 通道基本为常数，不能直接拿来做事件恢复。",
            "单受试者同时具备原始 EEG、预处理特征、overt audio、face animation units，模态丰富度明显强于 FEIS。",
            "speaking_inds 数量是 330，而 labels 只有 165；数据自身显示每个 trial 对应两个 speaking-like 片段，较短者接近刺激呈现，较长者更像 overt speaking。",
        ],
        "research_fit": {
            "phoneme_classification": "Yes - strong fit at phonological/syllabic prompt level",
            "word_classification": "Moderate - includes four words but vocabulary is small",
            "speech_decoding": "Yes - better than FEIS because EEG/audio/face are aligned per trial",
            "speech_reconstruction": "Moderate foundation only",
        },
    }
    assets = {
        "archive_sizes": relpath_str(karaone_archive_plot),
        "waveform": relpath_str(karaone_waveform_plot),
        "labels": relpath_str(karaone_label_plot),
        "audio_durations": relpath_str(karaone_audio_plot),
    }
    return ReportBundle("KaraOne", summary, assets)


def write_report_md(bundle: ReportBundle, path: Path) -> None:
    ensure_dir(path.parent)
    s = bundle.summary
    asset_refs = {
        key: os.path.relpath(str(Path(value).resolve()), start=str(path.parent.resolve()))
        if Path(value).is_absolute()
        else Path(value).as_posix()
        for key, value in bundle.assets.items()
    }
    if bundle.dataset_name == "FEIS":
        overview_table = markdown_table(
            ["Subject folder", "Trial count", "Unique labels", "Has full_eeg"],
            s["overview_rows"],
        )
        report = f"""# FEIS 数据分析报告

## 数据集概览

- 数据路径：`{s['root']}`
- 受试者文件夹数：`{s['subject_folder_count']}`，其中编号英文被试 `21` 个，中文补充被试 `2` 个。
- 代表性被试：`{s['representative_subject']}`
- 代表性被试 trial 数：`{s['representative_trial_count']}`
- 每个阶段单 trial 时长：`{s['trial_duration_sec_per_phase']:.1f}` 秒
- 通道：`{', '.join(s['channel_names'])}`

{overview_table}

![FEIS trial counts]({asset_refs['trial_counts']})

## 实验范式总结

直接从 `full_eeg.csv` 的 `Stage` 列可以恢复出稳定的 trial 流程：

1. `stimuli`：5 秒
2. `articulators`：1 秒
3. `thinking`：5 秒
4. `speaking`：5 秒
5. `resting`：5 秒

对代表性被试 `01` 来说，`full_eeg.csv` 中各阶段样本数为：

{markdown_table(["Stage", "Sample count"], [[k, v] for k, v in s["representative_stage_counts"].items()])}

这说明 FEIS 当前下载版本并不是“原始 EEG + 独立事件文件”，而是已经按阶段切好、并把 `Epoch / Label / Stage` 写进每个时间点的 CSV 派生版。

## 单受试者分析结果

### 波形与通道

![FEIS waveform]({asset_refs['waveform']})

![FEIS channel std]({asset_refs['channel_std']})

可以直接看到 `subject 01` 的 `full_eeg.csv` 在 5-32 秒区间内按阶段整齐切换，14 个通道都能连续观测到波形变化。

### Trial 与标签

![FEIS labels]({asset_refs['labels']})

- `subject 01` 共 `160` 个 trial
- `16` 个标签，每个标签 `10` 次
- 每个阶段文件中每个 epoch 长度固定为 `1280` 个采样点，即 `5.0` 秒

## 数据质量观察

"""
        for note in s["data_quality_notes"]:
            report += f"- {note}\n"
        report += f"""

## 与研究目标的匹配度分析

{markdown_table(
    ["研究任务", "判断"],
    [
        ["EEG → Phoneme Classification", s["research_fit"]["phoneme_classification"]],
        ["EEG → Word Classification", s["research_fit"]["word_classification"]],
        ["EEG → Speech Decoding", s["research_fit"]["speech_decoding"]],
        ["EEG → Speech Reconstruction", s["research_fit"]["speech_reconstruction"]],
    ],
)}

### 结论

- **适合做 EEG → phoneme / articulatory-class classification**：标签直接写在 CSV 里，trial 切分非常干净。
- **不太适合做 word classification**：语料核心是音素/音节单位，不是词汇系统。
- **可做 imagined vs spoken decoding**：阶段边界非常明确。
- **不适合作为语音重建主数据集**：只有 14 通道、公开音频更像提示/实验资产，缺少高质量 trial-synchronous overt speech ground truth。

## 下一步研究建议

1. 先在 FEIS 上建立 imagined phoneme baseline。
2. 利用 `Stage` 和 `Epoch` 做 subject-dependent / LOSO 分类基线。
3. 不把 FEIS 作为主要语音重建数据，而把它作为模型筛选与快速迭代数据集。
"""
    else:
        archive_table = markdown_table(["Archive", "Size (GB)"], s["archive_rows"])
        label_rows = [[label, count] for label, count in sorted(s["representative_label_counter"].items())]
        report = f"""# KaraOne 数据分析报告

## 数据集概览

- 数据路径：`{s['root']}`
- 本地已下载受试者归档数：`{s['downloaded_archive_count']}`
- 代表性受试者：`{s['representative_subject']}`
- `cnt` 原始 EEG 通道数：`{s['raw_channel_count_cnt']}`
- 采样率：`{s['sampling_rate_hz']:.0f} Hz`
- 单受试者连续 EEG 时长：`{s['raw_duration_sec'] / 60:.2f}` 分钟
- 单受试者标签数：`{s['representative_trial_count']}`
- 单受试者 overt 音频数：`{s['audio_file_count']}`

{archive_table}

![KaraOne archive sizes]({asset_refs['archive_sizes']})

## 实验范式总结

从本地数据本身可以恢复出下面这些关键结构：

- `labels.txt`：`165` 个 trial 标签
- `epoch_inds.mat`：
  - `clearing_inds`：`{s['clearing_interval_count']}` 段，平均 `5.0` 秒左右
  - `thinking_inds`：`{s['thinking_interval_count']}` 段，平均 `4.95` 秒左右
  - `speaking_inds`：`{s['speaking_interval_count']}` 段
- `kinect_data/*.wav`：`{s['audio_file_count']}` 段 overt speech 音频

一个非常重要的直接数据观察是：`speaking_inds` 的数量是 `330`，而 trial 标签只有 `165`。把 `speaking_inds` 按奇偶拆开后可以看到：

{markdown_table(
    ["Interval family", "Count", "Mean duration (s)"],
    [
        ["clearing", s["clearing_interval_count"], f"{s['clearing_duration_mean_sec']:.3f}"],
        ["thinking", s["thinking_interval_count"], f"{s['thinking_duration_mean_sec']:.3f}"],
        ["stimulus-like (odd speaking_inds)", s["stimulus_like_interval_count"], f"{s['stimulus_like_duration_mean_sec']:.3f}"],
        ["overt-like (even speaking_inds)", s["overt_like_interval_count"], f"{s['overt_like_duration_mean_sec']:.3f}"],
    ],
)}

这说明数据自身支持这样一个强推断：每个 trial 至少包含一个较短的刺激/提示片段，以及一个更长的 overt speaking 片段，再加一个 imagined thinking 片段。

## 单受试者分析结果

### EEG 波形

![KaraOne waveform]({asset_refs['waveform']})

图中展示了 `MM05` 第一个 trial 附近的连续 EEG，并按照 `epoch_inds.mat` 中的区间对 `clearing / stimulus-like / thinking / overt-like` 做了阴影标注。

### 标签与音频

{markdown_table(["Label", "Count"], label_rows)}

![KaraOne labels]({asset_refs['labels']})

![KaraOne audio durations]({asset_refs['audio_durations']})

- `11` 个唯一标签
- 每个标签在 `MM05` 中都是 `15` 次
- 单条 overt 音频平均时长约 `1-2` 秒量级，具体均值 `"{s['audio_duration_mean_sec']:.3f}"` 秒

## 事件与标签分析

- `labels.txt` 和 `MM05_p.txt` / `MM05.txt` 能把每个 trial 对应到具体 prompt。
- `epoch_inds.mat` 能把 trial 精确对齐到连续 EEG 的样本索引。
- **原始 Trigger 通道并不可靠**：MNE 读取 `.cnt` 后 `Trigger` 基本是常数，不能直接用它恢复事件。

这意味着：

1. **每个 trial 对应什么内容？**  
   可以，`labels.txt` / `MM05.txt` 已明确给出。
2. **是否能够定位到具体刺激？**  
   可以，用 `epoch_inds.mat` 可以定位 thinking / stimulus-like / overt-like 区段。
3. **EEG 与标签是否能够准确对应？**  
   可以，但关键依赖 `epoch_inds.mat`，而不是 `Trigger` 通道。

## 数据质量观察

"""
        for note in s["data_quality_notes"]:
            report += f"- {note}\n"
        report += f"""

## 与研究目标的匹配度分析

{markdown_table(
    ["研究任务", "判断"],
    [
        ["EEG → Phoneme Classification", s["research_fit"]["phoneme_classification"]],
        ["EEG → Word Classification", s["research_fit"]["word_classification"]],
        ["EEG → Speech Decoding", s["research_fit"]["speech_decoding"]],
        ["EEG → Speech Reconstruction", s["research_fit"]["speech_reconstruction"]],
    ],
)}

### 结论

- **很适合做 EEG → phonological / syllabic prompt classification**：标签结构清楚，trial 对齐信息强。
- **可以做有限词汇的 word classification**：数据里确实有词级 prompt，但词表很小。
- **比 FEIS 更适合做 speech decoding**：因为它同时有连续 EEG、overt audio、面部动画和样本级区间索引。
- **具备进一步探索语音重建的基础，但不是终局数据集**：有 overt audio，能支持 overt EEG → acoustic representation；但词表小、受试者规模仍有限，真实自然语音重建上限不高。

## 下一步研究建议

1. 先把 `epoch_inds.mat + labels.txt + wav` 对齐成统一 trial 表。
2. 优先做 imagined / overt prompt classification 与 overt-to-imagined transfer。
3. 如要做声学监督，先从 log-mel / SSL speech embedding 开始，而不是直接做 waveform reconstruction。
"""
    path.write_text(report, encoding="utf-8")


def save_bundle_json(bundle: ReportBundle, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps({"dataset_name": bundle.dataset_name, "summary": bundle.summary, "assets": bundle.assets}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )

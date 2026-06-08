#!/usr/bin/env python3
"""Prepare stage-labeled FEIS and KaraOne EEG segments plus waveform targets.

This script now exports all reliably recoverable EEG stages/windows rather than
only `thinking`.

Design goals:
- Keep only the original archives/source tree plus final processed outputs.
- FEIS: read already segmented phase files directly.
- KaraOne: optionally extract only the minimum required files from `.tar.bz2`
  into a temporary directory and delete that directory after processing.
- Preserve explicit stage separation so downstream work can filter
  `thinking`, `speaking`, `stimuli/hearing`, `resting`, etc.

Outputs per dataset:
- `{dataset}/subjects/*.npz`
- `{dataset}/audio/**.wav`
- `{dataset}/trials.csv`
- `{dataset}/segments.csv`
- `{dataset}/manifest.json`
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba-cache"))

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat, wavfile
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt
from tqdm import tqdm


DEFAULT_FEIS_ROOT = Path("/Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/data/FEIS")
DEFAULT_KARAONE_ROOT = Path("/Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/data/KaraOne")
DEFAULT_OUTPUT_ROOT = Path("/Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/data/processed/thinking_waveform_pairs")

FEIS_INNER_DIR = "scottwellington-FEIS-7e726fd"
FEIS_EEG_SFREQ = 256
FEIS_STAGE_ORDER = ["stimuli", "articulators", "thinking", "speaking", "resting"]
FEIS_STAGE_ROLE = {
    "stimuli": "hearing",
    "articulators": "articulator_preparation",
    "thinking": "imagined_speech",
    "speaking": "overt_speech",
    "resting": "rest",
}

KARAONE_RELATIVE_SUBJECT_DIR = Path("p/spoclab/users/szhao/EEG/data")
KARAONE_ALT_RELATIVE_SUBJECT_DIR = Path("p 2/spoclab/users/szhao/EEG/data")
KARAONE_DROP_CHANNELS = {"M1", "M2", "VEO", "HEO", "EKG", "EMG", "Trigger"}
KARAONE_STAGE_ORDER = ["clearing", "stimulus_like", "thinking", "overt_like"]
KARAONE_STAGE_ROLE = {
    "clearing": "baseline_reset",
    "stimulus_like": "hearing_or_prompt_processing",
    "thinking": "imagined_speech",
    "overt_like": "overt_speech",
}

FEIS_LABEL_FALLBACK_AUDIO_DIR = Path("B059691/decoded wavs/Original")
FEIS_AUDIO_SOURCE_SUBJECT_OVERRIDES = {
    "05": "04",
}
FEIS_SUBJECT_KNOWN_EVAL = "Protocol S: within-subject evaluation with subject-known template bank."
FEIS_SUBJECT_UNKNOWN_EVAL = "Protocol U: unseen-subject evaluation for disentangling subject identity from speech-relevant EEG information."


@dataclass
class DatasetStats:
    dataset: str
    subject_count: int
    trial_count: int
    segment_count: int
    notes: list[str]


@dataclass(frozen=True)
class FEISAudioSource:
    subject_id: str
    label: str
    path: Path
    audio_source_subject: str
    audio_source_kind: str
    is_fallback_audio: bool


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    headers = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def bandpass_filter(data: np.ndarray, sfreq: float, low_hz: float, high_hz: float) -> np.ndarray:
    sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def notch_filter(data: np.ndarray, sfreq: float, notch_hz: float, quality: float = 30.0) -> np.ndarray:
    if notch_hz <= 0 or notch_hz >= sfreq / 2.0:
        return data
    b, a = iirnotch(notch_hz, quality, fs=sfreq)
    return filtfilt(b, a, data, axis=-1)


def average_reference(data: np.ndarray) -> np.ndarray:
    return data - data.mean(axis=0, keepdims=True)


def resample_array(data: np.ndarray, src_sfreq: float, dst_sfreq: float) -> np.ndarray:
    if math.isclose(src_sfreq, dst_sfreq):
        return data
    src = int(round(src_sfreq))
    dst = int(round(dst_sfreq))
    gcd = math.gcd(src, dst)
    up = dst // gcd
    down = src // gcd
    return resample_poly(data, up=up, down=down, axis=-1)


def pad_or_crop(data: np.ndarray, target_len: int) -> np.ndarray:
    cur_len = data.shape[-1]
    if cur_len == target_len:
        return data
    if cur_len > target_len:
        start = (cur_len - target_len) // 2
        return data[..., start : start + target_len]
    pad_width = [(0, 0)] * data.ndim
    pad_width[-1] = (0, target_len - cur_len)
    return np.pad(data, pad_width, mode="constant", constant_values=0.0)


def zscore_against_baseline(epoch: np.ndarray, baseline: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = baseline.mean(axis=-1, keepdims=True)
    std = baseline.std(axis=-1, keepdims=True)
    std = np.where(std < eps, eps, std)
    return (epoch - mean) / std


def preprocess_epoch(
    epoch: np.ndarray,
    baseline: np.ndarray,
    src_sfreq: float,
    target_sfreq: int,
    target_len: int,
) -> tuple[np.ndarray, int]:
    epoch = resample_array(epoch, src_sfreq, target_sfreq)
    baseline = resample_array(baseline, src_sfreq, target_sfreq)
    valid_len = int(epoch.shape[-1])
    epoch = pad_or_crop(epoch, target_len)
    baseline = pad_or_crop(baseline, target_len)
    epoch = zscore_against_baseline(epoch, baseline).astype(np.float32)
    return epoch, min(valid_len, target_len)


def load_audio_mono(path: Path) -> tuple[int, np.ndarray]:
    sr, audio = wavfile.read(path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        scale = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32)
    return int(sr), audio


def resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32)
    gcd = math.gcd(src_sr, dst_sr)
    up = dst_sr // gcd
    down = src_sr // gcd
    return resample_poly(audio, up=up, down=down).astype(np.float32)


def write_audio(path: Path, sr: int, audio: np.ndarray) -> None:
    ensure_dir(path.parent)
    audio = np.clip(audio, -1.0, 1.0)
    int16_audio = (audio * 32767.0).astype(np.int16)
    wavfile.write(path, sr, int16_audio)


def preprocess_audio_file(src_path: Path, dst_path: Path, target_sr: int) -> None:
    if dst_path.exists():
        return
    src_sr, audio = load_audio_mono(src_path)
    audio = resample_audio(audio, src_sr, target_sr)
    write_audio(dst_path, target_sr, audio)


def feis_subject_dirs(feis_root: Path) -> list[Path]:
    exp_root = feis_root / FEIS_INNER_DIR / "experiments"
    return sorted(path for path in exp_root.iterdir() if path.is_dir() and path.name.isdigit())


def read_feis_phase_df(subject_dir: Path, phase: str) -> pd.DataFrame:
    zip_path = subject_dir / f"{phase}.zip"
    csv_path = subject_dir / f"{phase}.csv"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(f"{phase}.csv") as handle:
                return pd.read_csv(handle)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Neither {zip_path} nor {csv_path} exists")


def feis_channel_columns(df: pd.DataFrame) -> list[str]:
    non_channels = {"Time:256Hz", "Epoch", "Label", "Stage", "Flag"}
    return [col for col in df.columns if col not in non_channels]


def resolve_feis_audio_path(feis_root: Path, subject: str, label: str) -> Path:
    candidates = [
        feis_root / FEIS_INNER_DIR / "wavs" / subject / "wavs" / f"{label}.wav",
        feis_root / FEIS_INNER_DIR / "wavs" / subject / "combined_wavs" / f"{label}.wav",
        feis_root / FEIS_INNER_DIR / FEIS_LABEL_FALLBACK_AUDIO_DIR / f"{label}.wav",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve FEIS audio for subject={subject} label={label}")


def resolve_feis_audio_source(feis_root: Path, subject: str, label: str) -> FEISAudioSource:
    subject_str = str(subject).zfill(2) if str(subject).isdigit() else str(subject)
    candidates = [
        (
            feis_root / FEIS_INNER_DIR / "wavs" / subject_str / "wavs" / f"{label}.wav",
            subject_str,
            "subject_wavs",
            False,
        ),
        (
            feis_root / FEIS_INNER_DIR / "wavs" / subject_str / "combined_wavs" / f"{label}.wav",
            subject_str,
            "subject_combined_wavs",
            False,
        ),
        (
            feis_root / FEIS_INNER_DIR / FEIS_LABEL_FALLBACK_AUDIO_DIR / f"{label}.wav",
            FEIS_AUDIO_SOURCE_SUBJECT_OVERRIDES.get(subject_str, "fallback_unknown"),
            "fallback_original",
            True,
        ),
    ]
    for candidate, source_subject, source_kind, is_fallback in candidates:
        if candidate.exists():
            return FEISAudioSource(
                subject_id=subject_str,
                label=str(label),
                path=candidate,
                audio_source_subject=str(source_subject),
                audio_source_kind=str(source_kind),
                is_fallback_audio=bool(is_fallback),
            )
    raise FileNotFoundError(f"Could not resolve FEIS audio for subject={subject_str} label={label}")


def sha1_for_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def enrich_feis_rows(
    feis_root: Path,
    output_root: Path,
    trial_rows: list[dict[str, object]],
    segment_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    rows = [dict(row) for row in trial_rows]
    segs = [dict(row) for row in segment_rows]
    source_cache: dict[tuple[str, str], FEISAudioSource] = {}
    processed_audio_cache: dict[str, dict[str, object]] = {}
    label_hashes: dict[str, set[str]] = {}

    def get_audio_meta(subject_id: str, label: str, audio_path: str) -> dict[str, object]:
        key = (subject_id, label)
        if key not in source_cache:
            source_cache[key] = resolve_feis_audio_source(feis_root, subject_id, label)
        if audio_path not in processed_audio_cache:
            processed_path = output_root / "feis" / audio_path
            sr, audio = load_audio_mono(processed_path)
            processed_audio_cache[audio_path] = {
                "audio_sha1": sha1_for_file(processed_path),
                "audio_duration_sec": round(float(len(audio) / sr), 6),
            }
        source = source_cache[key]
        meta = processed_audio_cache[audio_path]
        label_hashes.setdefault(label, set()).add(str(meta["audio_sha1"]))
        return {
            "audio_source_subject": source.audio_source_subject,
            "audio_source_kind": source.audio_source_kind,
            "is_fallback_audio": source.is_fallback_audio,
            "audio_sha1": meta["audio_sha1"],
            "audio_duration_sec": meta["audio_duration_sec"],
            "unique_hashes_per_subject_label": 1,
            "subject_known_eval": FEIS_SUBJECT_KNOWN_EVAL,
            "subject_unknown_eval": FEIS_SUBJECT_UNKNOWN_EVAL,
            "is_clean_subject": not source.is_fallback_audio,
        }

    for row in rows:
        subject_id = str(row["subject_id"]).zfill(2) if str(row["subject_id"]).isdigit() else str(row["subject_id"])
        row["subject_id"] = subject_id
        row.update(get_audio_meta(subject_id, str(row["label"]), str(row["audio_path"])))

    for row in segs:
        subject_id = str(row["subject_id"]).zfill(2) if str(row["subject_id"]).isdigit() else str(row["subject_id"])
        row["subject_id"] = subject_id
        row.update(get_audio_meta(subject_id, str(row["label"]), str(row["audio_path"])))

    label_hash_counts = {label: len(hashes) for label, hashes in label_hashes.items()}
    for row in rows:
        row["unique_hashes_per_label_across_subjects"] = label_hash_counts[str(row["label"])]
    for row in segs:
        row["unique_hashes_per_label_across_subjects"] = label_hash_counts[str(row["label"])]

    clean_subjects = sorted(
        {
            str(row["subject_id"])
            for row in rows
            if not bool(row["is_fallback_audio"])
        }
    )
    anomaly_subjects = sorted(
        {
            str(row["subject_id"])
            for row in rows
            if bool(row["is_fallback_audio"])
        }
    )
    summary = {
        "clean_subject_ids": clean_subjects,
        "anomalous_subject_ids": anomaly_subjects,
        "unique_hashes_per_label_across_subjects": {
            label: int(count) for label, count in sorted(label_hash_counts.items())
        },
    }
    return rows, segs, summary


def list_karaone_subjects(karaone_root: Path) -> list[str]:
    archive_subjects = sorted(path.name.replace(".tar.bz2", "") for path in karaone_root.glob("*.tar.bz2"))
    existing_dirs = []
    for rel_base in [KARAONE_RELATIVE_SUBJECT_DIR, KARAONE_ALT_RELATIVE_SUBJECT_DIR]:
        base = karaone_root / rel_base
        if base.exists():
            existing_dirs.extend(path.name for path in base.iterdir() if path.is_dir())
    return sorted(set(archive_subjects) | set(existing_dirs))


def flatten_epoch_inds(path: Path) -> dict[str, np.ndarray]:
    mat = loadmat(path)
    out: dict[str, np.ndarray] = {}
    for key in ["clearing_inds", "thinking_inds", "speaking_inds"]:
        arr = mat[key].ravel()
        out[key] = np.asarray([[int(item[0, 0]), int(item[0, 1])] for item in arr], dtype=np.int32)
    return out


def locate_existing_karaone_subject_dir(karaone_root: Path, subject: str) -> Path | None:
    for rel_base in [KARAONE_RELATIVE_SUBJECT_DIR, KARAONE_ALT_RELATIVE_SUBJECT_DIR]:
        candidate = karaone_root / rel_base / subject
        if candidate.exists():
            return candidate
    return None


def extract_karaone_subject_to_temp(karaone_root: Path, subject: str, temp_root: Path) -> Path:
    archive_path = karaone_root / f"{subject}.tar.bz2"
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing KaraOne archive: {archive_path}")

    work_dir = Path(tempfile.mkdtemp(prefix=f"karaone_{subject}_", dir=str(temp_root)))
    subject_prefix = f"{KARAONE_RELATIVE_SUBJECT_DIR.as_posix()}/{subject}/"
    extract_members = []
    with tarfile.open(archive_path, "r:bz2") as tf:
        for member in tf.getmembers():
            name = member.name
            if not name.startswith(subject_prefix):
                continue
            if name.endswith(".cnt") or name.endswith("epoch_inds.mat"):
                extract_members.append(member)
                continue
            if "/kinect_data/" in name and (name.endswith(".wav") or name.endswith(".txt")):
                extract_members.append(member)
        tf.extractall(path=work_dir, members=extract_members)
    return work_dir / KARAONE_RELATIVE_SUBJECT_DIR / subject


def find_karaone_cnt_file(subject_dir: Path) -> Path:
    cnt_files = sorted(subject_dir.glob("*.cnt"))
    if len(cnt_files) == 1:
        return cnt_files[0]
    if len(cnt_files) > 1:
        return cnt_files[0]
    raise FileNotFoundError(f"No .cnt file found under {subject_dir}")


def preprocess_karaone_continuous(
    subject_dir: Path,
    bandpass_low: float,
    bandpass_high: float,
    notch_hz: float,
) -> tuple[np.ndarray, float, list[str]]:
    cnt_path = find_karaone_cnt_file(subject_dir)
    raw = mne.io.read_raw_cnt(str(cnt_path), preload=True, verbose="ERROR")
    keep_names = [name for name in raw.ch_names if name not in KARAONE_DROP_CHANNELS]
    data = raw.get_data(picks=keep_names).astype(np.float32)
    sfreq = float(raw.info["sfreq"])
    data = notch_filter(data, sfreq, notch_hz)
    data = bandpass_filter(data, sfreq, bandpass_low, bandpass_high)
    data = average_reference(data)
    return data.astype(np.float32), sfreq, keep_names


def feis_stage_target_len(df: pd.DataFrame, target_eeg_sfreq: int) -> int:
    samples = int(df.groupby("Epoch").size().iloc[0])
    seconds = samples / FEIS_EEG_SFREQ
    return int(round(seconds * target_eeg_sfreq))


def sanitize_stage_key(stage: str) -> str:
    return stage.replace("-", "_")


def robust_target_len_from_ranges(
    ranges: np.ndarray,
    src_sfreq: float,
    target_eeg_sfreq: int,
    percentile: float = 95.0,
) -> int:
    lengths = np.asarray(
        [int(round((int(end) - int(start) + 1) / src_sfreq * target_eeg_sfreq)) for start, end in ranges],
        dtype=np.int32,
    )
    return int(max(1, math.ceil(np.percentile(lengths, percentile))))


def preprocess_feis_subject(
    feis_root: Path,
    subject: str,
    output_root: Path,
    target_eeg_sfreq: int,
    target_audio_sfreq: int,
    bandpass_low: float,
    bandpass_high: float,
    notch_hz: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    subject_dir = feis_root / FEIS_INNER_DIR / "experiments" / subject
    phase_dfs = {stage: read_feis_phase_df(subject_dir, stage) for stage in FEIS_STAGE_ORDER}
    channels = feis_channel_columns(phase_dfs["thinking"])
    stage_target_lens = {stage: feis_stage_target_len(df, target_eeg_sfreq) for stage, df in phase_dfs.items()}

    grouped = {stage: df.groupby("Epoch", sort=True) for stage, df in phase_dfs.items()}
    epoch_ids = [int(epoch_id) for epoch_id, _ in grouped["thinking"]]
    labels = [str(grouped["thinking"].get_group(epoch_id)["Label"].iloc[0]) for epoch_id in epoch_ids]

    subject_bundle = output_root / "feis" / "subjects" / f"{subject}.npz"
    audio_root = output_root / "feis" / "audio" / subject
    audio_rel_cache: dict[str, str] = {}
    audio_relpaths: list[str] = []

    for label in labels:
        if label not in audio_rel_cache:
            src_audio = resolve_feis_audio_source(feis_root, subject, label).path
            dst_audio = audio_root / f"{label}.wav"
            preprocess_audio_file(src_audio, dst_audio, target_audio_sfreq)
            audio_rel_cache[label] = str(dst_audio.relative_to(output_root / "feis"))
        audio_relpaths.append(audio_rel_cache[label])

    trial_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    bundle_payload: dict[str, np.ndarray] = {
        "trial_indices": np.asarray(epoch_ids, dtype=np.int32),
        "labels": np.asarray(labels),
        "audio_relpaths": np.asarray(audio_relpaths),
        "channel_names": np.asarray(channels),
        "eeg_sfreq_hz": np.asarray([target_eeg_sfreq], dtype=np.int32),
        "stage_names": np.asarray(FEIS_STAGE_ORDER),
    }

    baseline_groups = grouped["resting"]
    for stage in FEIS_STAGE_ORDER:
        stage_groups = grouped[stage]
        stage_trials: list[np.ndarray] = []
        stage_valid_lengths: list[int] = []
        target_len = stage_target_lens[stage]
        stage_key = sanitize_stage_key(stage)
        for row_idx, epoch_id in enumerate(epoch_ids):
            epoch_df = stage_groups.get_group(epoch_id)
            baseline_df = baseline_groups.get_group(epoch_id)
            epoch = epoch_df[channels].to_numpy(dtype=np.float32).T
            baseline = baseline_df[channels].to_numpy(dtype=np.float32).T

            epoch = notch_filter(epoch, FEIS_EEG_SFREQ, notch_hz)
            epoch = bandpass_filter(epoch, FEIS_EEG_SFREQ, bandpass_low, bandpass_high)
            epoch = average_reference(epoch)

            baseline = notch_filter(baseline, FEIS_EEG_SFREQ, notch_hz)
            baseline = bandpass_filter(baseline, FEIS_EEG_SFREQ, bandpass_low, bandpass_high)
            baseline = average_reference(baseline)

            processed, valid_len = preprocess_epoch(
                epoch=epoch,
                baseline=baseline,
                src_sfreq=FEIS_EEG_SFREQ,
                target_sfreq=target_eeg_sfreq,
                target_len=target_len,
            )
            stage_trials.append(processed)
            stage_valid_lengths.append(valid_len)
            segment_rows.append(
                {
                    "dataset": "feis",
                    "subject_id": subject,
                    "trial_index": int(epoch_id),
                    "segment_stage": stage,
                    "segment_role": FEIS_STAGE_ROLE[stage],
                    "segment_array_key": f"stage__{stage_key}",
                    "label": labels[row_idx],
                    "audio_path": audio_relpaths[row_idx],
                    "eeg_subject_bundle": str(subject_bundle.relative_to(output_root / "feis")),
                    "baseline_window": "resting",
                    "eeg_num_channels": len(channels),
                    "eeg_num_samples": target_len,
                    "eeg_valid_num_samples": valid_len,
                    "eeg_sfreq_hz": target_eeg_sfreq,
                    "audio_sfreq_hz": target_audio_sfreq,
                    "audio_pairing": "subject-level canonical wav for same prompt label",
                }
            )
        bundle_payload[f"stage__{stage_key}"] = np.stack(stage_trials, axis=0).astype(np.float32)
        bundle_payload[f"stage__{stage_key}__valid_lengths"] = np.asarray(stage_valid_lengths, dtype=np.int32)

    for row_idx, epoch_id in enumerate(epoch_ids):
        trial_rows.append(
            {
                "dataset": "feis",
                "subject_id": subject,
                "trial_index": int(epoch_id),
                "label": labels[row_idx],
                "audio_path": audio_relpaths[row_idx],
                "eeg_subject_bundle": str(subject_bundle.relative_to(output_root / "feis")),
            }
        )

    ensure_dir(subject_bundle.parent)
    np.savez_compressed(subject_bundle, **bundle_payload)
    return trial_rows, segment_rows


def preprocess_karaone_subject(
    karaone_root: Path,
    subject: str,
    output_root: Path,
    temp_root: Path,
    target_eeg_sfreq: int,
    target_audio_sfreq: int,
    bandpass_low: float,
    bandpass_high: float,
    notch_hz: float,
    prefer_archive: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    temp_dir_to_remove: Path | None = None
    subject_dir: Path | None = None
    if prefer_archive and (karaone_root / f"{subject}.tar.bz2").exists():
        subject_dir = extract_karaone_subject_to_temp(karaone_root, subject, temp_root)
        temp_dir_to_remove = subject_dir.parents[6]
    else:
        subject_dir = locate_existing_karaone_subject_dir(karaone_root, subject)
        if subject_dir is None and (karaone_root / f"{subject}.tar.bz2").exists():
            subject_dir = extract_karaone_subject_to_temp(karaone_root, subject, temp_root)
            temp_dir_to_remove = subject_dir.parents[6]
    if subject_dir is None:
        raise FileNotFoundError(f"Could not locate KaraOne subject {subject}")

    try:
        data, src_sfreq, channel_names = preprocess_karaone_continuous(
            subject_dir=subject_dir,
            bandpass_low=bandpass_low,
            bandpass_high=bandpass_high,
            notch_hz=notch_hz,
        )
        epoch_inds = flatten_epoch_inds(subject_dir / "epoch_inds.mat")
        labels = (subject_dir / "kinect_data" / "labels.txt").read_text(errors="replace").splitlines()
        trial_indices = list(range(len(labels)))

        stage_ranges = {
            "clearing": epoch_inds["clearing_inds"],
            "stimulus_like": epoch_inds["speaking_inds"][0::2],
            "thinking": epoch_inds["thinking_inds"],
            "overt_like": epoch_inds["speaking_inds"][1::2],
        }
        stage_target_lens = {
            stage: robust_target_len_from_ranges(ranges, src_sfreq, target_eeg_sfreq)
            for stage, ranges in stage_ranges.items()
        }

        subject_bundle = output_root / "karaone" / "subjects" / f"{subject}.npz"
        audio_root = output_root / "karaone" / "audio" / subject
        audio_relpaths: list[str] = []
        for trial_index in trial_indices:
            src_audio = subject_dir / "kinect_data" / f"{trial_index}.wav"
            dst_audio = audio_root / f"{trial_index:03d}.wav"
            preprocess_audio_file(src_audio, dst_audio, target_audio_sfreq)
            audio_relpaths.append(str(dst_audio.relative_to(output_root / "karaone")))

        trial_rows: list[dict[str, object]] = []
        segment_rows: list[dict[str, object]] = []
        bundle_payload: dict[str, np.ndarray] = {
            "trial_indices": np.asarray(trial_indices, dtype=np.int32),
            "labels": np.asarray(labels),
            "audio_relpaths": np.asarray(audio_relpaths),
            "channel_names": np.asarray(channel_names),
            "eeg_sfreq_hz": np.asarray([target_eeg_sfreq], dtype=np.int32),
            "stage_names": np.asarray(KARAONE_STAGE_ORDER),
        }

        baseline_ranges = stage_ranges["clearing"]
        for stage in KARAONE_STAGE_ORDER:
            stage_key = sanitize_stage_key(stage)
            target_len = stage_target_lens[stage]
            stage_trials: list[np.ndarray] = []
            stage_valid_lengths: list[int] = []
            stage_src_ranges: list[list[int]] = []
            for row_idx, trial_index in enumerate(trial_indices):
                start, end = map(int, stage_ranges[stage][trial_index])
                baseline_start, baseline_end = map(int, baseline_ranges[trial_index])
                epoch = data[:, start : end + 1]
                baseline = data[:, baseline_start : baseline_end + 1]
                processed, valid_len = preprocess_epoch(
                    epoch=epoch,
                    baseline=baseline,
                    src_sfreq=src_sfreq,
                    target_sfreq=target_eeg_sfreq,
                    target_len=target_len,
                )
                stage_trials.append(processed)
                stage_valid_lengths.append(valid_len)
                stage_src_ranges.append([start, end])
                segment_rows.append(
                    {
                        "dataset": "karaone",
                        "subject_id": subject,
                        "trial_index": trial_index,
                        "segment_stage": stage,
                        "segment_role": KARAONE_STAGE_ROLE[stage],
                        "segment_array_key": f"stage__{stage_key}",
                        "label": labels[row_idx],
                        "audio_path": audio_relpaths[row_idx],
                        "eeg_subject_bundle": str(subject_bundle.relative_to(output_root / "karaone")),
                        "baseline_window": "clearing",
                        "eeg_num_channels": len(channel_names),
                        "eeg_num_samples": target_len,
                        "eeg_valid_num_samples": valid_len,
                        "eeg_sfreq_hz": target_eeg_sfreq,
                        "audio_sfreq_hz": target_audio_sfreq,
                        "segment_start_sample_src": start,
                        "segment_end_sample_src": end,
                        "baseline_start_sample_src": baseline_start,
                        "baseline_end_sample_src": baseline_end,
                        "audio_pairing": "same-trial overt wav",
                    }
                )
            bundle_payload[f"stage__{stage_key}"] = np.stack(stage_trials, axis=0).astype(np.float32)
            bundle_payload[f"stage__{stage_key}__valid_lengths"] = np.asarray(stage_valid_lengths, dtype=np.int32)
            bundle_payload[f"stage__{stage_key}__src_ranges"] = np.asarray(stage_src_ranges, dtype=np.int32)

        for row_idx, trial_index in enumerate(trial_indices):
            trial_rows.append(
                {
                    "dataset": "karaone",
                    "subject_id": subject,
                    "trial_index": trial_index,
                    "label": labels[row_idx],
                    "audio_path": audio_relpaths[row_idx],
                    "eeg_subject_bundle": str(subject_bundle.relative_to(output_root / "karaone")),
                }
            )

        ensure_dir(subject_bundle.parent)
        np.savez_compressed(subject_bundle, **bundle_payload)
        return trial_rows, segment_rows
    finally:
        if temp_dir_to_remove is not None and temp_dir_to_remove.exists():
            shutil.rmtree(temp_dir_to_remove)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["feis", "karaone"], default=["feis", "karaone"])
    parser.add_argument("--feis-root", type=Path, default=DEFAULT_FEIS_ROOT)
    parser.add_argument("--karaone-root", type=Path, default=DEFAULT_KARAONE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--temp-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--feis-subjects", nargs="*", default=None)
    parser.add_argument("--karaone-subjects", nargs="*", default=None)
    parser.add_argument("--target-eeg-sfreq", type=int, default=256)
    parser.add_argument("--target-audio-sfreq", type=int, default=16000)
    parser.add_argument("--bandpass-low", type=float, default=1.0)
    parser.add_argument("--bandpass-high", type=float, default=40.0)
    parser.add_argument("--feis-notch-hz", type=float, default=50.0)
    parser.add_argument("--karaone-notch-hz", type=float, default=60.0)
    parser.add_argument("--prefer-karaone-archives", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    dataset_stats: list[DatasetStats] = []

    if "feis" in args.datasets:
        feis_subjects = args.feis_subjects or [path.name for path in feis_subject_dirs(args.feis_root)]
        feis_trial_rows: list[dict[str, object]] = []
        feis_segment_rows: list[dict[str, object]] = []
        for subject in tqdm(feis_subjects, desc="FEIS subjects"):
            trial_rows, segment_rows = preprocess_feis_subject(
                feis_root=args.feis_root,
                subject=subject,
                output_root=args.output_root,
                target_eeg_sfreq=args.target_eeg_sfreq,
                target_audio_sfreq=args.target_audio_sfreq,
                bandpass_low=args.bandpass_low,
                bandpass_high=args.bandpass_high,
                notch_hz=args.feis_notch_hz,
            )
            feis_trial_rows.extend(trial_rows)
            feis_segment_rows.extend(segment_rows)
        feis_trial_rows, feis_segment_rows, feis_enrichment = enrich_feis_rows(
            feis_root=args.feis_root,
            output_root=args.output_root,
            trial_rows=feis_trial_rows,
            segment_rows=feis_segment_rows,
        )
        feis_manifest = {
            "dataset": "feis",
            "source_root": str(args.feis_root),
            "subject_count": len(feis_subjects),
            "trial_count": len(feis_trial_rows),
            "segment_count": len(feis_segment_rows),
            "stage_order": FEIS_STAGE_ORDER,
            "stage_role_map": FEIS_STAGE_ROLE,
            "hearing_equivalent_stage": "stimuli",
            "audio_pairing": "subject-level canonical wav for the same prompt label",
            "target_eeg_sfreq": args.target_eeg_sfreq,
            "target_audio_sfreq": args.target_audio_sfreq,
            "bandpass_hz": [args.bandpass_low, args.bandpass_high],
            "notch_hz": args.feis_notch_hz,
            "subject_known_eval": FEIS_SUBJECT_KNOWN_EVAL,
            "subject_unknown_eval": FEIS_SUBJECT_UNKNOWN_EVAL,
            "clean_subject_ids": feis_enrichment["clean_subject_ids"],
            "anomalous_subject_ids": feis_enrichment["anomalous_subject_ids"],
            "unique_hashes_per_label_across_subjects": feis_enrichment["unique_hashes_per_label_across_subjects"],
            "notes": [
                "Only numbered English subjects are processed by default.",
                "All five FEIS phases are exported separately: stimuli, articulators, thinking, speaking, resting.",
                "Each phase is baseline-normalized with the same trial's resting window.",
                "Processed metadata records subject-specific audio provenance and subject-05 fallback audio anomaly.",
            ],
        }
        write_csv(args.output_root / "feis" / "trials.csv", feis_trial_rows)
        write_csv(args.output_root / "feis" / "segments.csv", feis_segment_rows)
        write_json(args.output_root / "feis" / "manifest.json", feis_manifest)
        dataset_stats.append(
            DatasetStats("feis", len(feis_subjects), len(feis_trial_rows), len(feis_segment_rows), feis_manifest["notes"])
        )

    if "karaone" in args.datasets:
        karaone_subjects = args.karaone_subjects or list_karaone_subjects(args.karaone_root)
        karaone_trial_rows: list[dict[str, object]] = []
        karaone_segment_rows: list[dict[str, object]] = []
        for subject in tqdm(karaone_subjects, desc="KaraOne subjects"):
            trial_rows, segment_rows = preprocess_karaone_subject(
                karaone_root=args.karaone_root,
                subject=subject,
                output_root=args.output_root,
                temp_root=args.temp_root,
                target_eeg_sfreq=args.target_eeg_sfreq,
                target_audio_sfreq=args.target_audio_sfreq,
                bandpass_low=args.bandpass_low,
                bandpass_high=args.bandpass_high,
                notch_hz=args.karaone_notch_hz,
                prefer_archive=args.prefer_karaone_archives,
            )
            karaone_trial_rows.extend(trial_rows)
            karaone_segment_rows.extend(segment_rows)
        karaone_manifest = {
            "dataset": "karaone",
            "source_root": str(args.karaone_root),
            "subject_count": len(karaone_subjects),
            "trial_count": len(karaone_trial_rows),
            "segment_count": len(karaone_segment_rows),
            "stage_order": KARAONE_STAGE_ORDER,
            "stage_role_map": KARAONE_STAGE_ROLE,
            "hearing_equivalent_stage": "stimulus_like",
            "audio_pairing": "same-trial overt wav",
            "target_eeg_sfreq": args.target_eeg_sfreq,
            "target_audio_sfreq": args.target_audio_sfreq,
            "bandpass_hz": [args.bandpass_low, args.bandpass_high],
            "notch_hz": args.karaone_notch_hz,
            "notes": [
                "All reliably aligned KaraOne windows are exported separately: clearing, stimulus_like, thinking, overt_like.",
                "KaraOne does not expose a clean explicit hearing event label; stimulus_like is the closest reliable pre-thinking/hearing-like segment.",
                "Stage tensor lengths use a robust percentile target rather than the single longest trial, so one outlier interval does not inflate the whole array.",
                "With --prefer-karaone-archives, only the minimum needed files are extracted to a temp dir and then deleted.",
            ],
        }
        write_csv(args.output_root / "karaone" / "trials.csv", karaone_trial_rows)
        write_csv(args.output_root / "karaone" / "segments.csv", karaone_segment_rows)
        write_json(args.output_root / "karaone" / "manifest.json", karaone_manifest)
        dataset_stats.append(
            DatasetStats(
                "karaone",
                len(karaone_subjects),
                len(karaone_trial_rows),
                len(karaone_segment_rows),
                karaone_manifest["notes"],
            )
        )

    summary = {
        item.dataset: {
            "subject_count": item.subject_count,
            "trial_count": item.trial_count,
            "segment_count": item.segment_count,
            "notes": item.notes,
        }
        for item in dataset_stats
    }
    write_json(args.output_root / "summary.json", summary)


if __name__ == "__main__":
    main()

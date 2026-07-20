#!/usr/bin/env python3
"""Build a subject-disjoint, harmonised imagined-speech EEG dataset.

The script never edits ``data/``.  FEIS and KARA ONE are imported from their
existing stage bundles; ds004306 is read from its raw EEGLAB ``.set/.fdt``
files through a temporary symlink view, because OpenNeuro stores the FDT
payloads outside the BIDS subject directories.

The output contains only imagined-speech EEG.  All records have the same
shape: 14 common channels x 1280 samples (5 seconds at 256 Hz).  Padding is
recorded in ``valid_lengths`` and must be masked during model training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from scipy.io import loadmat

# MNE imports numba/matplotlib on import.  Give them writable caches even when
# this script is launched from a restricted shell or batch environment.
_CACHE_ROOT = Path(tempfile.gettempdir()) / "combined_eeg_preprocess_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(_CACHE_ROOT / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))

import mne  # noqa: E402

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.combined_0715.lineage import (  # noqa: E402
    file_sha256,
    preprocessing_sha256_from_rows,
)


COMMON_CHANNELS = [
    "F3", "FC5", "AF3", "F7", "T7", "P7", "O1", "O2", "P8", "T8", "F8", "AF4", "FC6", "F4",
]
TARGET_EEG_SFREQ = 256
TARGET_EEG_SAMPLES = 5 * TARGET_EEG_SFREQ
TARGET_AUDIO_SFREQ = 16000
DEFAULT_EXCLUDED_FEIS_SUBJECTS = {"05"}


@dataclass(frozen=True)
class AudioRecord:
    key: str
    output_relpath: str
    duration_sec: float
    n_samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory containing feis/, karaone/, and ds004306/ (default: bundle/data).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "eeg_output",
        help="Output directory. Raw input folders are never changed.",
    )
    parser.add_argument(
        "--ds-modalities",
        nargs="+",
        choices=("auditory", "text", "image"),
        default=("auditory",),
        help="ds004306 imagination modalities to export (default: auditory only).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("feis", "karaone", "ds004306"),
        default=("feis", "karaone", "ds004306"),
        help="Datasets to rebuild. Existing manifest/QC rows for omitted datasets are preserved.",
    )
    parser.add_argument(
        "--include-feis-subject-05",
        action="store_true",
        help="Include FEIS subject 05, which is marked anomalous in the source manifest.",
    )
    parser.add_argument("--split-seed", type=int, default=20260720)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing generated NPZ/CSV/JSON files. Existing cached audio is reused.",
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Only build EEG and manifests. Audio paths are recorded but converted WAVs are not created.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the input layout and print the planned work without writing output.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate existing EEG outputs and the locked training split without rebuilding data.",
    )
    parser.add_argument(
        "--locked-split",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "app" / "configs" / "split" / "combined_0715_v1_split.yaml",
        help="Authoritative subject split used for verification (default: app locked split YAML).",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_metadata_wav(path: Path) -> tuple[int, float]:
    import wave

    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        sfreq = wav.getframerate()
    return frames, frames / sfreq


def ensure_audio(source: Path, output_root: Path, skip_audio: bool) -> AudioRecord:
    """Create/reuse a lossless mono 16-kHz WAV cache entry for one source file."""
    require(source.is_file(), f"Audio file is missing: {source}")
    key = sha1_file(source)
    relpath = Path("audio_16k") / f"{key}.wav"
    destination = output_root / relpath
    if not skip_audio:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            command = [
                "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(source),
                "-vn", "-ac", "1", "-ar", str(TARGET_AUDIO_SFREQ), "-c:a", "pcm_s16le", str(destination),
            ]
            subprocess.run(command, check=True)
        samples, duration = audio_metadata_wav(destination)
    else:
        samples, duration = 0, float("nan")
    return AudioRecord(key=key, output_relpath=str(relpath), duration_sec=duration, n_samples=samples)


def select_common_channels(array: np.ndarray, channel_names: Iterable[str]) -> np.ndarray:
    names = [str(name) for name in channel_names]
    by_upper = {name.upper(): index for index, name in enumerate(names)}
    missing = [name for name in COMMON_CHANNELS if name not in by_upper]
    require(not missing, f"Missing common channels: {missing}; available: {names}")
    return np.asarray(array[:, [by_upper[name] for name in COMMON_CHANNELS], :], dtype=np.float32)


def common_average_reference(array: np.ndarray) -> np.ndarray:
    return array - np.mean(array, axis=1, keepdims=True, dtype=np.float32)


def robust_baseline_normalise(
    signal: np.ndarray,
    baseline: np.ndarray,
    baseline_valid_lengths: np.ndarray,
    clip: float = 12.0,
) -> np.ndarray:
    """CAR then baseline median/MAD normalization, independently for each trial/channel."""
    signal = common_average_reference(np.asarray(signal, dtype=np.float32))
    baseline = common_average_reference(np.asarray(baseline, dtype=np.float32))
    valid_lengths = np.asarray(baseline_valid_lengths, dtype=np.int32)
    require(len(valid_lengths) == len(baseline), "Baseline valid-length count does not match trials")
    center = np.zeros((len(baseline), baseline.shape[1], 1), dtype=np.float32)
    mad = np.zeros_like(center)
    fallback = np.zeros_like(center)
    for trial_index, raw_length in enumerate(valid_lengths):
        valid = min(max(int(raw_length), 1), baseline.shape[-1])
        values = baseline[trial_index, :, :valid]
        trial_center = np.median(values, axis=1, keepdims=True)
        center[trial_index] = trial_center
        mad[trial_index] = 1.4826 * np.median(np.abs(values - trial_center), axis=1, keepdims=True)
        fallback[trial_index] = np.std(values, axis=1, keepdims=True)
    scale = np.where(mad > 1e-8, mad, fallback)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return np.clip((signal - center) / scale, -clip, clip).astype(np.float32)


def pad_or_crop(data: np.ndarray, valid_length: int) -> tuple[np.ndarray, int]:
    actual = min(max(int(valid_length), 0), data.shape[-1], TARGET_EEG_SAMPLES)
    output = np.zeros((len(COMMON_CHANNELS), TARGET_EEG_SAMPLES), dtype=np.float32)
    if actual:
        output[:, :actual] = data[:, :actual]
    return output, actual


def write_npz(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite to replace generated data.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def normalize_stage_bundle(
    source_npz: Path,
    thinking_stage: str,
    baseline_stage: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one prepared FEIS/KARA subject and return normalized thinking trials."""
    raw = np.load(source_npz, allow_pickle=True)
    thinking = select_common_channels(raw[f"stage__{thinking_stage}"], raw["channel_names"])
    baseline = select_common_channels(raw[f"stage__{baseline_stage}"], raw["channel_names"])
    valid = np.asarray(raw[f"stage__{thinking_stage}__valid_lengths"], dtype=np.int32)
    baseline_valid = np.asarray(raw[f"stage__{baseline_stage}__valid_lengths"], dtype=np.int32)
    normalized = robust_baseline_normalise(thinking, baseline, baseline_valid)
    output = np.zeros((normalized.shape[0], len(COMMON_CHANNELS), TARGET_EEG_SAMPLES), dtype=np.float32)
    lengths = np.zeros(normalized.shape[0], dtype=np.int32)
    for index in range(normalized.shape[0]):
        output[index], lengths[index] = pad_or_crop(normalized[index], int(valid[index]))
    return output, lengths, raw["trial_indices"], raw["labels"], raw["audio_relpaths"]


def discover_ds_fdt(ds_root: Path, set_path: Path) -> Path:
    metadata = loadmat(set_path, squeeze_me=True, variable_names=["datfile", "data"])
    wanted = str(metadata.get("datfile", metadata.get("data", "")))
    match = re.search(r"sub-(\d+)/ses-(\d+)", str(set_path))
    require(match is not None, f"Cannot infer subject/session from {set_path}")
    subject, session = (int(match.group(1)), int(match.group(2)))
    # OpenNeuro's FDT sidecar directories use unpadded IDs for 10+ (sub10),
    # but two-digit padded IDs for 03 and 08 (sub03/sub08).  Try both forms.
    candidate_dirs = list(dict.fromkeys([
        ds_root / "derivatives" / "fdt_files" / f"sub{subject}_sess-{session:02d}",
        ds_root / "derivatives" / "fdt_files" / f"sub{subject:02d}_sess-{session:02d}",
    ]))
    candidates = [candidate for directory in candidate_dirs for candidate in sorted(directory.glob("*.fdt"))]
    require(candidates, f"No FDT payload found for {set_path}; searched {candidate_dirs}")
    exact = [candidate for candidate in candidates if candidate.name == wanted]
    # ds004306 has one known filename typo (sub-013); one FDT in its session
    # directory is nevertheless an unambiguous payload.
    require(len(exact) == 1 or len(candidates) == 1, f"Ambiguous FDT mapping for {set_path}: {candidates}")
    return exact[0] if exact else candidates[0]


@contextmanager
def staged_eeglab_raw(ds_root: Path, set_path: Path):
    """Open a set/FDT pair without adding symlinks to the immutable source tree."""
    fdt_path = discover_ds_fdt(ds_root, set_path)
    metadata = loadmat(set_path, squeeze_me=True, variable_names=["datfile", "data"])
    datfile = str(metadata.get("datfile", metadata.get("data", fdt_path.name)))
    with tempfile.TemporaryDirectory(prefix="ds004306_eeglab_") as temp:
        temp_dir = Path(temp)
        staged_set = temp_dir / (Path(datfile).with_suffix(".set").name)
        staged_fdt = temp_dir / datfile
        os.symlink(set_path.resolve(), staged_set)
        os.symlink(fdt_path.resolve(), staged_fdt)
        raw = mne.io.read_raw_eeglab(staged_set, preload=False, verbose="error")
        yield raw


def clean_trial_type(value: str) -> str:
    value = value.split("###", maxsplit=1)[0]
    return re.sub(r"^\d+,\s*", "", value)


def ds_modality_and_label(trial_type: str) -> tuple[str, str] | None:
    if not trial_type.startswith("Imagination"):
        return None
    if re.match(r"Imagination_(?:audio_)?a_", trial_type):
        modality, suffix = "auditory", re.sub(r"^Imagination_(?:audio_)?a_", "", trial_type)
    elif trial_type.startswith("Imagination_t_"):
        modality, suffix = "text", trial_type.removeprefix("Imagination_t_")
    elif trial_type.startswith("Imagination_image_"):
        modality, suffix = "image", trial_type.removeprefix("Imagination_image_")
    else:
        return None
    return modality, suffix.split("_", maxsplit=1)[0]


def ds_events(events_path: Path, allowed_modalities: set[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with events_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            events.append({"onset": float(row["onset"]), "trial_type": clean_trial_type(row["trial_type"])})
    trials: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        parsed = ds_modality_and_label(event["trial_type"])
        if parsed is None or parsed[0] not in allowed_modalities:
            continue
        require(index > 0 and events[index - 1]["trial_type"].startswith("Perception"),
                f"Imagination event is not preceded by Perception: {events_path} @ {event}")
        next_perception = next(
            (candidate["onset"] for candidate in events[index + 1:] if candidate["trial_type"].startswith("Perception")),
            event["onset"] + 5.0,
        )
        trials.append({
            "onset": event["onset"],
            "stop": min(event["onset"] + 5.0, next_perception),
            "modality": parsed[0],
            "label": parsed[1],
            "event_type": event["trial_type"],
        })
    return trials


def ds_baseline_interval(events_path: Path) -> tuple[float, float]:
    starts: list[float] = []
    ends: list[float] = []
    with events_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            trial_type = clean_trial_type(row["trial_type"])
            if trial_type.startswith("start_baseline"):
                starts.append(float(row["onset"]))
            elif trial_type.startswith("end_baseline"):
                ends.append(float(row["onset"]))
    require(starts and ends, f"No session baseline markers in {events_path}")
    return starts[0], ends[0]


def ds_audio_sources(ds_root: Path, label: str) -> list[Path]:
    # The published audio directory calls the guitar stimulus "hammer".  Keep
    # the original filenames and record this as weak/category-level pairing.
    folder = {"flower": "flower", "penguin": "penguin", "guitar": "hammer"}.get(label)
    require(folder is not None, f"Unsupported ds004306 category: {label}")
    sources = sorted((ds_root / "stimuli" / "audio" / folder).glob("*.ogg"))
    require(sources, f"No published audio files for ds004306 label {label}")
    return sources


def process_ds_session(
    ds_root: Path,
    set_path: Path,
    events_path: Path,
    allowed_modalities: set[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    trials = ds_events(events_path, allowed_modalities)
    baseline_start, baseline_stop = ds_baseline_interval(events_path)
    with staged_eeglab_raw(ds_root, set_path) as raw:
        by_upper = {name.upper(): name for name in raw.ch_names}
        missing = [name for name in COMMON_CHANNELS if name not in by_upper]
        require(not missing, f"{set_path} is missing common channels: {missing}")
        raw.pick([by_upper[name] for name in COMMON_CHANNELS])
        raw.load_data(verbose="error")
        raw.notch_filter(freqs=[50.0], verbose="error")
        raw.filter(l_freq=1.0, h_freq=40.0, verbose="error")
        raw.resample(TARGET_EEG_SFREQ, verbose="error")
        raw.set_eeg_reference("average", projection=False, verbose="error")
        baseline = raw.get_data(
            start=max(0, round(baseline_start * TARGET_EEG_SFREQ)),
            stop=min(raw.n_times, round(baseline_stop * TARGET_EEG_SFREQ)),
        ).astype(np.float32)
        require(baseline.shape[1] >= TARGET_EEG_SFREQ, f"Baseline is too short in {events_path}")
        center = np.median(baseline, axis=1, keepdims=True)
        mad = 1.4826 * np.median(np.abs(baseline - center), axis=1, keepdims=True)
        fallback = np.std(baseline, axis=1, keepdims=True)
        scale = np.where(mad > 1e-8, mad, fallback)
        scale = np.where(scale > 1e-8, scale, 1.0)
        windows: list[np.ndarray] = []
        lengths: list[int] = []
        for trial in trials:
            start = max(0, round(trial["onset"] * TARGET_EEG_SFREQ))
            stop = min(raw.n_times, round(trial["stop"] * TARGET_EEG_SFREQ))
            data = raw.get_data(start=start, stop=stop).astype(np.float32)
            # raw has already been average referenced; retain the same robust
            # baseline normalization used by FEIS/KARA exports.
            data = np.clip((data - center) / scale, -12.0, 12.0)
            window, valid = pad_or_crop(data, data.shape[1])
            windows.append(window)
            lengths.append(valid)
    return np.stack(windows), np.asarray(lengths, dtype=np.int32), trials


def write_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite to replace generated data.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _text_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def verify_preprocessed_outputs(
    output_root: Path,
    locked_split_path: Path,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Audit existing combined EEG outputs and always persist a JSONL report.

    Errors are accumulated instead of raised immediately so a corrupt output
    still leaves an actionable report.  Warnings cover 1--5% clipping; clipping
    at or above 5% is a critical error.
    """
    output_root = Path(output_root).resolve()
    locked_split_path = Path(locked_split_path).resolve()
    report_path = report_path or output_root / "qc" / "eeg_verification.jsonl"
    issues: list[dict[str, Any]] = []
    check_names = (
        "manifest_integrity",
        "npz_integrity",
        "manifest_npz_consistency",
        "finite_output",
        "padding_zero",
        "clip_fraction",
        "locked_split",
        "provenance",
    )

    def issue(check: str, severity: str, message: str, **context: Any) -> None:
        entry: dict[str, Any] = {"check": check, "severity": severity, "message": message}
        if context:
            entry["context"] = context
        issues.append(entry)

    manifest_path = output_root / "manifests" / "unified_trials.csv"
    rows: list[dict[str, Any]] = []
    required_manifest_fields = {
        "dataset", "subject_group_id", "subject_recording_id", "trial_index",
        "sample_key", "label", "eeg_relpath", "eeg_row", "eeg_valid_samples",
        "eeg_sfreq_hz",
    }
    try:
        rows = read_csv_rows(manifest_path)
        if not rows:
            issue("manifest_integrity", "error", "Manifest is missing or empty", path=str(manifest_path))
        else:
            missing_fields = required_manifest_fields - set(rows[0])
            if missing_fields:
                issue(
                    "manifest_integrity", "error", "Manifest is missing required fields",
                    fields=sorted(missing_fields),
                )
            sample_keys = [row.get("sample_key", "") for row in rows]
            if any(not key for key in sample_keys):
                issue("manifest_integrity", "error", "Manifest contains empty sample_key values")
            if len(sample_keys) != len(set(sample_keys)):
                issue("manifest_integrity", "error", "Manifest sample_key values are not unique")
    except Exception as error:
        issue("manifest_integrity", "error", "Could not read manifest", error=str(error), path=str(manifest_path))

    rows_by_relpath: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        relative = row.get("eeg_relpath", "")
        if not relative:
            issue("manifest_npz_consistency", "error", "Manifest row has no eeg_relpath", sample_key=row.get("sample_key"))
            continue
        rows_by_relpath[relative].append(row)

    n_trials_checked = 0
    n_npz_checked = 0
    clip_records: list[dict[str, Any]] = []
    for relative, bundle_rows in sorted(rows_by_relpath.items()):
        path = (output_root / relative).resolve()
        try:
            path.relative_to(output_root)
        except ValueError:
            issue("npz_integrity", "error", "EEG path escapes output root", eeg_relpath=relative)
            continue
        if not path.is_file():
            issue("npz_integrity", "error", "Referenced EEG NPZ is missing", eeg_relpath=relative)
            continue
        try:
            with np.load(path, allow_pickle=False) as raw:
                required_npz_fields = {
                    "eeg", "valid_lengths", "trial_indices", "labels", "channel_names", "eeg_sfreq_hz",
                }
                missing_npz_fields = required_npz_fields - set(raw.files)
                if missing_npz_fields:
                    issue(
                        "npz_integrity", "error", "EEG NPZ is missing required arrays",
                        eeg_relpath=relative, fields=sorted(missing_npz_fields),
                    )
                    continue
                eeg = np.asarray(raw["eeg"])
                valid_lengths = np.asarray(raw["valid_lengths"])
                trial_indices = np.asarray(raw["trial_indices"])
                labels = np.asarray(raw["labels"])
                channel_names = [_text_scalar(value) for value in np.asarray(raw["channel_names"]).reshape(-1)]
                sfreq_values = np.asarray(raw["eeg_sfreq_hz"]).reshape(-1)
                n_npz_checked += 1
        except Exception as error:
            issue("npz_integrity", "error", "Could not load EEG NPZ", eeg_relpath=relative, error=str(error))
            continue

        expected_shape = (len(eeg), len(COMMON_CHANNELS), TARGET_EEG_SAMPLES) if eeg.ndim >= 1 else None
        shape_ok = eeg.ndim == 3 and eeg.shape == expected_shape
        if not shape_ok:
            issue(
                "npz_integrity", "error", "EEG array must have shape [N,14,1280]",
                eeg_relpath=relative, observed_shape=list(eeg.shape),
            )
        if channel_names != COMMON_CHANNELS:
            issue(
                "npz_integrity", "error", "EEG channel order does not match COMMON_CHANNELS",
                eeg_relpath=relative, observed=channel_names, expected=COMMON_CHANNELS,
            )
        if len(sfreq_values) != 1 or int(sfreq_values[0]) != TARGET_EEG_SFREQ:
            issue(
                "npz_integrity", "error", "EEG sample rate must be 256 Hz",
                eeg_relpath=relative, observed=sfreq_values.tolist(),
            )

        n_trials = eeg.shape[0] if eeg.ndim >= 1 else 0
        for field_name, values in (
            ("valid_lengths", valid_lengths),
            ("trial_indices", trial_indices),
            ("labels", labels),
        ):
            if values.ndim == 0 or len(values) != n_trials:
                issue(
                    "npz_integrity", "error", f"{field_name} length does not match EEG rows",
                    eeg_relpath=relative, eeg_rows=n_trials,
                    observed_length=int(len(values)) if values.ndim else 0,
                )

        if len(bundle_rows) != n_trials:
            issue(
                "manifest_npz_consistency", "error", "Manifest does not cover every EEG row exactly once",
                eeg_relpath=relative, manifest_rows=len(bundle_rows), eeg_rows=n_trials,
            )
        parsed_rows: list[int] = []
        for row in bundle_rows:
            try:
                parsed_rows.append(int(row["eeg_row"]))
            except (KeyError, TypeError, ValueError):
                issue(
                    "manifest_npz_consistency", "error", "Manifest has an invalid eeg_row",
                    eeg_relpath=relative, sample_key=row.get("sample_key"), value=row.get("eeg_row"),
                )
        if sorted(parsed_rows) != list(range(n_trials)):
            issue(
                "manifest_npz_consistency", "error", "Manifest eeg_row values are not an exact 0..N-1 cover",
                eeg_relpath=relative,
            )

        if not shape_ok:
            continue
        if not np.isfinite(eeg).all():
            issue("finite_output", "error", "EEG array contains NaN or Inf", eeg_relpath=relative)

        valid_array_ok = valid_lengths.ndim > 0 and len(valid_lengths) == n_trials
        for index in range(n_trials):
            if not valid_array_ok:
                break
            try:
                valid = int(valid_lengths[index])
            except (TypeError, ValueError, OverflowError):
                issue(
                    "npz_integrity", "error", "valid_lengths contains a non-integer value",
                    eeg_relpath=relative, eeg_row=index,
                )
                continue
            if not 1 <= valid <= TARGET_EEG_SAMPLES:
                issue(
                    "npz_integrity", "error", "EEG valid length is outside [1,1280]",
                    eeg_relpath=relative, eeg_row=index, value=valid,
                )
                continue
            if np.count_nonzero(eeg[index, :, valid:]):
                issue(
                    "padding_zero", "error", "EEG padding tail is not exactly zero",
                    eeg_relpath=relative, eeg_row=index, valid_length=valid,
                )

        by_row = {int(row["eeg_row"]): row for row in bundle_rows if str(row.get("eeg_row", "")).lstrip("-").isdigit()}
        for index, row in sorted(by_row.items()):
            if not 0 <= index < n_trials:
                continue
            n_trials_checked += 1
            try:
                manifest_valid = int(row["eeg_valid_samples"])
                stored_valid = int(valid_lengths[index])
                if manifest_valid != stored_valid:
                    issue(
                        "manifest_npz_consistency", "error", "Manifest and NPZ valid lengths differ",
                        sample_key=row.get("sample_key"), manifest=manifest_valid, npz=stored_valid,
                    )
                if int(row["trial_index"]) != int(trial_indices[index]):
                    issue(
                        "manifest_npz_consistency", "error", "Manifest and NPZ trial indices differ",
                        sample_key=row.get("sample_key"),
                    )
                if row["label"] != _text_scalar(labels[index]):
                    issue(
                        "manifest_npz_consistency", "error", "Manifest and NPZ labels differ",
                        sample_key=row.get("sample_key"), manifest=row["label"], npz=_text_scalar(labels[index]),
                    )
                if int(row["eeg_sfreq_hz"]) != TARGET_EEG_SFREQ:
                    issue(
                        "manifest_npz_consistency", "error", "Manifest EEG sample rate is not 256 Hz",
                        sample_key=row.get("sample_key"), value=row.get("eeg_sfreq_hz"),
                    )
            except (KeyError, TypeError, ValueError, IndexError) as error:
                issue(
                    "manifest_npz_consistency", "error", "Could not validate manifest row against NPZ",
                    sample_key=row.get("sample_key"), error=str(error),
                )

        # Clipping is assessed per NPZ/channel over valid samples.  This keeps
        # memory bounded and points warnings back to an actionable recording.
        if valid_array_ok:
            for channel_index, channel_name in enumerate(COMMON_CHANNELS):
                total = 0
                clipped = 0
                for trial_index in range(n_trials):
                    try:
                        valid = int(valid_lengths[trial_index])
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if not 1 <= valid <= TARGET_EEG_SAMPLES:
                        continue
                    values = eeg[trial_index, channel_index, :valid]
                    total += int(values.size)
                    clipped += int(np.count_nonzero(np.abs(values) >= 11.999))
                fraction = clipped / total if total else float("nan")
                clip_records.append({
                    "eeg_relpath": relative,
                    "channel": channel_name,
                    "clip_fraction": fraction,
                })
                if np.isfinite(fraction) and fraction >= 0.05:
                    issue(
                        "clip_fraction", "error", "Channel clipping is at or above 5%",
                        eeg_relpath=relative, channel=channel_name, clip_fraction=fraction,
                    )
                elif np.isfinite(fraction) and fraction >= 0.01:
                    issue(
                        "clip_fraction", "warning", "Channel clipping is at or above 1%",
                        eeg_relpath=relative, channel=channel_name, clip_fraction=fraction,
                    )

    split_version: str | None = None
    try:
        split = yaml.safe_load(locked_split_path.read_text(encoding="utf-8"))
        split_version = str(split.get("version", "")) or None
        split_datasets = split.get("datasets", {})
        manifest_groups: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            if row.get("dataset") and row.get("subject_group_id"):
                manifest_groups[row["dataset"]].add(row["subject_group_id"])
        for dataset in sorted(set(manifest_groups) | set(split_datasets)):
            definition = split_datasets.get(dataset, {})
            groups = {
                name: set(definition.get(name, []))
                for name in ("train", "validation", "test")
            }
            for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
                overlap = sorted(groups[first] & groups[second])
                if overlap:
                    issue(
                        "locked_split", "error", "Locked split subject sets overlap",
                        dataset=dataset, splits=[first, second], subjects=overlap,
                    )
            covered = set().union(*groups.values())
            missing = sorted(manifest_groups.get(dataset, set()) - covered)
            extra = sorted(covered - manifest_groups.get(dataset, set()))
            if missing or extra:
                issue(
                    "locked_split", "error", "Locked split does not exactly cover manifest subjects",
                    dataset=dataset, missing=missing, extra=extra,
                )
    except Exception as error:
        issue(
            "locked_split", "error", "Could not read or validate locked split",
            path=str(locked_split_path), error=str(error),
        )

    provenance: dict[str, str] = {}
    try:
        provenance = {
            "locked_split_sha256": file_sha256(locked_split_path),
            "manifest_sha256": file_sha256(manifest_path),
            "preprocessing_sha256": preprocessing_sha256_from_rows(output_root, rows),
        }
    except Exception as error:
        issue(
            "provenance", "error", "Could not bind QC report to its exact input files",
            error=str(error),
        )

    errors = [entry for entry in issues if entry["severity"] == "error"]
    warnings = [entry for entry in issues if entry["severity"] == "warning"]
    checks: dict[str, dict[str, Any]] = {}
    for name in check_names:
        check_errors = [entry for entry in errors if entry["check"] == name]
        check_warnings = [entry for entry in warnings if entry["check"] == name]
        checks[name] = {
            "passed": not check_errors,
            "error_count": len(check_errors),
            "warning_count": len(check_warnings),
        }
    finite_output = checks["finite_output"]["passed"] and n_npz_checked > 0
    report: dict[str, Any] = {
        "task": "combined_eeg_preprocessing",
        "status": "passed" if not errors and rows and n_npz_checked else "failed",
        "authoritative_split": {
            "path": str(locked_split_path),
            "version": split_version,
        },
        "provenance": provenance,
        "checks": checks,
        "summary": {
            "manifest_rows": len(rows),
            "trials_checked": n_trials_checked,
            "npz_files_checked": n_npz_checked,
            "finite_output": finite_output,
            "critical_errors": len(errors),
            "warnings": len(warnings),
            "max_clip_fraction": max(
                (record["clip_fraction"] for record in clip_records if np.isfinite(record["clip_fraction"])),
                default=None,
            ),
        },
        "issues": issues,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def build_subject_splits(rows: list[dict[str, Any]], seed: int, val_fraction: float, test_fraction: float) -> dict[str, Any]:
    require(0 <= val_fraction < 0.5 and 0 <= test_fraction < 0.5 and val_fraction + test_fraction < 0.8,
            "Split fractions must be non-negative and leave a training majority.")
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        key = row["subject_group_id"]
        if key not in by_dataset[row["dataset"]]:
            by_dataset[row["dataset"]].append(key)
    result: dict[str, Any] = {
        "seed": seed,
        "unit": "subject_group_id",
        "purpose": "preprocessing_qc_only",
        "authoritative_for_training": False,
        "datasets": {},
    }
    for offset, (dataset, subjects) in enumerate(sorted(by_dataset.items())):
        subjects = sorted(subjects)
        rng = np.random.default_rng(seed + offset)
        ordered = [subjects[index] for index in rng.permutation(len(subjects))]
        n_test = max(1, round(len(ordered) * test_fraction))
        n_val = max(1, round(len(ordered) * val_fraction))
        n_val = min(n_val, len(ordered) - n_test - 1)
        result["datasets"][dataset] = {
            "train": ordered[n_val + n_test:],
            "validation": ordered[:n_val],
            "test": ordered[n_val:n_val + n_test],
        }
    return result


def qc_rows(dataset: str, subject_group_id: str, eeg: np.ndarray, valid_lengths: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for channel_index, channel_name in enumerate(COMMON_CHANNELS):
        valid_samples = [eeg[index, channel_index, :length] for index, length in enumerate(valid_lengths) if length > 0]
        values = np.concatenate(valid_samples) if valid_samples else np.asarray([], dtype=np.float32)
        rows.append({
            "dataset": dataset,
            "subject_group_id": subject_group_id,
            "channel": channel_name,
            "n_values": int(values.size),
            "mean": float(np.mean(values)) if values.size else float("nan"),
            "std": float(np.std(values)) if values.size else float("nan"),
            "min": float(np.min(values)) if values.size else float("nan"),
            "max": float(np.max(values)) if values.size else float("nan"),
            "clip_fraction": float(np.mean(np.abs(values) >= 11.999)) if values.size else float("nan"),
            "has_nan_or_inf": bool(not np.isfinite(values).all()) if values.size else True,
        })
    return rows


def add_prepared_dataset(
    dataset: str,
    dataset_root: Path,
    output_root: Path,
    thinking_stage: str,
    baseline_stage: str,
    skip_subjects: set[str],
    audio_cache: dict[Path, AudioRecord],
    skip_audio: bool,
    overwrite: bool,
    manifest_rows: list[dict[str, Any]],
    qcs: list[dict[str, Any]],
) -> None:
    subject_files = sorted((dataset_root / "subjects").glob("*.npz"))
    for number, source in enumerate(subject_files, start=1):
        subject_id = source.stem
        if subject_id in skip_subjects:
            print(f"[{dataset}] skip source-marked anomalous subject {subject_id}")
            continue
        print(f"[{dataset}] {number}/{len(subject_files)} subject {subject_id}")
        output_path = output_root / "subjects" / dataset / f"{subject_id}.npz"
        source_metadata = np.load(source, allow_pickle=True)
        trial_indices = source_metadata["trial_indices"]
        labels = source_metadata["labels"]
        audio_relpaths = source_metadata["audio_relpaths"]
        if output_path.exists() and not overwrite:
            print(f"[{dataset}] resume cached output for subject {subject_id}")
            cached = np.load(output_path, allow_pickle=False)
            eeg = np.asarray(cached["eeg"], dtype=np.float32)
            valid_lengths = np.asarray(cached["valid_lengths"], dtype=np.int32)
            require(len(eeg) == len(trial_indices), f"Cached output does not match source trial count: {output_path}")
        else:
            eeg, valid_lengths, _, _, _ = normalize_stage_bundle(source, thinking_stage, baseline_stage)
            write_npz(output_path, {
                "eeg": eeg,
                "valid_lengths": valid_lengths,
                "trial_indices": np.asarray(trial_indices, dtype=np.int32),
                "labels": np.asarray(labels),
                "channel_names": np.asarray(COMMON_CHANNELS),
                "eeg_sfreq_hz": np.asarray([TARGET_EEG_SFREQ], dtype=np.int32),
                "dataset": np.asarray([dataset]),
                "source_stage": np.asarray([thinking_stage]),
            }, overwrite)
        group_id = f"{dataset}:{subject_id}"
        qcs.extend(qc_rows(dataset, group_id, eeg, valid_lengths))
        for index, (trial_index, label, relpath) in enumerate(zip(trial_indices, labels, audio_relpaths)):
            source_audio = (dataset_root / str(relpath)).resolve()
            audio = audio_cache.get(source_audio)
            if audio is None:
                audio = ensure_audio(source_audio, output_root, skip_audio)
                audio_cache[source_audio] = audio
            manifest_rows.append({
                "dataset": dataset,
                "subject_group_id": group_id,
                "subject_recording_id": group_id,
                "trial_index": int(trial_index),
                "sample_key": f"{group_id}:{int(trial_index)}",
                "label": str(label),
                "modality": "imagined_speech",
                "source_stage": thinking_stage,
                "eeg_relpath": str(output_path.relative_to(output_root)),
                "eeg_row": index,
                "eeg_valid_samples": int(valid_lengths[index]),
                "eeg_sfreq_hz": TARGET_EEG_SFREQ,
                "audio_key": audio.key,
                "audio_relpath": audio.output_relpath,
                "audio_valid_samples": audio.n_samples,
                "audio_sfreq_hz": TARGET_AUDIO_SFREQ,
                "audio_pairing": "source-provided",
                "pairing_confidence": "feis_subject_label" if dataset == "feis" else "karaone_same_trial_overt",
            })


def add_ds004306(
    ds_root: Path,
    output_root: Path,
    allowed_modalities: set[str],
    audio_cache: dict[Path, AudioRecord],
    skip_audio: bool,
    overwrite: bool,
    manifest_rows: list[dict[str, Any]],
    qcs: list[dict[str, Any]],
) -> None:
    set_paths = sorted(ds_root.glob("sub-*/ses-*/eeg/*_eeg.set"))
    require(set_paths, f"No ds004306 EEGLAB sets under {ds_root}")
    for number, set_path in enumerate(set_paths, start=1):
        events_path = set_path.with_name(set_path.name.replace("_eeg.set", "_events.tsv"))
        require(events_path.is_file(), f"Missing events TSV for {set_path}")
        match = re.search(r"(sub-\d+)/(ses-\d+)", str(set_path))
        require(match is not None, f"Cannot parse recording ID from {set_path}")
        subject_id, session_id = match.groups()
        group_id = f"ds004306:{subject_id}"
        recording_id = f"ds004306:{subject_id}:{session_id}"
        output_path = output_root / "subjects" / "ds004306" / f"{subject_id}_{session_id}.npz"
        trials = ds_events(events_path, allowed_modalities)
        if output_path.exists() and not overwrite:
            print(f"[ds004306] {number}/{len(set_paths)} {subject_id} {session_id}: resume cached output")
            cached = np.load(output_path, allow_pickle=False)
            eeg = np.asarray(cached["eeg"], dtype=np.float32)
            valid_lengths = np.asarray(cached["valid_lengths"], dtype=np.int32)
            require(len(eeg) == len(trials), f"Cached output does not match event count: {output_path}")
        else:
            print(f"[ds004306] {number}/{len(set_paths)} {subject_id} {session_id}: reading/filtering/epoching")
            eeg, valid_lengths, trials = process_ds_session(ds_root, set_path, events_path, allowed_modalities)
            write_npz(output_path, {
                "eeg": eeg,
                "valid_lengths": valid_lengths,
                "trial_indices": np.arange(len(trials), dtype=np.int32),
                "labels": np.asarray([trial["label"] for trial in trials]),
                "modalities": np.asarray([trial["modality"] for trial in trials]),
                "event_types": np.asarray([trial["event_type"] for trial in trials]),
                "channel_names": np.asarray(COMMON_CHANNELS),
                "eeg_sfreq_hz": np.asarray([TARGET_EEG_SFREQ], dtype=np.int32),
                "dataset": np.asarray(["ds004306"]),
            }, overwrite)
        qcs.extend(qc_rows("ds004306", group_id, eeg, valid_lengths))
        for index, trial in enumerate(trials):
            candidates = ds_audio_sources(ds_root, trial["label"])
            records: list[AudioRecord] = []
            for candidate in candidates:
                record = audio_cache.get(candidate.resolve())
                if record is None:
                    record = ensure_audio(candidate, output_root, skip_audio)
                    audio_cache[candidate.resolve()] = record
                records.append(record)
            canonical = records[0]
            manifest_rows.append({
                "dataset": "ds004306",
                "subject_group_id": group_id,
                "subject_recording_id": recording_id,
                "trial_index": index,
                "sample_key": f"{recording_id}:{index}",
                "label": trial["label"],
                "modality": trial["modality"],
                "source_stage": "imagination",
                "event_type": trial["event_type"],
                "eeg_relpath": str(output_path.relative_to(output_root)),
                "eeg_row": index,
                "eeg_valid_samples": int(valid_lengths[index]),
                "eeg_sfreq_hz": TARGET_EEG_SFREQ,
                "audio_key": canonical.key,
                "audio_relpath": canonical.output_relpath,
                "audio_candidate_keys": ";".join(record.key for record in records),
                "audio_candidate_relpaths": ";".join(record.output_relpath for record in records),
                "audio_valid_samples": canonical.n_samples,
                "audio_sfreq_hz": TARGET_AUDIO_SFREQ,
                "audio_pairing": "published_category_level_candidates",
                "pairing_confidence": "weak_category_level",
            })


def validate_layout(data_root: Path) -> None:
    for name in ("feis", "karaone", "ds004306"):
        require((data_root / name).is_dir(), f"Expected {data_root / name}")
    require(list((data_root / "feis" / "subjects").glob("*.npz")), "No FEIS subject NPZ files")
    require(list((data_root / "karaone" / "subjects").glob("*.npz")), "No KARA ONE subject NPZ files")
    require(list((data_root / "ds004306").glob("sub-*/ses-*/eeg/*_eeg.set")), "No ds004306 SET files")
    require(shutil.which("ffmpeg") is not None, "ffmpeg is required for audio standardisation")


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    locked_split_path = args.locked_split.resolve()
    if args.verify_only:
        report = verify_preprocessed_outputs(output_root, locked_split_path)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "passed" else 2
    validate_layout(data_root)
    if args.dry_run:
        print(json.dumps({
            "data_root": str(data_root), "output_root": str(output_root),
            "common_channels": COMMON_CHANNELS, "target_eeg_shape": [14, TARGET_EEG_SAMPLES],
            "ds_modalities": list(args.ds_modalities),
            "message": "Input layout validated; no files written.",
        }, indent=2))
        return 0
    output_root.mkdir(parents=True, exist_ok=True)
    selected = set(args.datasets)
    all_datasets = {"feis", "karaone", "ds004306"}
    existing_manifest = read_csv_rows(output_root / "manifests" / "unified_trials.csv")
    existing_qc = read_csv_rows(output_root / "qc" / "channel_qc.csv")
    if selected != all_datasets and not existing_manifest:
        raise RuntimeError("Selective rebuild requires an existing unified_trials.csv to preserve omitted datasets")
    manifest_rows: list[dict[str, Any]] = [row for row in existing_manifest if row.get("dataset") not in selected]
    for row in manifest_rows:
        row.setdefault("sample_key", f"{row.get('subject_recording_id', row.get('subject_group_id', 'unknown'))}:{row.get('trial_index', 'unknown')}")
    quality_rows: list[dict[str, Any]] = [row for row in existing_qc if row.get("dataset") not in selected]
    audio_cache: dict[Path, AudioRecord] = {}
    skip_feis = set() if args.include_feis_subject_05 else DEFAULT_EXCLUDED_FEIS_SUBJECTS
    if "feis" in selected:
        add_prepared_dataset(
            "feis", data_root / "feis", output_root, "thinking", "resting", skip_feis,
            audio_cache, args.skip_audio, args.overwrite, manifest_rows, quality_rows,
        )
    if "karaone" in selected:
        add_prepared_dataset(
            "karaone", data_root / "karaone", output_root, "thinking", "clearing", set(),
            audio_cache, args.skip_audio, args.overwrite, manifest_rows, quality_rows,
        )
    if "ds004306" in selected:
        add_ds004306(
            data_root / "ds004306", output_root, set(args.ds_modalities), audio_cache,
            args.skip_audio, args.overwrite, manifest_rows, quality_rows,
        )
    write_csv(output_root / "manifests" / "unified_trials.csv", manifest_rows, args.overwrite)
    write_csv(output_root / "qc" / "channel_qc.csv", quality_rows, args.overwrite)
    splits = build_subject_splits(manifest_rows, args.split_seed, args.val_fraction, args.test_fraction)
    split_path = output_root / "manifests" / "subject_holdout_splits.json"
    if split_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {split_path}; pass --overwrite to replace generated data.")
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    summary = {
        "target_eeg_sfreq_hz": TARGET_EEG_SFREQ,
        "target_eeg_samples": TARGET_EEG_SAMPLES,
        "target_audio_sfreq_hz": TARGET_AUDIO_SFREQ,
        "common_channels": COMMON_CHANNELS,
        "trial_counts": dict(Counter(row["dataset"] for row in manifest_rows)),
        "modality_counts": dict(Counter(f"{row['dataset']}:{row['modality']}" for row in manifest_rows)),
        "n_subject_groups": len({row["subject_group_id"] for row in manifest_rows}),
        "n_audio_cache_entries": len({row.get("audio_key", "") for row in manifest_rows if row.get("audio_key")}),
        "excluded_feis_subjects": sorted(skip_feis),
        "ds_modalities": list(args.ds_modalities),
        "notes": [
            "All EEG arrays use COMMON_CHANNELS order and 5-second 256-Hz windows; V1 training uses the first 768 samples.",
            "ds004306 windows are stopped before the next perception event and padded; use eeg_valid_samples as a mask.",
            "ds004306 audio is category-level candidate supervision, not confirmed one-to-one trial audio.",
        ],
    }
    (output_root / "qc" / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    verification = verify_preprocessed_outputs(output_root, locked_split_path)
    print(json.dumps(summary, indent=2))
    if verification["status"] != "passed":
        print(json.dumps(verification, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise

#!/usr/bin/env python3
"""Build full-length derived files for downloaded OpenNeuro subsets.

This is the training-oriented OpenNeuro conversion entrypoint. It keeps the
full available recordings and writes derived arrays plus manifest files.

Default output:
  data/derived/openneuro_full/

The output directory is under `data/`, which is ignored by git in this project.
"""

from __future__ import annotations

import argparse
import json
import shutil
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_DS006104 = Path("data/raw/openneuro/ds006104_datalad")
DEFAULT_DS005345 = Path("data/raw/openneuro/ds005345_datalad")
DEFAULT_OUT = Path("data/derived/openneuro_full")


@dataclass
class RecordingArtifact:
    dataset: str
    subject: str
    session: str
    task: str
    source_eeg: str
    output_npz: str
    eeg_kind: str
    sfreq: float
    n_channels: int
    n_times: int
    duration_sec: float
    n_epochs: int | None = None
    events_source: str | None = None
    events_output: str | None = None
    channels_source: str | None = None
    channels_output: str | None = None


@dataclass
class AudioArtifact:
    dataset: str
    stream: str
    source: str
    output_features: str
    sample_rate: int
    channels: int
    duration_sec: float
    rms: float
    peak_abs: float


def relabel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_if_available(src: Path, dst: Path) -> str | None:
    if not src.exists() or src.stat().st_size == 0:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return str(dst)
    shutil.copy2(src, dst)
    return str(dst)


def iter_files(root: Path, suffix: str) -> Iterable[Path]:
    if not root.exists():
        return
    for path in sorted(root.rglob(f"*{suffix}")):
        if ".git" in path.parts or not path.is_file() or path.stat().st_size == 0:
            continue
        yield path


def parse_bids_name(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {"subject": "unknown", "session": "na", "task": "unknown", "run": "na"}
    stem = path.name
    for ext in [".edf", ".fif", ".vhdr", ".eeg", ".npz"]:
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    for part in stem.split("_"):
        if "-" not in part:
            continue
        key, value = part.split("-", 1)
        if key == "sub":
            fields["subject"] = f"sub-{value}"
        elif key == "ses":
            fields["session"] = f"ses-{value}"
        elif key == "task":
            fields["task"] = value
        elif key == "run":
            fields["run"] = f"run-{value}"
    return fields


def load_eeg(path: Path):
    import mne

    suffix = path.suffix.lower()
    if suffix == ".edf":
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        return raw.get_data().astype("float32"), float(raw.info["sfreq"]), list(raw.ch_names), "raw"
    if suffix == ".vhdr":
        raw = mne.io.read_raw_brainvision(path, preload=True, verbose=False)
        return raw.get_data().astype("float32"), float(raw.info["sfreq"]), list(raw.ch_names), "raw"
    if suffix == ".fif":
        try:
            raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
            return raw.get_data().astype("float32"), float(raw.info["sfreq"]), list(raw.ch_names), "raw"
        except ValueError as exc:
            if "No raw data" not in str(exc):
                raise
            epochs = mne.read_epochs(path, preload=True, verbose=False)
            return epochs.get_data().astype("float32"), float(epochs.info["sfreq"]), list(epochs.ch_names), "epochs"
    raise ValueError(f"Unsupported EEG format: {path}")


def maybe_resample_array(data: np.ndarray, sfreq: float, target_hz: float | None) -> tuple[np.ndarray, float]:
    if target_hz is None or float(sfreq) == float(target_hz):
        return data, sfreq
    from scipy.signal import resample_poly

    source = int(round(sfreq))
    target = int(round(target_hz))
    gcd = int(np.gcd(source, target))
    up = target // gcd
    down = source // gcd
    return resample_poly(data, up=up, down=down, axis=-1).astype("float32"), float(target_hz)


def export_raw_npz(
    source: Path,
    out_npz: Path,
    resample_hz: float | None,
    l_freq: float | None,
    h_freq: float | None,
) -> dict[str, object]:
    if l_freq is not None or h_freq is not None:
        import mne

        suffix = source.suffix.lower()
        if suffix == ".edf":
            raw = mne.io.read_raw_edf(source, preload=True, verbose=False)
        elif suffix == ".vhdr":
            raw = mne.io.read_raw_brainvision(source, preload=True, verbose=False)
        elif suffix == ".fif":
            raw = mne.io.read_raw_fif(source, preload=True, verbose=False)
        else:
            raise ValueError(f"Unsupported EEG format for filtering: {source}")
        raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
        data = raw.get_data().astype("float32")
        sfreq = float(raw.info["sfreq"])
        ch_names = list(raw.ch_names)
        eeg_kind = "raw"
    else:
        data, sfreq, ch_names, eeg_kind = load_eeg(source)
    data, sfreq = maybe_resample_array(data, sfreq, resample_hz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        eeg=data,
        sfreq=np.array(float(sfreq), dtype="float32"),
        ch_names=np.array(ch_names, dtype=object),
        eeg_kind=np.array(eeg_kind, dtype=object),
        source=np.array(str(source), dtype=object),
    )
    n_epochs = int(data.shape[0]) if data.ndim == 3 else None
    n_channels = int(data.shape[-2])
    n_times = int(data.shape[-1])
    return {
        "eeg_kind": eeg_kind,
        "sfreq": float(sfreq),
        "n_channels": n_channels,
        "n_times": n_times,
        "n_epochs": n_epochs,
        "duration_sec": float(n_times / sfreq),
    }


def matching_sidecar(eeg_path: Path, suffix: str) -> Path:
    name = eeg_path.name
    for eeg_suffix in ["_eeg.edf", "_eeg.fif", "_eeg.vhdr", "_eeg.eeg", "_eeg_preprocessed.fif"]:
        if name.endswith(eeg_suffix):
            return eeg_path.with_name(name[: -len(eeg_suffix)] + suffix)
    return eeg_path.with_suffix(suffix)


def process_ds006104(
    root: Path,
    out_dir: Path,
    resample_hz: float | None,
    l_freq: float | None,
    h_freq: float | None,
) -> list[RecordingArtifact]:
    artifacts: list[RecordingArtifact] = []
    for eeg_path in iter_files(root, ".edf"):
        fields = parse_bids_name(eeg_path)
        base = eeg_path.name.replace("_eeg.edf", "").replace(".edf", "")
        rec_dir = out_dir / "ds006104" / fields["subject"] / fields["session"]
        out_npz = rec_dir / f"{base}_full_eeg.npz"
        stats = export_raw_npz(eeg_path, out_npz, resample_hz, l_freq, h_freq)

        events_src = matching_sidecar(eeg_path, "_events.tsv")
        channels_src = matching_sidecar(eeg_path, "_channels.tsv")
        events_out = copy_if_available(events_src, rec_dir / events_src.name)
        channels_out = copy_if_available(channels_src, rec_dir / channels_src.name)

        artifacts.append(
            RecordingArtifact(
                dataset="ds006104",
                subject=fields["subject"],
                session=fields["session"],
                task=fields["task"],
                source_eeg=str(eeg_path),
                output_npz=str(out_npz),
                eeg_kind=str(stats["eeg_kind"]),
                sfreq=float(stats["sfreq"]),
                n_channels=int(stats["n_channels"]),
                n_times=int(stats["n_times"]),
                duration_sec=float(stats["duration_sec"]),
                n_epochs=stats["n_epochs"],
                events_source=str(events_src) if events_src.exists() else None,
                events_output=events_out,
                channels_source=str(channels_src) if channels_src.exists() else None,
                channels_output=channels_out,
            )
        )
    return artifacts


def read_wav_stats(path: Path) -> dict[str, object]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sr = wf.getframerate()
        width = wf.getsampwidth()
        frames = wf.getnframes()
        blob = wf.readframes(frames)
    if width == 2:
        vals = np.frombuffer(blob, dtype="<i2").astype("float32") / 32768.0
    elif width == 4:
        vals = np.frombuffer(blob, dtype="<i4").astype("float32") / 2147483648.0
    elif width == 1:
        vals = (np.frombuffer(blob, dtype="uint8").astype("float32") - 128.0) / 128.0
    else:
        vals = np.array([], dtype="float32")
    if channels > 1 and vals.size:
        vals = vals.reshape(-1, channels).mean(axis=1)
    return {
        "sample_rate": int(sr),
        "channels": int(channels),
        "duration_sec": float(frames / sr) if sr else 0.0,
        "rms": float(np.sqrt(np.mean(vals**2))) if vals.size else 0.0,
        "peak_abs": float(np.max(np.abs(vals))) if vals.size else 0.0,
    }


def process_ds005345(
    root: Path,
    out_dir: Path,
    resample_hz: float | None,
    l_freq: float | None,
    h_freq: float | None,
) -> tuple[list[RecordingArtifact], list[AudioArtifact]]:
    rec_artifacts: list[RecordingArtifact] = []
    audio_artifacts: list[AudioArtifact] = []
    ds_out = out_dir / "ds005345"

    for wav_path in sorted((root / "stimuli").glob("*.wav")):
        if not wav_path.is_file() or wav_path.stat().st_size == 0:
            continue
        stream = wav_path.stem
        audio_dir = ds_out / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        stats = read_wav_stats(wav_path)
        out_json = audio_dir / f"{stream}_audio_features.json"
        write_json(out_json, {"source": str(wav_path), **stats})
        audio_artifacts.append(
            AudioArtifact(
                dataset="ds005345",
                stream=stream,
                source=str(wav_path),
                output_features=str(out_json),
                sample_rate=int(stats["sample_rate"]),
                channels=int(stats["channels"]),
                duration_sec=float(stats["duration_sec"]),
                rms=float(stats["rms"]),
                peak_abs=float(stats["peak_abs"]),
            )
        )

    annotation_out = ds_out / "annotation"
    for csv_path in sorted((root / "annotation").glob("*.csv")):
        copy_if_available(csv_path, annotation_out / csv_path.name)

    for eeg_path in iter_files(root / "derivatives", ".fif"):
        fields = parse_bids_name(eeg_path)
        base = eeg_path.name.replace("_eeg_preprocessed.fif", "").replace(".fif", "")
        rec_dir = ds_out / fields["subject"] / fields["run"]
        out_npz = rec_dir / f"{base}_full_eeg.npz"
        stats = export_raw_npz(eeg_path, out_npz, resample_hz, l_freq, h_freq)
        rec_artifacts.append(
            RecordingArtifact(
                dataset="ds005345",
                subject=fields["subject"],
                session=fields["run"],
                task=fields["task"],
                source_eeg=str(eeg_path),
                output_npz=str(out_npz),
                eeg_kind=str(stats["eeg_kind"]),
                sfreq=float(stats["sfreq"]),
                n_channels=int(stats["n_channels"]),
                n_times=int(stats["n_times"]),
                duration_sec=float(stats["duration_sec"]),
                n_epochs=stats["n_epochs"],
            )
        )
    return rec_artifacts, audio_artifacts


def write_tables(out_dir: Path, recordings: list[RecordingArtifact], audio: list[AudioArtifact]) -> None:
    write_json(out_dir / "recordings_manifest.json", [asdict(x) for x in recordings])
    write_json(out_dir / "audio_manifest.json", [asdict(x) for x in audio])
    if recordings:
        pd.DataFrame([asdict(x) for x in recordings]).to_csv(out_dir / "recordings_manifest.csv", index=False)
    if audio:
        pd.DataFrame([asdict(x) for x in audio]).to_csv(out_dir / "audio_manifest.csv", index=False)
    lines = [
        "# Full OpenNeuro Derivatives",
        "",
        "This directory contains full-length derived arrays for the local DataLad downloads.",
        "",
        "## Files",
        "",
        "- `recordings_manifest.csv/json`: exported EEG recordings.",
        "- `audio_manifest.csv/json`: exported full-length audio statistics.",
        "- `*_full_eeg.npz`: compressed arrays with `eeg`, `sfreq`, `ch_names`, `source`.",
        "- copied `events.tsv`, `channels.tsv`, and ds005345 annotation CSV files where available.",
        "",
        "## EEG array schema",
        "",
        "```text",
        "eeg: float32 [channels, time] for raw or [epochs, channels, time] for epochs",
        "sfreq: float32 scalar",
        "ch_names: object array [channels]",
        "eeg_kind: `raw` or `epochs`",
        "source: original raw path",
        "```",
        "",
        "## Counts",
        "",
        f"- EEG recordings: {len(recordings)}",
        f"- audio streams: {len(audio)}",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds006104-root", type=Path, default=DEFAULT_DS006104)
    parser.add_argument("--ds005345-root", type=Path, default=DEFAULT_DS005345)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--datasets", default="ds006104,ds005345")
    parser.add_argument("--resample-hz", type=float, default=250.0)
    parser.add_argument("--l-freq", type=float, default=None)
    parser.add_argument("--h-freq", type=float, default=None)
    args = parser.parse_args()

    selected = {x.strip() for x in args.datasets.split(",") if x.strip()}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    recordings: list[RecordingArtifact] = []
    audio: list[AudioArtifact] = []
    if "ds006104" in selected:
        recordings.extend(
            process_ds006104(
                args.ds006104_root,
                args.out_dir,
                args.resample_hz,
                args.l_freq,
                args.h_freq,
            )
        )
    if "ds005345" in selected:
        ds005345_recordings, ds005345_audio = process_ds005345(
            args.ds005345_root,
            args.out_dir,
            args.resample_hz,
            args.l_freq,
            args.h_freq,
        )
        recordings.extend(ds005345_recordings)
        audio.extend(ds005345_audio)

    write_tables(args.out_dir, recordings, audio)
    print(f"Wrote {len(recordings)} full EEG recordings and {len(audio)} audio summaries to {args.out_dir}")


if __name__ == "__main__":
    main()

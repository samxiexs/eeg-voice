#!/usr/bin/env python3
"""Build variable-montage Track B without changing the 14-channel Track A."""
from __future__ import annotations

import argparse
import csv
import json
import re
import site
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm


BUNDLE = Path(__file__).resolve().parents[2]
APP = BUNDLE / "app"
if str(APP) not in sys.path: sys.path.insert(0, str(APP))
if str(BUNDLE / "scripts") not in sys.path: sys.path.insert(0, str(BUNDLE / "scripts"))

from src.open_vocab_0722.data import build_montage_registry_payload, channel_type, parse_asa_elc, read_csv, resolve_config_path  # noqa: E402


TARGET_RATE = 256
TARGET_SAMPLES = 1280


def standard_1005_path() -> Path:
    candidates = [Path(root) / "mne/channels/data/montages/standard_1005.elc" for root in site.getsitepackages()]
    for path in candidates:
        if path.is_file(): return path
    raise FileNotFoundError("MNE standard_1005.elc is required for Track B coordinates")


def valid_eeg_name(name: str) -> bool:
    upper = str(name).upper()
    return channel_type(upper) == "eeg" and "EOG" not in upper and not upper.startswith(("ECG", "EMG", "EXG"))


def pad(data: np.ndarray, valid: int) -> tuple[np.ndarray, int]:
    length = min(max(int(valid), 1), data.shape[-1], TARGET_SAMPLES)
    output = np.zeros((data.shape[0], TARGET_SAMPLES), dtype=np.float32)
    output[:, :length] = data[:, :length]
    return output, length


def prepared_subject(source: Path, thinking_stage: str, baseline_stage: str) -> dict[str, np.ndarray]:
    import preprocess_combined_eeg as base
    with np.load(source, allow_pickle=True) as raw:
        names = [str(value) for value in raw["channel_names"]]
        indices = [index for index, name in enumerate(names) if valid_eeg_name(name)]
        selected_names = [names[index] for index in indices]
        thinking = np.asarray(raw[f"stage__{thinking_stage}"][:, indices], dtype=np.float32)
        baseline = np.asarray(raw[f"stage__{baseline_stage}"][:, indices], dtype=np.float32)
        valid = np.asarray(raw[f"stage__{thinking_stage}__valid_lengths"], dtype=np.int32)
        baseline_valid = np.asarray(raw[f"stage__{baseline_stage}__valid_lengths"], dtype=np.int32)
        normalized = base.robust_baseline_normalise(thinking, baseline, baseline_valid)
        windows, lengths = zip(*(pad(normalized[index], int(valid[index])) for index in range(len(normalized))))
        return {
            "eeg": np.stack(windows), "valid_lengths": np.asarray(lengths),
            "trial_indices": np.asarray(raw["trial_indices"], dtype=np.int32), "labels": np.asarray(raw["labels"]),
            "channel_names": np.asarray(selected_names), "eeg_sfreq_hz": np.asarray([TARGET_RATE]),
        }


def ds_session(ds_root: Path, set_path: Path, events_path: Path) -> dict[str, np.ndarray]:
    import preprocess_combined_eeg as base
    trials = base.ds_events(events_path, {"auditory"})
    baseline_start, baseline_stop = base.ds_baseline_interval(events_path)
    with base.staged_eeglab_raw(ds_root, set_path) as raw:
        names = [name for name in raw.ch_names if valid_eeg_name(name)]
        raw.pick(names); raw.load_data(verbose="error")
        raw.notch_filter(freqs=[50.0], verbose="error"); raw.filter(l_freq=1.0, h_freq=40.0, verbose="error")
        raw.resample(TARGET_RATE, verbose="error"); raw.set_eeg_reference("average", projection=False, verbose="error")
        baseline = raw.get_data(start=max(0, round(baseline_start * TARGET_RATE)), stop=min(raw.n_times, round(baseline_stop * TARGET_RATE))).astype(np.float32)
        center = np.median(baseline, axis=1, keepdims=True)
        mad = 1.4826 * np.median(np.abs(baseline - center), axis=1, keepdims=True)
        fallback = np.std(baseline, axis=1, keepdims=True)
        scale = np.where(mad > 1e-8, mad, fallback); scale = np.where(scale > 1e-8, scale, 1.0)
        windows, lengths = [], []
        for trial in trials:
            start = max(0, round(trial["onset"] * TARGET_RATE)); stop = min(raw.n_times, round(trial["stop"] * TARGET_RATE))
            window, length = pad(np.clip((raw.get_data(start=start, stop=stop).astype(np.float32) - center) / scale, -12.0, 12.0), stop - start)
            windows.append(window); lengths.append(length)
    return {
        "eeg": np.stack(windows), "valid_lengths": np.asarray(lengths, dtype=np.int32),
        "trial_indices": np.arange(len(trials), dtype=np.int32), "labels": np.asarray([trial["label"] for trial in trials]),
        "channel_names": np.asarray(names), "eeg_sfreq_hz": np.asarray([TARGET_RATE]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--datasets", nargs="+", choices=("feis", "karaone", "ds004306"), default=("feis", "karaone", "ds004306"))
    args = parser.parse_args()
    config_path = args.config.resolve(); cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_root = resolve_config_path(config_path, cfg["data"]["eeg_output_root"])
    output_root = resolve_config_path(config_path, cfg["data"]["track_b_output_root"])
    source_rows = read_csv(source_root / "manifests/unified_trials.csv")
    data_root = BUNDLE / "data"; mapping: dict[str, str] = {}
    groups: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows: groups[row["eeg_relpath"]].append(row)
    for relative, rows in tqdm(sorted(groups.items()), desc="[0722 Track B] recordings"):
        dataset = rows[0]["dataset"]
        if dataset not in args.datasets: continue
        old = source_root / relative
        destination = output_root / relative
        if destination.exists() and not args.overwrite:
            mapping[relative] = relative; continue
        if dataset in {"feis", "karaone"}:
            subject = Path(relative).stem
            source = data_root / dataset / "subjects" / f"{subject}.npz"
            payload = prepared_subject(source, "thinking", "resting" if dataset == "feis" else "clearing")
        else:
            match = re.search(r"(sub-\d+)_(ses-\d+)", Path(relative).stem)
            if not match: raise ValueError(f"Cannot parse ds session from {relative}")
            subject, session = match.groups()
            set_path = next((data_root / "ds004306" / subject / session / "eeg").glob("*_eeg.set"))
            payload = ds_session(data_root / "ds004306", set_path, set_path.with_name(set_path.name.replace("_eeg.set", "_events.tsv")))
        if payload["eeg"].shape[0] != len(rows): raise ValueError(f"Track B trial count mismatch for {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(destination, **payload)
        mapping[relative] = str(destination.relative_to(output_root))
    missing = sorted(set(groups) - set(mapping))
    if missing: raise ValueError(f"Partial Track B rebuild would create an incomplete manifest: {missing}")
    new_rows = [{**row, "eeg_relpath": mapping[row["eeg_relpath"]]} for row in source_rows]
    manifest = output_root / "manifests/unified_trials.csv"; manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(new_rows[0])); writer.writeheader(); writer.writerows(new_rows)
    # Build the registry directly from the produced payloads. No interpolation
    # is permitted: every retained channel must have a standard coordinate.
    registry = build_montage_registry_payload(output_root, new_rows, parse_asa_elc(standard_1005_path()))
    registry_path = resolve_config_path(config_path, cfg["data"]["track_b_montage_registry"])
    registry_path.parent.mkdir(parents=True, exist_ok=True); registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "schema_version": "openvoice-track-b-preprocessing-v1", "output_root": str(output_root),
        "registry": str(registry_path), "recordings": len(mapping), "trials": len(new_rows),
        "channel_counts": sorted({len(value["channel_names"]) for value in registry["montages"].values()}),
        "interpolation_used": False, "eog_reference_aux_excluded": True,
    }
    track_b_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    track_b_cfg["version"] = "openvoice-eeg-0722-track-b-v1"
    track_b_cfg["data"]["track"] = "track_b_variable_montage"
    track_b_cfg["data"]["eeg_output_root"] = str(output_root)
    track_b_cfg["data"]["montage_registry"] = str(registry_path)
    track_b_artifacts = (BUNDLE / "artifacts/open_vocab_0722_track_b_v1").resolve()
    track_b_cfg["paths"]["output_root"] = str(track_b_artifacts)
    for name, relative_path in {
        "audio_checkpoint": "audio/checkpoints/best.pt",
        "eeg_pretrain_checkpoint": "eeg_pretrain/checkpoints/best.pt",
        "eeg_checkpoint": "eeg/checkpoints/selected.pt",
        "validation_report": "evaluation/validation_report.json",
        "validation_gate": "evaluation/validation_gate.json",
        "leakage_audit": "audits/data_leakage_audit.json",
    }.items():
        track_b_cfg["paths"][name] = str(track_b_artifacts / relative_path)
    for name in ("project_audio_cache", "teacher_cache", "public_audio_manifest", "public_audio_cache", "encodec_model"):
        track_b_cfg["paths"][name] = str(resolve_config_path(config_path, cfg["paths"][name]))
    runtime_config = output_root / "open_vocab_0722_track_b_v1.yaml"
    runtime_config.write_text(yaml.safe_dump(track_b_cfg, sort_keys=False), encoding="utf-8")
    report["runtime_config"] = str(runtime_config)
    (output_root / "track_b_preprocessing.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()

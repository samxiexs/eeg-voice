#!/usr/bin/env python3
"""Reproducible DS004940 training-data preparation (v3).

This program deliberately separates *audit* (inventory and immutable locks),
*make-splits*, *build*, *validate*, and *fit-normalizer*.  It never edits a
raw BIDS input.  The only preprocessing profile implemented here is the
project's ``harmonized_v3`` profile; it is not represented as official data
preprocessing.

Install the optional runtime with ``pip install -r requirements-preprocess.txt``.
The pure provenance/split helpers have no optional-dependency import at module
load time, so they are also directly unit-testable.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CONFIG = ROOT / "configs" / "training_data_v3.yaml"
SPLIT_ALGORITHM = "balanced-greedy-v3-subject-content-waveform-label"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def channel_order_hash(channels: Iterable[str]) -> str:
    return sha256_bytes(("\n".join(channels) + "\n").encode("utf-8"))


def round_half_up(value: float) -> int:
    """Round non-negative values; exact .5 is rounded upward, never bankers."""
    if value < 0:
        return -round_half_up(-value)
    return math.floor(value + 0.5)


def trial_id(dataset: str, subject: str, task: str, run: str, event_row: int, onset: float) -> str:
    text = f"{dataset}|{subject}|{task}|{run}|{event_row}|{onset:.9f}"
    return f"{dataset}-{sha256_bytes(text.encode())[:20]}"


def clean_perception_mask(target_length: int, start: int, end: int, mixed: bool) -> list[bool]:
    if not mixed:
        return [True] * target_length
    return [start <= i < end for i in range(target_length)]


def acoustic_supervision_mask(target_length: int, valid_length: int, zero_index: int,
                              duration_seconds: float | str, target_sfreq: int,
                              pairing: str) -> list[bool]:
    """Mask only the native presented-audio interval for verified pairs."""
    mask = [False] * target_length
    if pairing != "verified_exact":
        return mask
    try:
        duration = float(duration_seconds)
    except (TypeError, ValueError):
        return mask
    if not math.isfinite(duration) or duration <= 0:
        return mask
    start = max(0, int(zero_index))
    end = min(int(valid_length), target_length, start + math.ceil(duration * target_sfreq))
    for index in range(start, max(start, end)):
        mask[index] = True
    return mask


def balanced_group_assignment(weights: dict[str, int], folds: int, seed: str, namespace: str) -> dict[str, dict[str, Any]]:
    """Deterministic LPT placement; weights then stable hashes define ordering."""
    if folds < 2:
        raise ValueError("folds must be >= 2")
    fold_weight = [0] * folds
    fold_groups = [0] * folds
    groups = sorted(weights, key=lambda g: (-weights[g], sha256_bytes(f"{namespace}|{seed}|{g}".encode()), g))
    result: dict[str, dict[str, Any]] = {}
    for position, group in enumerate(groups):
        fold = min(range(folds), key=lambda f: (fold_weight[f], fold_groups[f], sha256_bytes(f"{namespace}|{seed}|{group}|{f}".encode()), f))
        result[group] = {"fold": fold, "trial_weight": int(weights[group]), "sort_position": position,
                         "tie_sha256": sha256_bytes(f"{namespace}|{seed}|{group}".encode())}
        fold_weight[fold] += weights[group]
        fold_groups[fold] += 1
    return result


def split_role(protocol: str, fold: int, subject_fold: int, content_fold: int | None,
               supervision_type: str, folds: int = 5) -> tuple[str, str]:
    """Return a frozen subject/content role for audio and label supervision.

    ``label_only`` examples do not have a waveform, but they still have an
    audited linguistic-content identity.  Routing them through that identity
    preserves the same subject x content leakage contract as paired audio.
    """
    val = (fold + 1) % folds
    if protocol == "subject_ood":
        return ("test", "") if subject_fold == fold else (("validation", "") if subject_fold == val else ("train", ""))
    if supervision_type not in {"paired_audio", "weak_audio", "label_only"}:
        return "excluded", "unsupported_supervision"
    if content_fold is None:
        return "excluded", "missing_linguistic_content_group"
    if protocol == "audio_ood":
        return ("test", "") if content_fold == fold else (("validation", "") if content_fold == val else ("train", ""))
    if protocol != "joint_ood":
        raise ValueError(f"unknown protocol {protocol}")
    if subject_fold == fold and content_fold == fold:
        return "test", ""
    if subject_fold == val and content_fold == val:
        return "validation", ""
    held_subject = subject_fold in (fold, val)
    held_content = content_fold in (fold, val)
    if held_subject or held_content:
        return "excluded", "joint_cross_quadrant"
    return "train", ""


def stimulus_content_id(dataset: str, value: str) -> str:
    """Content-level identity, deliberately coarser than the waveform file id."""
    return f"{dataset}:content:{Path(value).stem}"


def parse_bdf_header(path: Path) -> tuple[int, float]:
    """Read duration information without loading raw data, for split-run auditing."""
    with path.open("rb") as handle:
        header = handle.read(256)
    records = int(header[236:244].decode("ascii", "ignore").strip() or "0")
    duration = float(header[244:252].decode("ascii", "ignore").strip() or "1")
    return records, duration


def git_provenance() -> tuple[str, str]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"
    transformation_functions = (
        "round_half_up", "clean_perception_mask",
        "acoustic_supervision_mask", "channel_order_hash", "_descriptive_stats",
        "_atomic_shard", "_raw_to_canonical", "_rows_for_shards", "build",
    )
    current_sources = "".join(
        f"{name}\n{inspect.getsource(globals()[name])}\n" for name in transformation_functions
    )
    loader = HERE / "training_data_loader.py"
    if loader.exists():
        current_sources += f"training_data_loader.py\t{sha256_file(loader)}\n"
    return run(["git", "rev-parse", "HEAD"]), current_sources


def runtime():
    try:
        import yaml  # type: ignore
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install optional runtime: pip install -r requirements-preprocess.txt") from exc
    return yaml, pd


def progress(iterable, *, desc: str, total: int | None = None):
    """Use tqdm when installed, without making provenance helpers depend on it."""
    try:
        from tqdm.auto import tqdm  # type: ignore
        return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True, unit="item")
    except ImportError:
        return iterable


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    """Load a YAML config, optionally extending one local base config.

    The hash covers the fully resolved configuration.  This keeps the v3 file
    concise without allowing an edited v2 base file to evade provenance locks.
    """
    yaml, _ = runtime()
    path = path.resolve()
    raw = path.read_bytes()
    child = yaml.safe_load(raw) or {}
    if child.get("extends"):
        base_path = (path.parent / str(child.pop("extends"))).resolve()
        base, _ = load_config(base_path)
        base = {key: value for key, value in base.items() if not key.startswith("_")}
        config = _deep_merge(base, child)
        sources = [str(base_path), str(path)]
    else:
        config = child
        sources = [str(path)]
    config["_config_path"] = str(path)
    config["_config_sources"] = sources
    resolved = {key: value for key, value in config.items() if not key.startswith("_")}
    config["_config_sha256"] = sha256_bytes(stable_json(resolved))
    return config, config["_config_sha256"]


def output_root(config: dict[str, Any]) -> Path:
    return ROOT / config["output_root"]


def as_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def read_tsv(path: Path, pd):
    return pd.read_csv(path, sep="\t")


def normalise_ds004_channel(name: str) -> str:
    return name.split("_")[0].strip()


def pairing_level(supervision_type: str, audio_semantics: str) -> str:
    if supervision_type == "label_only":
        return "label_only"
    if audio_semantics == "presented_waveform":
        return "verified_exact"
    if supervision_type == "weak_audio":
        return "candidate_filename_timing"
    return "none"


def first_existing(paths: Iterable[Path]) -> Path | None:
    return next((p for p in paths if p.exists()), None)


def find_ds004_audio(root: Path, stim_file: str) -> Path | None:
    base = Path(str(stim_file)).name
    return first_existing([root / "stimuli" / base, root / "stimuli" / "audio" / base, *root.glob(f"**/{base}")])


def source_lock_entry(path: Path, kind: str) -> dict[str, Any]:
    return {"path": as_relative(path), "sha256": sha256_file(path), "bytes": path.stat().st_size, "kind": kind}


def inventory_sha256(root: Path, suffixes: tuple[str, ...] = (".wav",)) -> str:
    """Hash a deterministic path+content inventory, not merely concatenated WAV bytes."""
    entries = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes):
        entries.append(f"{path.relative_to(root).as_posix()}\t{sha256_file(path)}\n")
    return sha256_bytes("".join(entries).encode("utf-8"))


def resume_compatible(attrs: dict[str, Any], *, config_sha: str, source_lock_sha: str,
                      channel_hash: str | None = None, split_hash: str | None = None) -> bool:
    """Single gate for deterministic reuse of derived artifacts."""
    if attrs.get("preprocess_config_sha256", "") != config_sha or attrs.get("source_lock_sha256", "") != source_lock_sha:
        return False
    if channel_hash is not None and attrs.get("channel_order_hash", "") != channel_hash:
        return False
    return split_hash is None or attrs.get("split_index_sha256", "") == split_hash


def semantic_preprocessing_contract(*, config_sha: str, source_lock_sha: str,
                                    split_hash: str, code_diff_hash: str) -> dict[str, str]:
    """Return the provenance fields that can change the EEG values.

    ``code_commit`` is retained as useful lineage metadata, but a repository
    commit can contain a model, documentation, or runner-only edit.  It is
    therefore not a preprocessing compatibility key.  ``code_diff_hash`` is
    derived from the concrete transform functions and loader source and is the
    fail-closed key for an EEG-value-changing implementation edit.
    """
    return {
        "preprocess_config_sha256": str(config_sha),
        "source_lock_sha256": str(source_lock_sha),
        "split_index_sha256": str(split_hash),
        "code_diff_hash": str(code_diff_hash),
    }


def _event_column(frame, candidates: list[str]) -> str | None:
    lower = {str(c).lower(): c for c in frame.columns}
    return next((lower[c.lower()] for c in candidates if c.lower() in lower), None)


def _ds004_recording_runs(events_path: Path, source_sfreq: int) -> list[dict[str, Any]]:
    """Resolve one logical task recording to one or more physical BDF runs."""
    base = events_path.name.removesuffix("_events.tsv")
    exact = events_path.parent / f"{base}_eeg.bdf"
    split = sorted(
        events_path.parent.glob(f"{base}_run-*_eeg.bdf"),
        key=lambda path: int(re.search(r"_run-(\d+)_eeg", path.name).group(1)),
    )
    paths = split or ([exact] if exact.exists() else [])
    offset = 0
    runs = []
    for index, path in enumerate(paths, start=1):
        records, record_duration = parse_bdf_header(path)
        samples = round_half_up(records * record_duration * source_sfreq)
        match = re.search(r"_run-(\d+)_eeg", path.name)
        run = int(match.group(1)) if match else index
        runs.append({"path": path, "run": run, "start_sample": offset,
                     "end_sample": offset + samples, "samples": samples})
        offset += samples
    return runs


def _bad_channels_from_sidecar(path: Path | None, dataset: str) -> list[str]:
    if path is None or not path.exists():
        return []
    _, pd = runtime()
    frame = read_tsv(path, pd)
    status_col = _event_column(frame, ["status"])
    name_col = _event_column(frame, ["name"])
    if status_col is None or name_col is None:
        return []
    bad = frame[frame[status_col].astype(str).str.lower() == "bad"][name_col].astype(str).tolist()
    return [normalise_ds004_channel(name) for name in bad] if dataset == "ds004940" else bad


def _ds004_trial_rows(config: dict[str, Any], lock: dict[str, Any], qc: dict[str, Any]) -> list[dict[str, Any]]:
    _, pd = runtime()
    spec = config["sources"]["ds004940"]
    data_root = ROOT / spec["data_root"]
    epoch = config["harmonized"]["epoch"]["ds004940"]
    fixed_window = bool(epoch.get("fixed_window", False))
    audio_index = {path.name: path for path in (data_root / "stimuli").glob("*.wav")}
    audio_hashes: dict[Path, str] = {}
    rows: list[dict[str, Any]] = []
    event_files = sorted(data_root.glob("sub-*/eeg/*_events.tsv"))
    subject_seen = set()
    for events_path in progress(event_files, desc="audit ds004940 event files"):
        parts = events_path.name.split("_")
        task = next((x.removeprefix("task-") for x in parts if x.startswith("task-")), "unknown")
        subject = next((p.name for p in events_path.parents if p.name.startswith("sub-")), "unknown")
        session = next((p.name for p in events_path.parents if p.name.startswith("ses-")), "")
        subject_seen.add(subject)
        frame = read_tsv(events_path, pd)
        onset_col = _event_column(frame, ["stim_onset_s_", "stim_onset_s", "onset"])
        duration_col = _event_column(frame, ["stim_dur_s_", "stim_dur_s", "duration"])
        stim_col = _event_column(frame, ["stim_file", "stimulus_file"])
        type_col = _event_column(frame, ["type"])
        trial_type_col = _event_column(frame, ["trial_type"])
        if not onset_col:
            qc["exclusions"]["missing_onset"] += len(frame)
            continue
        runs = _ds004_recording_runs(events_path, 512)
        channels_path = events_path.with_name(events_path.name.replace("_events.tsv", "_channels.tsv"))
        channels_path = channels_path if channels_path.exists() else None
        bad_channels = _bad_channels_from_sidecar(channels_path, "ds004940")
        event_sha = sha256_file(events_path)
        channels_sha = sha256_file(channels_path) if channels_path else ""
        if not runs:
            qc["exclusions"]["missing_eeg"] += len(frame)
        for row_index, event in frame.iterrows():
            if type_col and str(event[type_col]) != "Sound":
                qc["exclusions"]["non_stimulus_event"] += 1
                continue
            condition = str(event[trial_type_col]).strip().upper() if trial_type_col else ""
            if condition not in {"NPC", "NPI"}:
                qc["exclusions"]["practice_or_nonexperimental_sound"] += 1
                continue
            try:
                onset = float(event[onset_col])
                stimulus_duration = float(event[duration_col]) if duration_col else float("nan")
            except (TypeError, ValueError):
                qc["exclusions"]["invalid_stimulus_onset"] += 1
                continue
            if not math.isfinite(stimulus_duration) or stimulus_duration <= 0:
                qc["exclusions"]["invalid_stimulus_duration"] += 1
                continue
            audio = audio_index.get(Path(str(event[stim_col])).name) if stim_col and str(event[stim_col]) not in ("nan", "") else None
            global_zero = round_half_up(onset * 512)
            global_start = global_zero - int(epoch["pre_samples_source"])
            # v3 ended the EEG epoch at presented-audio duration + post roll.
            # That made the model mask a near-perfect sentence-duration ID.
            # v4 always takes a real, fixed continuous window.  Any recording
            # that cannot supply it is rejected rather than zero padded.
            if fixed_window:
                global_end = global_start + int(epoch["total_samples_source"])
            else:
                global_end = global_zero + round_half_up((stimulus_duration + float(epoch["post_sentence_s"])) * 512)
            selected_run = next((item for item in runs if item["start_sample"] <= global_zero < item["end_sample"]), None)
            boundary = bool(selected_run and (global_start < selected_run["start_sample"] or global_end > selected_run["end_sample"]))
            bdf = selected_run["path"] if selected_run else None
            if bdf is None:
                reason = "missing_eeg_run"
            elif boundary:
                reason = "epoch_crosses_run_boundary"
            elif audio is None:
                reason = "missing_audio"
            else:
                reason = ""
            if reason:
                qc["exclusions"][reason] += 1
            run = str(selected_run["run"] if selected_run else "unknown")
            run_offset = int(selected_run["start_sample"]) if selected_run else 0
            identifier = trial_id("ds004940", subject, task, run, int(row_index), onset)
            if audio is not None and audio not in audio_hashes:
                audio_hashes[audio] = sha256_file(audio)
            audio_sha = audio_hashes.get(audio, "")
            content_id = stimulus_content_id("ds004940", str(event[stim_col])) if stim_col else ""
            entry = {
                "trial_id": identifier, "dataset": "ds004940", "dataset_version": spec["openneuro_version"],
                "subject": subject, "session": session, "task": task, "run": run,
                "source_eeg_path": as_relative(bdf) if bdf else "", "source_eeg_sha256": "",
                "source_channels_path": as_relative(channels_path) if channels_path else "",
                "source_channels_sha256": channels_sha,
                "bad_channels": json.dumps(bad_channels),
                "source_event_path": as_relative(events_path), "source_event_sha256": event_sha,
                "source_event_row": int(row_index), "event_onset_seconds": onset,
                "event_run_local_onset_seconds": (global_zero - run_offset) / 512,
                "event_to_sample_error": abs(onset * 512 - global_zero), "stimulus_duration_seconds": stimulus_duration,
                "condition": "active" if "active" in task.lower() else "passive",
                "stim_file": str(event[stim_col]) if stim_col else "", "waveform_id": audio_sha,
                "stimulus_content_id": content_id, "linguistic_content_id": content_id,
                "audio_path": as_relative(audio) if audio else "", "audio_sha256": audio_sha,
                "supervision_type": "paired_audio", "audio_semantics": "presented_waveform",
                "pairing_level": "verified_exact",
                "audio_semantics_evidence": "bids_stim_file_direct_reference" if audio else "missing_stim_file_target",
                "neural_task": "perception", "response_onset_relative_s": None,
                "response_onset_output_index": None, "response_onset_provenance": "",
                "production_contaminated": False, "clean_perception_start_index": 0,
                "clean_perception_end_index": int(spec.get("epoch_target_length", epoch["total_samples_target"])),
                "source_sfreq_hz": 512, "target_sfreq_hz": 256,
                "eeg_zero_index": 64, "audio_start_relative_to_eeg_samples": 64,
                "source_zero_sample": global_zero - run_offset,
                "source_start_sample": global_start - run_offset,
                "source_end_sample": global_end - run_offset,
                "eeg_valid_samples_target": (int(epoch["total_samples_target"]) if fixed_window else round_half_up((global_end - global_start) * 256 / 512)),
                "model_time_mask_policy": "fixed_full_epoch" if fixed_window else "variable_valid_epoch",
                "source_run_offsets": json.dumps([{"path": as_relative(item["path"]), "run": item["run"], "start_sample": item["start_sample"], "end_sample": item["end_sample"]} for item in runs]),
                "boundary_overlap": boundary,
                "qc_pass": not bool(reason), "build_status": "included" if not reason else "excluded",
                "exclusion_reason": reason, "channel_order_hash": channel_order_hash(spec["channel_order"]),
            }
            rows.append(entry)
    qc["actual_subjects"]["ds004940"] = len(subject_seen)
    return rows


def write_frame(frame, path: Path, pd) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path.with_suffix(".csv"), index=False)
    try:
        frame.to_parquet(path.with_suffix(".parquet"), index=False)
    except Exception as exc:
        # CSV is always written; parquet requires an optional engine.
        (path.with_suffix(".parquet.unavailable.txt")).write_text(str(exc) + "\n")


def audit(config: dict[str, Any], strict: bool) -> int:
    _, pd = runtime()
    root = output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    audit_timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    profile = config.get("preprocessing_profile", "harmonized_v3")
    qc: dict[str, Any] = {"schema_version": config["schema_version"], "created_at": time.time(), "actual_subjects": {},
                           "exclusions": Counter(), "warnings": [], "preprocessing_statement": f"{profile} is project preprocessing, not official preprocessing"}
    free_bytes = shutil.disk_usage(ROOT).free
    qc["disk_free_bytes"] = free_bytes
    qc["estimated_derived_bytes_range"] = [12 * 1024**3, 18 * 1024**3]
    if free_bytes < 18 * 1024**3:
        qc["warnings"].append("available disk is below the documented 18 GiB derived-data estimate")
    # Do not put wall-clock state in this immutable lock: its content hash must
    # remain identical when the same raw data are audited again.
    lock: dict[str, Any] = {"schema_version": config["schema_version"], "config_sha256": config["_config_sha256"], "files": [], "official_aux": {}}
    for dataset, spec in progress(config["sources"].items(), desc="audit pinned sources", total=len(config["sources"])):
        data_root = ROOT / spec["data_root"]
        description = data_root / "dataset_description.json"
        if description.exists():
            actual = sha256_file(description)
            lock["files"].append(source_lock_entry(description, "dataset_description"))
            if actual != spec["dataset_description_sha256"]:
                qc["warnings"].append(f"{dataset}: dataset_description hash mismatch")
        else:
            qc["warnings"].append(f"{dataset}: missing dataset_description")
        stimulus_root = data_root / "stimuli"
        if stimulus_root.exists():
            actual_inventory = inventory_sha256(stimulus_root)
            lock["stimulus_inventory"] = lock.get("stimulus_inventory", {}) | {dataset: {"root": as_relative(stimulus_root), "sha256": actual_inventory}}
            if actual_inventory != spec["stimulus_inventory_sha256"]:
                qc["warnings"].append(f"{dataset}: stimulus inventory hash mismatch")
        else:
            qc["warnings"].append(f"{dataset}: missing stimulus inventory")
    rows = _ds004_trial_rows(config, lock, qc)
    # Add every actually referenced source only once; raw hashes are intentionally computed here,
    # before build, so resume can reject a mutated raw dataset.
    sources = sorted({r.get(k, "") for r in rows for k in ("source_eeg_path", "source_channels_path", "source_event_path", "audio_path") if r.get(k)})
    for relative in progress(sources, desc="SHA256 source lock"):
        path = ROOT / relative
        if path.exists():
            entry = source_lock_entry(path, "source")
            lock["files"].append(entry)
            for row in rows:
                if row.get("source_eeg_path") == relative: row["source_eeg_sha256"] = entry["sha256"]
    for dataset, spec in config["sources"].items():
        observed = sum(r["dataset"] == dataset for r in rows)
        if observed != spec["expected_trials"]:
            qc["warnings"].append(f"{dataset}: observed {observed} trials; pinned expectation {spec['expected_trials']}")
        observed_subjects = qc["actual_subjects"].get(dataset, 0)
        if observed_subjects != spec["expected_subjects"]:
            qc["warnings"].append(f"{dataset}: observed {observed_subjects} subjects; pinned expectation {spec['expected_subjects']}")
    lock["files"].sort(key=lambda x: x["path"])
    lock["source_lock_sha256"] = sha256_bytes(stable_json({k: v for k, v in lock.items() if k != "source_lock_sha256"}))
    code_commit, code_diff = git_provenance()
    code_diff_hash = sha256_bytes(code_diff.encode())
    for row in rows:
        row["source_lock_sha256"] = lock["source_lock_sha256"]
        row["preprocess_config_sha256"] = config["_config_sha256"]
        row["code_commit"] = code_commit
        row["code_diff_hash"] = code_diff_hash
        row["audit_timestamp_utc"] = audit_timestamp_utc
    frame = pd.DataFrame(rows)
    qc["trial_counts"] = dict(Counter(row["dataset"] for row in rows))
    qc["included_counts"] = dict(Counter(row["dataset"] for row in rows if row.get("qc_pass")))
    qc["pairing_level_counts"] = dict(Counter(str(row.get("pairing_level", "none")) for row in rows))
    qc["boundary_overlap_counts"] = dict(Counter(row["dataset"] for row in rows if row.get("boundary_overlap")))
    write_frame(frame, root / "manifests" / "manifest_all", pd)
    (root / "source_lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    qc["exclusions"] = dict(qc["exclusions"])
    qc["status"] = "warning" if qc["warnings"] else "pass"
    (root / "qc").mkdir(exist_ok=True)
    (root / "qc" / "audit.json").write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n")
    pd.DataFrame([{"reason": k, "count": v} for k, v in qc["exclusions"].items()]).to_csv(root / "qc" / "exclusions.csv", index=False)
    print(f"audit wrote {len(rows)} inventoried trials, source lock {lock['source_lock_sha256']}")
    return 2 if strict and qc["warnings"] else 0


def read_manifest(config: dict[str, Any]):
    _, pd = runtime()
    path = output_root(config) / "manifests" / "manifest_all.csv"
    if not path.exists():
        raise RuntimeError("audit first: manifest_all.csv is missing")
    return pd.read_csv(path, keep_default_na=False, low_memory=False), pd


def make_splits(config: dict[str, Any]) -> int:
    frame, pd = read_manifest(config)
    root = output_root(config)
    lock = json.loads((root / "source_lock.json").read_text())
    commit, diff = git_provenance()
    eligible = frame[(frame["build_status"] == "included") & (frame["qc_pass"].astype(str).str.lower() == "true")].copy()
    eligible["subject_group"] = eligible.apply(lambda row: f"{row.dataset}:{row.subject}", axis=1)
    subject_weights = eligible.groupby("subject_group").size().astype(int).to_dict()
    content_eligible = eligible[eligible["supervision_type"].isin(["paired_audio", "weak_audio", "label_only"])].copy()
    content_eligible["content_group"] = content_eligible.apply(
        lambda r: r.get("linguistic_content_id") or r.get("stimulus_content_id") or
        (f"{r.dataset}:file:{r.audio_sha256}" if r.audio_sha256 else ""), axis=1
    )
    content_eligible["waveform_group"] = content_eligible.apply(
        lambda r: f"sha256:{r.audio_sha256}" if r.supervision_type in {"paired_audio", "weak_audio"} and r.audio_sha256 else "", axis=1
    )
    content_eligible = content_eligible[content_eligible.content_group != ""]
    content_weights = content_eligible.groupby("content_group").size().astype(int).to_dict()
    subject_assign = balanced_group_assignment(subject_weights, config["folds"], config["split_seed"], "subject")
    content_assign = balanced_group_assignment(content_weights, config["folds"], config["split_seed"], "linguistic_content")
    records = []
    content_groups_by_id = dict(zip(content_eligible["trial_id"], content_eligible["content_group"]))
    waveform_groups_by_id = dict(zip(content_eligible["trial_id"], content_eligible["waveform_group"]))
    for _, row in eligible.iterrows():
        subject_group = f"{row.dataset}:{row.subject}"
        s = subject_assign[subject_group]
        group = content_groups_by_id.get(row.trial_id)
        waveform_group = waveform_groups_by_id.get(row.trial_id, "")
        a = content_assign.get(group) if group else None
        for protocol in ("subject_ood", "audio_ood", "joint_ood"):
            for fold in range(config["folds"]):
                role, reason = split_role(protocol, fold, s["fold"], a["fold"] if a else None, row.supervision_type, config["folds"])
                records.append({"trial_id": row.trial_id, "protocol": protocol, "fold": fold, "role": role,
                                "exclusion_reason": reason, "subject_group": subject_group, "subject_fold": s["fold"],
                                "subject_group_trial_weight": s["trial_weight"], "subject_sort_position": s["sort_position"],
                                "audio_group": group or "", "linguistic_content_group": group or "", "waveform_group": waveform_group,
                                "supervision_axis": "label" if row.supervision_type == "label_only" else "audio",
                                "audio_fold": a["fold"] if a else "",
                                "audio_group_trial_weight": a["trial_weight"] if a else "", "audio_sort_position": a["sort_position"] if a else "",
                                "assignment_algorithm": SPLIT_ALGORITHM, "assignment_seed": config["split_seed"],
                                "source_lock_sha256": lock["source_lock_sha256"], "preprocess_config_sha256": config["_config_sha256"],
                                "code_commit": commit, "code_diff_hash": sha256_bytes(diff.encode())})
    result = pd.DataFrame(records)
    split_root = root / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    for protocol in ("subject_ood", "audio_ood", "joint_ood"):
        for fold in range(config["folds"]):
            subset = result[(result.protocol == protocol) & (result.fold == fold)].copy()
            content = subset.to_csv(index=False).encode()
            file_hash = sha256_bytes(content)
            subset["split_csv_sha256"] = file_hash
            target = split_root / f"{protocol}_fold-{fold}.csv"
            subset.to_csv(target, index=False)
    split_hashes = {p.name: sha256_file(p) for p in sorted(split_root.glob("*_fold-*.csv"))}
    split_index = {"algorithm": SPLIT_ALGORITHM, "seed": config["split_seed"], "subject": subject_assign,
                   "linguistic_content": content_assign, "split_csv_sha256": split_hashes,
                   "source_lock_sha256": lock["source_lock_sha256"], "preprocess_config_sha256": config["_config_sha256"], "code_commit": commit, "code_diff_hash": sha256_bytes(diff.encode())}
    split_index["split_index_sha256"] = sha256_bytes(stable_json(split_index))
    (split_root / "assignment.json").write_text(json.dumps(split_index, indent=2, sort_keys=True) + "\n")
    print(f"wrote frozen split CSVs for {len(eligible)} included QC-pass trials")
    return 0


def build_audio_bank(config: dict[str, Any], resume: bool) -> int:
    """Materialise a deduplicated, explicitly parameterised audio bank.

    This is intentionally a separate phase: it updates only derived manifest
    fields (audio_id and audio metadata), never BIDS stimuli.  HDF5 groups are
    ragged by audio_id, avoiding hidden padding or waveform normalization.
    """
    h5py, _, np = require_build_runtime()
    try:
        import torchaudio  # type: ignore
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("audio bank requires torch and torchaudio from requirements-preprocess.txt") from exc
    frame, pd = read_manifest(config)
    root = output_root(config)
    lock = json.loads((root / "source_lock.json").read_text())
    target = root / "audio_bank" / "audio_bank.h5"
    if target.exists() and resume:
        with h5py.File(target, "r") as bank:
            if not resume_compatible(dict(bank.attrs), config_sha=config["_config_sha256"], source_lock_sha=lock["source_lock_sha256"]):
                raise RuntimeError("resume refuses incompatible audio bank")
        print(f"reused {target}")
        return 0
    paired = frame[(frame.supervision_type.isin(["paired_audio", "weak_audio"])) & (frame.audio_path != "")].copy()
    # The same bytes can be cited under distinct experimental semantics.  Keep
    # separate logical bank records in that rare case rather than collapsing a
    # clean stimulus and a presented waveform into one semantic category.
    unique = paired.drop_duplicates(["audio_sha256", "audio_semantics"])
    partial = target.with_suffix(".h5.partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists(): partial.unlink()
    melcfg = config["audio"]["mel"]
    with h5py.File(partial, "w") as bank:
        bank.attrs["schema_version"] = config["schema_version"]
        bank.attrs["preprocess_config_sha256"] = config["_config_sha256"]
        bank.attrs["source_lock_sha256"] = lock["source_lock_sha256"]
        commit, diff = git_provenance()
        bank.attrs["code_commit"] = commit
        bank.attrs["code_diff_hash"] = sha256_bytes(diff.encode())
        bank.attrs["audio_config"] = json.dumps(config["audio"], sort_keys=True)
        index = []
        for _, row in progress(unique.iterrows(), desc="audio bank", total=len(unique)):
            source = ROOT / row.audio_path
            wave, source_rate = torchaudio.load(str(source))
            original_channels, original_samples = int(wave.shape[0]), int(wave.shape[1])
            # Explicit arithmetic mean; no peak/RMS/DC/clipping transformations.
            wave = wave.mean(dim=0, keepdim=True)
            if source_rate != config["audio"]["target_sample_rate"]:
                wave = torchaudio.functional.resample(wave, source_rate, config["audio"]["target_sample_rate"], resampling_method="sinc_interp_kaiser")
            mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=config["audio"]["target_sample_rate"], n_fft=melcfg["n_fft"], win_length=melcfg["win_length"],
                hop_length=melcfg["hop_length"], window_fn=torch.hann_window, power=melcfg["power"], normalized=melcfg["normalized"],
                center=melcfg["center"], pad=melcfg["pad"], pad_mode=melcfg["pad_mode"], onesided=melcfg["onesided"],
                n_mels=melcfg["n_mels"], f_min=melcfg["f_min"], f_max=melcfg["f_max"], norm=melcfg["norm"], mel_scale=melcfg["mel_scale"],
            )(wave)
            log_mel = torch.log(mel.clamp_min(float(melcfg["log_epsilon"]))).squeeze(0).cpu().numpy().astype("float32")
            waveform = wave.squeeze(0).cpu().numpy().astype("float32")
            audio_id = f"audio-{row.audio_sha256[:16]}-{row.audio_semantics}"
            group = bank.create_group(audio_id)
            group.create_dataset("waveform", data=waveform, compression="gzip", shuffle=True)
            group.create_dataset("log_mel", data=log_mel, compression="gzip", shuffle=True)
            for key, value in {"source_path":row.audio_path, "source_sha256":row.audio_sha256, "source_sample_rate_hz":int(source_rate),
                               "source_channels":original_channels, "source_samples":original_samples, "target_sample_rate_hz":config["audio"]["target_sample_rate"],
                               "audio_semantics":row.audio_semantics, "audio_semantics_evidence":row.audio_semantics_evidence,
                               "stimulus_content_id":row.stimulus_content_id}.items():
                group.attrs[key] = value
            index.append({"audio_id":audio_id, "audio_sha256":row.audio_sha256, "audio_semantics":row.audio_semantics, "source_path":row.audio_path,
                          "audio_semantics_evidence":row.audio_semantics_evidence,
                          "stimulus_content_id":row.stimulus_content_id, "source_sample_rate_hz":source_rate,
                          "source_channels":original_channels, "source_samples":original_samples, "target_samples":len(waveform), "target_duration_s":len(waveform) / config["audio"]["target_sample_rate"], "mel_frames":log_mel.shape[1]})
    os.replace(partial, target)
    index_frame = pd.DataFrame(index)
    write_frame(index_frame, root / "audio_bank" / "audio_inventory", pd)
    audio_id_map = {(r.audio_sha256, r.audio_semantics): r.audio_id for _, r in index_frame.iterrows()}
    frame["audio_id"] = frame.apply(lambda r: audio_id_map.get((r.audio_sha256, r.audio_semantics), ""), axis=1)
    audio_duration_map = {(r.audio_sha256, r.audio_semantics): r.target_duration_s for _, r in index_frame.iterrows()}
    frame["audio_target_duration_s"] = frame.apply(lambda r: audio_duration_map.get((r.audio_sha256, r.audio_semantics), ""), axis=1)
    write_frame(frame, root / "manifests" / "manifest_all", pd)
    print(f"audio bank wrote {len(index_frame)} deduplicated files: {target}")
    return 0


def require_build_runtime():
    try:
        import h5py  # type: ignore
        import mne  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install optional runtime: pip install -r requirements-preprocess.txt") from exc
    return h5py, mne, np


def _h5_write_strings(group, name: str, values: list[str], h5py) -> None:
    group.create_dataset(name, data=values, dtype=h5py.string_dtype("utf-8"))


def _descriptive_stats(array, np):
    finite = np.isfinite(array)
    safe = np.where(finite, array, np.nan)
    med = np.nanmedian(safe, axis=2)
    return {"count": finite.sum(axis=2), "mean": np.nanmean(safe, axis=2), "std": np.nanstd(safe, axis=2),
            "median": med, "mad": np.nanmedian(np.abs(safe - med[:, :, None]), axis=2),
            "min": np.nanmin(safe, axis=2), "max": np.nanmax(safe, axis=2),
            "rms": np.sqrt(np.nanmean(safe ** 2, axis=2)), "nonfinite_count": (~finite).sum(axis=2)}


def _atomic_shard(path: Path, arrays: dict[str, Any], attrs: dict[str, Any], strings: dict[str, list[str]]) -> str:
    h5py, _, np = require_build_runtime()
    partial = path.with_suffix(path.suffix + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        partial.unlink()
    with h5py.File(partial, "w") as output:
        for key, value in arrays.items():
            output.create_dataset(key, data=value, compression="gzip", shuffle=True)
        provenance = output.create_group("provenance")
        for key, values in strings.items():
            _h5_write_strings(provenance, key, values, h5py)
        stats = output.create_group("statistics")
        for key, value in _descriptive_stats(arrays["eeg"], np).items():
            stats.create_dataset(key, data=value.astype("float64"))
        stats.attrs["qc_descriptive_only"] = True
        for key, value in attrs.items():
            output.attrs[key] = json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
        output.flush()
    os.replace(partial, path)
    return sha256_file(path)


def _raw_to_canonical(raw, channels: list[str], dataset: str, config: dict[str, Any],
                      bad_channels: list[str] | None = None):
    """Preprocess one recording once, failing loudly on invalid montage/QC."""
    _, mne, np = require_build_runtime()
    canonical = channels
    aliases = {normalise_ds004_channel(c): c for c in raw.ch_names} if dataset == "ds004940" else {c: c for c in raw.ch_names}
    missing = [c for c in canonical if c not in aliases]
    picked = [aliases[c] for c in canonical if c in aliases]
    raw.pick(picked)
    rename = {actual: normalise_ds004_channel(actual) for actual in raw.ch_names} if dataset == "ds004940" else {}
    if rename:
        raw.rename_channels(rename)
    available_names = set(raw.ch_names)
    declared_bad = set(bad_channels or [])
    bad = sorted(c for c in declared_bad if c in available_names)
    affected_fraction = len(set(missing) | set(bad)) / max(len(canonical), 1)
    if affected_fraction > float(config["harmonized"]["interpolation"]["max_bad_fraction"]):
        raise ValueError(f"bad/missing channel fraction {affected_fraction:.3f} exceeds configured maximum")
    zero = list(missing)
    montage = mne.channels.make_standard_montage(config["harmonized"]["interpolation"][f"{dataset}_montage"])
    raw.set_montage(montage, on_missing="raise")
    interpolated = []
    if bad:
        raw.info["bads"] = bad
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
        interpolated = bad
    raw.set_eeg_reference("average", projection=False)
    raw.filter(*config["harmonized"]["bandpass_hz"], verbose="ERROR")
    raw.resample(config["harmonized"]["target_sfreq_hz"], npad="auto", verbose="ERROR")
    data = raw.get_data()
    aligned = np.zeros((len(canonical), data.shape[1]), dtype=np.float64)
    available = {normalise_ds004_channel(c) if dataset == "ds004940" else c: i for i, c in enumerate(raw.ch_names)}
    for i, c in enumerate(canonical):
        if c in available and c not in zero:
            aligned[i] = data[available[c]]
    # Use the complete template so a missing electrode can remain masked while
    # retaining its coordinate in the shared variable-channel contract.
    positions = montage.get_positions()["ch_pos"]
    xyz = np.stack([positions[c] if c in positions else np.zeros(3) for c in canonical]).astype("float32")
    if not np.isfinite(xyz).all() or np.linalg.norm(xyz, axis=1).min() <= 0:
        raise ValueError("montage produced missing/nonfinite channel coordinates")
    badmask = np.array([c in set(interpolated) | set(zero) for c in canonical], dtype=bool)
    return aligned, badmask, np.array([c in interpolated for c in canonical], bool), np.array([c in zero for c in canonical], bool), xyz


def _rows_for_shards(frame, requested_dataset: str, requested_subjects: set[str] | None,
                     requested_tasks: set[str] | None):
    selected = frame[(frame.build_status == "included") & (frame.qc_pass.astype(str).str.lower() == "true")]
    if requested_dataset != "all": selected = selected[selected.dataset == requested_dataset]
    if requested_subjects: selected = selected[selected.subject.isin(requested_subjects)]
    if requested_tasks: selected = selected[selected.task.isin(requested_tasks)]
    return selected


def build(config: dict[str, Any], dataset: str, subjects: str, tasks: str,
          limit_trials_per_group: int | None, common_contents: int | None,
          content_ids: str | None,
          split_role_filter: str, split_protocol: str,
          split_fold: int, resume: bool, allow_audit_warnings: bool,
          artifact_set: str = "built") -> int:
    h5py, mne, np = require_build_runtime()
    frame, pd = read_manifest(config)
    root = output_root(config)
    qc_audit = json.loads((root / "qc" / "audit.json").read_text())
    if qc_audit["status"] != "pass" and not allow_audit_warnings:
        raise RuntimeError("audit has warnings; inspect QC and use --allow-audit-warnings to record an explicit override")
    lock = json.loads((root / "source_lock.json").read_text())
    split_index_path = root / "splits" / "assignment.json"
    if not split_index_path.exists():
        raise RuntimeError("make-splits first; HDF5 shards must pin a frozen split index")
    split_index = json.loads(split_index_path.read_text())
    if not re.fullmatch(r"[a-z0-9_-]+", artifact_set):
        raise ValueError("artifact_set must contain only lowercase letters, digits, '_' or '-'")
    if artifact_set != "built" and split_role_filter == "any":
        raise ValueError("a named artifact_set requires an explicit split role")
    split_contract_hash = split_index["split_index_sha256"]
    requested = None if subjects == "all" else set(subjects.split(","))
    requested_tasks = None if tasks == "all" else set(tasks.split(","))
    selected = _rows_for_shards(frame, dataset, requested, requested_tasks)
    preprocessing_selected = selected.copy()
    if split_role_filter != "any":
        split_path = root / "splits" / f"{split_protocol}_fold-{split_fold}.csv"
        split_frame = pd.read_csv(split_path, keep_default_na=False)
        allowed = set(split_frame[split_frame.role == split_role_filter].trial_id)
        selected = selected[selected.trial_id.isin(allowed)]
        if artifact_set != "built":
            split_contract_hash = sha256_file(split_path)
    selected_contents: set[str] | None = None
    if content_ids:
        if common_contents is not None:
            raise ValueError("--content-ids and --common-contents are mutually exclusive")
        selected_contents = {value.strip() for value in content_ids.split(",") if value.strip()}
        if not selected_contents:
            raise ValueError("--content-ids supplied no non-empty IDs")
        selected = selected[selected.linguistic_content_id.isin(selected_contents)]
        missing = selected_contents - set(selected.linguistic_content_id)
        if missing:
            raise RuntimeError(f"requested content IDs are unavailable after filters: {sorted(missing)}")
    if common_contents is not None:
        if dataset == "all":
            raise ValueError("--common-contents requires one explicit dataset")
        required_subjects = selected.subject.nunique()
        coverage = selected[selected.linguistic_content_id != ""].groupby("linguistic_content_id").subject.nunique()
        common = [name for name, count in coverage.items() if int(count) == int(required_subjects)]
        common.sort(key=lambda name: sha256_bytes(f"pilot-common|{config['split_seed']}|{dataset}|{name}".encode()))
        if len(common) < common_contents:
            raise RuntimeError(f"only {len(common)} linguistic contents cover all {required_subjects} selected subjects")
        selected_contents = set(common[:common_contents])
        selected = selected[selected.linguistic_content_id.isin(selected_contents)]
    built_rows = []
    build_timestamp_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    shard_groups = list(selected.groupby(["dataset", "subject", "task"], sort=True))
    for (ds, subject, task), group in progress(shard_groups, desc="EEG shards", total=len(shard_groups)):
        preprocessing_group = preprocessing_selected[(preprocessing_selected.dataset == ds) & (preprocessing_selected.subject == subject) & (preprocessing_selected.task == task)]
        if selected_contents is not None:
            group = group.sort_values("trial_id").drop_duplicates("linguistic_content_id")
        if limit_trials_per_group is not None:
            group = group.sort_values("trial_id").head(limit_trials_per_group)
        spec = config["sources"][ds]
        canonical = spec["channel_order"]
        target_len = config["harmonized"]["epoch"][ds]["total_samples_target"]
        shard_root = root / "shards" if artifact_set == "built" else root / "shards" / artifact_set
        target = shard_root / ds / subject / f"task-{task}.h5"
        if target.exists() and resume:
            with h5py.File(target, "r") as previous:
                if not resume_compatible(dict(previous.attrs), config_sha=config["_config_sha256"], source_lock_sha=lock["source_lock_sha256"], channel_hash=channel_order_hash(canonical), split_hash=split_contract_hash):
                    raise RuntimeError(f"resume refuses incompatible shard {target}")
            continue
        # One recording is loaded/filtered/resampled once for all of its trials.
        raw_cache: dict[str, Any] = {}
        raw_errors: dict[str, str] = {}
        for raw_relative, recording_rows in preprocessing_group.groupby("source_eeg_path"):
            raw_path = ROOT / raw_relative
            declared_bad: set[str] = set()
            for text_value in recording_rows.get("bad_channels", []):
                text_value = str(text_value)
                if text_value.startswith("["):
                    declared_bad.update(json.loads(text_value))
            try:
                raw = mne.io.read_raw_bdf(raw_path, preload=True, verbose="ERROR") if raw_path.suffix.lower() == ".bdf" else mne.io.read_raw_edf(raw_path, preload=True, verbose="ERROR")
                expected_sfreq = float(recording_rows.iloc[0].source_sfreq_hz)
                if abs(float(raw.info["sfreq"]) - expected_sfreq) > 1e-6:
                    raise ValueError(f"source sampling rate {raw.info['sfreq']} != locked {expected_sfreq}")
                raw_cache[raw_relative] = _raw_to_canonical(raw, canonical, ds, config, sorted(declared_bad))
            except Exception as exc:
                raw_errors[raw_relative] = f"{type(exc).__name__}:{exc}"

        eegs=[]; valids=[]; cleans=[]; audio_losses=[]; bads=[]; interps=[]; zeros=[]; ids=[]; retained=[]
        shard_xyz = None
        for _, row in progress(group.iterrows(), desc=f"{ds}/{subject}/{task}", total=len(group)):
            try:
                if row.source_eeg_path in raw_errors:
                    raise ValueError(f"recording_preprocess:{raw_errors[row.source_eeg_path]}")
                data, bad, interp, zero, xyz = raw_cache[row.source_eeg_path]
                if shard_xyz is None:
                    shard_xyz = xyz
                elif not np.allclose(shard_xyz, xyz, atol=1e-6):
                    raise ValueError("recordings in one shard disagree on channel coordinates")
                target_sfreq = int(config["harmonized"]["target_sfreq_hz"])
                source_sfreq = int(row.source_sfreq_hz)
                start_target = round_half_up(int(row.source_start_sample) * target_sfreq / source_sfreq)
                valid_target = min(int(row.get("eeg_valid_samples_target", target_len)), target_len)
                end_target = start_target + valid_target
                if start_target < 0 or end_target > data.shape[1]:
                    raise ValueError("epoch crosses raw-run boundary")
                ep = np.zeros((len(canonical), target_len), dtype="float32")
                ep[:, :valid_target] = data[:, start_target:end_target].astype("float32")
                valid_mask = np.arange(target_len) < valid_target
                eegs.append(ep); valids.append(valid_mask)
                cleans.append(valid_mask.copy())
                audio_duration = row.get("audio_target_duration_s", "")
                if str(audio_duration).strip().lower() in {"", "nan", "none"}:
                    audio_duration = row.get("stimulus_duration_seconds", "")
                audio_losses.append(np.asarray(acoustic_supervision_mask(
                    target_len, valid_target, int(row.eeg_zero_index),
                    audio_duration, target_sfreq,
                    str(row.get("pairing_level", "")),
                ), dtype=bool))
                bads.append(bad); interps.append(interp); zeros.append(zero); ids.append(row.trial_id); retained.append(row.to_dict())
            except Exception as exc:
                row = row.copy(); row["build_status"] = "excluded"; row["exclusion_reason"] = f"build:{type(exc).__name__}:{exc}"; row["build_timestamp_utc"] = build_timestamp_utc; built_rows.append(row.to_dict())
        if not eegs:
            continue
        arrays={"eeg": np.stack(eegs), "channel_xyz": shard_xyz, "eeg_valid_mask": np.stack(valids), "clean_perception_mask": np.stack(cleans), "audio_loss_mask": np.stack(audio_losses), "bad_channel_mask":np.stack(bads), "interpolated_channel_mask":np.stack(interps), "zero_filled_channel_mask":np.stack(zeros), "channel_valid_mask":~np.stack(zeros)}
        commit, diff = git_provenance()
        fixed_policy = bool(config["harmonized"]["epoch"].get(ds, {}).get("fixed_window", False))
        attrs={"schema_version": config["schema_version"], "preprocessing_profile": config.get("preprocessing_profile", "harmonized_v3"), "artifact_set": artifact_set, "eeg_unit": "V", "eeg_dtype": "float32", "channel_order": canonical, "channel_order_hash": channel_order_hash(canonical), "preprocess_config_sha256":config["_config_sha256"], "source_lock_sha256":lock["source_lock_sha256"], "split_index_sha256":split_contract_hash, "split_hash_required": True, "code_commit":commit, "code_diff_hash":sha256_bytes(diff.encode()), "audit_override_allow_warnings": bool(allow_audit_warnings), "model_time_mask_policy": "fixed_full_epoch" if fixed_policy else "variable_valid_epoch"}
        provenance_keys = ("dataset", "subject", "task", "condition", "pairing_level", "supervision_type", "linguistic_content_id", "waveform_id", "phoneme_label", "audio_id")
        strings = {"trial_id": ids}
        for key in provenance_keys:
            strings[key] = [str(row.get(key, "")) for row in retained]
        checksum=_atomic_shard(target, arrays, attrs, strings)
        for index, row in enumerate(retained):
            row.update({"shard_path":as_relative(target), "shard_row":index, "shard_sha256":checksum, "build_status":"included", "source_lock_sha256":lock["source_lock_sha256"], "preprocess_config_sha256":config["_config_sha256"], "split_index_sha256":split_index["split_index_sha256"], "code_commit":commit, "code_diff_hash":attrs["code_diff_hash"], "audit_override_allow_warnings":bool(allow_audit_warnings), "build_timestamp_utc":build_timestamp_utc, "bad_channel_count":int(bads[index].sum()), "interpolated_channel_count":int(interps[index].sum()), "zero_filled_channel_count":int(zeros[index].sum())})
            built_rows.append(row)
    built = pd.DataFrame(built_rows)
    built_base = root / "manifests" / f"manifest_{artifact_set}"
    previous_path = built_base.with_suffix(".csv")
    if previous_path.exists() and not len(built):
        previous = pd.read_csv(previous_path, keep_default_na=False, low_memory=False)
        print(f"build resumed with 0 new trials; manifest contains {sum(previous.build_status == 'included')}")
        return 0
    if previous_path.exists() and len(built):
        previous = pd.read_csv(previous_path, keep_default_na=False, low_memory=False)
        rebuilt_shards = set(built.loc[built.get("shard_path", "") != "", "shard_path"]) if "shard_path" in built else set()
        previous = previous[(~previous.trial_id.isin(set(built.trial_id))) & (~previous.shard_path.isin(rebuilt_shards))]
        # Preserve old rows for auditability, but never leave an incompatible
        # shard eligible for a new loader/normalizer after code/config/split
        # changes.  The HDF5 file remains on disk and can still be inspected.
        _, merge_diff = git_provenance()
        current_contract = semantic_preprocessing_contract(
            config_sha=config["_config_sha256"],
            source_lock_sha=lock["source_lock_sha256"],
            # Named artifact sets pin their role-specific CSV rather than the
            # global split-index JSON.
            split_hash=split_contract_hash,
            code_diff_hash=sha256_bytes(merge_diff.encode()),
        )
        for shard_path, indices in previous[previous.build_status == "included"].groupby("shard_path").groups.items():
            compatible = False
            path = ROOT / shard_path
            if path.exists():
                try:
                    with h5py.File(path, "r") as old:
                        compatible = all(str(old.attrs.get(key, "")) == str(value) for key, value in current_contract.items())
                except OSError:
                    compatible = False
            if not compatible:
                previous.loc[list(indices), "build_status"] = "stale_incompatible"
                previous.loc[list(indices), "exclusion_reason"] = "stale_preprocessing_contract"
        combined = pd.concat([previous, built], ignore_index=True, sort=False).sort_values(["dataset", "subject", "task", "trial_id"])
    else:
        combined = built
    write_frame(combined, built_base, pd)
    print(f"build wrote {sum(built.get('build_status', []) == 'included') if len(built) else 0} new trials; manifest contains {sum(combined.get('build_status', []) == 'included') if len(combined) else 0}")
    return 0


def fit_normalizer(config: dict[str, Any], split_csv: Path, fold: int,
                   allow_mixed_production: bool, manifest_kind: str = "built",
                   normalizer_name: str | None = None) -> int:
    h5py, _, np = require_build_runtime()
    frame, pd = read_manifest(config)
    split = pd.read_csv(split_csv)
    train_ids = set(split[(split.fold == fold) & (split.role == "train")].trial_id)
    if not train_ids: raise RuntimeError("selected split/fold contains no training trials")
    if not re.fullmatch(r"[a-z0-9_-]+", manifest_kind):
        raise ValueError("manifest_kind contains unsafe characters")
    if normalizer_name is not None and not re.fullmatch(r"[a-z0-9_-]+", normalizer_name):
        raise ValueError("normalizer_name contains unsafe characters")
    built_path = output_root(config) / "manifests" / f"manifest_{manifest_kind}.csv"
    built = pd.read_csv(built_path, keep_default_na=False, low_memory=False) if built_path.exists() else frame
    chosen = built[(built.trial_id.isin(train_ids)) & (built.build_status == "included")]
    if not len(chosen):
        raise RuntimeError("no built train-fold trials are available for normalizer fitting")
    lock = json.loads((output_root(config) / "source_lock.json").read_text())
    split_index = json.loads((output_root(config) / "splits" / "assignment.json").read_text())
    # ``code_commit`` is reporting lineage, not a transform contract.  Requiring
    # it here invalidates a normalizer after an unrelated model/docs commit even
    # when every EEG sample was produced by exactly the same preprocessing code.
    contract_fields = ("preprocess_config_sha256", "source_lock_sha256", "split_index_sha256", "code_diff_hash")
    contracts: dict[str, set[str]] = {key: set() for key in contract_fields}
    for shard in sorted(set(chosen.shard_path)):
        with h5py.File(ROOT / shard, "r") as h5:
            for key in contract_fields:
                contracts[key].add(str(h5.attrs.get(key, "")))
    expected_split_hash = (split_index["split_index_sha256"] if manifest_kind == "built"
                           else sha256_file(split_csv))
    expected = {
        "preprocess_config_sha256": config["_config_sha256"],
        "source_lock_sha256": lock["source_lock_sha256"],
        "split_index_sha256": expected_split_hash,
    }
    for key, value in expected.items():
        if contracts[key] != {value}:
            raise RuntimeError(f"normalizer refuses incompatible {key}: {sorted(contracts[key])} != {value}")
    if bool(config.get("normalization", {}).get("require_single_preprocessing_contract", True)):
        if len(contracts["code_diff_hash"]) != 1 or "" in contracts["code_diff_hash"]:
            raise RuntimeError("normalizer refuses mixed preprocessing code_diff_hash: "
                               f"{sorted(contracts['code_diff_hash'])}")
    maximum = int(config.get("normalization", {}).get("max_samples_per_channel", 200000))
    dataset_results = {}
    for dataset, dataset_rows in chosen.groupby("dataset"):
        channels = config["sources"][dataset]["channel_order"]
        samples: list[list[Any]] = [[] for _ in channels]
        counts = np.zeros(len(channels), dtype=np.int64)
        for shard, entries in dataset_rows.groupby("shard_path"):
            with h5py.File(ROOT / shard, "r") as h5:
                if h5.attrs.get("eeg_unit") != "V": raise RuntimeError("normalizer refuses non-Volt EEG")
                for _, entry in entries.iterrows():
                    index = int(float(entry.shard_row))
                    valid = h5["eeg_valid_mask"][index].astype(bool)
                    x = h5["eeg"][index][:, valid].astype("float64")
                    if not np.isfinite(x).all():
                        raise RuntimeError(f"normalizer encountered nonfinite EEG in {entry.trial_id}")
                    for channel in range(len(channels)):
                        remaining = maximum - counts[channel]
                        if remaining <= 0 or x.shape[1] == 0:
                            continue
                        stride = max(1, math.ceil(x.shape[1] / remaining))
                        take = x[channel, ::stride][:remaining]
                        samples[channel].append(take)
                        counts[channel] += len(take)
        center = []
        scale = []
        for channel_samples in samples:
            values = np.concatenate(channel_samples) if channel_samples else np.array([0.0])
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median))) * 1.4826
            center.append(median)
            scale.append(max(mad, 1e-9))
        dataset_results[dataset] = {"channel_order": channels, "count": counts.tolist(), "center_median_v": center,
                                    "scale_mad_v": scale, "max_samples_per_channel": maximum}
    result={"schema_version":config["schema_version"], "normalization_fit_role":"train_only", "method":"dataset_channel_median_mad",
            "manifest_kind": manifest_kind,
            "protocol_split_csv":str(split_csv), "split_csv_sha256":sha256_file(split_csv), "fold":fold,
            "preprocessing_contract": {key: next(iter(values)) for key, values in contracts.items()},
            "datasets":dataset_results}
    # The split filename already includes the fold (for example,
    # joint_ood_fold-0.csv); avoid producing the ambiguous fold-0_fold-0 name.
    target_name = normalizer_name or split_csv.stem
    target=output_root(config)/"normalizers"/f"{target_name}.json"; target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(target)
    return 0


def migrate_provenance(config: dict[str, Any]) -> int:
    """Atomically migrate metadata-only preprocessing provenance.

    This migration is legal only when config/source/split hashes already match;
    it records the previous code hash in every HDF5 file and updates manifest
    checksums.  Any transform-contract mismatch still requires a real rebuild.
    """
    h5py, _, _ = require_build_runtime(); _, pd = runtime()
    root = output_root(config); manifest_path = root / "manifests" / "manifest_built.csv"
    frame = pd.read_csv(manifest_path, keep_default_na=False, low_memory=False)
    lock = json.loads((root / "source_lock.json").read_text())
    split_index = json.loads((root / "splits" / "assignment.json").read_text())
    commit, diff = git_provenance(); code_hash = sha256_bytes(diff.encode())
    expected = {"preprocess_config_sha256": config["_config_sha256"],
                "source_lock_sha256": lock["source_lock_sha256"],
                "split_index_sha256": split_index["split_index_sha256"]}
    migrated = 0
    for shard, indices in frame[frame.build_status == "included"].groupby("shard_path").groups.items():
        path = ROOT / shard; partial = path.with_suffix(path.suffix + ".provenance-migrate.partial")
        with h5py.File(path, "r") as source:
            for key, value in expected.items():
                if str(source.attrs.get(key, "")) != str(value):
                    raise RuntimeError(f"provenance migration refuses transform-incompatible shard {shard}: {key}")
            previous_hash = str(source.attrs.get("code_diff_hash", ""))
        shutil.copyfile(path, partial)
        try:
            with h5py.File(partial, "r+") as target:
                target.attrs["provenance_migrated_from_code_diff_hash"] = previous_hash
                target.attrs["provenance_migration_reason"] = "artifact_namespace_metadata_only_no_array_change"
                target.attrs["artifact_set"] = "built"
                target.attrs["code_commit"] = commit
                target.attrs["code_diff_hash"] = code_hash
                target.flush()
            os.replace(partial, path)
        finally:
            if partial.exists(): partial.unlink()
        checksum = sha256_file(path)
        frame.loc[list(indices), "code_commit"] = commit
        frame.loc[list(indices), "code_diff_hash"] = code_hash
        frame.loc[list(indices), "shard_sha256"] = checksum
        frame.loc[list(indices), "provenance_migration"] = "artifact_namespace_metadata_only_no_array_change"
        migrated += 1
    write_frame(frame, root / "manifests" / "manifest_built", pd)
    print(f"migrated provenance atomically for {migrated} included shards; EEG arrays were not rewritten")
    return 0


def validate(config: dict[str, Any], strict: bool) -> int:
    h5py, _, np = require_build_runtime()
    _, pd = runtime()
    root=output_root(config); errors=[]
    lock_path = root / "source_lock.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    current_commit, current_diff = git_provenance()
    current_code_hash = sha256_bytes(current_diff.encode())
    split_index_path = root / "splits" / "assignment.json"
    expected_split_hash = json.loads(split_index_path.read_text())["split_index_sha256"] if split_index_path.exists() else ""
    built_path=root/"manifests"/"manifest_built.csv"
    if not built_path.exists(): raise RuntimeError("build first")
    frame=pd.read_csv(built_path, keep_default_na=False, low_memory=False)
    observed_contracts: dict[str, set[str]] = {key: set() for key in ("preprocess_config_sha256", "source_lock_sha256", "split_index_sha256", "code_commit", "code_diff_hash")}
    for shard, rows in frame[frame.build_status == "included"].groupby("shard_path"):
        path=ROOT/shard
        if not path.exists(): errors.append(f"missing shard {shard}"); continue
        with h5py.File(path,"r") as h5:
            ds=rows.iloc[0].dataset; c=len(config["sources"][ds]["channel_order"]); t=config["harmonized"]["epoch"][ds]["total_samples_target"]
            for key in observed_contracts:
                observed_contracts[key].add(str(h5.attrs.get(key, "")))
            if h5.attrs.get("eeg_unit") != "V" or h5["eeg"].dtype != np.dtype("float32"): errors.append(f"unit/dtype {shard}")
            if h5["eeg"].shape[1:] != (c,t): errors.append(f"shape {shard}: {h5['eeg'].shape}")
            for required in ("channel_xyz", "eeg_valid_mask", "clean_perception_mask", "audio_loss_mask", "bad_channel_mask", "interpolated_channel_mask", "zero_filled_channel_mask", "channel_valid_mask"):
                if required not in h5: errors.append(f"missing {required} {shard}")
            if h5.attrs.get("channel_order_hash") != channel_order_hash(config["sources"][ds]["channel_order"]): errors.append(f"channel hash {shard}")
            if h5.attrs.get("preprocess_config_sha256", "") != config["_config_sha256"]: errors.append(f"config hash {shard}")
            if h5.attrs.get("source_lock_sha256", "") != lock.get("source_lock_sha256", ""): errors.append(f"source lock hash {shard}")
            if not expected_split_hash or h5.attrs.get("split_index_sha256", "") != expected_split_hash: errors.append(f"split hash {shard}")
            if h5.attrs.get("code_commit", "") != current_commit or h5.attrs.get("code_diff_hash", "") != current_code_hash:
                errors.append(f"preprocessing code provenance {shard}")
            if np.any(~np.isfinite(h5["eeg"][:])): errors.append(f"nonfinite {shard}")
            if h5["channel_xyz"].shape != (c, 3) or np.any(~np.isfinite(h5["channel_xyz"][:])): errors.append(f"channel xyz {shard}")
            for _, row in rows.iterrows():
                # CSV round-trips may promote this integer column to float when
                # excluded build rows contain an empty shard_row.
                index = int(float(row.shard_row))
                valid = h5["eeg_valid_mask"][index].astype(bool)
                if not valid.any() or (np.where(valid)[0][-1] + 1 != valid.sum()): errors.append(f"non-prefix time mask {row.trial_id}")
                acoustic = h5["audio_loss_mask"][index].astype(bool)
                if row.pairing_level != "verified_exact" and acoustic.any(): errors.append(f"weak pairing has acoustic loss {row.trial_id}")
    if len(frame[frame.build_status == "included"]):
        for key, values in observed_contracts.items():
            if len(values) != 1 or "" in values:
                errors.append(f"mixed shard preprocessing contract {key}: {sorted(values)}")
    for path in sorted((root/"splits").glob("*.csv")):
        split=pd.read_csv(path, keep_default_na=False)
        protocol = str(split.protocol.iloc[0]) if len(split) else ""
        columns = ["subject_group"] if protocol == "subject_ood" else ["linguistic_content_group", "waveform_group"]
        if protocol in {"joint_ood", "stage2_joint_ood"}: columns.insert(0, "subject_group")
        for role_a, role_b in (("train","test"),("train","validation"),("validation","test")):
            for column in columns:
                a=set(split[(split.role==role_a)&(split[column]!="")][column]); b=set(split[(split.role==role_b)&(split[column]!="")][column])
                if a & b: errors.append(f"split leakage {path.name} {column} {role_a}/{role_b}: {sorted(a & b)[:3]}")
        if protocol == "stage2_joint_ood":
            assignment_path = root / "splits" / "stage2_assignment.json"
            if not assignment_path.exists():
                errors.append("stage2 split is missing stage2_assignment.json")
            else:
                stage2_assignment = json.loads(assignment_path.read_text())
                for dataset, registered in stage2_assignment.get("datasets", {}).items():
                    dataset_rows = split[split.subject_group.str.startswith(f"{dataset}:")]
                    for role in ("train", "validation", "test"):
                        role_rows = dataset_rows[dataset_rows.role == role]
                        subject_count = sum(value == role for value in registered.get("subjects", {}).values())
                        for axis, content_key in (("audio", "contents"), ("label", "label_contents")):
                            content_count = sum(value == role for value in registered.get(content_key, {}).values())
                            expected = subject_count * content_count
                            actual = role_rows[role_rows.supervision_axis == axis]
                            cells = actual.groupby(["subject_group", "linguistic_content_group"]).size()
                            if len(actual) != expected or (len(cells) and not cells.eq(1).all()):
                                errors.append(
                                    f"stage2 exact grid {dataset}/{axis}/{role}: {len(actual)} != {expected}"
                                )
    # Content-supervised rows must have a real target; zeros are never a legal
    # fallback.  Label-only rows intentionally need no audio target.
    target_path = root / "speech_targets" / "speech_targets.h5"
    supervised = frame[(frame.build_status == "included") & frame.supervision_type.isin(["paired_audio", "weak_audio"])]
    if len(supervised):
        if not target_path.exists():
            errors.append("missing speech target cache")
        else:
            with h5py.File(target_path, "r") as targets:
                if targets.attrs.get("preprocess_config_sha256", "") != config["_config_sha256"]:
                    errors.append("speech target config hash mismatch")
                if targets.attrs.get("source_lock_sha256", "") != lock.get("source_lock_sha256", ""):
                    errors.append("speech target source lock mismatch")
                if targets.attrs.get("code_commit", "") != current_commit or targets.attrs.get("code_diff_hash", "") != current_code_hash:
                    errors.append("speech target code provenance mismatch")
                target_implementation = HERE / "cache_speech_targets.py"
                if targets.attrs.get("target_code_sha256", "") != sha256_file(target_implementation):
                    errors.append("speech target implementation hash mismatch")
                missing_targets = []
                for _, row in supervised.iterrows():
                    audio_id = str(row.get("audio_id", "")) or (f"audio-{row.audio_sha256[:16]}-{row.audio_semantics}" if row.audio_sha256 else "")
                    if not audio_id or audio_id not in targets:
                        missing_targets.append(audio_id or f"trial:{row.trial_id}")
                missing_targets = sorted(set(missing_targets))
                if missing_targets:
                    errors.append(f"missing speech targets: {missing_targets[:5]} (n={len(missing_targets)})")
    normalizer_path = root / "normalizers" / "joint_ood_fold-0.json"
    if normalizer_path.exists():
        normalizer = json.loads(normalizer_path.read_text())
        expected_normalizer = {
            "preprocess_config_sha256": config["_config_sha256"],
            "source_lock_sha256": lock.get("source_lock_sha256", ""),
            "split_index_sha256": expected_split_hash,
            "code_commit": current_commit,
            "code_diff_hash": current_code_hash,
        }
        if normalizer.get("preprocessing_contract") != expected_normalizer:
            errors.append("normalizer preprocessing contract mismatch")
    # Deterministic 20-pair DS004940 review sheet.  Machine-verifiable fields
    # are checked here; listening/transcript semantics remain explicitly human.
    candidates = frame[(frame.dataset == "ds004940") & (frame.build_status == "included")].copy()
    candidates["review_order"] = candidates.trial_id.map(
        lambda value: sha256_bytes(f"pair-review|{config['split_seed']}|{value}".encode())
    )
    review_path = root/"qc"/"ds004940_pair_review_20.csv"
    prior_human_status = {}
    if review_path.exists():
        prior = pd.read_csv(review_path, keep_default_na=False)
        if {"trial_id", "human_listen_transcript_status"}.issubset(prior.columns):
            prior_human_status = dict(zip(prior.trial_id, prior.human_listen_transcript_status))
    review_rows = []
    for _, row in candidates.sort_values("review_order").head(20).iterrows():
        audio_path = ROOT / row.audio_path
        event_path = ROOT / row.source_event_path
        eeg_path = ROOT / row.source_eeg_path
        audio_ok = audio_path.exists() and sha256_file(audio_path) == row.audio_sha256
        event_ok = event_path.exists() and sha256_file(event_path) == row.source_event_sha256
        mapping_ok = int(row.source_start_sample) <= int(row.source_zero_sample) < int(row.source_end_sample)
        machine_pass = bool(audio_ok and event_ok and eeg_path.exists() and mapping_ok and row.pairing_level == "verified_exact")
        if not machine_pass:
            errors.append(f"DS004940 pair review failed {row.trial_id}")
        review_rows.append({"trial_id":row.trial_id,"subject":row.subject,"task":row.task,"condition":row.condition,
                            "stim_file":row.stim_file,"audio_path":row.audio_path,"audio_sha256":row.audio_sha256,
                            "source_eeg_path":row.source_eeg_path,"event_onset_seconds":row.event_onset_seconds,
                            "event_run_local_onset_seconds":row.event_run_local_onset_seconds,
                            "stimulus_duration_seconds":row.stimulus_duration_seconds,"audio_hash_pass":audio_ok,
                            "event_hash_pass":event_ok,"epoch_mapping_pass":mapping_ok,"machine_pair_review_pass":machine_pass,
                            "human_listen_transcript_status":prior_human_status.get(row.trial_id,"pending")})
    pd.DataFrame(review_rows).to_csv(review_path,index=False)
    approved = {"pass", "passed", "verified", "approved"}
    human_status = "pass" if review_rows and all(
        str(row["human_listen_transcript_status"]).strip().lower() in approved for row in review_rows
    ) else "pending"
    psd_path = root / "qc" / "preprocessing_psd.json"
    psd_status = json.loads(psd_path.read_text()).get("status", "fail") if psd_path.exists() else "pending"
    if strict and psd_status != "pass":
        errors.append(f"preprocessing PSD QC is {psd_status}")
    formal_ready = not errors and human_status == "pass" and psd_status == "pass"
    report={"status":"pass" if not errors else "fail", "errors":errors,
            "ds004940_machine_pair_review_count":len(review_rows),
            "ds004940_human_listen_transcript_status":human_status,
            "preprocessing_psd_status": psd_status,
            "formal_m0_ready": formal_ready,
            "formal_m0_blockers": [name for name, passed in {
                "machine_validation": not errors,
                "ds004940_human_pair_review": human_status == "pass",
                "preprocessing_psd_qc": psd_status == "pass",
            }.items() if not passed],
            "created_at":time.time()}; (root/"qc"/"validate.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return 2 if errors and strict else 0


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("audit"); a.add_argument("--strict",action="store_true")
    ab=sub.add_parser("build-audio-bank"); ab.add_argument("--resume", action="store_true")
    sub.add_parser("make-splits")
    b=sub.add_parser("build"); b.add_argument("--dataset",choices=["all","ds004940"],default="all"); b.add_argument("--subjects",default="all"); b.add_argument("--tasks",default="all"); b.add_argument("--limit-trials-per-group",type=int); b.add_argument("--common-contents",type=int); b.add_argument("--content-ids"); b.add_argument("--split-role",choices=["any","train","validation","test"],default="any"); b.add_argument("--split-protocol",choices=["subject_ood","audio_ood","joint_ood","stage2_joint_ood"],default="joint_ood"); b.add_argument("--split-fold",type=int,default=0); b.add_argument("--artifact-set",default="built"); b.add_argument("--resume",action="store_true"); b.add_argument("--allow-audit-warnings",action="store_true")
    n=sub.add_parser("fit-normalizer"); n.add_argument("--split-csv",type=Path,required=True); n.add_argument("--fold",type=int,required=True); n.add_argument("--manifest-kind",default="built"); n.add_argument("--normalizer-name"); n.add_argument("--allow-mixed-production",action="store_true")
    sub.add_parser("migrate-provenance")
    v=sub.add_parser("validate"); v.add_argument("--strict",action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args=parser().parse_args(argv); config,_=load_config(args.config)
    if args.command=="audit": return audit(config,args.strict)
    if args.command=="build-audio-bank": return build_audio_bank(config,args.resume)
    if args.command=="make-splits": return make_splits(config)
    if args.command=="build": return build(config,args.dataset,args.subjects,args.tasks,args.limit_trials_per_group,args.common_contents,args.content_ids,args.split_role,args.split_protocol,args.split_fold,args.resume,args.allow_audit_warnings,args.artifact_set)
    if args.command=="fit-normalizer": return fit_normalizer(config,args.split_csv,args.fold,args.allow_mixed_production,args.manifest_kind,args.normalizer_name)
    if args.command=="migrate-provenance": return migrate_provenance(config)
    return validate(config,args.strict)


if __name__ == "__main__":
    try: raise SystemExit(main())
    except RuntimeError as exc: print(f"error: {exc}",file=sys.stderr); raise SystemExit(2)

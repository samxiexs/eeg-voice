"""Strict subject/content splits and timestamp-preserving BrainVision preparation."""
from __future__ import annotations
import csv
from functools import lru_cache
from pathlib import Path
import re
import numpy as np
from scipy import signal
import soundfile as sf
import torch
from torch.utils.data import Dataset
from .perceived_audio import acoustic_features, read_wave
from .utils import Moments, atomic_json, digest, fingerprint, read_json


def textgrid_words(path: Path) -> list[dict]:
    from praatio import textgrid
    grid = textgrid.openTextgrid(str(path), includeEmptyIntervals=True)
    names = [n for n in grid.tierNames if "word" in n.lower()]
    if len(names) != 1: raise ValueError(f"Ambiguous word tier: {path}: {grid.tierNames}")
    words = []
    for start, end, text in grid.getTier(names[0]).entries:
        text = text.strip()
        if text and text.lower() not in {"sil", "sp", "spn", "<sil>", "<eps>"}:
            words.append({"start": float(start), "end": float(end), "word": text})
    if not words: raise ValueError(f"No word annotations in {path}")
    return words


def reference(words, start, seconds=4):
    return " ".join(w["word"] for w in words if start <= (w["start"] + w["end"]) / 2 < start + seconds)


def roi_mapping(names: list[str], centers: list[str]) -> dict:
    import mne
    from mne.transforms import apply_trans
    actual = mne.channels.make_standard_montage("biosemi128")
    anchors = mne.channels.make_standard_montage("standard_1020")
    def positions(montage, selected):
        source = montage.get_positions()["ch_pos"]
        trans = mne.channels.compute_native_head_t(montage)
        xyz = apply_trans(trans, np.array([source[n] for n in selected]))
        return xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    xyz, anchor = positions(actual, names), positions(anchors, centers)
    groups = np.linalg.norm(xyz[:, None] - anchor[None], axis=-1).argmin(1)
    if set(groups.tolist()) != set(range(len(centers))): raise ValueError("Empty scalp group")
    return {"names": names, "positions_head_unit": xyz.tolist(), "centers": centers, "groups": groups.tolist(), "source": "MNE biosemi128 montage; acquisition A1-D32 names"}


def zero_phase_fir(x, kernel):
    half = len(kernel) // 2
    padded = np.pad(x, ((0, 0), (half, half)), mode="reflect")
    return signal.fftconvolve(padded, kernel[None], mode="valid", axes=-1)


def filter_eeg(eeg_uv: np.ndarray, rate: float, cfg):
    p = cfg["preprocessing"]
    taps = int(round(p["fir_seconds"] * rate)) | 1
    kernel = signal.firwin(taps, [p["highpass"], p["lowpass"]], pass_zero=False, fs=rate, window="hamming")
    clean = zero_phase_fir(eeg_uv, kernel)
    from fractions import Fraction
    ratio = Fraction(p["eeg_rate"] / rate).limit_denominator(10000)
    main = signal.resample_poly(clean, ratio.numerator, ratio.denominator, axis=-1)
    neural_kernel = signal.firwin(p["eeg_rate"] + 1, p["neural_lowpass"], fs=p["eeg_rate"], window="hamming")
    neural = signal.resample_poly(zero_phase_fir(main, neural_kernel), 1, 2, axis=-1)
    return main.astype(np.float32), neural.astype(np.float32), {"main_kernel_samples": taps, "main_support_seconds": (taps - 1) / rate, "neural_kernel_samples": len(neural_kernel)}


def recording_dir(cfg, subject, audio):
    return cfg.dir("artifacts") / "recordings" / subject / audio


def calibrated_microvolts(nominal_volts, cfg):
    multiplier=cfg["dataset"]["header_unit_multiplier"]
    audit=read_json(cfg.root/cfg["dataset"]["unit_calibration_report"])
    if audit["dataset_revision"]!=cfg["dataset"]["revision"] or not audit["source_matches"] or multiplier!=audit["header_unit_multiplier"]:
        raise ValueError("Source-unit calibration is missing or belongs to a different snapshot")
    return nominal_volts*1e6*multiplier


def prepare_recording(cfg, vhdr: Path) -> dict:
    import mne
    match = re.search(r"(sub-\d+)_task-listening_run-(\d+)_eeg", vhdr.name)
    if not match: raise ValueError(f"Unexpected BrainVision filename: {vhdr}")
    subject, run = match.group(1), int(match.group(2))
    audio = cfg["dataset"]["run_audio"][run]
    stem = vhdr.name.removesuffix("_eeg.vhdr")
    tsv = vhdr.with_name(stem + "_channels.tsv")
    with tsv.open() as handle: channels = list(csv.DictReader(handle, delimiter="\t"))
    picked = [row["name"] for row in channels if row["type"].upper() == "EEG"]
    if len(picked) != cfg["preprocessing"]["channels"]: raise ValueError(f"Expected 128 EEG channels: {vhdr}")
    canonical = [f"{letter}{i}" for letter in "ABCD" for i in range(1,33)]
    if set(picked) != set(canonical): raise ValueError("Unknown BioSemi channel mapping; do not infer positions by index")
    picked = canonical
    destination = recording_dir(cfg, subject, audio)
    meta_path = destination / "meta.json"
    if meta_path.exists():
        saved = read_json(meta_path)
        if saved["config_hash"] == cfg.hash: return saved
        raise ValueError("Stale preprocessing cache; change output root or explicitly remove cache")
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
    raw.pick(picked)
    raw.reorder_channels(picked)
    # Read the header normally, then apply the explicitly audited corpus export
    # correction. Never guess a per-record multiplier from amplitudes.
    uv = calibrated_microvolts(raw.get_data(), cfg)
    if not np.isfinite(uv).all(): raise ValueError(f"Nonfinite EEG: {vhdr}")
    good = np.array([next(row["status"] for row in channels if row["name"] == n).lower() != "bad" for n in picked])
    if not good.any(): raise ValueError("All EEG channels marked bad")
    uv -= uv[good].mean(0, keepdims=True)
    uv[~good] = 0
    wave_path = cfg.dir("raw") / "stimuli" / f"{audio}.wav"
    grid_path = wave_path.with_suffix(".TextGrid")
    if not wave_path.exists() or not grid_path.exists(): raise FileNotFoundError("EEG preparation requires paired real audio and TextGrid")
    info = sf.info(wave_path)
    offset = cfg["dataset"]["eeg_start_seconds"]
    if raw.n_times / raw.info["sfreq"] < info.duration + offset: raise ValueError("EEG is shorter than its mapped audio")
    main, neural, filters = filter_eeg(uv, raw.info["sfreq"], cfg)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "eeg.npy", main)
    np.save(destination / "neural.npy", neural)
    mapping = roi_mapping(picked, cfg["preprocessing"]["roi_centers"])
    meta = {"subject": subject, "run": run, "audio": audio, "duration": info.duration, "offset": offset, "clock_drift": 0.0, "alignment_source": cfg["dataset"]["alignment_source"], "eeg_source": str(vhdr.relative_to(cfg.dir("raw"))), "eeg_sha256": digest(vhdr.with_suffix(".eeg")), "audio_sha256": digest(wave_path), "raw_rate": raw.info["sfreq"], "units": "uV", "good_channels": good.tolist(), "montage": mapping, "filters": filters, "config_hash": cfg.hash, "dataset_revision": cfg["dataset"]["revision"]}
    atomic_json(meta_path, meta)
    return meta


def qc_window(core: np.ndarray, known_good: np.ndarray, cfg):
    p = cfg["preprocessing"]
    good = known_good & (core.std(-1) >= p["flat_uv"]) & (np.ptp(core, axis=-1) <= p["peak_to_peak_uv"])
    good &= np.isfinite(core).all(-1)
    return good, (1 - good.mean()) <= p["max_bad_fraction"]


def windows_for_recording(cfg, meta) -> tuple[list[dict], int]:
    p, w = cfg["preprocessing"], cfg["window"]
    eeg = np.load(recording_dir(cfg, meta["subject"], meta["audio"]) / "eeg.npy", mmap_mode="r")
    good = np.array(meta["good_channels"])
    stride = w["train_stride"] if meta["audio"] in cfg.splits["audio"]["train"] else w["evaluation_stride"]
    guard = max(p["guard_seconds"], meta["filters"]["main_support_seconds"] / 2 + w["eeg_right"] + 1)
    rows, rejected = [], 0
    for a in np.arange(guard, meta["duration"] - guard - w["seconds"] + 1e-6, stride):
        start = round((a + meta["offset"]) * p["eeg_rate"])
        core = eeg[:, start:start + round(w["seconds"] * p["eeg_rate"])]
        mask, accepted = qc_window(core, good, cfg)
        if not accepted: rejected += 1; continue
        audio_key = f"{meta['audio']}_{a:.3f}"
        rows.append({"id": f"{meta['subject']}/{audio_key}", "subject": meta["subject"], "audio": meta["audio"], "start": float(a), "offset": meta["offset"], "key": audio_key, "good_channels": mask.tolist()})
    return rows, rejected


def prepare(cfg, audio_model, subjects=None) -> dict:
    acquisition = read_json(cfg.dir("raw") / "acquisition.json")
    if not acquisition["complete"] or acquisition["revision"] != cfg["dataset"]["revision"]:
        raise ValueError("Complete checksum-verified paired download required")
    subjects = subjects or cfg["dataset"]["subjects"]
    manifest_path = cfg.dir("artifacts") / "manifest.json"
    previous = read_json(manifest_path) if manifest_path.exists() else None
    if previous and previous["config_hash"] != cfg.hash: raise ValueError("Preprocessing config changed under existing manifest")
    recordings, rows, rejections = [], [], {}
    for subject in subjects:
        sources = sorted((cfg.dir("raw") / subject).rglob("*_eeg.vhdr"))
        if not sources: raise FileNotFoundError(f"No EEG recordings: {subject}")
        for vhdr in sources:
            meta = prepare_recording(cfg, vhdr)
            new_rows, rejected = windows_for_recording(cfg, meta)
            recordings.append(meta); rows.extend(new_rows)
            rejections[f"{subject}/{meta['audio']}"] = {"accepted": len(new_rows), "rejected": rejected}
    # Verify globally unique acoustic content across content partitions.
    audio_hashes = {}
    words = {}
    transcript_hashes = {}
    for split, clips in cfg.splits["audio"].items():
        for clip in clips:
            path = cfg.dir("raw") / "stimuli" / f"{clip}.wav"
            # Compare decoded canonical samples, not WAV headers or encodings.
            sha = __import__("hashlib").sha256(read_wave(path,24000).tobytes()).hexdigest()
            if sha in audio_hashes: raise ValueError("Repeated audio across named stimuli; revise split before training")
            audio_hashes[sha] = clip
            words[clip] = textgrid_words(path.with_suffix(".TextGrid"))
            transcript = " ".join(w["word"].lower() for w in words[clip])
            if transcript in transcript_hashes:
                raise ValueError("Repeated full stimulus transcript; review content grouping before training")
            transcript_hashes[transcript] = clip
    metadata = {"config_hash": cfg.hash, "dataset_revision": acquisition["revision"], "subjects": subjects, "complete_subjects": sorted(subjects) == sorted(cfg["dataset"]["subjects"]), "recordings": recordings, "windows": rows, "rejections": rejections, "words": words, "unique_audio_seconds": sum(sf.info(cfg.dir('raw')/'stimuli'/f'{a}.wav').duration for a in words)}
    atomic_json(manifest_path, metadata)
    from .pipeline import prepare_audio_targets
    prepare_audio_targets(cfg, audio_model)
    return metadata


@lru_cache(maxsize=16)
def load_array(path: str): return np.load(path, mmap_mode="r")


class WindowDataset(Dataset):
    def __init__(self, cfg, fold, split, subjects=None, clips=None, targets=True, fixed_validation=False):
        self.cfg, self.fold, self.split, self.targets = cfg, fold, split, targets
        self.manifest = read_json(cfg.dir("artifacts") / "manifest.json")
        if self.manifest["config_hash"] != cfg.hash: raise ValueError("Stale window manifest")
        people = set(subjects or cfg.fold_subjects(fold, split))
        audio = set(clips or cfg.splits["audio"][split])
        self.rows = [r for r in self.manifest["windows"] if r["subject"] in people and r["audio"] in audio]
        if fixed_validation:
            from .utils import stable_int
            keys = {}
            for clip in sorted(audio):
                candidate = sorted({r["key"] for r in self.rows if r["audio"] == clip}, key=stable_int)
                keys[clip] = set(candidate[:cfg["training"]["validation_windows_per_audio"]])
            self.rows = [r for r in self.rows if r["key"] in keys[r["audio"]]]
        if not self.rows: raise ValueError(f"No accepted windows for {fold}/{split}")
        self.stats = read_json(cfg.dir("artifacts") / "stats" / f"{fold}.json")
        if self.stats["config_hash"] != cfg.hash: raise ValueError("Stale fold normalization")
        self.audio_stats = read_json(cfg.dir("artifacts") / "stats" / "audio.json")

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row, cfg = self.rows[index], self.cfg
        p, w = cfg["preprocessing"], cfg["window"]
        folder = recording_dir(cfg, row["subject"], row["audio"])
        eeg = load_array(str(folder / "eeg.npy"))
        start = round((row["start"] + row["offset"] - w["eeg_left"]) * p["eeg_rate"])
        samples = round((w["seconds"] + w["eeg_left"] + w["eeg_right"]) * p["eeg_rate"])
        x = np.array(eeg[:, start:start + samples].T)
        mask = np.array(row["good_channels"])
        x = (x - np.array(self.stats["eeg"]["mean"])) / np.array(self.stats["eeg"]["std"])
        x[:, ~mask] = 0
        neural = load_array(str(folder / "neural.npy"))
        low_start = round((row["start"] + row["offset"]) * p["neural_rate"])
        low = np.array(neural[:, low_start:low_start + round(w["seconds"] * p["neural_rate"])].T)
        low = (low - np.array(self.stats["neural"]["mean"])) / np.array(self.stats["neural"]["std"])
        low[:, ~mask] = 0
        target_path = cfg.dir("artifacts") / "audio_targets" / f"{row['key']}.npz"
        with np.load(target_path) as target:
            times = torch.from_numpy(target["times"].copy())
            result = {"eeg": torch.tensor(x, dtype=torch.float32), "neural": torch.tensor(low, dtype=torch.float32), "channel_mask": torch.from_numpy(mask), "times": times, "id": row["id"], "subject": row["subject"], "audio": row["audio"], "key": row["key"], "start": row["start"]}
            if self.targets:
                for name, stats_key in [("teacher", "teacher"), ("phi", "phi")]:
                    value = target[name].copy()
                    stat = self.audio_stats[stats_key]
                    result[name] = torch.tensor((value - stat["mean"]) / stat["std"], dtype=torch.float32)
                stat = self.audio_stats["mel"]
                result["mel"] = torch.tensor((target["mel"].copy() - np.array(stat["mean"])[:, None]) / np.array(stat["std"])[:, None], dtype=torch.float32)
        return result


def fit_fold_statistics(cfg, fold):
    manifest = read_json(cfg.dir("artifacts") / "manifest.json")
    people = set(cfg.fold_subjects(fold, "train"))
    clips = set(cfg.splits["audio"]["train"])
    main, low = Moments(128), Moments(128)
    count, rejected = 0, 0
    for meta in manifest["recordings"]:
        if meta["subject"] not in people or meta["audio"] not in clips: continue
        path = recording_dir(cfg, meta["subject"], meta["audio"])
        guard = cfg["preprocessing"]["guard_seconds"]
        accepted = [r for r in manifest["windows"] if r["subject"] == meta["subject"] and r["audio"] == meta["audio"]]
        for filename, rate, moments in [("eeg.npy", 128, main), ("neural.npy", 64, low)]:
            x = np.load(path / filename, mmap_mode="r")
            # Fit only accepted training windows; bad-channel observations are excluded.
            for row in accepted:
                first = round((row["start"] + row["offset"]) * rate)
                core = x[:, first:first + round(4*rate)].T
                moments.add(core, np.array(row["good_channels"])[None])
        qc = manifest["rejections"][f"{meta['subject']}/{meta['audio']}"]
        count += qc["accepted"]; rejected += qc["rejected"]
    if not count: raise ValueError("No train windows in this fold")
    if rejected / (count + rejected) > cfg["preprocessing"]["max_rejection_fraction"]:
        raise ValueError("Training QC rejection exceeds protocol gate; inspect units/recordings before training")
    value = {"config_hash": cfg.hash, "subjects": sorted(people), "clips": sorted(clips), "eeg": main.result(), "neural": low.result(), "accepted": count, "rejected": rejected}
    atomic_json(cfg.dir("artifacts") / "stats" / f"{fold}.json", value)
    return value

#!/usr/bin/env python3
"""Deterministic raw-versus-harmonized EEG PSD quality-control report."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

from prepare_training_data import ROOT, load_config, normalise_ds004_channel, output_root


def band_power(signal: np.ndarray, sample_rate: float, bands: dict[str, list[float]], seconds: float = 1.0) -> dict[str, float]:
    if signal.ndim != 2 or not signal.shape[0] or signal.shape[1] < 8 or not np.isfinite(signal).all():
        raise ValueError("PSD input must be finite [channels,time] EEG")
    frequencies, density = welch(signal, fs=sample_rate, axis=-1,
                                 nperseg=min(signal.shape[-1], max(8, int(round(sample_rate * seconds)))))
    density = np.median(density, axis=0)
    result = {}
    for name, (low, high) in bands.items():
        selected = (frequencies >= float(low)) & (frequencies < min(float(high), sample_rate / 2.0 + 1e-9))
        result[name] = float(np.trapezoid(density[selected], frequencies[selected])) if selected.sum() >= 2 else 0.0
    return result


def _stable(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _raw_epoch(row, channels: list[str]) -> tuple[np.ndarray, float]:
    path = ROOT / row.source_eeg_path
    raw = mne.io.read_raw_bdf(path, preload=False, verbose="ERROR") if path.suffix.lower() == ".bdf" else mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    aliases = {normalise_ds004_channel(name): name for name in raw.ch_names} if row.dataset == "ds004940" else {name: name for name in raw.ch_names}
    picks = [aliases[name] for name in channels if name in aliases]
    if not picks:
        raise RuntimeError(f"no canonical channels found in {path}")
    start = max(0, int(row.source_start_sample)); end = min(raw.n_times, int(row.source_end_sample))
    return raw.get_data(picks=picks, start=start, stop=end), float(raw.info["sfreq"])


def run(config: dict, trials_per_dataset: int | None = None) -> dict:
    root = output_root(config)
    frame = pd.read_csv(root / "manifests" / "manifest_built.csv", keep_default_na=False, low_memory=False)
    qc = config["preprocessing_qc"]
    bands = qc["bands_hz"]; requested = trials_per_dataset or int(qc["trials_per_dataset"])
    channel_limit = int(qc["channels_per_trial"]); seconds = float(qc["welch_seconds"])
    records = []
    for dataset in ("ds004940",):
        selected = frame[(frame.dataset == dataset) & (frame.build_status == "included")].copy()
        selected["_order"] = selected.trial_id.map(lambda value: _stable(f"preprocessing-psd|{value}"))
        selected = selected.sort_values("_order").head(requested)
        if len(selected) < requested:
            raise RuntimeError(f"PSD QC needs {requested} built {dataset} trials, found {len(selected)}")
        canonical = list(config["sources"][dataset]["channel_order"][:channel_limit])
        for _, row in selected.iterrows():
            raw_epoch, raw_rate = _raw_epoch(row, canonical)
            with h5py.File(ROOT / row.shard_path, "r") as shard:
                index = int(float(row.shard_row)); valid = shard["eeg_valid_mask"][index].astype(bool)
                processed = shard["eeg"][index][:channel_limit, valid].astype(np.float64)
            before = band_power(raw_epoch[:channel_limit], raw_rate, bands, seconds)
            after = band_power(processed, float(config["harmonized"]["target_sfreq_hz"]), bands, seconds)
            pass_before = max(before["passband"], 1e-30); pass_after = max(after["passband"], 1e-30)
            records.append({"trial_id": row.trial_id, "dataset": dataset, "raw_sfreq_hz": raw_rate,
                            "processed_sfreq_hz": config["harmonized"]["target_sfreq_hz"],
                            "raw_band_power": before, "processed_band_power": after,
                            "raw_dc_to_passband": before["dc"] / pass_before,
                            "processed_dc_to_passband": after["dc"] / pass_after,
                            "raw_high_to_passband": before["high"] / pass_before,
                            "processed_high_to_passband": after["high"] / pass_after,
                            "finite": bool(np.isfinite(processed).all())})
    high_reduced = [row["processed_high_to_passband"] < row["raw_high_to_passband"] for row in records]
    finite = all(row["finite"] for row in records)
    status = "pass" if finite and np.mean(high_reduced) >= 0.75 else "fail"
    report = {"status": status, "method": "scipy.signal.welch_median_across_locked_channels",
              "trials": records, "checks": {"all_finite": finite,
              "high_frequency_ratio_reduced_fraction": float(np.mean(high_reduced)),
              "required_reduced_fraction": 0.75}}
    target = root / "qc" / "preprocessing_psd.json"; target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2)); return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "training_data_v3.yaml")
    parser.add_argument("--trials-per-dataset", type=int)
    args = parser.parse_args(); config, _ = load_config(args.config)
    return 0 if run(config, args.trials_per_dataset)["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

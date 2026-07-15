from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.karaone_0715.data import LABELS, SplitManifest0715, robust_baseline_normalise, write_json  # noqa: E402


BANDS = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 20.0), (20.0, 30.0), (30.0, 40.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible 0715 leakage/signal probe; MM21 is never read.")
    parser.add_argument("--config", default=str(APP_DIR / "configs" / "karaone_0715.yaml"))
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else APP_DIR / path


class BundleCache:
    def __init__(self, root: Path):
        self.root = root
        self.values: dict[str, dict[str, np.ndarray]] = {}

    def get(self, relative_path: str) -> dict[str, np.ndarray]:
        if relative_path not in self.values:
            raw = np.load(self.root / relative_path, allow_pickle=False)
            required = ["trial_indices"]
            for stage_name in ("clearing", "stimulus_like", "thinking", "overt_like"):
                required.extend([f"stage__{stage_name}", f"stage__{stage_name}__valid_lengths"])
            self.values[relative_path] = {key: raw[key] for key in required}
            raw.close()
        return self.values[relative_path]


def stage(bundle: dict[str, np.ndarray], row: int, name: str) -> np.ndarray:
    key = f"stage__{name}"
    valid = int(bundle[f"{key}__valid_lengths"][row])
    return np.asarray(bundle[key][row, :, :valid], dtype=np.float32)


def spectral_temporal_features(eeg: np.ndarray, sample_rate: int = 256) -> np.ndarray:
    eeg = np.asarray(eeg, dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(eeg, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(eeg.shape[1], d=1.0 / float(sample_rate))
    band_power = np.stack(
        [np.log(spectrum[:, (frequencies >= low) & (frequencies < high)].mean(axis=1) + 1e-8) for low, high in BANDS],
        axis=1,
    )
    relative = band_power - band_power.mean(axis=1, keepdims=True)
    chunks = np.array_split(eeg, 4, axis=1)
    temporal = np.stack([np.sqrt(np.mean(np.square(chunk), axis=1) + 1e-8) for chunk in chunks], axis=1)
    return np.concatenate(([eeg.shape[1] / sample_rate], relative.reshape(-1), temporal.reshape(-1))).astype(np.float32)


def fit_probe(features: np.ndarray, labels: np.ndarray, train_mask: np.ndarray) -> dict[str, float]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1500, C=0.3, class_weight="balanced", random_state=15),
    )
    model.fit(features[train_mask], labels[train_mask])
    train_prediction = model.predict(features[train_mask])
    val_prediction = model.predict(features[~train_mask])
    return {
        "train_balanced_accuracy": float(balanced_accuracy_score(labels[train_mask], train_prediction)),
        "p02_balanced_accuracy": float(balanced_accuracy_score(labels[~train_mask], val_prediction)),
    }


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = resolve(cfg["data"]["root"])
    manifest = SplitManifest0715.build(root)
    with (root / "trials.csv").open("r", encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = [row for row in all_rows if row["subject_id"] in manifest.train_subjects or row["subject_id"] == manifest.subject_val]
    if any(row["subject_id"] == manifest.subject_test for row in rows):
        raise RuntimeError("Signal probe must not access MM21")
    labels = np.asarray([LABELS.index(row["label"]) for row in rows], dtype=np.int64)
    train_mask = np.asarray([row["subject_id"] in manifest.train_subjects for row in rows], dtype=bool)
    cache = BundleCache(root)
    feature_sets: dict[str, list[np.ndarray]] = {"length_only": [], "clearing": [], "stimulus_like": [], "thinking": [], "overt_instance": [], "overt_clearing_calibrated": []}
    for record in tqdm(rows, desc="[0715 probe] extract features", unit="trial", dynamic_ncols=True):
        bundle = cache.get(record["eeg_subject_bundle"])
        index = int(np.flatnonzero(np.asarray(bundle["trial_indices"], dtype=np.int64) == int(record["trial_index"]))[0])
        clearing = stage(bundle, index, "clearing")
        overt = stage(bundle, index, "overt_like")
        feature_sets["length_only"].append(np.asarray([overt.shape[1] / 256.0], dtype=np.float32))
        for stage_name in ("clearing", "stimulus_like", "thinking"):
            value = stage(bundle, index, stage_name)
            value = (value - value.mean(axis=1, keepdims=True)) / np.maximum(value.std(axis=1, keepdims=True), 1e-5)
            feature_sets[stage_name].append(spectral_temporal_features(value))
        instance = (overt - overt.mean(axis=1, keepdims=True)) / np.maximum(overt.std(axis=1, keepdims=True), 1e-5)
        feature_sets["overt_instance"].append(spectral_temporal_features(instance))
        feature_sets["overt_clearing_calibrated"].append(spectral_temporal_features(robust_baseline_normalise(overt, clearing)))
    results = {name: {"feature_dim": int(np.stack(values).shape[1]), **fit_probe(np.stack(values), labels, train_mask)} for name, values in feature_sets.items()}
    report: dict[str, Any] = {
        "version": "0715",
        "phase": "signal_probe",
        "purpose": "Estimate cross-subject decodability and detect timing/stage leakage before neural generation.",
        "train_n": int(train_mask.sum()),
        "p02_n": int((~train_mask).sum()),
        "chance_accuracy": 1.0 / len(LABELS),
        "subject_test_accessed": False,
        "model": "StandardScaler + class-balanced multinomial logistic regression",
        "results": results,
        "interpretation_rule": "A useful overt signal should exceed length-only and non-overt stages on P02; train accuracy alone is not evidence.",
    }
    destination = Path(args.output) if args.output else resolve(cfg["paths"]["output_root"]) / "karaone_0715_signal_probe.json"
    write_json(destination, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

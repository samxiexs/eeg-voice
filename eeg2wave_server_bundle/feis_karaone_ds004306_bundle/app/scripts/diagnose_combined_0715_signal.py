from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.combined_0715.data import DATASETS, CombinedContext, load_context  # noqa: E402


BANDS_HZ: tuple[tuple[float, float], ...] = (
    (1.0, 4.0),
    (4.0, 8.0),
    (8.0, 13.0),
    (13.0, 20.0),
    (20.0, 30.0),
    (30.0, 40.0),
)
N_CHANNELS = 14
N_TIME_BLOCKS = 4
FEATURE_DIM = N_CHANNELS * (len(BANDS_HZ) + N_TIME_BLOCKS) + 1


def balanced_accuracy(target: np.ndarray, prediction: np.ndarray, classes: int) -> float:
    recalls = []
    for label in range(classes):
        selected = target == label
        if selected.any():
            recalls.append(float(np.mean(prediction[selected] == label)))
    return float(np.mean(recalls)) if recalls else float("nan")


def extract_signal_features(
    eeg: np.ndarray,
    valid_length: int,
    sfreq: int = 256,
) -> np.ndarray:
    """Return channel-resolved spectral/RMS features plus valid-duration.

    Only the valid prefix is inspected, so padding values cannot leak dataset or
    class information into the probe.  The feature order is channel-major:
    six relative log-band powers, four log-RMS blocks, then one duration value.
    """
    eeg = np.asarray(eeg, dtype=np.float32)
    if eeg.ndim != 2 or eeg.shape[0] != N_CHANNELS:
        raise ValueError(f"Expected EEG [14,T], got {eeg.shape}")
    valid = min(max(int(valid_length), 1), eeg.shape[1])
    signal = np.asarray(eeg[:, :valid], dtype=np.float64)
    if not np.isfinite(signal).all():
        raise ValueError("Signal probe input contains NaN or Inf in its valid prefix")

    # Match the KaraOne 0715 probe definition: log FFT band power is centred
    # across the six bands within each channel, retaining spectral shape while
    # removing each channel's global log-power offset.
    spectrum = np.abs(np.fft.rfft(signal, axis=-1)) ** 2
    frequencies = np.fft.rfftfreq(valid, d=1.0 / float(sfreq))
    log_band_values: list[np.ndarray] = []
    for low, high in BANDS_HZ:
        selected = (frequencies >= low) & (frequencies < high)
        power = spectrum[:, selected].mean(axis=-1) if selected.any() else np.zeros(N_CHANNELS)
        log_band_values.append(np.log(power + 1.0e-8))
    log_band_power = np.stack(log_band_values, axis=1)
    relative_log_band_power = log_band_power - log_band_power.mean(axis=1, keepdims=True)

    features: list[float] = []
    for channel in range(N_CHANNELS):
        features.extend(relative_log_band_power[channel].tolist())
        for block in np.array_split(signal[channel], N_TIME_BLOCKS):
            rms = float(np.sqrt(np.mean(np.square(block)) + 1.0e-8)) if block.size else 0.0
            features.append(rms)
    features.append(float(valid / float(sfreq)))
    output = np.asarray(features, dtype=np.float32)
    if output.shape != (FEATURE_DIM,) or not np.isfinite(output).all():
        raise RuntimeError(f"Invalid signal feature vector: shape={output.shape}")
    return output


class _BundleReader:
    def __init__(self, context: CombinedContext, eeg_len: int, max_open: int = 8):
        self.context = context
        self.eeg_len = int(eeg_len)
        self.max_open = max_open
        self.cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def trial(self, row: dict[str, str]) -> tuple[np.ndarray, int]:
        relative = row["eeg_relpath"]
        if relative not in self.cache:
            self.cache[relative] = dict(np.load(self.context.root / relative, allow_pickle=False))
            while len(self.cache) > self.max_open:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(relative)
        bundle = self.cache[relative]
        eeg = np.asarray(bundle["eeg"][int(row["eeg_row"])], dtype=np.float32)[:, : self.eeg_len]
        valid = min(max(int(row["eeg_valid_samples"]), 1), eeg.shape[-1])
        return eeg, valid


def _split_arrays(
    context: CombinedContext,
    reader: _BundleReader,
    dataset: str,
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [
        row for row in context.rows
        if row["dataset"] == dataset and context.split_for(row) == split
    ]
    if not rows:
        raise ValueError(f"No {dataset} rows in locked {split} split")
    features: list[np.ndarray] = []
    labels: list[int] = []
    lengths: list[float] = []
    for row in rows:
        eeg, valid = reader.trial(row)
        features.append(extract_signal_features(eeg, valid))
        labels.append(context.label_to_local[(dataset, row["label"])])
        lengths.append(valid / max(eeg.shape[-1], 1))
    return (
        np.stack(features),
        np.asarray(labels, dtype=np.int64),
        np.asarray(lengths, dtype=np.float32).reshape(-1, 1),
    )


def _fit_classifier(features: np.ndarray, labels: np.ndarray, seed: int) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=seed,
        ),
    ).fit(features, labels)


def run_probe(config_path: str | Path, seed: int = 15) -> dict[str, object]:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    context = load_context(config_path)
    eeg_len = int(config["data"]["eeg_len"])
    reader = _BundleReader(context, eeg_len=eeg_len)
    results: dict[str, object] = {
        "feature_schema": {
            "dimension": FEATURE_DIM,
            "channels": N_CHANNELS,
            "bands_hz": [list(band) for band in BANDS_HZ],
            "time_rms_blocks_per_channel": N_TIME_BLOCKS,
            "valid_duration_features": 1,
            "eeg_samples": eeg_len,
            "padding_ignored": True,
        },
        "split_version": context.split.get("version"),
        "test_subjects_read": False,
        "datasets": {},
    }
    dataset_results = results["datasets"]
    assert isinstance(dataset_results, dict)
    for dataset in DATASETS:
        x_train, y_train, length_train = _split_arrays(context, reader, dataset, "train")
        x_validation, y_validation, length_validation = _split_arrays(context, reader, dataset, "validation")
        classes = len({local for (name, _), local in context.label_to_local.items() if name == dataset})

        signal_model = _fit_classifier(x_train, y_train, seed)
        length_model = _fit_classifier(length_train, y_train, seed)
        train_prediction = signal_model.predict(x_train)
        validation_prediction = signal_model.predict(x_validation)
        length_validation_prediction = length_model.predict(length_validation)
        train_ba = balanced_accuracy(y_train, train_prediction, classes)
        validation_ba = balanced_accuracy(y_validation, validation_prediction, classes)
        length_ba = balanced_accuracy(y_validation, length_validation_prediction, classes)
        chance = 1.0 / classes
        dataset_results[dataset] = {
            "n_train": int(len(y_train)),
            "n_validation": int(len(y_validation)),
            "n_classes": classes,
            "train_balanced_accuracy": train_ba,
            "validation_balanced_accuracy": validation_ba,
            "chance": chance,
            "chance_margin": validation_ba - chance,
            "length_only_balanced_accuracy": length_ba,
            "gain_over_length_only": validation_ba - length_ba,
        }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe train-subject to validation-subject EEG signal probe.")
    parser.add_argument("--config", default=str(APP_DIR / "configs" / "combined_0715_v1.yaml"))
    parser.add_argument(
        "--cache",
        default=None,
        help="Deprecated compatibility argument; the signal probe no longer reads the audio cache.",
    )
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_probe(args.config, seed=args.seed)
    destination = Path(args.output) if args.output else Path(args.config).resolve().parents[2] / "artifacts" / "combined_0715_v1" / "signal_probe.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

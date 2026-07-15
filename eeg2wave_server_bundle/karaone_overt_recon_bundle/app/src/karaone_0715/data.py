from __future__ import annotations

import csv
import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from torch.utils.data import Dataset


VERSION = "0715"
TRAIN_SUBJECTS = ("MM05", "MM08", "MM09", "MM10", "MM11", "MM12", "MM14", "MM15", "MM16", "MM18", "MM19", "MM20")
SUBJECT_VAL = "P02"
SUBJECT_TEST = "MM21"
LABELS = ("/diy/", "/iy/", "/m/", "/n/", "/piy/", "/tiy/", "/uw/", "gnaw", "knew", "pat", "pot")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class TrialRecord0715:
    subject: str
    trial_index: int
    label: str
    audio_path: str
    eeg_subject_bundle: str

    @property
    def key(self) -> str:
        return f"{self.subject}:{self.trial_index}"


@dataclass(frozen=True)
class SplitManifest0715:
    version: str
    train_subjects: tuple[str, ...]
    subject_val: str
    subject_test: str
    trial_csv_sha256: str
    n_trials: int

    @classmethod
    def build(cls, root: str | Path) -> "SplitManifest0715":
        root = Path(root)
        records = read_trial_records(root)
        observed = {record.subject for record in records}
        expected = set(TRAIN_SUBJECTS) | {SUBJECT_VAL, SUBJECT_TEST}
        if observed != expected:
            raise ValueError(f"Unexpected KaraOne subjects: {sorted(observed)}")
        return cls(
            version=VERSION,
            train_subjects=TRAIN_SUBJECTS,
            subject_val=SUBJECT_VAL,
            subject_test=SUBJECT_TEST,
            trial_csv_sha256=sha256_bytes((root / "trials.csv").read_bytes()),
            n_trials=len(records),
        )

    def split_for(self, subject: str) -> str:
        if subject in self.train_subjects:
            return "subject_train"
        if subject == self.subject_val:
            return "subject_val"
        if subject == self.subject_test:
            return "subject_test"
        raise ValueError(f"Unknown subject: {subject}")

    @property
    def checksum(self) -> str:
        return sha256_bytes(json.dumps(asdict(self), sort_keys=True).encode("utf-8"))

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**asdict(self), "checksum": self.checksum}, indent=2) + "\n", encoding="utf-8")
        return path


def read_trial_records(root: str | Path) -> list[TrialRecord0715]:
    root = Path(root)
    with (root / "trials.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = [
        TrialRecord0715(
            subject=str(row["subject_id"]),
            trial_index=int(row["trial_index"]),
            label=str(row["label"]),
            audio_path=str(row["audio_path"]),
            eeg_subject_bundle=str(row["eeg_subject_bundle"]),
        )
        for row in rows
    ]
    if len(records) != 1913:
        raise ValueError(f"Expected 1913 trials, got {len(records)}")
    if {record.label for record in records} != set(LABELS):
        raise ValueError("Unexpected label vocabulary")
    return records


def records_for_split(root: str | Path, manifest: SplitManifest0715, split: str) -> list[TrialRecord0715]:
    if split not in {"subject_train", "subject_val", "subject_test"}:
        raise ValueError(f"Unsupported split: {split}")
    return [record for record in read_trial_records(root) if manifest.split_for(record.subject) == split]


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_audio(path: str | Path, sample_rate: int = 16000, duration_sec: float = 2.0) -> np.ndarray:
    source_rate, audio = wavfile.read(str(path))
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / float(np.iinfo(audio.dtype).max)
    audio = np.asarray(audio, dtype=np.float32)
    if int(source_rate) != int(sample_rate):
        gcd = int(np.gcd(int(source_rate), int(sample_rate)))
        audio = resample_poly(audio, int(sample_rate) // gcd, int(source_rate) // gcd).astype(np.float32)
    target = int(round(float(sample_rate) * float(duration_sec)))
    audio = audio[:target]
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-8))
    gain = min(0.08 / max(rms, 1e-8), 10.0)
    return np.clip(audio * gain, -0.95, 0.95).astype(np.float32)


def audio_envelope(audio: np.ndarray, steps: int = 150) -> tuple[np.ndarray, float, float]:
    chunks = np.array_split(np.asarray(audio, dtype=np.float32), int(steps))
    envelope = np.asarray([np.sqrt(np.mean(np.square(chunk), dtype=np.float64) + 1e-8) for chunk in chunks], dtype=np.float32)
    envelope = envelope / max(float(envelope.max(initial=0.0)), 1e-6)
    median = float(np.median(envelope))
    mad = float(np.median(np.abs(envelope - median)))
    active = envelope >= max(median + 1.5 * mad, 0.15)
    if active.any():
        start = int(np.flatnonzero(active)[0])
        end = int(np.flatnonzero(active)[-1]) + 1
    else:
        start, end = 0, 0
    return envelope, float(start / steps), float((end - start) / steps)


class AudioCodeBank:
    """Read-only 0715 EnCodec discrete-code target cache."""

    def __init__(self, path: str | Path, manifest: SplitManifest0715):
        self.path = Path(path)
        raw = np.load(self.path, allow_pickle=False)
        required = {
            "keys", "subjects", "labels", "audio_paths", "fit_split", "encodec_codes",
            "encodec_scale", "encodec_scale_valid", "audio_envelope", "onset", "duration",
        }
        missing = required - set(raw.files)
        if missing:
            raise ValueError(f"0715 cache is missing fields: {sorted(missing)}")
        self.keys = np.asarray(raw["keys"]).astype(str)
        self.subjects = np.asarray(raw["subjects"]).astype(str)
        self.labels = np.asarray(raw["labels"]).astype(str)
        self.audio_paths = np.asarray(raw["audio_paths"]).astype(str)
        self.fit_split = np.asarray(raw["fit_split"], dtype=bool)
        self.codes = np.asarray(raw["encodec_codes"], dtype=np.int64)
        self.scale = np.asarray(raw["encodec_scale"], dtype=np.float32)
        self.scale_valid = np.asarray(raw["encodec_scale_valid"], dtype=bool)
        self.envelope = np.asarray(raw["audio_envelope"], dtype=np.float32)
        self.onset = np.asarray(raw["onset"], dtype=np.float32)
        self.duration = np.asarray(raw["duration"], dtype=np.float32)
        if self.codes.ndim != 3:
            raise ValueError(f"encodec_codes must be [N,Q,T], got {self.codes.shape}")
        if self.codes.min() < 0 or self.codes.max() >= 1024:
            raise ValueError("EnCodec code index outside [0,1023]")
        n = len(self.keys)
        if any(len(value) != n for value in (self.subjects, self.labels, self.audio_paths, self.fit_split, self.codes, self.scale, self.scale_valid, self.envelope, self.onset, self.duration)):
            raise ValueError("0715 cache arrays do not share a trial count")
        self.splits = np.asarray([manifest.split_for(subject) for subject in self.subjects])
        if not np.array_equal(self.fit_split, self.splits == "subject_train"):
            raise ValueError("fit_split disagrees with the 0715 subject manifest")
        self.key_to_index = {key: index for index, key in enumerate(self.keys.tolist())}
        if len(self.key_to_index) != n:
            raise ValueError("Duplicate trial keys in 0715 cache")

    @property
    def codebooks(self) -> int:
        return int(self.codes.shape[1])

    @property
    def code_steps(self) -> int:
        return int(self.codes.shape[2])

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.splits == split).astype(np.int64)


class _SubjectBundleCache:
    def __init__(self, root: Path, max_items: int = 12):
        self.root = root
        self.max_items = int(max_items)
        self.items: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, relative_path: str) -> dict[str, np.ndarray]:
        if relative_path in self.items:
            self.items.move_to_end(relative_path)
            return self.items[relative_path]
        raw = np.load(self.root / relative_path, allow_pickle=False)
        required = (
            "trial_indices",
            "stage__clearing",
            "stage__clearing__valid_lengths",
            "stage__overt_like",
            "stage__overt_like__valid_lengths",
        )
        missing = set(required) - set(raw.files)
        if missing:
            raise ValueError(f"EEG subject bundle is missing 0715 fields: {sorted(missing)}")
        clearing = np.asarray(raw["stage__clearing"], dtype=np.float32)
        clearing_lengths = np.asarray(raw["stage__clearing__valid_lengths"], dtype=np.int64)
        centers = np.empty((len(clearing), clearing.shape[1], 1), dtype=np.float32)
        scales = np.empty_like(centers)
        for row, valid in enumerate(clearing_lengths.tolist()):
            baseline = clearing[row, :, : int(valid)]
            center = np.median(baseline, axis=1, keepdims=True)
            mad = 1.4826 * np.median(np.abs(baseline - center), axis=1, keepdims=True)
            fallback = np.std(baseline, axis=1, keepdims=True)
            centers[row] = center
            scales[row] = np.where(mad > 1e-5, mad, fallback)
        value = {
            "trial_indices": raw["trial_indices"],
            "stage__overt_like": raw["stage__overt_like"],
            "stage__overt_like__valid_lengths": raw["stage__overt_like__valid_lengths"],
            "baseline_center": centers,
            "baseline_scale": scales,
        }
        raw.close()
        self.items[relative_path] = value
        if len(self.items) > self.max_items:
            self.items.popitem(last=False)
        return value


def robust_baseline_normalise(overt: np.ndarray, clearing: np.ndarray, clip_value: float = 12.0) -> np.ndarray:
    """Same-trial EEG-only robust baseline normalization.

    Clearing is never an audio/reference input. It supplies channel-wise location
    and scale only; overt samples remain the sole content-bearing sequence.
    """

    overt = np.asarray(overt, dtype=np.float32)
    clearing = np.asarray(clearing, dtype=np.float32)
    center = np.median(clearing, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(clearing - center), axis=1, keepdims=True)
    fallback = np.std(clearing, axis=1, keepdims=True)
    scale = np.where(mad > 1e-5, mad, fallback)
    normalized = (overt - center) / np.maximum(scale, 1e-5)
    return np.clip(normalized, -float(clip_value), float(clip_value)).astype(np.float32)


def apply_baseline_stats(overt: np.ndarray, center: np.ndarray, scale: np.ndarray, clip_value: float = 12.0) -> np.ndarray:
    normalized = (np.asarray(overt, dtype=np.float32) - np.asarray(center, dtype=np.float32)) / np.maximum(np.asarray(scale, dtype=np.float32), 1e-5)
    return np.clip(normalized, -float(clip_value), float(clip_value)).astype(np.float32)


class KaraOne0715Dataset(Dataset[dict[str, Any]]):
    """Independent 0715 EEG reader with clearing-calibrated overt EEG."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        bank: AudioCodeBank | None = None,
        manifest: SplitManifest0715 | None = None,
        eeg_len: int = 768,
        baseline_mode: str = "robust_clearing",
        clip_value: float = 12.0,
    ):
        self.root = Path(root)
        self.manifest = manifest or SplitManifest0715.build(self.root)
        self.split = split
        self.records = records_for_split(self.root, self.manifest, split)
        self.bank = bank
        self.eeg_len = int(eeg_len)
        self.baseline_mode = str(baseline_mode)
        self.clip_value = float(clip_value)
        self.labels = {label: index for index, label in enumerate(LABELS)}
        self.subjects = {subject: index for index, subject in enumerate(TRAIN_SUBJECTS)}
        self.bundle_cache = _SubjectBundleCache(self.root)

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _trial_row(bundle: dict[str, np.ndarray], trial_index: int) -> int:
        matches = np.flatnonzero(np.asarray(bundle["trial_indices"], dtype=np.int64) == int(trial_index))
        if len(matches) != 1:
            raise KeyError(f"No unique EEG row for trial {trial_index}")
        return int(matches[0])

    @staticmethod
    def _stage(bundle: dict[str, np.ndarray], row: int, stage: str) -> np.ndarray:
        key = f"stage__{stage}"
        length_key = f"{key}__valid_lengths"
        valid = int(bundle[length_key][row])
        return np.asarray(bundle[key][row, :, :valid], dtype=np.float32)

    def __getitem__(self, item: int) -> dict[str, Any]:
        record = self.records[item]
        bundle = self.bundle_cache.get(record.eeg_subject_bundle)
        row = self._trial_row(bundle, record.trial_index)
        overt = self._stage(bundle, row, "overt_like")
        if self.baseline_mode == "robust_clearing":
            overt = apply_baseline_stats(overt, bundle["baseline_center"][row], bundle["baseline_scale"][row], self.clip_value)
        elif self.baseline_mode == "instance":
            mean = overt.mean(axis=1, keepdims=True)
            std = overt.std(axis=1, keepdims=True)
            overt = np.clip((overt - mean) / np.maximum(std, 1e-5), -self.clip_value, self.clip_value)
        else:
            raise ValueError(f"Unsupported baseline_mode: {self.baseline_mode}")
        valid_len = min(overt.shape[1], self.eeg_len)
        eeg = np.zeros((overt.shape[0], self.eeg_len), dtype=np.float32)
        eeg[:, :valid_len] = overt[:, :valid_len]
        result: dict[str, Any] = {
            "eeg": torch.from_numpy(eeg),
            "eeg_valid_len": torch.tensor(valid_len, dtype=torch.long),
            "label_idx": torch.tensor(self.labels[record.label], dtype=torch.long),
            "subject_idx": torch.tensor(self.subjects.get(record.subject, -1), dtype=torch.long),
            "label": record.label,
            "subject": record.subject,
            "key": record.key,
            "trial_index": torch.tensor(record.trial_index, dtype=torch.long),
            "audio_path": record.audio_path,
            "split": self.split,
        }
        if self.bank is not None:
            cache_index = self.bank.key_to_index.get(record.key)
            if cache_index is None:
                raise KeyError(f"No audio-code target for {record.key}")
            result.update(
                {
                    "codes": torch.from_numpy(np.ascontiguousarray(self.bank.codes[cache_index])).long(),
                    "audio_envelope": torch.from_numpy(np.ascontiguousarray(self.bank.envelope[cache_index])).float(),
                    "onset": torch.tensor(self.bank.onset[cache_index], dtype=torch.float32),
                    "duration": torch.tensor(self.bank.duration[cache_index], dtype=torch.float32),
                    "encodec_scale": torch.from_numpy(np.ascontiguousarray(self.bank.scale[cache_index])).float(),
                    "encodec_scale_valid": torch.tensor(bool(self.bank.scale_valid[cache_index]), dtype=torch.bool),
                }
            )
        return result


def fit_audit(records: Iterable[TrialRecord0715], manifest: SplitManifest0715, artifact: str) -> dict[str, Any]:
    rows = tuple(records)
    subjects = tuple(sorted({row.subject for row in rows}))
    offenders = sorted(set(subjects) - set(manifest.train_subjects))
    if offenders:
        raise ValueError(f"{artifact} attempted to fit held-out subjects: {offenders}")
    return {
        "artifact": artifact,
        "fit_split": "subject_train",
        "fit_subjects": list(subjects),
        "split_checksum": manifest.checksum,
        "n_fit_trials": len(rows),
        "source_keys_sha256": sha256_bytes("\n".join(sorted(row.key for row in rows)).encode("utf-8")),
    }

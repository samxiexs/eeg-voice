from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from torch.utils.data import Dataset


VERSION = "0711v1"
TRAIN_SUBJECTS = ("MM05", "MM08", "MM09", "MM10", "MM11", "MM12", "MM14", "MM15", "MM16", "MM18", "MM19", "MM20")
SUBJECT_VAL = "P02"
SUBJECT_TEST = "MM21"
LABELS = ("/diy/", "/iy/", "/m/", "/n/", "/piy/", "/tiy/", "/uw/", "gnaw", "knew", "pat", "pot")
BANDS = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 60.0))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalise_subject(subject: str) -> str:
    return str(subject).strip()


@dataclass(frozen=True)
class TrialRecord:
    subject: str
    trial_index: int
    label: str
    audio_path: str
    eeg_subject_bundle: str

    @property
    def key(self) -> str:
        return f"{self.subject}:{self.trial_index}"


@dataclass(frozen=True)
class SplitManifest:
    """The only split definition accepted by 0711v1."""

    version: str
    train_subjects: tuple[str, ...]
    subject_val: str
    subject_test: str
    trial_csv_sha256: str
    n_trials: int

    @classmethod
    def build(cls, root: str | Path) -> "SplitManifest":
        root = Path(root)
        csv_path = root / "trials.csv"
        rows = read_trial_records(root)
        subjects = {row.subject for row in rows}
        expected = set(TRAIN_SUBJECTS) | {SUBJECT_VAL, SUBJECT_TEST}
        if subjects != expected:
            raise ValueError(f"Unexpected KaraOne subject set: {sorted(subjects)}")
        return cls(
            version=VERSION,
            train_subjects=TRAIN_SUBJECTS,
            subject_val=SUBJECT_VAL,
            subject_test=SUBJECT_TEST,
            trial_csv_sha256=_sha256_bytes(csv_path.read_bytes()),
            n_trials=len(rows),
        )

    @property
    def heldout_subjects(self) -> tuple[str, str]:
        return (self.subject_val, self.subject_test)

    @property
    def checksum(self) -> str:
        return _sha256_bytes(json.dumps(asdict(self), sort_keys=True).encode("utf-8"))

    def split_for(self, subject: str) -> str:
        subject = _normalise_subject(subject)
        if subject in self.train_subjects:
            return "subject_train"
        if subject == self.subject_val:
            return "subject_val"
        if subject == self.subject_test:
            return "subject_test"
        raise ValueError(f"Unknown subject: {subject}")

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**asdict(self), "checksum": self.checksum}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path


@dataclass(frozen=True)
class FitAudit:
    """Proof that a learned object was fit exclusively on train subjects."""

    artifact: str
    fit_subjects: tuple[str, ...]
    fit_split: str
    split_checksum: str
    n_fit_trials: int
    source_keys_sha256: str

    @classmethod
    def from_records(cls, artifact: str, manifest: SplitManifest, records: Iterable[TrialRecord]) -> "FitAudit":
        rows = tuple(records)
        subjects = tuple(sorted({_normalise_subject(row.subject) for row in rows}))
        if set(subjects) - set(manifest.train_subjects):
            raise ValueError(f"{artifact} attempted to fit heldout subjects: {subjects}")
        keys = "\n".join(sorted(row.key for row in rows)).encode("utf-8")
        return cls(
            artifact=artifact,
            fit_subjects=subjects,
            fit_split="subject_train",
            split_checksum=manifest.checksum,
            n_fit_trials=len(rows),
            source_keys_sha256=_sha256_bytes(keys),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assert_train_only(records: Iterable[TrialRecord], manifest: SplitManifest, artifact: str) -> None:
    offenders = sorted({_normalise_subject(row.subject) for row in records} - set(manifest.train_subjects))
    if offenders:
        raise ValueError(f"{artifact} may only fit subject_train; found {offenders}")


def read_trial_records(root: str | Path) -> list[TrialRecord]:
    root = Path(root)
    with (root / "trials.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = [
        TrialRecord(
            subject=_normalise_subject(row["subject_id"]),
            trial_index=int(row["trial_index"]),
            label=str(row["label"]),
            audio_path=str(row["audio_path"]),
            eeg_subject_bundle=str(row["eeg_subject_bundle"]),
        )
        for row in rows
    ]
    if len(records) != 1913:
        raise ValueError(f"Expected 1913 trials, found {len(records)}")
    if {row.label for row in records} != set(LABELS):
        raise ValueError("Unexpected KaraOne label vocabulary")
    return records


def records_for_split(root: str | Path, manifest: SplitManifest, split: str) -> list[TrialRecord]:
    if split not in {"subject_train", "subject_val", "subject_test"}:
        raise ValueError(f"Unsupported split: {split}")
    return [row for row in read_trial_records(root) if manifest.split_for(row.subject) == split]


def git_commit(repo_root: str | Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except Exception:  # noqa: BLE001 - manifests should still work outside git
        return None


def make_run_manifest(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    split_manifest: SplitManifest,
    phase: str,
    stage: str,
    seed: int,
    input_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    config_path = Path(config_path)
    inputs = {}
    for item in input_paths:
        path = Path(item)
        if path.exists() and path.is_file():
            inputs[str(path)] = _sha256_bytes(path.read_bytes())
    return {
        "version": VERSION,
        "run_name": run_name(stage, phase, seed),
        "phase": phase,
        "stage": stage,
        "seed": int(seed),
        "git_commit": git_commit(repo_root),
        "config_sha256": _sha256_bytes(config_path.read_bytes()),
        "split_checksum": split_manifest.checksum,
        "train_subjects": list(split_manifest.train_subjects),
        "subject_val": split_manifest.subject_val,
        "subject_test": split_manifest.subject_test,
        "input_sha256": inputs,
        "test_accessed": False,
    }


def run_name(stage: str, phase: str, seed: int) -> str:
    allowed = {"audit", "audio_ssl", "eeg_ssl", "align_global", "align_token", "flow", "evaluate"}
    if phase not in allowed:
        raise ValueError(f"Unknown 0711v1 phase: {phase}")
    if stage not in {"overt_like", "thinking"}:
        raise ValueError(f"Unknown KaraOne stage: {stage}")
    return f"karaone_{VERSION}_{stage}_{phase}_s{int(seed)}"


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _audio_float(audio: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / float(np.iinfo(audio.dtype).max)
    return np.asarray(audio, dtype=np.float32)


def load_audio(path: str | Path, sample_rate: int = 16000, duration_sec: float = 2.0) -> np.ndarray:
    sr, audio = wavfile.read(str(path))
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = _audio_float(audio)
    if sr != sample_rate:
        gcd = np.gcd(int(sr), int(sample_rate))
        audio = resample_poly(audio, sample_rate // gcd, int(sr) // gcd).astype(np.float32)
    expected = int(round(sample_rate * duration_sec))
    audio = audio[:expected]
    if len(audio) < expected:
        audio = np.pad(audio, (0, expected - len(audio)))
    rms = np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-8)
    return (audio * min(0.08 / rms, 10.0)).clip(-0.95, 0.95).astype(np.float32)


class _SubjectBundleCache:
    def __init__(self, root: Path, max_items: int = 3):
        self.root = root
        self.max_items = max_items
        self.items: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, relative_path: str) -> dict[str, np.ndarray]:
        if relative_path in self.items:
            self.items.move_to_end(relative_path)
            return self.items[relative_path]
        raw = np.load(self.root / relative_path, allow_pickle=False)
        value = {key: raw[key] for key in raw.files}
        self.items[relative_path] = value
        if len(self.items) > self.max_items:
            self.items.popitem(last=False)
        return value


class TopographicProjector:
    """Inverse-distance scalp interpolation into a fixed non-PNG tensor."""

    def __init__(self, channel_names: Iterable[str], positions_xy: np.ndarray, grid_size: int = 9):
        self.channel_names = tuple(str(item).upper() for item in channel_names)
        self.positions_xy = np.asarray(positions_xy, dtype=np.float32)
        if self.positions_xy.shape != (len(self.channel_names), 2):
            raise ValueError("positions_xy must be [n_channels, 2]")
        self.grid_size = int(grid_size)
        axis = np.linspace(-1.0, 1.0, self.grid_size, dtype=np.float32)
        gx, gy = np.meshgrid(axis, axis, indexing="xy")
        grid = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)
        distance = np.linalg.norm(grid[:, None, :] - self.positions_xy[None, :, :], axis=-1)
        weights = 1.0 / np.maximum(distance, 0.05) ** 2
        scalp = (grid[:, 0] ** 2 + grid[:, 1] ** 2) <= 1.05
        weights[~scalp] = 0.0
        self.grid_weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
        self.scalp_mask = scalp.reshape(self.grid_size, self.grid_size).astype(np.float32)

    @classmethod
    def from_standard_montage(cls, channel_names: Iterable[str], grid_size: int = 9) -> "TopographicProjector":
        names = tuple(str(item).upper() for item in channel_names)
        try:
            import mne

            montage = mne.channels.make_standard_montage("standard_1005")
            pos = montage.get_positions()["ch_pos"]
            lookup = {str(key).upper(): value for key, value in pos.items()}
            missing = [name for name in names if name not in lookup]
            if missing:
                raise ValueError(f"No standard_1005 positions for channels: {missing}")
            xyz = np.asarray([lookup[name] for name in names], dtype=np.float32)
            xy = xyz[:, :2]
            xy = xy / np.maximum(np.linalg.norm(xy, axis=1).max(), 1e-6)
        except Exception:  # noqa: BLE001 - MNE/numba is optional at runtime
            xy = _fallback_1010_xy(names)
        return cls(names, xy, grid_size=grid_size)

    def transform(self, eeg: np.ndarray, valid_len: int, sample_rate: float = 256.0, time_bins: int = 32) -> np.ndarray:
        eeg = np.asarray(eeg, dtype=np.float32)
        valid_len = max(1, min(int(valid_len), eeg.shape[1]))
        n_fft = min(128, valid_len)
        padded = eeg[:, :valid_len]
        out = np.zeros((len(BANDS), int(time_bins), self.grid_size, self.grid_size), dtype=np.float32)
        centers = np.linspace(0, valid_len - 1, num=int(time_bins)).astype(int)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))
        for time_idx, center in enumerate(centers):
            left = center - n_fft // 2
            right = left + n_fft
            indices = np.clip(np.arange(left, right), 0, valid_len - 1)
            segment = padded[:, indices] * np.hanning(n_fft)[None, :]
            power = np.abs(np.fft.rfft(segment, axis=1)) ** 2
            for band_idx, (lo, hi) in enumerate(BANDS):
                band = (freqs >= lo) & (freqs < hi)
                values = np.log(power[:, band].mean(axis=1) + 1e-6) if np.any(band) else np.zeros(eeg.shape[0])
                image = self.grid_weights @ values
                out[band_idx, time_idx] = image.reshape(self.grid_size, self.grid_size) * self.scalp_mask
        return out


def _fallback_1010_xy(channel_names: tuple[str, ...]) -> np.ndarray:
    """Deterministic 10-10 layout fallback for the 62-channel KaraOne montage.

    It is used only when MNE's standard montage cannot initialise (for example
    when a local numba cache is unavailable). The coordinate order remains the
    same across every split and is recorded in the run manifest through config.
    """
    y_by_prefix = {
        "FP": 0.94, "AF": 0.78, "F": 0.60, "FT": 0.46, "FC": 0.32,
        "T": 0.00, "C": 0.00, "TP": -0.46, "CP": -0.32, "P": -0.60,
        "PO": -0.78, "O": -0.94, "CB": -0.98,
    }
    # 10-10 lateral order; odd electrodes are left and even electrodes right.
    lateral = {1: 0.18, 2: 0.18, 3: 0.38, 4: 0.38, 5: 0.60, 6: 0.60, 7: 0.84, 8: 0.84}
    prefixes = sorted(y_by_prefix, key=len, reverse=True)
    positions = []
    for name in channel_names:
        prefix = next((item for item in prefixes if name.startswith(item)), None)
        if prefix is None:
            raise ValueError(f"No fallback 10-10 coordinate for channel {name}")
        suffix = name[len(prefix) :]
        if suffix == "Z":
            x = 0.0
        else:
            try:
                number = int(suffix)
            except ValueError as error:
                raise ValueError(f"No fallback 10-10 coordinate for channel {name}") from error
            if number not in lateral:
                raise ValueError(f"No fallback lateral coordinate for channel {name}")
            x = lateral[number] * (-1.0 if number % 2 else 1.0)
        positions.append((x, y_by_prefix[prefix]))
    return np.asarray(positions, dtype=np.float32)


def compute_time_anchor(audio: np.ndarray, sample_rate: int = 16000, steps: int = 200) -> dict[str, np.ndarray | float]:
    audio = np.asarray(audio, dtype=np.float32)
    hop = max(1, int(round(len(audio) / steps)))
    envelope = np.asarray([
        np.sqrt(np.mean(np.square(audio[idx * hop : min(len(audio), (idx + 1) * hop)]), dtype=np.float64) + 1e-8)
        for idx in range(steps)
    ], dtype=np.float32)
    threshold = max(float(np.median(envelope) + 1.5 * np.median(np.abs(envelope - np.median(envelope)))), float(0.15 * envelope.max(initial=0.0)))
    active = envelope >= threshold
    if active.any():
        start, end = int(np.flatnonzero(active)[0]), int(np.flatnonzero(active)[-1]) + 1
    else:
        start, end = 0, 0
    return {
        "active_mask": active.astype(np.float32),
        "envelope": envelope,
        "onset_sec": float(start * hop / sample_rate),
        "duration_sec": float((end - start) * hop / sample_rate),
    }


class KaraOne0711Dataset(Dataset):
    """Direct KaraOne reader; it never imports older KaraOne versions."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        stage: str,
        *,
        manifest: SplitManifest | None = None,
        eeg_len: int = 1280,
        sample_rate: int = 256,
        topology: TopographicProjector | None = None,
        include_audio: bool = False,
        include_topography: bool = True,
    ):
        self.root = Path(root)
        self.manifest = manifest or SplitManifest.build(root)
        self.split = split
        self.stage = stage
        self.eeg_len = int(eeg_len)
        self.sample_rate = int(sample_rate)
        self.include_audio = bool(include_audio)
        self.include_topography = bool(include_topography)
        self.records = records_for_split(root, self.manifest, split)
        self.labels = {label: idx for idx, label in enumerate(LABELS)}
        self.bundle_cache = _SubjectBundleCache(self.root)
        first = self.bundle_cache.get(self.records[0].eeg_subject_bundle)
        names = tuple(str(item) for item in first["channel_names"].tolist())
        self.topology = topology or TopographicProjector.from_standard_montage(names)

    def __len__(self) -> int:
        return len(self.records)

    def _load_eeg(self, record: TrialRecord) -> tuple[np.ndarray, int]:
        bundle = self.bundle_cache.get(record.eeg_subject_bundle)
        trials = bundle["trial_indices"].astype(np.int64)
        matches = np.flatnonzero(trials == int(record.trial_index))
        if len(matches) != 1:
            raise KeyError(f"No unique EEG trial for {record.key}")
        idx = int(matches[0])
        stage_key = f"stage__{self.stage}"
        lengths_key = f"{stage_key}__valid_lengths"
        if stage_key not in bundle or lengths_key not in bundle:
            raise KeyError(f"Missing {stage_key} in {record.eeg_subject_bundle}")
        raw = np.asarray(bundle[stage_key][idx], dtype=np.float32)
        valid = min(int(bundle[lengths_key][idx]), raw.shape[1], self.eeg_len)
        eeg = np.zeros((raw.shape[0], self.eeg_len), dtype=np.float32)
        eeg[:, :valid] = raw[:, :valid]
        return eeg, valid

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        eeg, valid_len = self._load_eeg(record)
        item: dict[str, Any] = {
            "eeg": torch.from_numpy(eeg),
            "eeg_valid_len": torch.tensor(valid_len, dtype=torch.long),
            "label_idx": torch.tensor(self.labels[record.label], dtype=torch.long),
            "label": record.label,
            "subject": record.subject,
            "trial_index": torch.tensor(record.trial_index, dtype=torch.long),
            "key": record.key,
            "split": self.split,
        }
        if self.include_topography:
            item["topography"] = torch.from_numpy(self.topology.transform(eeg, valid_len, self.sample_rate))
        if self.include_audio:
            audio = load_audio(self.root / record.audio_path)
            item["audio"] = torch.from_numpy(audio)
            item.update({key: torch.as_tensor(value) if isinstance(value, np.ndarray) else torch.tensor(value) for key, value in compute_time_anchor(audio).items()})
        return item


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)

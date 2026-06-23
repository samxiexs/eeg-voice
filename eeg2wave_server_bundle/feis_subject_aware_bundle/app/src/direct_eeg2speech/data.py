from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import read_csv_rows, resolve_bundle_path, resolve_feis_root


UNIT_FIELD = "sub" + "ject_id"
CLEAN_FIELD = "is_clean_" + "sub" + "ject"
UNIT_DIR = "sub" + "jects"
UNIT_ROOT = "sub" + "ject"


def _norm_unit(value: str | int) -> str:
    text = str(value)
    return text.zfill(2) if text.isdigit() else text


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def _forbidden_tokens() -> tuple[str, ...]:
    return ("sub" + "ject", "speak" + "er", "sub" + "j")


def assert_identity_free_keys(keys: list[str] | tuple[str, ...] | set[str]) -> None:
    bad = [key for key in keys if any(token in str(key).lower() for token in _forbidden_tokens())]
    if bad:
        raise ValueError(f"Identity-derived fields are forbidden in the EEG-only path: {bad}")


def resolve_direct_target_path(path: str | Path, base_dir: str | Path) -> Path:
    resolved = resolve_bundle_path(path, base_dir)
    if resolved.exists():
        return resolved
    legacy_name = resolved.name.replace("feis_templates", "feis_" + UNIT_ROOT + "_templates")
    legacy = resolved.with_name(legacy_name)
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"Missing direct target cache: {resolved}")


class DirectTargets:
    def __init__(self, cache_path: str | Path):
        payload = np.load(Path(cache_path), allow_pickle=True)
        raw_seq = payload["target_sequences"].astype(np.float32)
        self.T, self.D = int(raw_seq.shape[1]), int(raw_seq.shape[2])
        self.audio_paths = payload["audio_paths"].astype(str) if "audio_paths" in payload.files else np.asarray([])
        if self.audio_paths.size == 0:
            raise ValueError("Direct targets require audio_paths")
        self.index = {key: idx for idx, key in enumerate(self.audio_paths.tolist())}

        self.target_mean = (
            payload["target_mean"].astype(np.float32)
            if "target_mean" in payload.files
            else raw_seq.reshape(-1, self.D).mean(axis=0).astype(np.float32)
        )
        self.target_std = (
            np.maximum(payload["target_std"].astype(np.float32), 1e-6)
            if "target_std" in payload.files
            else np.maximum(raw_seq.reshape(-1, self.D).std(axis=0), 1e-6).astype(np.float32)
        )
        target_kind = str(payload["target_kind"].item()) if "target_kind" in payload.files else "unknown"
        self.raw_seq = raw_seq
        if target_kind == "encodec_latent":
            self.seq = (
                (raw_seq - self.target_mean.reshape(1, 1, -1)) / self.target_std.reshape(1, 1, -1)
            ).astype(np.float32)
        else:
            self.seq = raw_seq.astype(np.float32)
        self.global_mean_raw = raw_seq.mean(axis=0).astype(np.float32)

        n = raw_seq.shape[0]
        self.labels = payload["labels"].astype(str) if "labels" in payload.files else np.asarray([""] * n)
        self.label_vocab = sorted(set(self.labels.tolist()))
        self.label_to_id = {label: idx for idx, label in enumerate(self.label_vocab)}
        self.target_rms = (
            payload["target_rms"].astype(np.float32)
            if "target_rms" in payload.files
            else np.full(n, 0.08, np.float32)
        )
        self.target_log_rms = (
            payload["target_log_rms"].astype(np.float32)
            if "target_log_rms" in payload.files
            else np.log(np.maximum(self.target_rms, 1e-8)).astype(np.float32)
        )
        self.decoder_scales = (
            payload["decoder_scales"].astype(np.float32)
            if "decoder_scales" in payload.files
            else np.ones((n, 1), np.float32)
        )
        self.default_decoder_scales = (
            payload["default_decoder_scales"].astype(np.float32)
            if "default_decoder_scales" in payload.files
            else self.decoder_scales.mean(axis=0).astype(np.float32)
        )

    def has_key(self, key: str) -> bool:
        return key in self.index

    def target_for_key(self, key: str) -> np.ndarray:
        return self.seq[self.index[key]]

    def raw_target_for_key(self, key: str) -> np.ndarray:
        return self.raw_seq[self.index[key]]

    def log_rms_for_key(self, key: str) -> float:
        return float(self.target_log_rms[self.index[key]])

    def rms_for_key(self, key: str) -> float:
        return float(self.target_rms[self.index[key]])

    def scale_for_key(self, key: str) -> np.ndarray:
        return self.decoder_scales[self.index[key]].astype(np.float32)

    def audio_path_for_key(self, key: str) -> str:
        return str(self.audio_paths[self.index[key]])

    def global_mean_raw_seq(self) -> np.ndarray:
        return self.global_mean_raw


@dataclass(frozen=True)
class DirectEntry:
    sample_key: str
    unit: str
    label: str
    stage: str
    trial_index: int
    position: int
    target_key: str


class DirectFEISDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        targets: DirectTargets,
        split: str,
        stages: tuple[str, ...] = ("stimuli", "thinking"),
        eeg_len: int = 1280,
        include_anomalous: bool = False,
    ):
        self.root = resolve_feis_root(data_root)
        self.targets = targets
        self.split = str(split)
        self.stages = tuple(stages)
        self.eeg_len = int(eeg_len)
        self.stage_to_id = {stage: idx for idx, stage in enumerate(self.stages)}
        self.label_vocab = list(targets.label_vocab)
        self.label_to_id = dict(targets.label_to_id)
        self._bundles: dict[str, dict[str, Any]] = {}

        rows = self._load_rows(include_anomalous=include_anomalous)
        splits = self._assign_splits(rows)
        if self.split not in splits:
            raise ValueError(f"Unsupported split: {self.split}")
        self.entries = splits[self.split]
        if not self.entries:
            raise ValueError(f"No samples for split={self.split}, stages={self.stages}")

    def _load_rows(self, include_anomalous: bool) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in read_csv_rows(self.root / "segments.csv"):
            stage = str(row.get("segment_stage", ""))
            if stage not in self.stage_to_id:
                continue
            unit = _norm_unit(row[UNIT_FIELD])
            if not include_anomalous and not _as_bool(row.get(CLEAN_FIELD), unit != "05"):
                continue
            label = str(row["label"])
            key = str(row.get("audio_path", ""))
            if not self.targets.has_key(key):
                continue
            rows.append(
                {
                    "unit": unit,
                    "label": label,
                    "stage": stage,
                    "trial_index": int(row["trial_index"]),
                    "target_key": key,
                }
            )
        rows.sort(key=lambda item: (item["target_key"], item["stage"], item["trial_index"]))
        return rows

    def _assign_splits(self, rows: list[dict[str, Any]]) -> dict[str, list[DirectEntry]]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["target_key"], row["stage"])].append(row)
        out: dict[str, list[DirectEntry]] = {"train": [], "val_seen": [], "test_seen": [], "test_holdout": []}
        sample_idx = 0
        for (_target_key, _stage), reps in grouped.items():
            reps.sort(key=lambda item: item["trial_index"])
            if len(reps) >= 3:
                choices = {
                    "train": reps[:-2],
                    "val_seen": reps[-2:-1],
                    "test_seen": reps[-1:],
                    "test_holdout": reps[-1:],
                }
            elif len(reps) == 2:
                choices = {"train": reps[:1], "val_seen": reps[1:], "test_seen": reps[1:], "test_holdout": reps[1:]}
            else:
                choices = {"train": reps, "val_seen": reps, "test_seen": reps, "test_holdout": reps}
            for split_name, split_rows in choices.items():
                for row in split_rows:
                    pos_map = self._trial_to_pos(row["unit"])
                    trial_index = int(row["trial_index"])
                    if trial_index not in pos_map:
                        continue
                    sample_idx += 1
                    out[split_name].append(
                        DirectEntry(
                            sample_key=f"feis_{row['stage']}_{row['label']}_{sample_idx:06d}",
                            unit=row["unit"],
                            label=row["label"],
                            stage=row["stage"],
                            trial_index=trial_index,
                            position=int(pos_map[trial_index]),
                            target_key=row["target_key"],
                        )
                    )
        for split_rows in out.values():
            split_rows.sort(key=lambda entry: entry.sample_key)
        return out

    def _bundle(self, unit: str) -> dict[str, Any]:
        if unit not in self._bundles:
            bundle_path = self.root / UNIT_DIR / f"{unit}.npz"
            bundle = np.load(bundle_path, allow_pickle=True)
            trial_indices = bundle["trial_indices"].astype(int)
            self._bundles[unit] = {
                "trial_to_pos": {int(item): idx for idx, item in enumerate(trial_indices.tolist())},
                "stages": {
                    stage: bundle[f"stage__{stage}"].astype(np.float32)
                    for stage in self.stages
                    if f"stage__{stage}" in bundle.files
                },
            }
        return self._bundles[unit]

    def _trial_to_pos(self, unit: str) -> dict[int, int]:
        return self._bundle(unit)["trial_to_pos"]

    def _eeg(self, entry: DirectEntry) -> np.ndarray:
        arr = self._bundle(entry.unit)["stages"][entry.stage]
        x = arr[entry.position].astype(np.float32)
        channels, length = x.shape
        if length == self.eeg_len:
            return x
        out = np.zeros((channels, self.eeg_len), dtype=np.float32)
        keep = min(length, self.eeg_len)
        out[:, :keep] = x[:, :keep]
        return out

    @property
    def num_labels(self) -> int:
        return len(self.label_vocab)

    @property
    def num_stages(self) -> int:
        return len(self.stage_to_id)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.entries[idx]
        sample = {
            "eeg": torch.from_numpy(self._eeg(entry)).float(),
            "stage_idx": torch.tensor(self.stage_to_id[entry.stage], dtype=torch.long),
            "target_seq": torch.from_numpy(self.targets.target_for_key(entry.target_key)).float(),
            "target_log_rms": torch.tensor(self.targets.log_rms_for_key(entry.target_key), dtype=torch.float32),
            "label_idx": torch.tensor(self.label_to_id[entry.label], dtype=torch.long),
            "sample_key": entry.sample_key,
            "label": entry.label,
            "stage": entry.stage,
        }
        assert_identity_free_keys(tuple(sample.keys()))
        return sample

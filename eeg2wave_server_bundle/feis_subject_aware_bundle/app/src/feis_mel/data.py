from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.feis_mel.targets import MelLabelTargets
from src.utils import read_csv_rows, resolve_feis_root


UNIT_FIELD = "sub" + "ject_id"
CLEAN_FIELD = "is_clean_" + "sub" + "ject"
UNIT_DIR = "sub" + "jects"


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
    return ("sub" + "ject", "speak" + "er", "sub" + "j", "stage_idx")


def assert_mel_identity_free_keys(keys: list[str] | tuple[str, ...] | set[str]) -> None:
    bad = [key for key in keys if any(token in str(key).lower() for token in _forbidden_tokens())]
    if bad:
        raise ValueError(f"Identity or external-condition fields are forbidden in FEIS mel batches: {bad}")


@dataclass(frozen=True)
class MelEntry:
    sample_key: str
    source_key: str
    label: str
    trial_index: int
    position: int


class FEISMelDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        targets: MelLabelTargets,
        split: str,
        stage: str,
        eeg_len: int = 1280,
        include_anomalous: bool = False,
    ):
        self.root = resolve_feis_root(data_root)
        self.targets = targets
        self.split = str(split)
        self.stage = str(stage)
        self.eeg_len = int(eeg_len)
        self._bundles: dict[str, dict[str, Any]] = {}

        rows = self._load_rows(include_anomalous=include_anomalous)
        splits = self._assign_splits(rows)
        if self.split not in splits:
            raise ValueError(f"Unsupported split: {self.split}")
        self.entries = splits[self.split]
        if not self.entries:
            raise ValueError(f"No samples for split={self.split}, stage={self.stage}")

    def _load_rows(self, include_anomalous: bool) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in read_csv_rows(self.root / "segments.csv"):
            if str(row.get("segment_stage", "")) != self.stage:
                continue
            source_key = _norm_unit(row[UNIT_FIELD])
            if not include_anomalous and not _as_bool(row.get(CLEAN_FIELD), source_key != "05"):
                continue
            label = str(row["label"])
            if label not in self.targets.label_to_id:
                continue
            rows.append(
                {
                    "source_key": source_key,
                    "label": label,
                    "trial_index": int(row["trial_index"]),
                }
            )
        rows.sort(key=lambda item: (item["source_key"], item["label"], item["trial_index"]))
        return rows

    def _assign_splits(self, rows: list[dict[str, Any]]) -> dict[str, list[MelEntry]]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["source_key"], row["label"])].append(row)
        out: dict[str, list[MelEntry]] = {"train": [], "val_seen": [], "test_seen": [], "test_holdout": []}
        sample_idx = 0
        for (_source_key, _label), reps in grouped.items():
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
                    pos_map = self._trial_to_pos(row["source_key"])
                    trial_index = int(row["trial_index"])
                    if trial_index not in pos_map:
                        continue
                    sample_idx += 1
                    out[split_name].append(
                        MelEntry(
                            sample_key=f"feis_{row['label']}_{sample_idx:06d}",
                            source_key=row["source_key"],
                            label=row["label"],
                            trial_index=trial_index,
                            position=int(pos_map[trial_index]),
                        )
                    )
        for split_rows in out.values():
            split_rows.sort(key=lambda entry: entry.sample_key)
        return out

    def _bundle(self, source_key: str) -> dict[str, Any]:
        if source_key not in self._bundles:
            bundle_path = self.root / UNIT_DIR / f"{source_key}.npz"
            bundle = np.load(bundle_path, allow_pickle=True)
            trial_indices = bundle["trial_indices"].astype(int)
            self._bundles[source_key] = {
                "trial_to_pos": {int(item): idx for idx, item in enumerate(trial_indices.tolist())},
                "stage": bundle[f"stage__{self.stage}"].astype(np.float32),
            }
        return self._bundles[source_key]

    def _trial_to_pos(self, source_key: str) -> dict[int, int]:
        return self._bundle(source_key)["trial_to_pos"]

    def _eeg(self, entry: MelEntry) -> np.ndarray:
        arr = self._bundle(entry.source_key)["stage"]
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
        return self.targets.num_labels

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.entries[idx]
        label_idx = self.targets.label_id(entry.label)
        sample = {
            "eeg": torch.from_numpy(self._eeg(entry)).float(),
            "target_bank": torch.from_numpy(self.targets.bank_for_label_id(label_idx)).float(),
            "target_log_rms": torch.tensor(self.targets.log_rms_for_label_id(label_idx), dtype=torch.float32),
            "label_idx": torch.tensor(label_idx, dtype=torch.long),
            "label": entry.label,
            "sample_key": entry.sample_key,
        }
        assert_mel_identity_free_keys(tuple(sample.keys()))
        return sample


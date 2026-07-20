from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, WeightedRandomSampler

from .audio_eval import CACHE_SCHEMA_VERSION, validate_cache_arrays


DATASETS = ("feis", "karaone", "ds004306")
DATASET_IDS = {name: index for index, name in enumerate(DATASETS)}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class CombinedContext:
    root: Path
    manifest_path: Path
    split_path: Path
    rows: tuple[dict[str, str], ...]
    split: dict[str, Any]
    label_to_global: dict[tuple[str, str], int]
    label_to_local: dict[tuple[str, str], int]
    subject_to_index: dict[str, int]
    config_sha256: str

    def split_for(self, row: dict[str, str]) -> str:
        group = row["subject_group_id"]
        for name in ("train", "validation", "test"):
            if group in self.split["datasets"].get(row["dataset"], {}).get(name, []):
                return name
        raise ValueError(f"Subject {group} is not present in locked split")


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))
    required = {"dataset", "subject_group_id", "subject_recording_id", "trial_index", "eeg_relpath", "eeg_row", "eeg_valid_samples", "audio_key", "label"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"Unified manifest is missing fields: {sorted(missing)}")
    keys = [row.get("sample_key") or f"{row['subject_recording_id']}:{row['trial_index']}" for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("sample_key is not unique; use subject_recording_id + trial_index")
    return rows


def load_context(config_path: str | Path) -> CombinedContext:
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = (config_path.parent / cfg["data"]["eeg_output_root"]).resolve()
    manifest = root / "manifests" / "unified_trials.csv"
    split_path = config_path.parent / cfg["data"]["split_file"]
    rows = _read_rows(manifest)
    split = yaml.safe_load(split_path.read_text(encoding="utf-8"))
    for dataset in DATASETS:
        declared = split.get("datasets", {}).get(dataset, {})
        groups = {name: set(declared.get(name, [])) for name in ("train", "validation", "test")}
        overlaps = (groups["train"] & groups["validation"]) | (groups["train"] & groups["test"]) | (groups["validation"] & groups["test"])
        if overlaps:
            raise ValueError(f"Locked split has overlapping {dataset} subjects: {sorted(overlaps)}")
        observed = {row["subject_group_id"] for row in rows if row["dataset"] == dataset}
        covered = set().union(*groups.values())
        if observed != covered:
            raise ValueError(
                f"Locked split coverage mismatch for {dataset}: missing={sorted(observed - covered)}, extra={sorted(covered - observed)}"
            )
    labels_by_dataset: dict[str, list[str]] = {}
    for dataset in DATASETS:
        labels_by_dataset[dataset] = sorted({row["label"] for row in rows if row["dataset"] == dataset})
    expected = {"feis": 16, "karaone": 11, "ds004306": 3}
    for dataset, count in expected.items():
        if len(labels_by_dataset.get(dataset, [])) != count:
            raise ValueError(f"Expected {count} {dataset} labels, got {len(labels_by_dataset.get(dataset, []))}")
    label_to_global: dict[tuple[str, str], int] = {}
    label_to_local: dict[tuple[str, str], int] = {}
    offset = 0
    for dataset in DATASETS:
        for local, label in enumerate(labels_by_dataset[dataset]):
            label_to_global[(dataset, label)] = offset + local
            label_to_local[(dataset, label)] = local
        offset += len(labels_by_dataset[dataset])
    train_groups = [group for dataset in DATASETS for group in split["datasets"][dataset]["train"]]
    subject_to_index = {group: index for index, group in enumerate(sorted(train_groups))}
    return CombinedContext(
        root,
        manifest,
        split_path,
        rows,
        split,
        label_to_global,
        label_to_local,
        subject_to_index,
        sha256_bytes(config_path.read_bytes()),
    )


class AudioCodeBank:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as raw:
            checks = validate_cache_arrays(raw)
            failed = sorted(name for name, passed in checks.items() if not passed)
            if failed:
                raise ValueError(
                    "Combined code cache failed schema-v2 validation; rebuild it with "
                    "`bash app/run_combined_0715_v1.sh cache --rebuild`. "
                    f"Failed checks: {failed}"
                )
            self.version = str(np.asarray(raw["version"]).item())
            if self.version != CACHE_SCHEMA_VERSION:
                raise ValueError(f"Expected {CACHE_SCHEMA_VERSION}, got {self.version!r}; rebuild the cache")
            self.keys = raw["keys"].astype(str)
            self.labels = raw["labels"].astype(str)
            self.datasets = raw["datasets"].astype(str)
            self.audio_relpaths = raw["audio_relpaths"].astype(str)
            self.audio_valid_samples = np.asarray(raw["audio_valid_samples"], dtype=np.int64)
            self.codes = np.asarray(raw["encodec_codes"], dtype=np.int64)
            self.scale = np.asarray(raw["encodec_scale"], dtype=np.float32)
            self.scale_valid = np.asarray(raw["encodec_scale_valid"], dtype=bool)
            self.envelope = np.asarray(raw["audio_envelope"], dtype=np.float32)
            self.onset = np.asarray(raw["onset"], dtype=np.float32)
            self.duration = np.asarray(raw["duration"], dtype=np.float32)
            self.code_valid_steps = np.asarray(raw["code_valid_steps"], dtype=np.int64)
            self.fit_split = np.asarray(raw["fit_split"], dtype=bool)
        self.index_by_key = {key: index for index, key in enumerate(self.keys.tolist())}

    def indices(self, split: str) -> np.ndarray:
        if split == "train":
            return np.flatnonzero(self.fit_split)
        if split in {"validation", "test"}:
            raise ValueError("Validation/test cache indices require the locked manifest; use bank_indices_for_split")
        raise ValueError(f"Unknown audio split {split}")

    def index_for_key(self, key: str) -> int:
        try:
            return int(self.index_by_key[str(key)])
        except KeyError as error:
            raise KeyError(f"audio_key {key} is absent from code cache") from error


class AudioCodeDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, bank: AudioCodeBank, indices: Iterable[int], context: CombinedContext):
        self.bank = bank
        self.indices_array = np.asarray(list(indices), dtype=np.int64)
        self.labels = np.asarray([context.label_to_global[(str(bank.datasets[index]), str(bank.labels[index]))] for index in self.indices_array], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices_array)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = int(self.indices_array[item])
        valid = int(self.bank.code_valid_steps[index])
        mask = np.zeros((8, 150), dtype=bool)
        mask[:, :valid] = True
        return torch.from_numpy(np.ascontiguousarray(self.bank.codes[index])).long(), torch.tensor(self.labels[item]), torch.from_numpy(mask)


class CombinedEEGDataset(Dataset[dict[str, Any]]):
    def __init__(self, context: CombinedContext, bank: AudioCodeBank, dataset: str, split: str, eeg_len: int = 768):
        self.context = context
        self.bank = bank
        self.dataset = dataset
        self.split = split
        self.eeg_len = int(eeg_len)
        self.rows = tuple(row for row in context.rows if row["dataset"] == dataset and context.split_for(row) == split)
        self.bundle_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self.audio_indices = np.asarray([bank.index_for_key(row["audio_key"]) for row in self.rows], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def _bundle(self, relative: str) -> dict[str, np.ndarray]:
        if relative not in self.bundle_cache:
            bundle = dict(np.load(self.context.root / relative, allow_pickle=False))
            self.bundle_cache[relative] = bundle
            while len(self.bundle_cache) > 8:
                self.bundle_cache.popitem(last=False)
        else:
            self.bundle_cache.move_to_end(relative)
        return self.bundle_cache[relative]

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = self.rows[item]
        bundle = self._bundle(row["eeg_relpath"])
        eeg = np.asarray(bundle["eeg"][int(row["eeg_row"])], dtype=np.float32)
        valid = min(max(int(row["eeg_valid_samples"]), 1), eeg.shape[-1])
        eeg = eeg[:, : self.eeg_len]
        if eeg.shape[-1] < self.eeg_len:
            eeg = np.pad(eeg, ((0, 0), (0, self.eeg_len - eeg.shape[-1])))
        eeg_mask = np.zeros(self.eeg_len, dtype=bool)
        eeg_mask[: min(valid, self.eeg_len)] = True
        audio_index = int(self.audio_indices[item])
        code_valid = int(self.bank.code_valid_steps[audio_index])
        code_mask = np.zeros((8, 150), dtype=bool)
        code_mask[:, :code_valid] = True
        label_global = self.context.label_to_global[(self.dataset, row["label"])]
        label_local = self.context.label_to_local[(self.dataset, row["label"])]
        subject_index = self.context.subject_to_index.get(row["subject_group_id"], -1)
        return {
            "eeg": torch.from_numpy(np.ascontiguousarray(eeg)),
            "eeg_valid_len": torch.tensor(min(valid, self.eeg_len), dtype=torch.long),
            "eeg_mask": torch.from_numpy(eeg_mask),
            "codes": torch.from_numpy(np.ascontiguousarray(self.bank.codes[audio_index])).long(),
            "code_mask": torch.from_numpy(code_mask),
            "audio_envelope": torch.from_numpy(np.ascontiguousarray(self.bank.envelope[audio_index])),
            "onset": torch.tensor(self.bank.onset[audio_index]),
            "duration": torch.tensor(self.bank.duration[audio_index]),
            "label_idx": torch.tensor(label_global, dtype=torch.long),
            "label_local": torch.tensor(label_local, dtype=torch.long),
            "dataset_idx": torch.tensor(DATASET_IDS[self.dataset], dtype=torch.long),
            "subject_idx": torch.tensor(subject_index, dtype=torch.long),
            "audio_idx": torch.tensor(audio_index, dtype=torch.long),
            "pairing_level": torch.tensor({"strong": 2, "medium": 1, "weak": 0}.get("strong" if self.dataset == "karaone" else "medium" if self.dataset == "feis" else "weak"), dtype=torch.long),
            "sample_key": row.get("sample_key") or f"{row['subject_recording_id']}:{row['trial_index']}",
            "audio_key": row["audio_key"],
            "subject_group_id": row["subject_group_id"],
            "subject_recording_id": row["subject_recording_id"],
            "trial_index": int(row["trial_index"]),
            "label": row["label"],
            "audio_relpath": row["audio_relpath"],
            "audio_valid_samples": int(row.get("audio_valid_samples") or self.bank.audio_valid_samples[audio_index]),
            "audio_pairing": row.get("audio_pairing", "unknown"),
            "pairing_confidence": row.get("pairing_confidence", "unknown"),
        }


def balanced_weights(dataset: CombinedEEGDataset) -> torch.DoubleTensor:
    groups = [(row["subject_group_id"], row["label"]) for row in dataset.rows]
    counts = Counter(groups)
    return torch.tensor([1.0 / counts[group] for group in groups], dtype=torch.double)

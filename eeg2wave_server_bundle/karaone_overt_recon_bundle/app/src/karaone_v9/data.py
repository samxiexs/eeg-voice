from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from src.karaone_recon.semantic_tokens import KaraOneSemanticTokenTargets
from src.karaone_recon.targets import KaraOneTargets


@dataclass(frozen=True)
class KaraOneV9Entry:
    subject: str
    label: str
    stage: str
    trial_index: int
    position: int
    split_kind: str


class KaraOneV9TargetBank:
    """Canonical v9 target view over speech semantic, prosody, and codec caches."""

    def __init__(
        self,
        semantic_cache: str | Path,
        *,
        codec_cache: str | Path | None = None,
        prosody_cache: str | Path | None = None,
        semantic_token_cache: str | Path | None = None,
        data_root: str | Path | None = None,
    ):
        self.semantic = KaraOneTargets(semantic_cache, data_root=data_root)
        self.codec = KaraOneTargets(codec_cache, data_root=data_root) if codec_cache else None
        self.prosody = KaraOneTargets(prosody_cache, data_root=data_root) if prosody_cache else None
        self.semantic_tokens = (
            KaraOneSemanticTokenTargets(semantic_token_cache) if semantic_token_cache and Path(semantic_token_cache).exists() else None
        )

        self.subject_vocab = list(self.semantic.subject_vocab)
        self.label_vocab = list(self.semantic.label_vocab)
        self.subject_to_id = dict(self.semantic.subject_to_id)
        self.label_to_id = dict(self.semantic.label_to_id)
        self.semantic_steps = int(self.semantic.T)
        self.semantic_dim = int(self.semantic.D)
        self.codec_steps = int(self.codec.T) if self.codec is not None else 1
        self.codec_dim = int(self.codec.D) if self.codec is not None else 1
        self.prosody_steps = int(self.prosody.core_len_frames) if self.prosody is not None else self.semantic_steps
        self.semantic_token_steps = int(self.semantic_tokens.T) if self.semantic_tokens is not None else self.semantic_steps
        self.semantic_token_vocab = int(self.semantic_tokens.vocab_size) if self.semantic_tokens is not None else 1

    @staticmethod
    def key(subject: str, trial_index: int) -> str:
        return f"{subject}:{int(trial_index)}"

    def has_trial(self, subject: str, trial_index: int, *, require_codec: bool = False) -> bool:
        if not self.semantic.has_trial(subject, trial_index):
            return False
        if require_codec and (self.codec is None or not self.codec.has_trial(subject, trial_index)):
            return False
        return True

    def semantic_seq(self, subject: str, trial_index: int) -> np.ndarray:
        return self.semantic.target(subject, trial_index).astype(np.float32)

    def semantic_summary(self, subject: str, trial_index: int) -> np.ndarray:
        return self.semantic.target_summary(subject, trial_index).astype(np.float32)

    def codec_seq(self, subject: str, trial_index: int) -> np.ndarray:
        if self.codec is None:
            return np.zeros((1, 1), dtype=np.float32)
        return self.codec.target(subject, trial_index).astype(np.float32)

    def prosody_targets(self, subject: str, trial_index: int) -> dict[str, np.ndarray | float]:
        if self.prosody is None or not self.prosody.has_trial(subject, trial_index):
            return {
                "active": np.zeros(self.prosody_steps, dtype=np.float32),
                "energy": np.zeros(self.prosody_steps, dtype=np.float32),
                "duration": 0.0,
                "onset": 0.0,
            }
        core_steps = max(float(self.prosody.core_len_frames), 1.0)
        full_steps = max(float(self.prosody.full_target_steps), 1.0)
        active = (
            self.prosody.field(subject, trial_index, "core_active_mask").astype(np.float32)
            if self.prosody.has_field("core_active_mask")
            else np.ones(self.prosody.core_len_frames, dtype=np.float32)
        )
        energy = (
            self.prosody.field(subject, trial_index, "active_envelope_norm").astype(np.float32)
            if self.prosody.has_field("active_envelope_norm")
            else np.zeros_like(active, dtype=np.float32)
        )
        duration = (
            float(self.prosody.field(subject, trial_index, "active_duration_frames")) / full_steps
            if self.prosody.has_field("active_duration_frames")
            else float(np.mean(active))
        )
        onset = (
            float(self.prosody.field(subject, trial_index, "active_start_frame")) / full_steps
            if self.prosody.has_field("active_start_frame")
            else 0.0
        )
        if active.shape[0] != int(self.prosody_steps):
            active = _resample_1d(active, int(self.prosody_steps))
        if energy.shape[0] != int(self.prosody_steps):
            energy = _resample_1d(energy, int(self.prosody_steps))
        return {
            "active": active.astype(np.float32),
            "energy": energy.astype(np.float32),
            "duration": float(np.clip(duration, 0.0, 1.0)),
            "onset": float(np.clip(onset, 0.0, 1.0)),
        }

    def semantic_token_targets(self, subject: str, trial_index: int) -> tuple[np.ndarray, np.ndarray]:
        if self.semantic_tokens is None or not self.semantic_tokens.has_trial(subject, trial_index):
            return (
                np.zeros(self.semantic_token_steps, dtype=np.int64),
                np.zeros(self.semantic_token_steps, dtype=np.float32),
            )
        return (
            self.semantic_tokens.tokens(subject, trial_index).astype(np.int64),
            self.semantic_tokens.mask(subject, trial_index).astype(np.float32),
        )


class KaraOneV9Dataset(Dataset):
    """Subject-holdout aware KaraOne dataset for the v9 canonical pipeline."""

    def __init__(
        self,
        data_root: str | Path,
        targets: KaraOneV9TargetBank,
        split: str,
        *,
        stages: Iterable[str] = ("overt_like",),
        subject_val: str = "P02",
        subject_test: str = "MM21",
        eeg_len: int = 1280,
        train_split_mode: str = "subject_train",
        require_codec: bool = False,
    ):
        self.root = Path(data_root)
        self.targets = targets
        self.split = str(split)
        self.stages = tuple(str(stage) for stage in stages)
        self.stage_to_id = {stage: idx for idx, stage in enumerate(self.stages)}
        self.subject_val = str(subject_val)
        self.subject_test = str(subject_test)
        self.heldout_subjects = {self.subject_val, self.subject_test}
        self.eeg_len = int(eeg_len)
        self.train_split_mode = str(train_split_mode)
        self.require_codec = bool(require_codec)
        self.subject_vocab = list(targets.subject_vocab)
        self.label_vocab = list(targets.label_vocab)
        self.subject_to_id = dict(targets.subject_to_id)
        self.label_to_id = dict(targets.label_to_id)
        self._bundle_cache: dict[str, dict] = {}
        rows = self._load_rows()
        self.entries = self._assign_split(rows)
        if not self.entries:
            raise ValueError(f"No KaraOne v9 samples for split={split}, stages={self.stages}")

    @property
    def num_subjects(self) -> int:
        return len(self.subject_vocab)

    @property
    def num_labels(self) -> int:
        return len(self.label_vocab)

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    def _load_rows(self) -> list[dict]:
        rows: list[dict] = []
        with (self.root / "segments.csv").open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                stage = str(row["segment_stage"])
                subject = str(row["subject_id"])
                trial_index = int(row["trial_index"])
                if stage not in self.stage_to_id:
                    continue
                if not self.targets.has_trial(subject, trial_index, require_codec=self.require_codec):
                    continue
                rows.append(
                    {
                        "subject": subject,
                        "label": str(row["label"]),
                        "stage": stage,
                        "trial_index": trial_index,
                    }
                )
        rows.sort(key=lambda item: (item["subject"], item["label"], item["stage"], int(item["trial_index"])))
        return rows

    def _assign_split(self, rows: list[dict]) -> list[KaraOneV9Entry]:
        split = self.split
        if split in {"subject_train", "train_full"}:
            return self._rows_to_entries([row for row in rows if row["subject"] not in self.heldout_subjects], "subject_train")
        if split == "subject_val":
            return self._rows_to_entries([row for row in rows if row["subject"] == self.subject_val], "subject_val")
        if split in {"subject_test", "test_holdout"}:
            return self._rows_to_entries([row for row in rows if row["subject"] == self.subject_test], "subject_test")
        if split in {"train", "val", "test"}:
            return self._assign_trial_split([row for row in rows if row["subject"] not in self.heldout_subjects], split)
        raise ValueError(f"Unsupported v9 split={split}")

    def _assign_trial_split(self, rows: list[dict], split: str) -> list[KaraOneV9Entry]:
        grouped: dict[tuple[str, str, str], list[dict]] = {}
        for row in rows:
            grouped.setdefault((row["subject"], row["label"], row["stage"]), []).append(row)
        selected: list[dict] = []
        for group in grouped.values():
            group.sort(key=lambda item: int(item["trial_index"]))
            if len(group) == 1:
                selected.extend(group if split == "train" else [])
            elif len(group) == 2:
                if split == "train":
                    selected.extend(group[:1])
                elif split == "test":
                    selected.extend(group[-1:])
            else:
                if split == "train":
                    selected.extend(group[:-2])
                elif split == "val":
                    selected.extend(group[-2:-1])
                elif split == "test":
                    selected.extend(group[-1:])
        return self._rows_to_entries(selected, split)

    def _rows_to_entries(self, rows: list[dict], split_kind: str) -> list[KaraOneV9Entry]:
        entries: list[KaraOneV9Entry] = []
        for row in rows:
            pos = self._trial_to_position(row["subject"]).get(int(row["trial_index"]))
            if pos is None:
                continue
            entries.append(
                KaraOneV9Entry(
                    subject=str(row["subject"]),
                    label=str(row["label"]),
                    stage=str(row["stage"]),
                    trial_index=int(row["trial_index"]),
                    position=int(pos),
                    split_kind=str(split_kind),
                )
            )
        entries.sort(key=lambda item: (item.subject, item.label, item.stage, item.trial_index))
        return entries

    def _load_bundle(self, subject: str) -> dict:
        if subject not in self._bundle_cache:
            path = self.root / "subjects" / f"{subject}.npz"
            payload = np.load(path, allow_pickle=True)
            trial_indices = payload["trial_indices"].astype(np.int32)
            stages: dict[str, np.ndarray] = {}
            valid_lengths: dict[str, np.ndarray] = {}
            for stage in self.stages:
                key = f"stage__{stage}"
                if key not in payload.files:
                    continue
                stages[stage] = payload[key].astype(np.float32)
                valid_key = f"{key}__valid_lengths"
                valid_lengths[stage] = (
                    payload[valid_key].astype(np.int32)
                    if valid_key in payload.files
                    else np.full(stages[stage].shape[0], stages[stage].shape[-1], dtype=np.int32)
                )
            self._bundle_cache[subject] = {
                "trial_to_position": {int(trial): idx for idx, trial in enumerate(trial_indices.tolist())},
                "stages": stages,
                "valid_lengths": valid_lengths,
            }
        return self._bundle_cache[subject]

    def _trial_to_position(self, subject: str) -> dict[int, int]:
        return self._load_bundle(subject)["trial_to_position"]

    def _eeg(self, entry: KaraOneV9Entry) -> tuple[np.ndarray, int]:
        bundle = self._load_bundle(entry.subject)
        if entry.stage not in bundle["stages"]:
            raise KeyError(f"Subject {entry.subject} has no stage {entry.stage}")
        arr = bundle["stages"][entry.stage][entry.position]
        valid_len = int(bundle["valid_lengths"][entry.stage][entry.position])
        channels, length = arr.shape
        if length == self.eeg_len:
            return arr.astype(np.float32), min(valid_len, self.eeg_len)
        out = np.zeros((channels, self.eeg_len), dtype=np.float32)
        n = min(length, self.eeg_len)
        out[:, :n] = arr[:, :n]
        return out, min(valid_len, self.eeg_len)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        eeg, valid_len = self._eeg(entry)
        prosody = self.targets.prosody_targets(entry.subject, entry.trial_index)
        tokens, token_mask = self.targets.semantic_token_targets(entry.subject, entry.trial_index)
        return {
            "eeg": torch.from_numpy(eeg).float(),
            "eeg_valid_len": torch.tensor(valid_len, dtype=torch.long),
            "stage_idx": torch.tensor(self.stage_to_id[entry.stage], dtype=torch.long),
            "subject_idx": torch.tensor(self.subject_to_id[entry.subject], dtype=torch.long),
            "label_idx": torch.tensor(self.label_to_id[entry.label], dtype=torch.long),
            "semantic_seq": torch.from_numpy(self.targets.semantic_seq(entry.subject, entry.trial_index)).float(),
            "semantic_summary": torch.from_numpy(self.targets.semantic_summary(entry.subject, entry.trial_index)).float(),
            "semantic_token_targets": torch.from_numpy(tokens).long(),
            "semantic_token_mask": torch.from_numpy(token_mask).float(),
            "codec_seq": torch.from_numpy(self.targets.codec_seq(entry.subject, entry.trial_index)).float(),
            "prosody_active": torch.from_numpy(prosody["active"]).float(),
            "prosody_energy": torch.from_numpy(prosody["energy"]).float(),
            "prosody_duration": torch.tensor(float(prosody["duration"]), dtype=torch.float32),
            "prosody_onset": torch.tensor(float(prosody["onset"]), dtype=torch.float32),
            "subject": entry.subject,
            "label": entry.label,
            "stage": entry.stage,
            "trial_index": entry.trial_index,
            "template_id": KaraOneV9TargetBank.key(entry.subject, entry.trial_index),
            "split_kind": entry.split_kind,
        }


def _resample_1d(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == int(length):
        return values.astype(np.float32)
    if values.size == 0:
        return np.zeros(int(length), dtype=np.float32)
    src = np.linspace(0.0, 1.0, values.size)
    dst = np.linspace(0.0, 1.0, int(length))
    return np.interp(dst, src, values).astype(np.float32)

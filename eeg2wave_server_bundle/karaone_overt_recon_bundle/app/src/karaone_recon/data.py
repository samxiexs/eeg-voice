from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .targets import KaraOneTargets


@dataclass(frozen=True)
class KaraOneEntry:
    subject: str
    label: str
    stage: str
    trial_index: int
    position: int
    split_kind: str


class KaraOneTrialDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        targets: KaraOneTargets,
        split: str,
        stages: Iterable[str] = ("overt_like",),
        split_protocol: str = "trial",
        heldout_subjects: Iterable[str] = ("P02", "MM21"),
        eeg_len: int = 1280,
        aux_targets: KaraOneTargets | None = None,
    ):
        self.root = Path(data_root)
        self.targets = targets
        # Optional auxiliary HuBERT-feature targets (same subject:trial keys). Used for
        # the content-bearing auxiliary head + retrieval metrics; the main `targets`
        # (mel/encodec) still drives waveform rendering.
        self.aux_targets = aux_targets
        self.split = str(split)
        self.stages = tuple(stages)
        self.split_protocol = str(split_protocol)
        self.heldout_subjects = set(str(item) for item in heldout_subjects)
        self.eeg_len = int(eeg_len)
        self.stage_to_id = {stage: idx for idx, stage in enumerate(self.stages)}
        self.subject_vocab = list(targets.subject_vocab)
        self.label_vocab = list(targets.label_vocab)
        self.subject_to_id = dict(targets.subject_to_id)
        self.label_to_id = dict(targets.label_to_id)
        self._bundle_cache: dict[str, dict] = {}

        rows = self._load_rows()
        self.entries = self._assign_split(rows)
        if not self.entries:
            raise ValueError(f"No KaraOne samples for split={split}, protocol={split_protocol}, stages={self.stages}")

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
        out: list[dict] = []
        with (self.root / "segments.csv").open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                stage = str(row["segment_stage"])
                subject = str(row["subject_id"])
                trial_index = int(row["trial_index"])
                if stage not in self.stage_to_id:
                    continue
                if not self.targets.has_trial(subject, trial_index):
                    continue
                out.append(
                    {
                        "subject": subject,
                        "label": str(row["label"]),
                        "stage": stage,
                        "trial_index": trial_index,
                    }
                )
        return out

    def _assign_split(self, rows: list[dict]) -> list[KaraOneEntry]:
        if self.split_protocol == "subject_holdout":
            return self._assign_subject_split(rows)
        if self.split_protocol != "trial":
            raise ValueError(f"Unsupported split_protocol={self.split_protocol}")
        return self._assign_trial_split(rows)

    def _assign_subject_split(self, rows: list[dict]) -> list[KaraOneEntry]:
        entries: list[KaraOneEntry] = []
        want_heldout = self.split in {"val", "test", "subject_test"}
        want_train = self.split in {"train", "subject_train"}
        for row in rows:
            subject = row["subject"]
            if (subject in self.heldout_subjects and want_heldout) or (subject not in self.heldout_subjects and want_train):
                pos = self._trial_to_position(subject).get(int(row["trial_index"]))
                if pos is not None:
                    entries.append(
                        KaraOneEntry(subject, row["label"], row["stage"], int(row["trial_index"]), pos, "subject_holdout")
                    )
        entries.sort(key=lambda item: (item.subject, item.label, item.stage, item.trial_index))
        return entries

    def _assign_trial_split(self, rows: list[dict]) -> list[KaraOneEntry]:
        # Reserve heldout subjects for the subject_holdout protocol ONLY. If they were
        # left in the trial split, they would be in trial-train AND in subject_test,
        # leaking training data into the cross-subject eval (inflating its metrics).
        grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for row in rows:
            if row["subject"] in self.heldout_subjects:
                continue
            grouped[(row["subject"], row["label"], row["stage"])].append(row)
        for group in grouped.values():
            group.sort(key=lambda item: int(item["trial_index"]))

        entries: list[KaraOneEntry] = []
        for (subject, label, stage), group in grouped.items():
            if len(group) == 1:
                chosen = group if self.split == "train" else []
            elif len(group) == 2:
                if self.split == "train":
                    chosen = group[:1]
                elif self.split == "test":
                    chosen = group[-1:]
                else:
                    chosen = []
            else:
                if self.split == "train":
                    chosen = group[:-2]
                elif self.split == "val":
                    chosen = group[-2:-1]
                elif self.split == "test":
                    chosen = group[-1:]
                else:
                    chosen = []
            pos_map = self._trial_to_position(subject)
            for row in chosen:
                trial_index = int(row["trial_index"])
                if trial_index in pos_map:
                    entries.append(KaraOneEntry(subject, label, stage, trial_index, pos_map[trial_index], self.split))
        entries.sort(key=lambda item: (item.subject, item.label, item.stage, item.trial_index))
        return entries

    def _load_bundle(self, subject: str) -> dict:
        if subject not in self._bundle_cache:
            path = self.root / "subjects" / f"{subject}.npz"
            bundle = np.load(path, allow_pickle=True)
            trial_indices = bundle["trial_indices"].astype(np.int32)
            stages = {}
            valid_lengths = {}
            for stage in self.stages:
                key = f"stage__{stage}"
                if key not in bundle.files:
                    raise KeyError(f"Missing {key} in {path}")
                stages[stage] = bundle[key].astype(np.float32)
                valid_key = f"{key}__valid_lengths"
                valid_lengths[stage] = (
                    bundle[valid_key].astype(np.int32)
                    if valid_key in bundle.files
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

    def _eeg(self, entry: KaraOneEntry) -> tuple[np.ndarray, int]:
        bundle = self._load_bundle(entry.subject)
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
        item = {
            "eeg": torch.from_numpy(eeg).float(),
            "eeg_valid_len": torch.tensor(valid_len, dtype=torch.long),
            "subject_idx": torch.tensor(self.subject_to_id[entry.subject], dtype=torch.long),
            "label_idx": torch.tensor(self.label_to_id[entry.label], dtype=torch.long),
            "stage_idx": torch.tensor(self.stage_to_id[entry.stage], dtype=torch.long),
            "target_seq": torch.from_numpy(self.targets.target(entry.subject, entry.trial_index)).float(),
            "target_summary": torch.from_numpy(self.targets.target_summary(entry.subject, entry.trial_index)).float(),
            "content_proto": torch.from_numpy(self.targets.content_prototype(entry.label)).float(),
            "subject_proto": torch.from_numpy(self.targets.subject_prototype(entry.subject)).float(),
            "target_log_rms": torch.tensor(self.targets.target_log_rms_value(entry.subject, entry.trial_index), dtype=torch.float32),
            "subject": entry.subject,
            "label": entry.label,
            "stage": entry.stage,
            "trial_index": entry.trial_index,
            "template_id": KaraOneTargets.key(entry.subject, entry.trial_index),
        }
        if self.aux_targets is not None:
            if self.aux_targets.has_trial(entry.subject, entry.trial_index):
                item["hubert_seq"] = torch.from_numpy(self.aux_targets.target(entry.subject, entry.trial_index)).float()
                item["hubert_summary"] = torch.from_numpy(self.aux_targets.target_summary(entry.subject, entry.trial_index)).float()
            else:
                item["hubert_seq"] = torch.zeros(self.aux_targets.T, self.aux_targets.D, dtype=torch.float32)
                item["hubert_summary"] = torch.zeros(self.aux_targets.D, dtype=torch.float32)
        return item


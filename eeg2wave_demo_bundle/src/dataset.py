from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import load_wav_fixed, resolve_feis_root


@dataclass(frozen=True)
class TrialEntry:
    trial_index: int
    position: int
    label: str
    audio_relpath: str


class FEISThinkingDataset(Dataset):
    def __init__(
        self,
        subject_id: str,
        data_root: str | Path,
        stage: str = "thinking",
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        audio_sr: int = 16000,
        audio_dur: float = 1.5,
    ):
        self.subject_id = str(subject_id)
        self.stage = str(stage)
        self.split = str(split)
        self.feis_root = resolve_feis_root(data_root)
        self.audio_root = self.feis_root
        self.audio_sr = int(audio_sr)
        self.audio_n_samples = int(round(audio_sr * audio_dur))
        self._audio_cache: dict[str, np.ndarray] = {}

        bundle_path = self.feis_root / "subjects" / f"{self.subject_id}.npz"
        if not bundle_path.exists():
            raise FileNotFoundError(f"Missing FEIS subject bundle: {bundle_path}")
        bundle = np.load(bundle_path, allow_pickle=True)
        stage_key = f"stage__{self.stage}"
        if stage_key not in bundle.files:
            raise KeyError(f"Stage {self.stage} is not present in {bundle_path.name}")

        self.eeg_array = bundle[stage_key].astype(np.float32)
        self.labels = bundle["labels"].astype(str)
        self.audio_relpaths = bundle["audio_relpaths"].astype(str)
        self.trial_indices = bundle["trial_indices"].astype(np.int32)
        self.channel_names = bundle["channel_names"].astype(str)
        self.trial_to_position = {int(trial_idx): pos for pos, trial_idx in enumerate(self.trial_indices.tolist())}

        entries = self._load_entries()
        self.entries = self._split_entries(entries, split, float(train_ratio), float(val_ratio))
        if not self.entries:
            raise ValueError(f"No FEIS samples left for subject={subject_id}, stage={stage}, split={split}")

    def _load_entries(self) -> list[TrialEntry]:
        rows: list[TrialEntry] = []
        segments_path = self.feis_root / "segments.csv"
        with segments_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row["subject_id"] != self.subject_id or row["segment_stage"] != self.stage:
                    continue
                trial_index = int(row["trial_index"])
                position = self.trial_to_position[trial_index]
                audio_relpath = row.get("audio_path") or self.audio_relpaths[position]
                rows.append(
                    TrialEntry(
                        trial_index=trial_index,
                        position=position,
                        label=str(row["label"]),
                        audio_relpath=str(audio_relpath),
                    )
                )
        rows.sort(key=lambda item: item.trial_index)
        return rows

    @staticmethod
    def _split_entries(entries: list[TrialEntry], split: str, train_ratio: float, val_ratio: float) -> list[TrialEntry]:
        count = len(entries)
        train_end = max(1, min(count, int(count * train_ratio)))
        val_end = max(train_end, min(count, int(count * (train_ratio + val_ratio))))
        if split == "train":
            return entries[:train_end]
        if split == "val":
            return entries[train_end:val_end]
        if split == "test":
            return entries[val_end:]
        raise ValueError(f"Unsupported split: {split}")

    def canonical_wavs_by_label(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for entry in self._load_entries():
            if entry.label not in out:
                out[entry.label] = self._load_audio(entry.audio_relpath)
        return out

    def _load_audio(self, relpath: str) -> np.ndarray:
        if relpath not in self._audio_cache:
            self._audio_cache[relpath] = load_wav_fixed(
                self.audio_root / relpath,
                sample_rate=self.audio_sr,
                n_samples=self.audio_n_samples,
            )
        return self._audio_cache[relpath]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, object]:
        entry = self.entries[index]
        eeg = torch.from_numpy(self.eeg_array[entry.position]).float()
        wav = torch.from_numpy(self._load_audio(entry.audio_relpath)).float()
        return {
            "eeg": eeg,
            "waveform": wav,
            "label": entry.label,
            "trial_index": entry.trial_index,
        }

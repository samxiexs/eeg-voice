"""v3 dataset wrapper.

Reuses `FEISProtocolDataset` (protocols G/S/U, EnCodec-latent target cache,
cross-subject indices) and adds an optional paired teacher-stage EEG channel
for cross-stage knowledge distillation (e.g. speaking-stage teacher feeding a
thinking-stage student for the same trial).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..dataset import FEISProtocolDataset, _normalize_subject_id


class V3Dataset(FEISProtocolDataset):
    def __init__(self, *args, teacher_stage: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_stage = None if teacher_stage in (None, "", "none") else str(teacher_stage)
        self._teacher_cache: dict[str, np.ndarray] = {}

    def _load_teacher_eeg(self, subject_id: str, position: int) -> np.ndarray:
        subject_id = _normalize_subject_id(subject_id)
        if subject_id not in self._teacher_cache:
            bundle_path = Path(self.feis_root) / "subjects" / f"{subject_id}.npz"
            bundle = np.load(bundle_path, allow_pickle=True)
            key = f"stage__{self.teacher_stage}"
            if key not in bundle.files:
                raise KeyError(f"Teacher stage {self.teacher_stage} missing in {bundle_path.name}")
            self._teacher_cache[subject_id] = bundle[key].astype(np.float32)
        return self._teacher_cache[subject_id][position]

    def __getitem__(self, index: int) -> dict[str, object]:
        item = super().__getitem__(index)
        if self.teacher_stage is not None:
            entry = self.entries[index]
            teacher_eeg = self._load_teacher_eeg(entry.subject_id, entry.position)
            item["teacher_eeg"] = torch.from_numpy(teacher_eeg).float()
        return item

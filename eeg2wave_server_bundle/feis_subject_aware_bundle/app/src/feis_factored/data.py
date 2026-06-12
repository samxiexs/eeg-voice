"""Grid dataset for the factored FEIS model.

Grid axes: subject x label x stage(hear/imagine) x repetition.
Each (subject,label) cell -> one target latent (the subject's own recording).

Splits:
  - train        : reps[:-2] of SEEN cells
  - val_seen     : 2nd-last rep of SEEN cells        (model selection; NOT test)
  - test_seen    : last rep of SEEN cells            (within-cell recognition)
  - test_holdout : ALL reps of HELD-OUT cells        (unseen subject x label combo)

Hold-out cells are chosen by a Latin-square (one cell per subject) so every
subject and every label still appears in train on OTHER cells -> tests factored
generalisation. `holdout_random=True` randomises the per-subject held-out label
(seeded) so a constant/zero-EEG predictor cannot exploit the deterministic
i%16 arrangement (the v1 "zeroeeg beats EEG on holdout" artifact).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .targets import FactoredTargets


def _norm_subject(s: str) -> str:
    s = str(s)
    return s.zfill(2) if s.isdigit() else s


@dataclass(frozen=True)
class Entry:
    subject: str
    label: str
    stage: str
    trial_index: int
    position: int          # row in the subject npz
    is_holdout: bool


class FactoredFEISDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        targets: FactoredTargets,
        split: str,                       # train | val_seen | test_seen | test_holdout
        stages: tuple[str, ...] = ("stimuli", "thinking"),
        eeg_len: int = 1280,
        include_anomalous: bool = False,
        holdout_offset: int = 0,
        holdout_random: bool = False,
        seed: int = 7,
    ):
        self.root = Path(data_root)
        self.tg = targets
        self.split = split
        self.stages = tuple(stages)
        self.eeg_len = int(eeg_len)
        self.stage_to_id = {s: i for i, s in enumerate(self.stages)}
        self._bundles: dict[str, dict] = {}

        # subjects actually present + clean filter
        subjects = sorted(self.tg.subject_vocab)
        if not include_anomalous:
            subjects = [s for s in subjects if s != "05"]
        self.subject_vocab = subjects
        self.subject_to_id = {s: i for i, s in enumerate(self.subject_vocab)}
        self.label_vocab = list(self.tg.label_vocab)
        self.label_to_id = dict(self.tg.label_to_id)

        # --- choose hold-out cells: one cell per subject.
        # Latin square (default): subject i -> label (i+offset) % n_label.
        # Random (holdout_random=True): seeded random label per subject, so a
        # constant/zero-EEG predictor cannot game the deterministic arrangement.
        n_lab = len(self.label_vocab)
        self.holdout_cells: set[tuple[str, str]] = set()
        rng = np.random.RandomState(int(seed) + 991)
        for i, sub in enumerate(self.subject_vocab):
            if holdout_random:
                lab = self.label_vocab[int(rng.randint(0, n_lab))]
            else:
                lab = self.label_vocab[(i + holdout_offset) % n_lab]
            if self.tg.has_cell(sub, lab):
                self.holdout_cells.add((sub, lab))

        rows = self._load_rows()
        self.entries = self._assign_split(rows, split)
        if not self.entries:
            raise ValueError(f"No samples for split={split}, stages={stages}")

    # --- loading ---
    def _load_rows(self) -> list[dict]:
        out = []
        with (self.root / "segments.csv").open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                stage = r.get("segment_stage")
                if stage not in self.stages:
                    continue
                sub = _norm_subject(r["subject_id"])
                if sub not in self.subject_to_id:
                    continue
                lab = str(r["label"])
                if not self.tg.has_cell(sub, lab):
                    continue
                out.append({"subject": sub, "label": lab, "stage": stage,
                            "trial_index": int(r["trial_index"])})
        return out

    def _assign_split(self, rows: list[dict], split: str) -> list[Entry]:
        # group reps per (subject,label,stage), order by trial
        grouped: dict[tuple, list[dict]] = defaultdict(list)
        for r in rows:
            grouped[(r["subject"], r["label"], r["stage"])].append(r)
        for g in grouped.values():
            g.sort(key=lambda x: x["trial_index"])

        entries: list[Entry] = []
        for (sub, lab, stage), reps in grouped.items():
            is_ho = (sub, lab) in self.holdout_cells
            pos_map = self._trial_to_pos(sub, stage)
            if is_ho:
                chosen = reps if split == "test_holdout" else []
            else:
                # SEEN cells: last rep -> test_seen, 2nd-last -> val_seen, rest -> train.
                if len(reps) == 1:
                    chosen = reps if split == "train" else []
                elif len(reps) == 2:
                    if split == "train":
                        chosen = reps[:-1]
                    elif split == "test_seen":
                        chosen = reps[-1:]
                    else:  # val_seen / test_holdout
                        chosen = []
                else:  # >= 3 reps
                    if split == "train":
                        chosen = reps[:-2]
                    elif split == "val_seen":
                        chosen = reps[-2:-1]
                    elif split == "test_seen":
                        chosen = reps[-1:]
                    else:  # test_holdout
                        chosen = []
            for r in chosen:
                ti = r["trial_index"]
                if ti not in pos_map:
                    continue
                entries.append(Entry(sub, lab, stage, ti, pos_map[ti], is_ho))
        entries.sort(key=lambda e: (e.subject, e.label, e.stage, e.trial_index))
        return entries

    def _bundle(self, subject: str) -> dict:
        if subject not in self._bundles:
            b = np.load(self.root / "subjects" / f"{subject}.npz", allow_pickle=True)
            ti = b["trial_indices"].astype(int)
            self._bundles[subject] = {
                "trial_to_pos": {int(t): i for i, t in enumerate(ti.tolist())},
                "stages": {s: b[f"stage__{s}"].astype(np.float32) for s in self.stages
                           if f"stage__{s}" in b.files},
            }
        return self._bundles[subject]

    def _trial_to_pos(self, subject: str, stage: str) -> dict[int, int]:
        return self._bundle(subject)["trial_to_pos"]

    def _eeg(self, e: Entry) -> np.ndarray:
        arr = self._bundle(e.subject)["stages"][e.stage]          # [n, C, L]
        x = arr[e.position]                                       # [C, L]
        c, l = x.shape
        if l == self.eeg_len:
            return x
        out = np.zeros((c, self.eeg_len), dtype=np.float32)
        n = min(l, self.eeg_len)
        out[:, :n] = x[:, :n]
        return out

    # --- api ---
    @property
    def num_subjects(self) -> int:
        return len(self.subject_vocab)

    @property
    def num_labels(self) -> int:
        return len(self.label_vocab)

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        e = self.entries[idx]
        eeg = self._eeg(e)
        target_seq = self.tg.cell_target(e.subject, e.label)               # [T, D]
        content_proto = self.tg.content_prototype(e.label)                 # [D]
        speaker_proto = self.tg.speaker_prototype(e.subject)               # [D]
        coarse = self.tg.coarse_ids(e.label)
        target_log_rms = self.tg.cell_log_rms(e.subject, e.label)          # scalar
        return {
            "eeg": torch.from_numpy(eeg).float(),
            "subject_idx": torch.tensor(self.subject_to_id[e.subject], dtype=torch.long),
            "label_idx": torch.tensor(self.label_to_id[e.label], dtype=torch.long),
            "stage_idx": torch.tensor(self.stage_to_id[e.stage], dtype=torch.long),
            "target_seq": torch.from_numpy(target_seq).float(),
            "content_proto": torch.from_numpy(content_proto).float(),
            "speaker_proto": torch.from_numpy(speaker_proto).float(),
            "target_log_rms": torch.tensor(target_log_rms, dtype=torch.float32),
            "manner_idx": torch.tensor(coarse["manner"], dtype=torch.long),
            "voicing_idx": torch.tensor(coarse["voicing"], dtype=torch.long),
            "vc_idx": torch.tensor(coarse["vc"], dtype=torch.long),
            "subject": e.subject,
            "label": e.label,
            "stage": e.stage,
            "cell_key": f"{e.subject}:{e.label}",
            "is_holdout": bool(e.is_holdout),
        }

"""Multi-dataset registry + unified EEG->speech dataset (v4).

A dataset is a pluggable unit described by a `DatasetSpec` (channels, stage,
how to key its EnCodec-latent targets). Both FEIS and KaraOne share the same
processed `segments.csv` schema, so one generic loader covers both:

- FEIS targets are *template-level* (key = "subject:label"; canonical wav).
- KaraOne targets are *trial-level* (key = "subject:trial_index"; trial-sync wav).

All datasets emit the same sample contract and land in one shared latent space.
Subject ids are made global across the active datasets so the subject embedding
is unambiguous.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    processed_root: str            # dir containing segments.csv + subjects/ + audio/
    n_channels: int
    stage: str = "thinking"
    target_cache_path: str = ""    # npz with target_keys/target_sequences/...
    target_key: str = "template"   # "template" (subject:label) | "trial" (subject:trial)
    eeg_len: int = 1280


# Default registry. Paths are resolved relative to the bundle at load time.
REGISTRY: dict[str, DatasetSpec] = {
    "feis": DatasetSpec(
        name="feis",
        processed_root="../data/feis",
        n_channels=14,
        stage="thinking",
        target_cache_path="../artifacts/audio_targets/feis_subject_templates_encodec_latents.npz",
        target_key="template",
        eeg_len=1280,
    ),
    "karaone": DatasetSpec(
        name="karaone",
        processed_root="../data/karaone",
        n_channels=62,
        stage="thinking",
        target_cache_path="../artifacts/audio_targets/karaone_trial_encodec_latents.npz",
        target_key="trial",
        eeg_len=1280,
    ),
}


def _normalize_subject(sid: str) -> str:
    return sid.zfill(2) if str(sid).isdigit() else str(sid)


def _pad_or_crop_time(seq: np.ndarray, target_t: int, valid: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """[T, D] -> [target_t, D] with a 0/1 validity mask over frames.

    `valid` is the number of genuinely-present frames (for variable-length
    trials padded in storage); defaults to the full sequence length.
    """
    t, d = seq.shape
    valid = t if valid is None else int(valid)
    out = np.zeros((target_t, d), dtype=np.float32)
    mask = np.zeros((target_t,), dtype=np.float32)
    n = min(t, target_t)
    out[:n] = seq[:n]
    mask[: min(valid, target_t)] = 1.0
    return out, mask


class _TargetCache:
    """Loads an EnCodec-latent target cache and serves padded, normalised frames."""

    def __init__(self, path: Path, target_steps: int):
        payload = np.load(path, allow_pickle=True)
        # Key field: prefer explicit target_keys, else template_ids.
        if "target_keys" in payload.files:
            keys = payload["target_keys"].astype(str)
        elif "template_ids" in payload.files:
            keys = payload["template_ids"].astype(str)
        else:
            raise KeyError(f"Target cache {path} lacks target_keys/template_ids")
        seqs = payload["target_sequences"].astype(np.float32)            # [N, T, D]
        self.target_steps = int(target_steps)
        self.dim = int(seqs.shape[2])
        # Normalisation stats (compute if absent).
        if "target_mean" in payload.files:
            self.mean = payload["target_mean"].astype(np.float32)
            self.std = np.maximum(payload["target_std"].astype(np.float32), 1e-6)
        else:
            self.mean = seqs.reshape(-1, self.dim).mean(0).astype(np.float32)
            self.std = np.maximum(seqs.reshape(-1, self.dim).std(0), 1e-6).astype(np.float32)
        self.index = {k: i for i, k in enumerate(keys.tolist())}
        self._raw = seqs
        self._valid = (
            payload["target_valid_steps"].astype(int) if "target_valid_steps" in payload.files else None
        )

    def get(self, key: str) -> dict[str, np.ndarray]:
        idx = self.index[key]
        raw = self._raw[idx]                                            # [T, D]
        valid = None if self._valid is None else int(self._valid[idx])
        norm = (raw - self.mean.reshape(1, -1)) / self.std.reshape(1, -1)
        seq, mask = _pad_or_crop_time(norm.astype(np.float32), self.target_steps, valid=valid)
        summary = (seq * mask.reshape(-1, 1)).sum(0) / max(mask.sum(), 1.0)
        return {"target_sequence": seq, "target_mask": mask, "target_summary": summary.astype(np.float32)}


@dataclass
class _Entry:
    subject_id: str
    trial_index: int
    position: int
    label: str
    target_key: str


class UnifiedEEGSpeechDataset(Dataset):
    """Generic per-dataset loader emitting the shared v4 sample contract."""

    def __init__(
        self,
        spec: DatasetSpec,
        bundle_dir: Path,
        split: str = "train",
        target_steps: int = 150,
        subject_offset: int = 0,
        subject_global_index: dict[str, int] | None = None,
    ):
        self.spec = spec
        self.split = split
        self.root = _resolve(spec.processed_root, bundle_dir)
        self.target_steps = int(target_steps)
        self.subject_offset = int(subject_offset)

        rows = self._load_stage_rows()
        self.label_vocab = sorted({r["label"] for r in rows})
        self.label_to_id = {l: i for i, l in enumerate(self.label_vocab)}
        self.subject_vocab = sorted({r["subject_id"] for r in rows})
        # Global subject index map (shared across datasets when provided).
        if subject_global_index is None:
            self.subject_global_index = {
                s: subject_offset + i for i, s in enumerate(self.subject_vocab)
            }
        else:
            self.subject_global_index = subject_global_index

        self.entries = self._split_rows(rows)[split]
        self._bundles: dict[str, np.ndarray] = {}
        self._trial_pos: dict[str, dict[int, int]] = {}
        self.targets = _TargetCache(_resolve(spec.target_cache_path, bundle_dir), target_steps)

    # --- row loading / splitting -------------------------------------------------
    def _load_stage_rows(self) -> list[dict]:
        path = self.root / "segments.csv"
        out = []
        with path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("segment_stage") != self.spec.stage:
                    continue
                sid = _normalize_subject(row["subject_id"])
                label = str(row["label"])
                trial = int(row["trial_index"])
                key = f"{sid}:{label}" if self.spec.target_key == "template" else f"{sid}:{trial}"
                out.append({"subject_id": sid, "trial_index": trial, "label": label, "target_key": key})
        out.sort(key=lambda r: (r["subject_id"], r["label"], r["trial_index"]))
        return out

    def _split_rows(self, rows: list[dict]) -> dict[str, list[_Entry]]:
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in rows:
            grouped[(r["subject_id"], r["label"])].append(r)
        out = {"train": [], "val": [], "test": []}
        for (_sid, _lbl), grp in grouped.items():
            grp.sort(key=lambda r: r["trial_index"])
            if len(grp) < 3:
                out["train"].extend(grp)
                continue
            for r in grp[:-2]:
                out["train"].append(r)
            out["val"].append(grp[-2])
            out["test"].append(grp[-1])
        entries = {k: [] for k in out}
        for split, rs in out.items():
            for r in rs:
                entries[split].append(
                    _Entry(r["subject_id"], r["trial_index"], -1, r["label"], r["target_key"])
                )
            entries[split].sort(key=lambda e: (e.subject_id, e.trial_index))
        return entries

    # --- eeg loading -------------------------------------------------------------
    def _eeg(self, subject_id: str, trial_index: int) -> np.ndarray:
        if subject_id not in self._bundles:
            b = np.load(self.root / "subjects" / f"{subject_id}.npz", allow_pickle=True)
            self._bundles[subject_id] = b[f"stage__{self.spec.stage}"].astype(np.float32)
            ti = b["trial_indices"].astype(int)
            self._trial_pos[subject_id] = {int(t): i for i, t in enumerate(ti.tolist())}
        arr = self._bundles[subject_id]
        pos = self._trial_pos[subject_id][int(trial_index)]
        eeg = arr[pos]                                          # [C, L_native]
        # Pad/crop to spec.eeg_len for a consistent within-dataset tensor.
        c, l = eeg.shape
        if l == self.spec.eeg_len:
            return eeg
        out = np.zeros((c, self.spec.eeg_len), dtype=np.float32)
        n = min(l, self.spec.eeg_len)
        out[:, :n] = eeg[:, :n]
        return out

    # --- dataset api -------------------------------------------------------------
    @property
    def num_labels(self) -> int:
        return len(self.label_vocab)

    def unique_target_keys(self, split: str) -> list[str]:
        return sorted({e.target_key for e in self._split_rows(self._load_stage_rows())[split]})

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        e = self.entries[idx]
        eeg = self._eeg(e.subject_id, e.trial_index)
        tgt = self.targets.get(e.target_key)
        gsid = self.subject_global_index.get(e.subject_id, self.subject_offset)
        return {
            "eeg": torch.from_numpy(eeg).float(),
            "subject_index": torch.tensor(gsid, dtype=torch.long),
            "label_id": torch.tensor(self.label_to_id[e.label], dtype=torch.long),
            "label": e.label,
            "subject_id": e.subject_id,
            "trial_index": torch.tensor(e.trial_index, dtype=torch.long),
            "target_key": e.target_key,
            "dataset_name": self.spec.name,
            "target_sequence": torch.from_numpy(tgt["target_sequence"]).float(),
            "target_mask": torch.from_numpy(tgt["target_mask"]).float(),
            "target_summary": torch.from_numpy(tgt["target_summary"]).float(),
        }


def _resolve(path: str, bundle_dir: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (bundle_dir / p).resolve()


def build_global_subjects(specs: list[DatasetSpec], bundle_dir: Path, stage_override: dict | None = None):
    """Assign a contiguous global subject id across all active datasets.

    Returns (per_dataset_index_map, total_subject_count).
    """
    index_maps: dict[str, dict[str, int]] = {}
    cursor = 0
    for spec in specs:
        root = _resolve(spec.processed_root, bundle_dir)
        subs = set()
        with (root / "segments.csv").open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("segment_stage") != spec.stage:
                    continue
                subs.add(_normalize_subject(row["subject_id"]))
        mapping = {s: cursor + i for i, s in enumerate(sorted(subs))}
        index_maps[spec.name] = mapping
        cursor += len(mapping)
    return index_maps, cursor

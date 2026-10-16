"""Isolated manifests, content-disjoint splits and fixed-time feature caches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .aligned import CONTRACT, EEG_SAMPLES, sha256, atomic_json


def truth(values):
    return values.astype(str).str.lower().isin(["true", "1", "yes"])


def content_split(frame: pd.DataFrame, *, seed: int = 31, reserved_train_ids=(),
                  joint_ood: bool = False) -> pd.DataFrame:
    """Union content and identical audio hashes before splitting across people."""
    frame = frame.copy()
    required = ["trial_id", "subject", "linguistic_content_id", "audio_sha256"]
    if frame.empty or frame[required].isna().any().any() or (frame[required] == "").any().any():
        raise ValueError("missing trial/content/subject/audio identity")
    if frame.trial_id.duplicated().any():
        raise ValueError("duplicate trial IDs")
    parent = {}

    def root(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for row in frame.itertuples():
        a, b = root("c:" + row.linguistic_content_id), root("a:" + row.audio_sha256)
        parent[max(a, b)] = min(a, b)
    frame["content_group"] = [root("c:" + c) for c in frame.linguistic_content_id]
    reserved = set(frame.loc[frame.trial_id.isin(reserved_train_ids), "content_group"])
    groups = sorted(set(frame.content_group) - reserved,
                    key=lambda c: hashlib.sha256(f"{seed}:{c}".encode()).hexdigest())
    n = len(set(frame.content_group)); holdout = max(1, int(np.ceil(n * .1)))
    if len(groups) < 2 * holdout + 1:
        raise ValueError("not enough independent contents for three roles")
    roles = {c: "train" for c in frame.content_group}
    roles.update({c: "test" for c in groups[:holdout]})
    roles.update({c: "validation" for c in groups[holdout:2 * holdout]})
    frame["role"] = frame.content_group.map(roles)
    if joint_ood:
        subjects = sorted(set(frame.subject), key=lambda s: hashlib.sha256(f"{seed}:subject:{s}".encode()).hexdigest())
        count = max(1, int(np.ceil(len(subjects) * .1)))
        if len(subjects) < 2 * count + 1:
            raise ValueError("not enough subjects for joint OOD")
        subject_roles = {s: "train" for s in subjects}
        subject_roles.update({s: "test" for s in subjects[:count]})
        subject_roles.update({s: "validation" for s in subjects[count:2 * count]})
        frame.loc[frame.role != frame.subject.map(subject_roles), "role"] = "excluded"
    frame["fold"] = 0
    validate_split(frame)
    return frame


def validate_split(frame: pd.DataFrame) -> None:
    active = frame[frame.role != "excluded"]
    if set(active.role) != {"train", "validation", "test"}:
        raise ValueError("split must contain all three roles")
    for column in ("linguistic_content_id", "audio_sha256", "content_group"):
        if active.groupby(column).role.nunique().max() != 1:
            raise ValueError(f"split leaks {column}")


def local_path(root: Path, name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else root / path


def check_shard(h5: h5py.File, row) -> int:
    index = int(float(row.shard_row))
    if h5.attrs.get("eeg_unit") != "V" or h5.attrs.get("model_time_mask_policy") != "fixed_full_epoch":
        raise ValueError("requires Volt, fixed_full_epoch EEG shard")
    if h5["eeg"].shape[-1] != EEG_SAMPLES or not h5["eeg_valid_mask"][index].all():
        raise ValueError("requires 1178 physically observed EEG samples")
    value = h5["provenance/trial_id"][index]
    value = value.decode() if isinstance(value, bytes) else str(value)
    if value != str(row.trial_id):
        raise ValueError("manifest/shard trial mismatch")
    for key in ("preprocess_config_sha256", "source_lock_sha256", "channel_order_hash", "split_index_sha256"):
        expected = str(row.get(key, ""))
        if not expected or str(h5.attrs.get(key, "")) != expected:
            raise ValueError(f"shard provenance mismatch: {key}")
    return index


def fit_eeg_normalizer(root: Path, manifest: Path, output: Path, maximum: int = 200000) -> None:
    frame = pd.read_csv(manifest, keep_default_na=False)
    train = frame[frame.role == "train"].sort_values("trial_id")
    if train.empty:
        raise ValueError("empty train fold")
    arrays, counts, order = None, None, None
    for name, rows in train.groupby("shard_path", sort=True):
        with h5py.File(local_path(root, name), "r") as h5:
            current_order = str(h5.attrs["channel_order_hash"])
            if order is not None and current_order != order:
                raise ValueError("inconsistent channel order")
            order = current_order
            channels = h5["eeg"].shape[1]
            if arrays is None:
                arrays = [[] for _ in range(channels)]; counts = np.zeros(channels, dtype=int)
            for _, row in rows.iterrows():
                index = check_shard(h5, row)
                x = h5["eeg"][index]
                if not np.isfinite(x).all():
                    raise ValueError("nonfinite EEG")
                mask = h5["channel_valid_mask"][index].astype(bool)
                for c in range(channels):
                    remaining = maximum - counts[c]
                    if mask[c] and remaining > 0:
                        take = x[c, :remaining]; arrays[c].append(take); counts[c] += len(take)
    if arrays is None or any(not a for a in arrays):
        raise ValueError("cannot normalize a channel with no valid training samples")
    center, scale = [], []
    for a in arrays:
        x = np.concatenate(a); median = np.median(x)
        center.append(float(median)); scale.append(max(float(1.4826 * np.median(np.abs(x - median))), 1e-9))
    atomic_json(output, {"contract": CONTRACT, "manifest_sha256": sha256(manifest), "fit_role": "train",
                         "channel_order_hash": order, "center": center, "scale": scale,
                         "training_trial_ids": train.trial_id.tolist()})


class AlignedDataset(Dataset):
    def __init__(self, root: Path, manifest: Path, cache: Path, role: str,
                 normalizer: Path | None = None, *, audio_only: bool = False, m0: bool = False):
        self.root = root
        self.frame = pd.read_csv(manifest, keep_default_na=False)
        self.frame = self.frame[self.frame.role == role]
        if m0:
            self.frame = self.frame[truth(self.frame.is_m0)]
        if audio_only:
            self.frame = self.frame.drop_duplicates("audio_key")
        self.frame = self.frame.reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"no examples for role={role}, m0={m0}")
        self.cache = cache; self.audio_only = audio_only
        self.normalizer = json.loads(normalizer.read_text()) if normalizer else None
        if not audio_only:
            if not self.normalizer or self.normalizer.get("fit_role") != "train" or self.normalizer["manifest_sha256"] != sha256(manifest):
                raise ValueError("EEG normalizer was not fitted for this manifest")
            for name, rows in self.frame.groupby("shard_path"):
                if "shard_sha256" not in rows or set(rows.shard_sha256) != {sha256(local_path(root, name))}:
                    raise ValueError(f"EEG shard checksum mismatch: {name}")
        with h5py.File(cache, "r") as h5:
            if h5.attrs.get("contract") != CONTRACT or h5.attrs.get("manifest_sha256") != sha256(manifest):
                raise ValueError("target cache provenance mismatch")
            missing = set(self.frame.audio_key) - set(h5["targets"])
            if missing:
                raise ValueError(f"missing targets: {sorted(missing)[:3]}")
            self.speech_times = torch.from_numpy(h5["speech_times"][:])
            self.mel_times = torch.from_numpy(h5["mel_times"][:])
            self.teacher_sha256 = str(h5.attrs["teacher_sha256"])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        # Open per item: safe under worker forks, no inherited h5py handles.
        with h5py.File(self.cache, "r") as h5:
            target = h5["targets"][row.audio_key]
            record = {k: torch.from_numpy(target[k][:].astype("float32")) for k in ("teacher", "mel", "mfcc", "wave")}
            # Used exclusively by the legacy audio-only upper-bound model.
            # AlignedEEGModel's inference signature cannot consume this field.
            record["oracle_duration_frames"] = torch.tensor(int(target.attrs["source_samples_16k"]) // 256 + 1, dtype=torch.long)
        record.update(trial_id=str(row.trial_id), content=str(row.content_group), subject=str(row.subject))
        if not self.audio_only:
            with h5py.File(local_path(self.root, row.shard_path), "r") as h5:
                i = check_shard(h5, row)
                if str(h5.attrs["channel_order_hash"]) != self.normalizer["channel_order_hash"]:
                    raise ValueError("normalizer channel mismatch")
                valid = h5["channel_valid_mask"][i].astype(bool)
                eeg = h5["eeg"][i].astype("float32")
                eeg = (eeg - np.asarray(self.normalizer["center"], dtype="float32")[:, None]) / np.asarray(self.normalizer["scale"], dtype="float32")[:, None]
                eeg *= valid[:, None]
                record.update(eeg=torch.from_numpy(eeg), channel_xyz=torch.from_numpy(h5["channel_xyz"][:].astype("float32")),
                              channel_mask=torch.from_numpy(valid), time_mask=torch.ones(EEG_SAMPLES, dtype=torch.bool))
        if any(not torch.isfinite(v).all() for v in record.values() if torch.is_tensor(v)):
            raise ValueError(f"nonfinite data: {row.trial_id}")
        return record


def fixed_bank(dataset: AlignedDataset, normalizer):
    """One frozen prototype per content, averaging distinct audio realizations."""
    labels, bank = [], []
    with h5py.File(dataset.cache, "r") as h5:
        for label, rows in dataset.frame.groupby("content_group", sort=True):
            values = [normalizer(torch.from_numpy(h5["targets"][key]["teacher"][:])).mean(0)
                      for key in sorted(set(rows.audio_key))]
            labels.append(str(label)); bank.append(torch.stack(values).mean(0))
    if len(bank) < 2:
        raise ValueError("contrastive training needs at least two independent contents")
    return labels, torch.stack(bank).detach()


def wrong_trial_indices(frame: pd.DataFrame) -> list[int]:
    """Same-person, different-content controls independent of minibatch size."""
    result = []
    for i, row in frame.iterrows():
        candidates = frame.index[(frame.subject == row.subject) & (frame.content_group != row.content_group)].tolist()
        if not candidates:
            raise ValueError(f"no within-subject wrong trial for {row.trial_id}")
        candidates.sort(key=lambda j: hashlib.sha256(f"{row.trial_id}:{frame.iloc[j].trial_id}".encode()).hexdigest())
        result.append(candidates[0])
    return result

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from src.karaone_v9.data import KaraOneV9Dataset, KaraOneV9Entry, KaraOneV9TargetBank


class KaraOneV91ClusterBank:
    """Train-only fitted cluster bank with heldout assignment metadata."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.available = bool(self.path and self.path.exists())
        self.by_key: dict[str, dict[str, int | bool]] = {}
        self.n_eeg_clusters = 1
        self.n_speech_clusters = 1
        self.n_cross_clusters = 1
        self.eeg_to_speech_soft = np.ones((1, 1), dtype=np.float32)
        if not self.available:
            return
        payload = np.load(self.path, allow_pickle=True)
        keys = [str(item) for item in payload["keys"].tolist()]
        eeg = payload["eeg_cluster_id"].astype(np.int64)
        speech = payload["speech_cluster_id"].astype(np.int64)
        cross = payload["cross_modal_cluster_id"].astype(np.int64)
        fit = payload["fit_split"].astype(bool) if "fit_split" in payload.files else np.zeros(len(keys), dtype=bool)
        for idx, key in enumerate(keys):
            self.by_key[key] = {
                "eeg_cluster_id": int(eeg[idx]),
                "speech_cluster_id": int(speech[idx]),
                "cross_modal_cluster_id": int(cross[idx]),
                "cluster_fit_split": bool(fit[idx]),
            }
        self.n_eeg_clusters = int(eeg.max()) + 1 if eeg.size else 1
        self.n_speech_clusters = int(speech.max()) + 1 if speech.size else 1
        self.n_cross_clusters = int(cross.max()) + 1 if cross.size else 1
        if "eeg_to_speech_soft" in payload.files:
            self.eeg_to_speech_soft = payload["eeg_to_speech_soft"].astype(np.float32)

    @staticmethod
    def key(subject: str, stage: str, trial_index: int) -> str:
        return f"{subject}:{stage}:{int(trial_index)}"

    def lookup(self, subject: str, stage: str, trial_index: int) -> dict[str, int | bool]:
        return self.by_key.get(
            self.key(subject, stage, trial_index),
            {
                "eeg_cluster_id": 0,
                "speech_cluster_id": 0,
                "cross_modal_cluster_id": 0,
                "cluster_fit_split": False,
            },
        )


class KaraOneV91ClusteredDataset(Dataset):
    """Thin v9 dataset wrapper that attaches v9.1 cluster metadata."""

    def __init__(
        self,
        data_root: str | Path,
        targets: KaraOneV9TargetBank,
        split: str,
        *,
        cluster_bank: KaraOneV91ClusterBank | None = None,
        stages: tuple[str, ...] = ("overt_like",),
        subject_val: str = "P02",
        subject_test: str = "MM21",
        eeg_len: int = 1280,
        train_split_mode: str = "subject_train",
        require_codec: bool = False,
    ):
        self.base = KaraOneV9Dataset(
            data_root,
            targets,
            split,
            stages=stages,
            subject_val=subject_val,
            subject_test=subject_test,
            eeg_len=eeg_len,
            train_split_mode=train_split_mode,
            require_codec=require_codec,
        )
        self.cluster_bank = cluster_bank or KaraOneV91ClusterBank(None)
        self.entries = self.base.entries
        self.targets = targets
        self.subject_vocab = self.base.subject_vocab
        self.label_vocab = self.base.label_vocab
        self.subject_to_id = self.base.subject_to_id
        self.label_to_id = self.base.label_to_id
        self.stage_to_id = self.base.stage_to_id
        self.stages = self.base.stages

    @property
    def num_subjects(self) -> int:
        return self.base.num_subjects

    @property
    def num_labels(self) -> int:
        return self.base.num_labels

    @property
    def num_stages(self) -> int:
        return self.base.num_stages

    @property
    def heldout_subjects(self) -> set[str]:
        return self.base.heldout_subjects

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        cluster = self.cluster_bank.lookup(str(item["subject"]), str(item["stage"]), int(item["trial_index"]))
        item["eeg_cluster_id"] = torch.tensor(int(cluster["eeg_cluster_id"]), dtype=torch.long)
        item["speech_cluster_id"] = torch.tensor(int(cluster["speech_cluster_id"]), dtype=torch.long)
        item["cross_modal_cluster_id"] = torch.tensor(int(cluster["cross_modal_cluster_id"]), dtype=torch.long)
        item["cluster_fit_split"] = torch.tensor(bool(cluster["cluster_fit_split"]), dtype=torch.bool)
        return item

    def entry(self, idx: int) -> KaraOneV9Entry:
        return self.entries[idx]


class KaraOneV91ClusterBalancedBatchSampler(Sampler[list[int]]):
    """Simple multi-subject, multi-cluster sampler for v9.1 training.

    It samples one item from different (subject, eeg_cluster, speech_cluster)
    buckets per batch when possible.  This keeps batches usable for
    same-cluster/different-subject positives and hard negatives.
    """

    def __init__(self, dataset: KaraOneV91ClusteredDataset, *, batch_size: int, seed: int = 7, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.buckets: dict[tuple[str, int, int], list[int]] = {}
        for idx, entry in enumerate(dataset.entries):
            cluster = dataset.cluster_bank.lookup(entry.subject, entry.stage, entry.trial_index)
            key = (entry.subject, int(cluster["eeg_cluster_id"]), int(cluster["speech_cluster_id"]))
            self.buckets.setdefault(key, []).append(idx)
        self.bucket_keys = list(self.buckets)
        if not self.bucket_keys:
            raise ValueError("ClusterBalancedBatchSampler received an empty dataset")

    def __len__(self) -> int:
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else int(math.ceil(n / max(self.batch_size, 1)))

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed)
        pools = {key: rng.permutation(values).tolist() for key, values in self.buckets.items()}
        remaining = sum(len(values) for values in pools.values())
        while remaining > 0:
            batch: list[int] = []
            used_subjects: set[str] = set()
            used_clusters: set[tuple[int, int]] = set()
            keys = self.bucket_keys[:]
            rng.shuffle(keys)

            # First pass keeps batches diverse.  It must be finite; otherwise
            # the final sparse buckets can deadlock when every remaining bucket
            # violates the diversity preference.
            for key in keys:
                if len(batch) >= self.batch_size:
                    break
                if not pools[key]:
                    continue
                subject, eeg_cluster, speech_cluster = key
                cluster_pair = (int(eeg_cluster), int(speech_cluster))
                if subject in used_subjects and cluster_pair in used_clusters and len(batch) < max(2, self.batch_size // 2):
                    continue
                batch.append(int(pools[key].pop()))
                remaining -= 1
                used_subjects.add(subject)
                used_clusters.add(cluster_pair)

            # Relax the diversity constraint to drain the tail of the epoch.
            while len(batch) < self.batch_size and remaining > 0:
                nonempty = [key for key in self.bucket_keys if pools[key]]
                if not nonempty:
                    break
                key = nonempty[int(rng.integers(0, len(nonempty)))]
                while pools[key] and len(batch) < self.batch_size:
                    batch.append(int(pools[key].pop()))
                    remaining -= 1
            if not batch:
                break
            if self.drop_last and len(batch) < self.batch_size:
                break
            rng.shuffle(batch)
            yield batch


def load_channel_names(data_root: str | Path, subject: str | None = None, n_channels: int = 62) -> list[str]:
    root = Path(data_root)
    subjects_dir = root / "subjects"
    candidates = []
    if subject:
        candidates.append(subjects_dir / f"{subject}.npz")
    candidates.extend(sorted(subjects_dir.glob("*.npz")))
    for path in candidates:
        if not path.exists():
            continue
        payload = np.load(path, allow_pickle=True)
        if "channel_names" in payload.files:
            names = [str(item) for item in payload["channel_names"].tolist()]
            if names:
                return names
    return [f"Ch{idx + 1:03d}" for idx in range(int(n_channels))]

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.karaone_v10.data import (
    KaraOneV10ClusterBalancedBatchSampler,
    KaraOneV10ClusterBank,
    KaraOneV10ClusteredDataset,
    load_channel_names,
)


class KaraOneV11TokenBank:
    """Train-only token/codebook cache for v11.

    The bank intentionally stores fitted codebooks and cluster assignments
    outside heldout statistics.  Missing banks are allowed for synthetic tests
    and smoke planning, but real training should run the token builder first.
    """

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.available = bool(self.path and self.path.exists())
        self.by_key: dict[str, dict[str, np.ndarray | bool]] = {}
        self.codec_codebook = np.zeros((1, 1), dtype=np.float32)
        self.channel_cluster_id = np.zeros(62, dtype=np.int64)
        self.n_channel_clusters = 1
        self.codec_token_vocab = 1
        self.codec_steps = 1
        self.codec_dim = 1
        if not self.available:
            return
        payload = np.load(self.path, allow_pickle=True)
        keys = [str(item) for item in payload["keys"].tolist()]
        codec_tokens = payload["codec_token_ids"].astype(np.int64)
        codec_mask = payload["codec_token_mask"].astype(np.float32)
        fit = payload["fit_split"].astype(bool) if "fit_split" in payload.files else np.zeros(len(keys), dtype=bool)
        for idx, key in enumerate(keys):
            self.by_key[key] = {
                "codec_token_ids": codec_tokens[idx],
                "codec_token_mask": codec_mask[idx],
                "token_fit_split": bool(fit[idx]),
            }
        self.codec_codebook = payload["codec_codebook"].astype(np.float32)
        self.channel_cluster_id = payload["channel_cluster_id"].astype(np.int64)
        self.n_channel_clusters = int(payload["n_channel_clusters"]) if "n_channel_clusters" in payload.files else int(self.channel_cluster_id.max()) + 1
        self.codec_token_vocab = int(self.codec_codebook.shape[0])
        self.codec_steps = int(codec_tokens.shape[1]) if codec_tokens.ndim == 2 else 1
        self.codec_dim = int(self.codec_codebook.shape[1]) if self.codec_codebook.ndim == 2 else 1

    @staticmethod
    def key(subject: str, stage: str, trial_index: int) -> str:
        return f"{subject}:{stage}:{int(trial_index)}"

    def lookup(self, subject: str, stage: str, trial_index: int) -> dict[str, np.ndarray | bool]:
        return self.by_key.get(
            self.key(subject, stage, trial_index),
            {
                "codec_token_ids": np.zeros(self.codec_steps, dtype=np.int64),
                "codec_token_mask": np.zeros(self.codec_steps, dtype=np.float32),
                "token_fit_split": False,
            },
        )


class KaraOneV11Dataset(torch.utils.data.Dataset):
    """v11 token-first dataset wrapper over the v10 clustered dataset."""

    def __init__(
        self,
        data_root: str | Path,
        targets,
        split: str,
        *,
        cluster_bank: KaraOneV10ClusterBank | None = None,
        token_bank: KaraOneV11TokenBank | None = None,
        stages: tuple[str, ...] = ("overt_like",),
        subject_val: str = "P02",
        subject_test: str = "MM21",
        eeg_len: int = 1280,
        require_codec: bool = False,
    ):
        self.base = KaraOneV10ClusteredDataset(
            data_root,
            targets,
            split,
            cluster_bank=cluster_bank,
            stages=stages,
            subject_val=subject_val,
            subject_test=subject_test,
            eeg_len=eeg_len,
            require_codec=require_codec,
        )
        self.token_bank = token_bank or KaraOneV11TokenBank(None)
        self.entries = self.base.entries
        self.targets = targets
        self.subject_vocab = self.base.subject_vocab
        self.label_vocab = self.base.label_vocab
        self.subject_to_id = self.base.subject_to_id
        self.label_to_id = self.base.label_to_id
        self.stage_to_id = self.base.stage_to_id
        self.stages = self.base.stages
        self.cluster_bank = self.base.cluster_bank

    @property
    def num_subjects(self) -> int:
        return self.base.num_subjects

    @property
    def num_labels(self) -> int:
        return self.base.num_labels

    @property
    def num_stages(self) -> int:
        return self.base.num_stages

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.base[idx]
        token = self.token_bank.lookup(str(item["subject"]), str(item["stage"]), int(item["trial_index"]))
        channel_clusters = self.token_bank.channel_cluster_id
        if channel_clusters.shape[0] != item["eeg"].shape[0]:
            channel_clusters = np.resize(channel_clusters, item["eeg"].shape[0]).astype(np.int64)
        item["audio_semantic_tokens"] = item["semantic_token_targets"]
        item["audio_semantic_token_mask"] = item["semantic_token_mask"]
        item["codec_token_targets"] = torch.from_numpy(np.asarray(token["codec_token_ids"], dtype=np.int64)).long()
        item["codec_token_mask"] = torch.from_numpy(np.asarray(token["codec_token_mask"], dtype=np.float32)).float()
        item["token_fit_split"] = torch.tensor(bool(token["token_fit_split"]), dtype=torch.bool)
        item["channel_cluster_id"] = torch.from_numpy(channel_clusters.astype(np.int64)).long()
        return item


def outputs_to_token_bank(outputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "pred_semantic_token_hist": outputs["pred_semantic_token_hist"],
        "target_semantic_token_hist": outputs["target_semantic_token_hist"],
        "target_semantic_tokens": outputs["target_semantic_tokens"],
        "target_semantic_token_mask": outputs["target_semantic_token_mask"],
        "labels": outputs["labels"],
        "subjects": outputs["subjects"],
        "semantic_summary": outputs["target"],
    }


__all__ = [
    "KaraOneV10ClusterBalancedBatchSampler",
    "KaraOneV10ClusterBank",
    "KaraOneV11Dataset",
    "KaraOneV11TokenBank",
    "load_channel_names",
    "outputs_to_token_bank",
]

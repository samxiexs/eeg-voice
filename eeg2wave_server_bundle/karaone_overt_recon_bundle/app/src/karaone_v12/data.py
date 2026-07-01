from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.karaone_v11.data import (
    KaraOneV10ClusterBalancedBatchSampler,
    KaraOneV10ClusterBank,
    KaraOneV11Dataset,
    KaraOneV11TokenBank,
    load_channel_names,
    outputs_to_token_bank,
)


class KaraOneV12TimeAnchorBank:
    """Train-only fitted time-anchor cache with heldout diagnostic assignments."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.available = bool(self.path and self.path.exists())
        self.by_key: dict[str, dict[str, Any]] = {}
        self.active_steps = 200
        self.sample_rate = 16000
        self.duration_sec = 2.0
        self.stage_lag_prior: dict[str, float] = {}
        if not self.available:
            return
        payload = np.load(self.path, allow_pickle=True)
        keys = [str(item) for item in payload["keys"].tolist()]
        self.active_steps = int(payload["active_mask"].shape[1])
        self.sample_rate = int(payload["sample_rate"]) if "sample_rate" in payload.files else 16000
        self.duration_sec = float(payload["duration_sec"]) if "duration_sec" in payload.files else 2.0
        fit = payload["fit_split"].astype(bool) if "fit_split" in payload.files else np.zeros(len(keys), dtype=bool)
        for idx, key in enumerate(keys):
            self.by_key[key] = {
                "onset_sec": float(payload["onset_sec"][idx]),
                "duration_sec": float(payload["duration_active_sec"][idx]),
                "center_sec": float(payload["center_sec"][idx]),
                "lag_sec": float(payload["lag_sec"][idx]),
                "confidence": float(payload["confidence"][idx]),
                "active_mask": payload["active_mask"][idx].astype(np.float32),
                "envelope": payload["envelope"][idx].astype(np.float32),
                "time_fit_split": bool(fit[idx]),
            }
        if "stage_lag_prior_keys" in payload.files:
            keys_stage = [str(item) for item in payload["stage_lag_prior_keys"].tolist()]
            vals = payload["stage_lag_prior_values"].astype(np.float32)
            self.stage_lag_prior = {key: float(vals[idx]) for idx, key in enumerate(keys_stage)}

    @staticmethod
    def key(subject: str, stage: str, trial_index: int) -> str:
        return f"{subject}:{stage}:{int(trial_index)}"

    def lookup(self, subject: str, stage: str, trial_index: int) -> dict[str, Any]:
        default = {
            "onset_sec": 0.0,
            "duration_sec": 0.0,
            "center_sec": 0.0,
            "lag_sec": float(self.stage_lag_prior.get(str(stage), 0.0)),
            "confidence": 0.0,
            "active_mask": np.zeros(self.active_steps, dtype=np.float32),
            "envelope": np.zeros(self.active_steps, dtype=np.float32),
            "time_fit_split": False,
        }
        return self.by_key.get(self.key(subject, stage, trial_index), default)


class KaraOneV12Dataset(KaraOneV11Dataset):
    """v12 dataset: v11 token data plus time-anchor targets."""

    def __init__(
        self,
        data_root: str | Path,
        targets,
        split: str,
        *,
        cluster_bank: KaraOneV10ClusterBank | None = None,
        token_bank: KaraOneV11TokenBank | None = None,
        time_anchor_bank: KaraOneV12TimeAnchorBank | None = None,
        stages: tuple[str, ...] = ("overt_like",),
        subject_val: str = "P02",
        subject_test: str = "MM21",
        eeg_len: int = 1280,
        require_codec: bool = False,
    ):
        super().__init__(
            data_root,
            targets,
            split,
            cluster_bank=cluster_bank,
            token_bank=token_bank,
            stages=stages,
            subject_val=subject_val,
            subject_test=subject_test,
            eeg_len=eeg_len,
            require_codec=require_codec,
        )
        self.time_anchor_bank = time_anchor_bank or KaraOneV12TimeAnchorBank(None)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = super().__getitem__(idx)
        anchor = self.time_anchor_bank.lookup(str(item["subject"]), str(item["stage"]), int(item["trial_index"]))
        item["time_onset_sec"] = torch.tensor(float(anchor["onset_sec"]), dtype=torch.float32)
        item["time_duration_sec"] = torch.tensor(float(anchor["duration_sec"]), dtype=torch.float32)
        item["time_center_sec"] = torch.tensor(float(anchor["center_sec"]), dtype=torch.float32)
        item["time_lag_sec"] = torch.tensor(float(anchor["lag_sec"]), dtype=torch.float32)
        item["time_confidence"] = torch.tensor(float(anchor["confidence"]), dtype=torch.float32)
        item["time_active_mask"] = torch.from_numpy(np.asarray(anchor["active_mask"], dtype=np.float32)).float()
        item["time_envelope"] = torch.from_numpy(np.asarray(anchor["envelope"], dtype=np.float32)).float()
        item["time_fit_split"] = torch.tensor(bool(anchor["time_fit_split"]), dtype=torch.bool)
        return item


__all__ = [
    "KaraOneV10ClusterBalancedBatchSampler",
    "KaraOneV10ClusterBank",
    "KaraOneV11TokenBank",
    "KaraOneV12Dataset",
    "KaraOneV12TimeAnchorBank",
    "load_channel_names",
    "outputs_to_token_bank",
]

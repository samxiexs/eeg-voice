from __future__ import annotations

from pathlib import Path

import numpy as np


class MelLabelTargets:
    def __init__(self, cache_path: str | Path):
        payload = np.load(Path(cache_path), allow_pickle=True)
        self.label_vocab = payload["label_vocab"].astype(str).tolist()
        self.label_to_id = {label: idx for idx, label in enumerate(self.label_vocab)}
        self.target_kind = str(payload["target_kind"].item()) if "target_kind" in payload.files else "mel"
        self.raw_banks = payload["target_banks"].astype(np.float32)
        self.target_mean = payload["target_mean"].astype(np.float32)
        self.target_std = np.maximum(payload["target_std"].astype(np.float32), 1e-6)
        self.T = int(self.raw_banks.shape[2])
        self.D = int(self.raw_banks.shape[3])
        self.banks = ((self.raw_banks - self.target_mean.reshape(1, 1, 1, -1)) / self.target_std.reshape(1, 1, 1, -1)).astype(np.float32)
        self.canonical_audio_paths = payload["canonical_audio_paths"].astype(str)
        self.target_rms = payload["target_rms"].astype(np.float32)
        self.target_log_rms = np.log(np.maximum(self.target_rms, 1e-8)).astype(np.float32)
        self.global_mean_raw = self.raw_banks.mean(axis=(0, 1)).astype(np.float32)
        self.label_prototypes = self.banks.mean(axis=(1, 2)).astype(np.float32)
        self.decoder_scales = (
            payload["decoder_scales"].astype(np.float32)
            if "decoder_scales" in payload.files
            else np.ones((self.num_labels, self.refs_per_label, 1), dtype=np.float32)
        )
        self.default_decoder_scales = (
            payload["default_decoder_scales"].astype(np.float32)
            if "default_decoder_scales" in payload.files
            else self.decoder_scales.reshape(-1, self.decoder_scales.shape[-1]).mean(axis=0).astype(np.float32)
        )

    @property
    def num_labels(self) -> int:
        return len(self.label_vocab)

    @property
    def refs_per_label(self) -> int:
        return int(self.banks.shape[1])

    def bank_for_label_id(self, label_idx: int) -> np.ndarray:
        return self.banks[int(label_idx)]

    def raw_bank_for_label_id(self, label_idx: int) -> np.ndarray:
        return self.raw_banks[int(label_idx)]

    def label_id(self, label: str) -> int:
        return self.label_to_id[str(label)]

    def canonical_path_for_label_id(self, label_idx: int, ref_idx: int = 0) -> str:
        return str(self.canonical_audio_paths[int(label_idx), int(ref_idx)])

    def decoder_scale_for_label_id(self, label_idx: int, ref_idx: int = 0) -> np.ndarray:
        return self.decoder_scales[int(label_idx), int(ref_idx)].astype(np.float32)

    def log_rms_for_label_id(self, label_idx: int) -> float:
        return float(self.target_log_rms[int(label_idx)].mean())

    def rms_for_label_id(self, label_idx: int) -> float:
        return float(self.target_rms[int(label_idx)].mean())

    def denormalize(self, norm_mel: np.ndarray) -> np.ndarray:
        return (np.asarray(norm_mel, dtype=np.float32) * self.target_std.reshape(1, -1) + self.target_mean.reshape(1, -1)).astype(np.float32)

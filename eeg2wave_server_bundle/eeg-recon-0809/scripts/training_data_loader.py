"""Fail-closed HDF5 reader for harmonized v3 EEG shards.

The loader exposes the variable-channel/time contract used by the joint model.
It never infers acoustic supervision from the presence of a WAV: supervision
is read from the audited ``pairing_level`` recorded in the shard.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _channel_hash(channels: list[str]) -> str:
    import hashlib
    return hashlib.sha256(("\n".join(channels) + "\n").encode()).hexdigest()


def _decode(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


class TrainingShardDataset:
    def __init__(self, shard_path: str | Path, channel_order: list[str] | None,
                 split_index_sha256: str, normalizer: dict[str, Any] | None = None,
                 allow_mixed_production: bool | None = None):
        import h5py
        import numpy as np

        self.path = Path(shard_path)
        self.h5 = h5py.File(self.path, "r")
        if self.h5.attrs.get("eeg_unit") != "V":
            raise ValueError("refusing non-Volt EEG shard")
        stored_order = json.loads(self.h5.attrs["channel_order"]) if isinstance(self.h5.attrs.get("channel_order"), str) else list(self.h5.attrs["channel_order"])
        self.channel_order = [_decode(value) for value in stored_order]
        if channel_order is not None and self.h5.attrs.get("channel_order_hash") != _channel_hash(channel_order):
            raise ValueError("refusing channel-order-incompatible shard")
        if not split_index_sha256 or self.h5.attrs.get("split_index_sha256", "") != split_index_sha256:
            raise ValueError("refusing shard without the requested frozen split hash")
        if "channel_xyz" not in self.h5:
            raise ValueError("refusing shard without channel coordinates")
        self.xyz = self.h5["channel_xyz"][:].astype("float32")
        if self.xyz.shape != (len(self.channel_order), 3) or not np.isfinite(self.xyz).all():
            raise ValueError("invalid channel coordinate contract")
        self.provenance = self.h5["provenance"]
        self.ids = [_decode(value) for value in self.provenance["trial_id"][:]]
        self.normalizer = normalizer
        if allow_mixed_production:
            raise ValueError("mixed-production cropping was retired: both datasets are perception tasks")

    def __len__(self):
        return len(self.ids)

    def _provenance(self, key: str, index: int) -> str:
        return _decode(self.provenance[key][index]) if key in self.provenance else ""

    def __getitem__(self, index: int):
        import numpy as np

        eeg = self.h5["eeg"][index].astype("float32")
        time_mask = self.h5["eeg_valid_mask"][index].astype(bool)
        channel_mask = self.h5["channel_valid_mask"][index].astype(bool)
        if self.normalizer is not None:
            center = np.asarray(self.normalizer["center_median_v"], dtype="float32")[:, None]
            scale = np.asarray(self.normalizer["scale_mad_v"], dtype="float32")[:, None]
            if center.shape[0] != eeg.shape[0]:
                raise ValueError("normalizer/channel mismatch")
            eeg = (eeg - center) / np.maximum(scale, 1e-9)
        # Keep padding and absent channels inert even when a non-zero robust
        # center is subtracted.  The model re-applies these masks defensively.
        eeg *= time_mask[None, :]
        eeg *= channel_mask[:, None]
        pairing = self._provenance("pairing_level", index)
        supervision = self._provenance("supervision_type", index)
        audio_loss_mask = self.h5["audio_loss_mask"][index].astype(bool)
        record = {
            "trial_id": self.ids[index],
            "eeg": eeg,
            "channel_xyz": self.xyz.copy(),
            "channel_mask": channel_mask,
            "time_mask": time_mask,
            "eeg_valid_mask": time_mask,
            "clean_perception_mask": self.h5["clean_perception_mask"][index].astype(bool),
            "audio_loss_mask": audio_loss_mask,
            "bad_channel_mask": self.h5["bad_channel_mask"][index].astype(bool),
            "pairing_level": pairing,
            "supervision_type": supervision,
            "content_supervision": supervision in {"paired_audio", "weak_audio"},
            "acoustic_supervision": bool(audio_loss_mask.any()),
            "label_supervision": supervision in {"weak_audio", "label_only"},
        }
        for key in ("dataset", "subject", "task", "condition", "linguistic_content_id", "waveform_id", "phoneme_label", "audio_id"):
            record[key] = self._provenance(key, index)
        return record

    def close(self):
        if getattr(self, "h5", None) is not None:
            self.h5.close()
            self.h5 = None

    def __del__(self):
        self.close()

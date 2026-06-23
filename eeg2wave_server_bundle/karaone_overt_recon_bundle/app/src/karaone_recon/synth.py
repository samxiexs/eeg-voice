from __future__ import annotations

import numpy as np

from ..audio_features import AudioFeatureConfig, load_codec_backend


def denormalize_latent(pred_norm: np.ndarray, target_mean: np.ndarray, target_std: np.ndarray) -> np.ndarray:
    return (pred_norm * target_std.reshape(1, -1) + target_mean.reshape(1, -1)).astype(np.float32)


def build_codec_backend(
    codec_model_path: str,
    duration_sec: float = 2.0,
    bandwidth: float = 6.0,
    local_files_only: bool = True,
):
    cfg = AudioFeatureConfig(
        duration_sec=duration_sec,
        target_kind="encodec_latent",
        backend="encodec_latent",
        codec_model_name_or_path=codec_model_path,
        codec_bandwidth=bandwidth,
        local_files_only=local_files_only,
    )
    return load_codec_backend(cfg)


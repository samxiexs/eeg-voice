"""v3 synthesis: predicted EnCodec latents -> frozen EnCodec decoder -> wav.

The model predicts in the *normalised* latent space of the target cache. We
denormalise with the cache statistics and hand the result to the frozen
EnCodec decoder. Because the decoder is a trained neural codec, the output is
clean, natural speech regardless of how good the EEG prediction is — this is
what makes v3 outputs sound like real voice instead of the old averaged hum.
"""

from __future__ import annotations

import numpy as np

from ..audio_features import AudioFeatureConfig, load_codec_backend


def denormalize_latent(
    pred_norm: np.ndarray, target_mean: np.ndarray, target_std: np.ndarray
) -> np.ndarray:
    """[T, D] normalised -> raw EnCodec latent space."""
    return (pred_norm * target_std.reshape(1, -1) + target_mean.reshape(1, -1)).astype(np.float32)


def build_codec_backend(
    codec_model_path: str,
    duration_sec: float = 1.0,
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


def latent_to_wav(
    backend,
    pred_norm_latent: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    decoder_scales: np.ndarray | None = None,
) -> np.ndarray:
    raw = denormalize_latent(pred_norm_latent, target_mean, target_std)
    return backend.decode(raw, decoder_scales=decoder_scales)

"""Factory functions for EEGVoiceTokenV1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_simple_yaml
from .tokenizer import EEGVoiceV1Config, default_quantizer_groups
from .voice_model import EEGVoiceTokenV1


def _as_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return tuple(int(x) for x in value)
    if isinstance(value, list):
        return tuple(int(x) for x in value)
    raise TypeError(f"Expected list/tuple, got {type(value)!r}")


def v1_config_from_dict(cfg: dict[str, Any]) -> EEGVoiceV1Config:
    tokenizer = cfg.get("tokenizer", {})
    heads = cfg.get("heads", {})
    retrieval = cfg.get("retrieval", {})
    quantizer_groups = default_quantizer_groups()
    if "quantizer_groups" in tokenizer:
        quantizer_groups = {name: _as_tuple(values) for name, values in tokenizer["quantizer_groups"].items()}
    return EEGVoiceV1Config(
        sample_rate=int(tokenizer.get("sample_rate", 250)),
        window_sec=float(tokenizer.get("window_sec", 2.0)),
        dim=int(tokenizer.get("dim", 256)),
        latent_queries=int(tokenizer.get("latent_queries", 32)),
        codebook_dim=int(tokenizer.get("codebook_dim", tokenizer.get("dim", 256))),
        codebook_size=int(tokenizer.get("codebook_size", 1024)),
        num_quantizers=int(tokenizer.get("num_quantizers", 8)),
        quantizer_groups=quantizer_groups,
        encoder_channels=int(tokenizer.get("encoder_channels", 96)),
        downsample_rates=_as_tuple(tokenizer.get("downsample_rates", [2, 2, 2, 2])),
        n_heads=int(tokenizer.get("n_heads", 8)),
        dropout=float(tokenizer.get("dropout", 0.1)),
        sensor_pos_dim=int(tokenizer.get("sensor_pos_dim", 6)),
        n_sensor_types=int(tokenizer.get("n_sensor_types", 3)),
        mask_ratio=float(tokenizer.get("mask_ratio", 0.25)),
        noise_std=float(tokenizer.get("noise_std", 0.05)),
        encoder_residual_layers=int(tokenizer.get("encoder_residual_layers", 2)),
        temporal_layers=int(tokenizer.get("temporal_layers", 2)),
        q7_full_recon_weight=float(tokenizer.get("q7_full_recon_weight", 0.25)),
        q7_group_dropout=float(tokenizer.get("q7_group_dropout", 0.5)),
        retrieval_queue_size=int(retrieval.get("queue_size", 4096)),
        retrieval_queue_negatives=int(retrieval.get("queue_negatives", 256)),
        retrieval_temperature=float(retrieval.get("temperature", 0.07)),
        audio_embedding_dim=int(heads.get("audio_embedding_dim", 11)),
        projection_dim=int(heads.get("projection_dim", 256)),
        content_classes=int(heads.get("content_classes", 64)),
        phoneme_classes=int(heads.get("phoneme_classes", 48)),
        pitch_dim=int(heads.get("pitch_dim", 1)),
        prosody_dim=int(heads.get("prosody_dim", 2)),
        timbre_dim=int(heads.get("timbre_dim", 8)),
        style_classes=int(heads.get("style_classes", 5)),
        mode_labels=tuple(str(x) for x in heads.get("mode_labels", ["heard", "imagined", "inner", "overt", "visualized_control"])),
        dataset_adapter_count=int(heads.get("dataset_adapter_count", 128)),
    )


def build_eeg_voice_token_v1(config: EEGVoiceV1Config | dict[str, Any] | str | Path | None = None) -> EEGVoiceTokenV1:
    if config is None:
        return EEGVoiceTokenV1()
    if isinstance(config, EEGVoiceV1Config):
        return EEGVoiceTokenV1(config)
    if isinstance(config, dict):
        return EEGVoiceTokenV1(v1_config_from_dict(config))
    cfg = load_simple_yaml(Path(config))
    return EEGVoiceTokenV1(v1_config_from_dict(cfg))


def build_model_v1_bundle(config_path: str | Path = "configs/model_v1.yaml") -> dict[str, Any]:
    cfg = load_simple_yaml(Path(config_path))
    config = v1_config_from_dict(cfg)
    return {"config": cfg, "model_config": config, "model": EEGVoiceTokenV1(config)}

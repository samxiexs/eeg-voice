"""Factory functions for assembling token-centric EEG voice models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_simple_yaml
from .heads import AudioContrastiveHead, ProbeHead, VoiceAttributeHead
from .tokenizer import BrainStyleEEGTokenizerConfig, BrainStyleEEGTokenizerV0
from .voice_model import TokenCentricEEGVoiceConfig, TokenCentricEEGVoiceModelV01


def tokenizer_config_from_dict(cfg: dict[str, Any]) -> BrainStyleEEGTokenizerConfig:
    """Convert the `tokenizer:` section of `configs/model_v0.yaml`."""
    return BrainStyleEEGTokenizerConfig(
        sample_rate=int(cfg["sample_rate"]),
        window_sec=float(cfg["window_sec"]),
        dim=int(cfg["dim"]),
        latent_queries=int(cfg["latent_queries"]),
        codebook_dim=int(cfg.get("codebook_dim", cfg["dim"])),
        codebook_size=int(cfg["codebook_size"]),
        num_quantizers=int(cfg["num_quantizers"]),
        encoder_channels=int(cfg["encoder_channels"]),
        downsample_rates=tuple(int(x) for x in cfg["downsample_rates"]),
        n_heads=int(cfg["n_heads"]),
        dropout=float(cfg["dropout"]),
        sensor_pos_dim=int(cfg.get("sensor_pos_dim", 6)),
        n_sensor_types=int(cfg.get("n_sensor_types", 3)),
        mask_ratio=float(cfg.get("mask_ratio", 0.25)),
        noise_std=float(cfg.get("noise_std", 0.05)),
        encoder_residual_layers=int(cfg.get("encoder_residual_layers", 2)),
        temporal_layers=int(cfg.get("temporal_layers", 2)),
    )


def build_tokenizer_v0(config: BrainStyleEEGTokenizerConfig | dict[str, Any] | None = None) -> BrainStyleEEGTokenizerV0:
    """Build only the BrainOmni-style EEG tokenizer."""
    if config is None:
        return BrainStyleEEGTokenizerV0()
    if isinstance(config, dict):
        config = tokenizer_config_from_dict(config)
    return BrainStyleEEGTokenizerV0(config)


def build_ds006104_probe_heads(dim: int, label_sizes: dict[str, int]) -> dict[str, ProbeHead]:
    """Build classification probes for ds006104 labels."""
    return {name: ProbeHead(dim, num_classes=size) for name, size in label_sizes.items()}


def build_ds005345_retrieval_head(dim: int, audio_dim: int = 11, proj_dim: int = 256) -> AudioContrastiveHead:
    """Build the EEG/audio contrastive head for speaker stream retrieval."""
    return AudioContrastiveHead(eeg_dim=dim, audio_dim=audio_dim, proj_dim=proj_dim)


def build_voice_attribute_heads(dim: int) -> dict[str, VoiceAttributeHead]:
    """Build optional pitch/intensity/timbre attribute heads."""
    return {
        "f0_bin": VoiceAttributeHead(dim, output_dim=2, task="classification"),
        "intensity_bin": VoiceAttributeHead(dim, output_dim=2, task="classification"),
        "voice_stats": VoiceAttributeHead(dim, output_dim=11, task="regression"),
    }


def build_token_centric_model_v01(config_path: str | Path = "configs/model_v0.yaml") -> TokenCentricEEGVoiceModelV01:
    """Build the v0.1 token-centric model with downstream token evaluation heads."""
    cfg = load_simple_yaml(Path(config_path))
    tokenizer_cfg = tokenizer_config_from_dict(cfg["tokenizer"])
    heads_cfg = cfg.get("heads", {})
    model_cfg = TokenCentricEEGVoiceConfig(
        tokenizer=tokenizer_cfg,
        audio_embedding_dim=int(heads_cfg.get("audio_embedding_dim", 11)),
        projection_dim=int(heads_cfg.get("projection_dim", 256)),
        contrastive_temperature=float(heads_cfg.get("contrastive_temperature", 0.07)),
        phoneme_classes=int(heads_cfg.get("phoneme_classes", 48)),
        pitch_dim=int(heads_cfg.get("pitch_dim", 1)),
        timbre_dim=int(heads_cfg.get("timbre_dim", 8)),
        speaker_dim=int(heads_cfg.get("speaker_dim", 128)),
        style_classes=int(heads_cfg.get("style_classes", 5)),
        dropout=float(heads_cfg.get("dropout", tokenizer_cfg.dropout)),
    )
    return TokenCentricEEGVoiceModelV01(model_cfg)


def build_model_v0_bundle(config_path: str | Path = "configs/model_v0.yaml") -> dict[str, Any]:
    """Build tokenizer and default heads from a YAML config.

    The bundle is intentionally a plain dict so notebooks and demo scripts can
    show each component without a large training framework.
    """
    cfg = load_simple_yaml(Path(config_path))
    tokenizer_cfg = tokenizer_config_from_dict(cfg["tokenizer"])
    tokenizer = build_tokenizer_v0(tokenizer_cfg)
    return {
        "config": cfg,
        "tokenizer": tokenizer,
        "ds005345_retrieval": build_ds005345_retrieval_head(
            dim=tokenizer_cfg.dim,
            audio_dim=int(cfg["heads"]["audio_embedding_dim"]),
            proj_dim=int(cfg["heads"]["projection_dim"]),
        ),
        "voice_attributes": build_voice_attribute_heads(tokenizer_cfg.dim),
    }


def build_model_v01_bundle(config_path: str | Path = "configs/model_v0.yaml") -> dict[str, Any]:
    """Build the token-centric v0.1 wrapper plus its parsed config."""
    cfg = load_simple_yaml(Path(config_path))
    return {
        "config": cfg,
        "model": build_token_centric_model_v01(config_path),
    }

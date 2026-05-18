import torch

from src.eeg_voice_model import EEGVoiceBatch, EEGVoiceTokenV1, EEGVoiceV1Config, VoiceAlignmentTargets
from src.eeg_voice_model.builders import build_eeg_voice_token_v1


def small_config() -> EEGVoiceV1Config:
    return EEGVoiceV1Config(
        sample_rate=64,
        window_sec=1.0,
        dim=32,
        latent_queries=3,
        codebook_dim=16,
        codebook_size=24,
        encoder_channels=12,
        downsample_rates=(2, 2),
        n_heads=4,
        dropout=0.0,
        mask_ratio=0.0,
        noise_std=0.0,
        encoder_residual_layers=1,
        temporal_layers=1,
        q7_group_dropout=0.0,
        retrieval_queue_size=16,
        retrieval_queue_negatives=3,
        audio_embedding_dim=11,
        projection_dim=16,
        content_classes=9,
        phoneme_classes=7,
        pitch_dim=1,
        prosody_dim=2,
        timbre_dim=4,
        style_classes=3,
        dataset_adapter_count=16,
    )


def make_batch(config: EEGVoiceV1Config, targets: VoiceAlignmentTargets | None = None) -> EEGVoiceBatch:
    batch = 3
    channels = 6
    return EEGVoiceBatch(
        eeg=torch.randn(batch, channels, config.window_samples),
        sensor_pos=torch.randn(batch, channels, 3),
        channel_mask=torch.ones(batch, channels, dtype=torch.bool),
        dataset_id=["ds004408", "ds006104", "kara_one"],
        language=["en", "en", "en"],
        domain_group=["english_first_core", "english_first_core", "english_first_core"],
        speaker_id=["ds004408_spk01", "ds006104_spk01", "kara_one_p02"],
        audio_embedding=torch.randn(batch, config.audio_embedding_dim),
        targets=targets,
    )


def test_eeg_voice_token_v1_grouped_forward_without_labels():
    config = small_config()
    model = EEGVoiceTokenV1(config)
    out = model(make_batch(config))

    assert out["tokens"].shape[-1] == 8
    assert tuple(out["group_names"]) == ("base", "content", "prosody", "voice", "residual")
    assert set(out["group_tokens"]) == {"base", "content", "prosody", "voice", "residual"}
    assert out["group_tokens"]["base"].shape[-1] == 2
    assert out["group_tokens"]["residual"].shape[-1] == 1
    assert out["recon_aligned"].shape == out["recon_full"].shape == (3, 6, config.window_samples)
    assert out["retrieval_logits"].shape == (3, 3)
    assert "residual" not in out["head_groups"]["content"]
    assert "residual" not in out["head_groups"]["voice"]
    assert "residual" not in out["head_groups"]["mode"]
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["q7_metrics"]["q7_usage"])
    assert torch.isfinite(out["q7_metrics"]["q7_perplexity"])
    assert torch.isfinite(out["q7_metrics"]["q7_dataset_predictability"])


def test_eeg_voice_token_v1_all_optional_losses_and_queue():
    config = small_config()
    model = EEGVoiceTokenV1(config)
    first = model(make_batch(config))
    steps = first["content_logits"].shape[1]
    targets = VoiceAlignmentTargets(
        content_labels=torch.randint(0, config.content_classes, (3, steps)),
        phoneme_labels=torch.randint(0, config.phoneme_classes, (3, steps)),
        pitch_target=torch.randn(3, config.pitch_dim),
        prosody_target=torch.randn(3, config.prosody_dim),
        timbre_target=torch.randn(3, config.timbre_dim),
        style_labels=torch.tensor([0, 1, 2]),
        mode_labels=torch.tensor([0, 1, 3]),
    )
    out = model(make_batch(config, targets=targets))

    assert out["retrieval_logits"].shape == (3, 6)
    assert out["retrieval_queue_filled"].item() == 6
    for key in [
        "content_loss",
        "phoneme_loss",
        "pitch_loss",
        "prosody_loss",
        "timbre_loss",
        "style_loss",
        "mode_loss",
        "retrieval_loss",
    ]:
        assert key in out["losses"]
        assert torch.isfinite(out["losses"][key])
    assert torch.isfinite(out["losses"]["recon_aligned_loss"])
    assert torch.isfinite(out["losses"]["recon_full_loss"])
    assert torch.isfinite(out["loss"])


def test_builder_loads_model_v1_yaml():
    model = build_eeg_voice_token_v1("configs/model_v1.yaml")
    assert isinstance(model, EEGVoiceTokenV1)
    assert model.config.quantizer_groups["residual"] == (7,)

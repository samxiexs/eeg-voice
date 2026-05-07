import torch

from src.eeg_voice_model.heads import AudioContrastiveHead, ProbeHead, TokenMetrics
from src.eeg_voice_model.tokenizer import BrainStyleEEGTokenizerConfig, BrainStyleEEGTokenizerV0
from src.eeg_voice_model.voice_model import TokenCentricEEGVoiceConfig, TokenCentricEEGVoiceModelV01


def test_tokenizer_and_heads_synthetic_forward():
    cfg = BrainStyleEEGTokenizerConfig(
        sample_rate=250,
        window_sec=2.0,
        dim=64,
        latent_queries=4,
        codebook_dim=16,
        codebook_size=32,
        num_quantizers=2,
        encoder_channels=16,
        n_heads=4,
        dropout=0.0,
    )
    model = BrainStyleEEGTokenizerV0(cfg)
    eeg = torch.randn(2, 8, cfg.window_samples)
    sensor_pos = torch.randn(2, 8, 3)
    mask = torch.ones(2, 8, dtype=torch.bool)
    out = model(eeg, sensor_pos, mask)

    assert out["z"].shape[:2] == (2, 4)
    assert out["tokens"].shape[-1] == 2
    assert int(out["tokens"].max()) < 32
    assert out["x_rec"].shape == eeg.shape
    assert torch.isfinite(out["losses"]["loss"])

    probe = ProbeHead(cfg.dim, num_classes=3)
    probe_out = probe(out["z_q"], torch.tensor([0, 1]))
    assert probe_out["logits"].shape == (2, 3)
    assert torch.isfinite(probe_out["loss"])

    contrast = AudioContrastiveHead(cfg.dim, audio_dim=11, proj_dim=32)
    contrast_out = contrast(out["z_q"], torch.randn(2, 11))
    assert contrast_out["logits"].shape == (2, 2)
    assert torch.isfinite(contrast_out["loss"])

    metrics = TokenMetrics(cfg.codebook_size)(out["tokens"])
    assert torch.isfinite(metrics["codebook_usage"])
    assert torch.isfinite(metrics["token_perplexity"])

    dense_tokens = model.tokenize(eeg, sensor_pos, mask, overlap_ratio=0.5)
    assert dense_tokens.shape[2] >= out["tokens"].shape[2]


def test_token_centric_v01_synthetic_forward():
    tokenizer_cfg = BrainStyleEEGTokenizerConfig(
        sample_rate=250,
        window_sec=2.0,
        dim=64,
        latent_queries=4,
        codebook_dim=16,
        codebook_size=32,
        num_quantizers=2,
        encoder_channels=16,
        n_heads=4,
        dropout=0.0,
    )
    cfg = TokenCentricEEGVoiceConfig(
        tokenizer=tokenizer_cfg,
        audio_embedding_dim=11,
        projection_dim=32,
        phoneme_classes=12,
        pitch_dim=1,
        timbre_dim=4,
        speaker_dim=8,
        style_classes=3,
        dropout=0.0,
    )
    model = TokenCentricEEGVoiceModelV01(cfg)
    eeg = torch.randn(2, 8, tokenizer_cfg.window_samples)
    sensor_pos = torch.randn(2, 8, 3)
    mask = torch.ones(2, 8, dtype=torch.bool)
    first = model(eeg, sensor_pos, mask, audio_embedding=torch.randn(2, 11))
    phoneme_labels = torch.randint(0, cfg.phoneme_classes, (2, first["phoneme_logits"].shape[1]))
    voice_targets = {
        "pitch": torch.randn(2, cfg.pitch_dim),
        "timbre": torch.randn(2, cfg.timbre_dim),
        "speaker": torch.randn(2, cfg.speaker_dim),
        "style": torch.tensor([0, 1]),
    }
    out = model(
        eeg,
        sensor_pos,
        mask,
        audio_embedding=torch.randn(2, 11),
        phoneme_labels=phoneme_labels,
        voice_targets=voice_targets,
    )

    assert out["tokens"].shape[-1] == tokenizer_cfg.num_quantizers
    assert out["phoneme_logits"].shape[:2] == phoneme_labels.shape
    assert out["pitch_pred"].shape == (2, cfg.pitch_dim)
    assert out["timbre_pred"].shape == (2, cfg.timbre_dim)
    assert out["speaker_embedding"].shape == (2, cfg.speaker_dim)
    assert out["style_logits"].shape == (2, cfg.style_classes)
    assert out["retrieval_logits"].shape == (2, 2)
    assert torch.isfinite(out["loss"])

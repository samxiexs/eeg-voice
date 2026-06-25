from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.feis_mel.audio import MelConfig, wav_to_logmel
from src.feis_mel.data import assert_mel_identity_free_keys
from src.feis_mel.diffusion import FEISAcousticDiffusionConfig, FEISDiffusionInference, build_feis_acoustic_diffusion
from src.feis_mel.losses import compute_feis_mel_losses, softmin_dtw_mel_loss
from src.feis_mel.model import FEISEEGToMel, FEISMelConfig


def test_mel_extraction_shape_and_finite():
    cfg = MelConfig(sample_rate=16000, n_mels=24, n_fft=256, hop_length=64, target_frames=20)
    wav = np.sin(2 * np.pi * 220 * np.arange(16000, dtype=np.float32) / 16000.0).astype(np.float32) * 0.05
    mel = wav_to_logmel(wav, cfg)
    assert mel.shape == (20, 24)
    assert np.isfinite(mel).all()


def test_feis_mel_forward_backward_identity_free():
    for channel_moe in (True, False):
        cfg = FEISMelConfig(
            d_model=48,
            cond_dim=8,
            num_labels=16,
            target_steps=12,
            mel_dim=16,
            num_blocks=4,
            num_heads=4,
            num_cross_layers=1,
            ff_mult=2,
            channel_moe=channel_moe,
            moe_num_experts=3,
            moe_top_k=2,
        )
        model = FEISEEGToMel(cfg)
        batch = 4
        eeg = torch.randn(batch, 14, 256)
        target_bank = torch.randn(batch, 3, 12, 16)
        labels = torch.randint(0, 16, (batch,))
        log_rms = torch.randn(batch)
        prototypes = torch.randn(16, 16)
        out = model(eeg)
        assert out["pred_mel"].shape == (batch, 12, 16)
        assert out["content_logits"].shape == (batch, 16)
        assert out["pred_log_rms"].shape == (batch,)
        losses = compute_feis_mel_losses(
            out,
            target_bank,
            labels,
            log_rms,
            prototypes,
            dtw_band=3,
            dtw_top_k=2,
        )
        losses["total"].backward()
        assert torch.isfinite(losses["total"])
        assert "mel_dtw" in losses
        assert "retrieval_acc" in losses


def test_feis_mel_diffusion_smoke():
    cfg = FEISMelConfig(
        d_model=32,
        cond_dim=8,
        num_labels=16,
        target_steps=10,
        mel_dim=12,
        num_blocks=3,
        num_heads=4,
        num_cross_layers=1,
        ff_mult=2,
        channel_moe=True,
        moe_num_experts=2,
        moe_top_k=1,
    )
    base = FEISEEGToMel(cfg)
    diffusion_cfg = FEISAcousticDiffusionConfig(
        target_dim=12,
        target_steps=10,
        cond_dim=32,
        d_model=32,
        num_steps=8,
        sample_steps=2,
        eval_steps=2,
        num_layers=1,
        num_heads=4,
    )
    diffusion = build_feis_acoustic_diffusion(diffusion_cfg)
    eeg = torch.randn(2, 14, 128)
    target = torch.randn(2, 10, 12)
    out = base(eeg)
    losses = diffusion.training_losses(target, out["eeg_tokens"], coarse_latent=out["pred_mel"])
    losses["diffusion_loss"].backward()
    wrapper = FEISDiffusionInference(base, diffusion, target_steps=10, target_dim=12, sample_steps=2)
    sampled = wrapper(eeg)
    assert sampled["pred_mel"].shape == (2, 10, 12)
    assert torch.isfinite(sampled["pred_mel"]).all()


def test_dtw_handles_temporal_shift_better_than_naive_l1():
    base = torch.zeros(1, 16, 4)
    base[:, 4:8, :] = 1.0
    shifted = torch.zeros(1, 1, 16, 4)
    shifted[:, :, 6:10, :] = 1.0
    dtw, _ = softmin_dtw_mel_loss(base, shifted, band=4, top_k=1)
    naive = F.l1_loss(base, shifted[:, 0])
    assert float(dtw) < float(naive)


def test_mel_identity_guard_rejects_external_fields():
    assert_mel_identity_free_keys(("eeg", "target_bank", "label_idx", "sample_key"))
    try:
        assert_mel_identity_free_keys(("eeg", "stage_idx"))
    except ValueError:
        return
    raise AssertionError("identity/external-condition guard accepted stage_idx")


if __name__ == "__main__":
    test_mel_extraction_shape_and_finite()
    test_feis_mel_forward_backward_identity_free()
    test_feis_mel_diffusion_smoke()
    test_dtw_handles_temporal_shift_better_than_naive_l1()
    test_mel_identity_guard_rejects_external_fields()
    print("FEIS mel smoke passed")

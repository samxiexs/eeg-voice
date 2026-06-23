"""Smoke tests for the EEG-only direct EEG->speech path."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.direct_eeg2speech.losses import compute_direct_losses
from src.direct_eeg2speech.model import DirectEEG2Speech, DirectEEG2SpeechConfig
from src.direct_eeg2speech.data import assert_identity_free_keys


def test_direct_forward_backward_identity_free():
    cfg = DirectEEG2SpeechConfig(
        d_model=64,
        cond_dim=16,
        num_labels=16,
        num_stages=2,
        target_steps=20,
        target_dim=32,
        num_blocks=4,
        num_transformer_layers=1,
        num_heads=4,
        ff_mult=2,
        use_channel_moe=True,
        moe_num_experts=3,
        moe_top_k=2,
        use_latent_diffusion=True,
        diffusion_num_steps=8,
        diffusion_sample_steps=3,
        diffusion_layers=1,
        diffusion_time_dim=32,
    )
    model = DirectEEG2Speech(cfg)
    batch = 4
    eeg = torch.randn(batch, 14, 256)
    stage = torch.randint(0, 2, (batch,))
    target = torch.randn(batch, 20, 32)
    out = model(eeg, stage, target_seq=target)
    assert out["pred_latent"].shape == (batch, 20, 32)
    assert out["content_logits"].shape == (batch, 16)
    assert out["pred_log_rms"].shape == (batch,)
    assert out["latent_summary"].shape == (batch, 32)
    assert "moe_load_balance" in out
    assert "moe_cluster_cohesion" in out
    assert "diffusion_loss" in out

    labels = torch.randint(0, 16, (batch,))
    log_rms = torch.randn(batch)
    mean_latent = torch.zeros(20, 32)
    losses = compute_direct_losses(
        out,
        target,
        labels,
        target_log_rms=log_rms,
        mean_latent=mean_latent,
        lambda_std=0.5,
        lambda_diversity=0.5,
        lambda_mean_margin=0.5,
        lambda_moe_load_balance=0.1,
        lambda_moe_sparsity=0.01,
        lambda_moe_route_entropy=0.01,
        lambda_moe_cluster=0.1,
    )
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert "std_ratio" in losses
    assert "mean_distance" in losses
    assert "moe_gate_mean" in losses
    assert "diffusion_loss" in losses

    gen_latent, gen_log_rms = model.generate_full(eeg[:2], stage[:2], sample_steps=2)
    assert gen_latent.shape == (2, 20, 32)
    assert gen_log_rms.shape == (2,)


def test_direct_model_rejects_extra_identity_argument():
    model = DirectEEG2Speech(DirectEEG2SpeechConfig(d_model=32, num_heads=4, target_dim=16))
    eeg = torch.randn(2, 14, 128)
    stage = torch.zeros(2, dtype=torch.long)
    identity = torch.zeros(2, dtype=torch.long)
    try:
        model(eeg, identity, stage)  # type: ignore[misc]
    except TypeError:
        return
    raise AssertionError("DirectEEG2Speech unexpectedly accepted an extra identity input")


def test_identity_free_key_guard():
    assert_identity_free_keys(("eeg", "stage_idx", "target_seq", "label_idx", "sample_key"))
    try:
        assert_identity_free_keys(("eeg", "sub" + "ject_idx"))
    except ValueError:
        return
    raise AssertionError("Identity guard accepted a forbidden key")


if __name__ == "__main__":
    test_direct_forward_backward_identity_free()
    test_direct_model_rejects_extra_identity_argument()
    test_identity_free_key_guard()
    print("direct EEG-only smoke passed")

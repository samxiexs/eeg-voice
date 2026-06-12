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


def test_direct_forward_backward_no_subject_input():
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
    )
    model = DirectEEG2Speech(cfg)
    batch = 4
    eeg = torch.randn(batch, 14, 256)
    stage = torch.randint(0, 2, (batch,))
    out = model(eeg, stage)
    assert out["pred_latent"].shape == (batch, 20, 32)
    assert out["content_logits"].shape == (batch, 16)
    assert out["pred_log_rms"].shape == (batch,)
    assert out["voice_embed"].shape == (batch, 32)

    target = torch.randn(batch, 20, 32)
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
    )
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert "std_ratio" in losses
    assert "mean_distance" in losses


def test_direct_model_rejects_subject_argument():
    model = DirectEEG2Speech(DirectEEG2SpeechConfig(d_model=32, num_heads=4, target_dim=16))
    eeg = torch.randn(2, 14, 128)
    stage = torch.zeros(2, dtype=torch.long)
    subject = torch.zeros(2, dtype=torch.long)
    try:
        model(eeg, subject, stage)  # type: ignore[misc]
    except TypeError:
        return
    raise AssertionError("DirectEEG2Speech unexpectedly accepted subject input")


if __name__ == "__main__":
    test_direct_forward_backward_no_subject_input()
    test_direct_model_rejects_subject_argument()
    print("direct EEG-only smoke passed")

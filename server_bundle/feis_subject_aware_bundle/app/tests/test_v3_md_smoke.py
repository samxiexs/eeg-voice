"""Smoke test: two datasets with different channel counts share one trunk."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.v3.losses import compute_v3_losses
from src.v3.model import DatasetHead, EEG2SpeechMD, EEG2SpeechMDConfig


def test_shared_trunk_two_datasets():
    cfg = EEG2SpeechMDConfig(
        datasets=[DatasetHead("feis", 14, 16), DatasetHead("karaone", 62, 11)],
        d_model=64, cond_dim=32, num_subjects=35, target_steps=150, target_dim=128, num_blocks=4,
    )
    model = EEG2SpeechMD(cfg)

    # FEIS-shaped batch (14ch).
    out_f = model(torch.randn(6, 14, 1280), torch.randint(0, 35, (6,)), "feis")
    assert out_f["speech_sequence"].shape == (6, 150, 128)
    assert out_f["label_logits"].shape == (6, 16)

    # KaraOne-shaped batch (62ch) through the SAME trunk.
    out_k = model(torch.randn(4, 62, 1280), torch.randint(0, 35, (4,)), "karaone")
    assert out_k["speech_sequence"].shape == (4, 150, 128)
    assert out_k["label_logits"].shape == (4, 11)

    # Backward on a joint pseudo-step.
    lf = compute_v3_losses(out_f, torch.randn(6, 150, 128), torch.randn(6, 128), torch.randint(0, 16, (6,)),
                           target_mask=torch.ones(6, 150))
    lk = compute_v3_losses(out_k, torch.randn(4, 150, 128), torch.randn(4, 128), torch.randint(0, 11, (4,)),
                           target_mask=torch.ones(4, 150))
    (lf["total"] + lk["total"]).backward()
    # Shared trunk must receive gradients from both datasets.
    trunk_grad = sum(p.grad.abs().sum() for p in model.trunk.parameters() if p.grad is not None)
    assert trunk_grad > 0


if __name__ == "__main__":
    test_shared_trunk_two_datasets()
    print("v3 MD smoke test passed")

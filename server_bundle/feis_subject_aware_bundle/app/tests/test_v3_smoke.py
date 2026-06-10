"""Smoke tests for the v3 pipeline: forward/backward + dataset wiring.

Run:  python -m pytest app/tests/test_v3_smoke.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.v3.losses import compute_v3_losses
from src.v3.model import EEG2SpeechV3, EEG2SpeechV3Config


def test_forward_backward_no_collapse_signal():
    cfg = EEG2SpeechV3Config(num_subjects=4, target_steps=75, target_dim=128, num_labels=16, d_model=64, num_blocks=4)
    model = EEG2SpeechV3(cfg)
    b = 8
    eeg = torch.randn(b, 14, 1280)
    subj = torch.randint(0, 4, (b,))
    out = model(eeg, subj)
    assert out["speech_sequence"].shape == (b, 75, 128)
    assert out["label_logits"].shape == (b, 16)
    assert out["contrastive_embedding"].shape == (b, 128)

    target_seq = torch.randn(b, 75, 128)
    target_sum = torch.randn(b, 128)
    label_ids = torch.randint(0, 16, (b,))
    losses = compute_v3_losses(out, target_seq, target_sum, label_ids)
    losses["total"].backward()
    grad_norm = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
    assert torch.isfinite(losses["total"])
    assert grad_norm > 0


def test_unknown_subject_index_is_safe():
    cfg = EEG2SpeechV3Config(num_subjects=3, d_model=32, num_blocks=3)
    model = EEG2SpeechV3(cfg)
    eeg = torch.randn(2, 14, 1280)
    # index == num_subjects is the reserved "unknown" row; out-of-range clamps.
    out = model(eeg, torch.tensor([3, 99]))
    assert out["speech_sequence"].shape[0] == 2


if __name__ == "__main__":
    test_forward_backward_no_collapse_signal()
    test_unknown_subject_index_is_safe()
    print("v3 smoke tests passed")

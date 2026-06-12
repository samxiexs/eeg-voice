"""Smoke test for the factored FEIS pipeline (model + losses, no data needed)."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.feis_factored.model import FactoredConfig, FactoredEEG2Speech
from src.feis_factored.losses import compute_factored_losses, supervised_contrastive


def test_forward_backward():
    cfg = FactoredConfig(num_subjects=4, num_labels=16, num_stages=2, target_steps=75,
                         target_dim=128, d_model=64, num_blocks=4)
    m = FactoredEEG2Speech(cfg)
    b = 8
    eeg = torch.randn(b, 14, 1280)
    subj = torch.randint(0, 4, (b,)); stage = torch.randint(0, 2, (b,))
    out = m(eeg, subj, stage)
    assert out["pred_latent"].shape == (b, 75, 128)
    assert out["content_logits"].shape == (b, 16)
    assert out["pred_log_rms"].shape == (b,)
    label = torch.randint(0, 16, (b,))
    tgt = torch.randn(b, 75, 128); proto = torch.randn(b, 128)
    log_rms = torch.randn(b)
    losses = compute_factored_losses(out, tgt, label, proto, target_log_rms=log_rms,
                                     subject_idx=subj, lambda_std=0.5)
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert "log_rms_loss" in losses and "std_ratio" in losses
    # generate_full returns both latent and predicted log-rms
    lat, lr = m.generate_full(eeg, subj, stage)
    assert lat.shape == (b, 75, 128) and lr.shape == (b,)


def test_log_rms_loss_drops_with_correct_energy():
    # the log_rms term is lower when pred_log_rms matches the target
    out = {"pred_latent": torch.zeros(4, 5, 8), "pred_log_rms": torch.zeros(4),
           "content_embed": torch.zeros(4, 8), "content_logits": torch.zeros(4, 16)}
    tgt_seq = torch.zeros(4, 5, 8); label = torch.zeros(4, dtype=torch.long)
    proto = torch.zeros(4, 8)
    good = compute_factored_losses(out, tgt_seq, label, proto,
                                   target_log_rms=torch.zeros(4))["log_rms_loss"]
    bad = compute_factored_losses(out, tgt_seq, label, proto,
                                  target_log_rms=torch.full((4,), 3.0))["log_rms_loss"]
    assert good < bad


def test_supcon_pulls_same_label():
    # embeds form two clusters; supcon is LOW when labels match the clusters,
    # HIGH when labels are assigned across clusters (deterministic, no RNG).
    emb = torch.tensor([[1.0, 0], [1.0, 0], [0, 1.0], [0, 1.0]])
    good = supervised_contrastive(emb, torch.tensor([0, 0, 1, 1]), 0.1)  # labels match clusters
    bad = supervised_contrastive(emb, torch.tensor([0, 1, 0, 1]), 0.1)   # labels split clusters
    assert good < bad


if __name__ == "__main__":
    test_forward_backward(); test_supcon_pulls_same_label()
    test_log_rms_loss_drops_with_correct_energy()
    print("factored smoke passed")

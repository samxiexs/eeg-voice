from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.karaone_v9.losses import compute_v9_alignment_losses, compute_v9_pretrain_losses, compute_v9_transport_losses
from src.karaone_v9.model import KaraOneV9Config, KaraOneV9NeuralSemanticTransport
from src.karaone_v9.eval import same_label_cross_subject_gain


def _batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    semantic = torch.randn(batch_size, 20, 24)
    return {
        "eeg": torch.randn(batch_size, 62, 256),
        "eeg_valid_len": torch.full((batch_size,), 220, dtype=torch.long),
        "stage_idx": torch.randint(0, 2, (batch_size,)),
        "subject_idx": torch.tensor([0, 1, 2, 3])[:batch_size],
        "label_idx": torch.tensor([0, 1, 2, 3])[:batch_size],
        "semantic_seq": semantic,
        "semantic_summary": semantic.mean(dim=1),
        "semantic_token_targets": torch.randint(0, 8, (batch_size, 20)),
        "semantic_token_mask": torch.ones(batch_size, 20),
        "codec_seq": torch.randn(batch_size, 15, 12),
        "prosody_active": torch.randint(0, 2, (batch_size, 12)).float(),
        "prosody_energy": torch.randn(batch_size, 12),
        "prosody_duration": torch.rand(batch_size),
        "prosody_onset": torch.rand(batch_size),
    }


def _model() -> KaraOneV9NeuralSemanticTransport:
    cfg = KaraOneV9Config(
        eeg_len=256,
        patch_size=16,
        patch_stride=8,
        d_model=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        channel_dropout=0.0,
        cond_dim=8,
        num_stages=2,
        num_subjects=4,
        num_labels=5,
        semantic_dim=24,
        semantic_token_vocab=8,
        codec_dim=12,
        transport_layers=1,
        transport_heads=4,
    )
    return KaraOneV9NeuralSemanticTransport(cfg)


def test_v9_forward_alignment_backward():
    model = _model()
    batch = _batch()
    out = model(batch["eeg"], batch["stage_idx"], batch["eeg_valid_len"], lambda_subject_adv=0.1)
    assert out["pred_semantic_summary"].shape == (4, 24)
    assert out["pred_semantic_seq"].shape[-1] == 24
    assert out["semantic_token_logits"].shape[-1] == 8
    assert out["prompt_ctc_logits"].shape[-1] == 6
    assert out["subject_logits"].shape == (4, 4)
    losses = compute_v9_alignment_losses(out, batch)
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert "seq_ot" in losses
    assert "prosody" in losses


def test_v9_pretrain_and_transport_backward():
    model = _model()
    batch = _batch()
    pre = model(batch["eeg"], batch["stage_idx"], batch["eeg_valid_len"], mask_ratio=0.5)
    pre_losses = compute_v9_pretrain_losses(pre)
    pre_losses["total"].backward()
    assert torch.isfinite(pre_losses["total"])

    model.zero_grad(set_to_none=True)
    out = model(batch["eeg"], batch["stage_idx"], batch["eeg_valid_len"], codec_seq=batch["codec_seq"])
    transport_losses = compute_v9_transport_losses(out, batch)
    transport_losses["total"].backward()
    assert torch.isfinite(transport_losses["total"])
    sampled = model.generate_codec(batch["eeg"][:2], batch["stage_idx"][:2], batch["eeg_valid_len"][:2], steps=2, codec_steps=15)
    assert sampled["pred_codec_seq"].shape == (2, 15, 12)
    assert torch.isfinite(sampled["pred_codec_seq"]).all()


def test_v9_forward_has_no_subject_input():
    params = inspect.signature(KaraOneV9NeuralSemanticTransport.forward).parameters
    assert "subject_idx" not in params
    assert "speaker_id" not in params


def test_v9_same_label_cross_subject_gain_returns_scalar():
    pred = torch.randn(3, 5).numpy()
    mean = torch.zeros(3, 5).numpy()
    bank = torch.randn(4, 5).numpy()
    labels = torch.tensor([0, 1, 0]).numpy().astype(str)
    label_bank = torch.tensor([0, 0, 1, 1]).numpy().astype(str)
    subjects = torch.tensor([0, 1, 2]).numpy().astype(str)
    subject_bank = torch.tensor([3, 4, 3, 4]).numpy().astype(str)
    value = same_label_cross_subject_gain(pred, mean, bank, labels, label_bank, subjects, subject_bank)
    assert isinstance(value, float)


if __name__ == "__main__":
    test_v9_forward_alignment_backward()
    test_v9_pretrain_and_transport_backward()
    test_v9_forward_has_no_subject_input()
    test_v9_same_label_cross_subject_gain_returns_scalar()
    print("KaraOne v9 smoke passed")

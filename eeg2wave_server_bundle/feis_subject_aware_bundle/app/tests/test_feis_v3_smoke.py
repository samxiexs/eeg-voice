from __future__ import annotations

import sys
from pathlib import Path

import torch

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.feis_v3.losses import compute_feis_v3_losses
from src.feis_v3.model import FEISV3ModelConfig, FEISV3TokenGenerator


def _batch(batch=3, semantic_steps=8, codec_steps=8, semantic_vocab=11, codec_vocab=13):
    return {
        "eeg": torch.randn(batch, 14, 128),
        "stage_idx": torch.randint(0, 3, (batch,)),
        "eeg_valid_len": torch.full((batch,), 128),
        "channel_cluster_id": torch.randint(0, 2, (batch,)),
        "semantic_token_ids": torch.randint(0, semantic_vocab, (batch, semantic_steps)),
        "semantic_token_mask": torch.ones(batch, semantic_steps),
        "codec_token_ids": torch.randint(0, codec_vocab, (batch, codec_steps)),
        "codec_token_mask": torch.ones(batch, codec_steps),
        "prosody_active": torch.rand(batch, semantic_steps).round(),
        "prosody_duration": torch.rand(batch, semantic_steps),
        "prosody_energy": torch.randn(batch, semantic_steps),
        "prosody_onset": torch.zeros(batch, semantic_steps),
        "audio_variant_cluster_id": torch.randint(0, 5, (batch,)),
        "label_idx": torch.randint(0, 4, (batch,)),
        "subject_idx": torch.arange(batch),
        "subject_id": [f"{idx + 1:02d}" for idx in range(batch)],
        "label": ["a", "a", "b"][:batch],
    }


def test_feis_v3_forward_backward_all_aligners():
    cfg = FEISV3ModelConfig(
        d_model=32,
        num_heads=4,
        num_layers=1,
        semantic_vocab=11,
        codec_vocab=13,
        semantic_steps=8,
        codec_steps=8,
        num_labels=4,
        channel_clusters=2,
        audio_variant_clusters=5,
        channel_experts=2,
        channel_top_k=4,
        num_subjects_for_adversary=3,
    )
    for aligner in ["mlp", "clip", "ctc", "ot", "perceiver", "hybrid"]:
        model = FEISV3TokenGenerator(cfg)
        batch = _batch()
        out = model(
            batch["eeg"],
            stage_idx=batch["stage_idx"],
            eeg_valid_len=batch["eeg_valid_len"],
            channel_cluster_id=batch["channel_cluster_id"],
        )
        assert out["semantic_logits"].shape == (3, 8, 11)
        assert out["codec_logits"].shape == (3, 8, 13)
        losses = compute_feis_v3_losses(out, batch, aligner=aligner)
        assert torch.isfinite(losses["total"])
        losses["total"].backward()


if __name__ == "__main__":
    test_feis_v3_forward_backward_all_aligners()
    print("FEIS v3 smoke passed")

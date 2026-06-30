from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.karaone_v91.data import KaraOneV91ClusterBalancedBatchSampler
from src.karaone_v91.eval import compute_v91_metrics, write_channel_reports
from src.karaone_v91.losses import compute_v91_alignment_losses, compute_v91_pretrain_losses, compute_v91_transport_losses
from src.karaone_v91.model import KaraOneV91ClusteredChannelMoEFlow, KaraOneV91Config


def _batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    semantic = torch.randn(batch_size, 20, 24)
    return {
        "eeg": torch.randn(batch_size, 62, 256),
        "eeg_valid_len": torch.full((batch_size,), 220, dtype=torch.long),
        "stage_idx": torch.randint(0, 2, (batch_size,)),
        "subject_idx": torch.tensor([0, 1, 2, 3])[:batch_size],
        "label_idx": torch.tensor([0, 1, 2, 3])[:batch_size],
        "eeg_cluster_id": torch.tensor([0, 0, 1, 1])[:batch_size],
        "speech_cluster_id": torch.tensor([0, 0, 1, 1])[:batch_size],
        "cross_modal_cluster_id": torch.tensor([0, 1, 2, 3])[:batch_size],
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


def _model() -> KaraOneV91ClusteredChannelMoEFlow:
    cfg = KaraOneV91Config(
        eeg_len=256,
        patch_size=16,
        patch_stride=8,
        d_model=36,
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
        channel_experts=4,
        channel_top_k=8,
        channel_desc_dim=16,
        channel_embedding_dim=8,
        shared_dim=36,
        domain_dim=12,
        transport_layers=1,
        transport_heads=4,
        heun_steps=2,
    )
    return KaraOneV91ClusteredChannelMoEFlow(cfg)


def test_v91_forward_alignment_backward():
    model = _model()
    batch = _batch()
    out = model(batch["eeg"], batch["stage_idx"], batch["eeg_valid_len"], lambda_subject_adv=0.1)
    assert out["pred_semantic_summary"].shape == (4, 24)
    assert out["channel_gate"].shape == (4, 62)
    assert out["channel_assign"].shape == (4, 62, 4)
    assert out["channel_load"].shape == (4,)
    assert torch.isfinite(out["channel_gate_entropy"]).all()
    losses = compute_v91_alignment_losses(out, batch)
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert "cluster_nce" in losses
    assert "channel_balance" in losses


def test_v91_pretrain_and_transport_backward():
    model = _model()
    batch = _batch()
    pre = model(batch["eeg"], batch["stage_idx"], batch["eeg_valid_len"], mask_ratio=0.5)
    pre_losses = compute_v91_pretrain_losses(pre, batch)
    pre_losses["total"].backward()
    assert torch.isfinite(pre_losses["total"])

    model.zero_grad(set_to_none=True)
    out = model(batch["eeg"], batch["stage_idx"], batch["eeg_valid_len"], codec_seq=batch["codec_seq"])
    transport_losses = compute_v91_transport_losses(out, batch)
    transport_losses["total"].backward()
    assert torch.isfinite(transport_losses["total"])
    sampled = model.generate_codec(batch["eeg"][:2], batch["stage_idx"][:2], batch["eeg_valid_len"][:2], steps=2, codec_steps=15)
    assert sampled["pred_codec_seq"].shape == (2, 15, 12)
    assert torch.isfinite(sampled["pred_codec_seq"]).all()


def test_v91_forward_has_no_subject_input():
    params = inspect.signature(KaraOneV91ClusteredChannelMoEFlow.forward).parameters
    assert "subject_idx" not in params
    assert "speaker_id" not in params


def test_v91_cluster_sampler_and_channel_reports():
    entries = [
        SimpleNamespace(subject=f"S{idx % 3}", stage="overt_like", trial_index=idx)
        for idx in range(18)
    ]

    class Bank:
        available = True

        def lookup(self, subject, stage, trial_index):
            return {
                "eeg_cluster_id": int(trial_index) % 4,
                "speech_cluster_id": int(trial_index) % 5,
                "cross_modal_cluster_id": int(trial_index) % 6,
                "cluster_fit_split": True,
            }

    class Dataset:
        def __init__(self, rows):
            self.cluster_bank = Bank()
            self.entries = rows

        def __len__(self):
            return len(self.entries)

    batches = list(KaraOneV91ClusterBalancedBatchSampler(Dataset(entries), batch_size=5, seed=3))
    assert batches
    assert all(len(batch) <= 5 for batch in batches)

    gate = np.random.default_rng(7).random((6, 62), dtype=np.float32)
    outputs = {
        "channel_gate": gate,
        "labels": ["a", "b", "a", "b", "c", "c"],
        "stages": ["overt_like"] * 6,
        "speech_cluster_id": np.asarray([0, 1, 0, 1, 2, 2]),
    }
    with tempfile.TemporaryDirectory() as tmp:
        paths = write_channel_reports(tmp, outputs, [f"Ch{idx + 1:03d}" for idx in range(62)])
        for path in paths.values():
            assert Path(path).exists()


def test_v91_metrics_numpy_scalar_compatibility():
    rng = np.random.default_rng(11)
    outputs = {
        "pred": rng.normal(size=(5, 8)).astype(np.float32),
        "zero": np.zeros((5, 8), dtype=np.float32),
        "target": rng.normal(size=(5, 8)).astype(np.float32),
        "prompt_logits": rng.normal(size=(5, 3)).astype(np.float32),
        "subject_logits": rng.normal(size=(5, 4)).astype(np.float32),
        "label_idx": np.asarray([0, 1, 2, 0, 1], dtype=np.int64),
        "subject_idx": np.asarray([0, 1, 2, 3, 0], dtype=np.int64),
        "labels": ["a", "b", "c", "a", "b"],
        "subjects": ["s1", "s2", "s3", "s4", "s1"],
        "eeg_cluster_id": np.asarray([0, 1, 0, 1, 2], dtype=np.int64),
        "speech_cluster_id": np.asarray([0, 1, 2, 0, 1], dtype=np.int64),
        "channel_gate": rng.random(size=(5, 62)).astype(np.float32),
    }
    metrics = compute_v91_metrics(outputs, train_bank={"target": outputs["target"], "labels": outputs["labels"], "subjects": outputs["subjects"], "speech_cluster_id": outputs["speech_cluster_id"]})
    assert isinstance(metrics["same_label_cross_subject_gain"], float)
    assert "v91_research_gate_pass" in metrics


if __name__ == "__main__":
    test_v91_forward_alignment_backward()
    test_v91_pretrain_and_transport_backward()
    test_v91_forward_has_no_subject_input()
    test_v91_cluster_sampler_and_channel_reports()
    test_v91_metrics_numpy_scalar_compatibility()
    print("KaraOne v9.1 smoke passed")

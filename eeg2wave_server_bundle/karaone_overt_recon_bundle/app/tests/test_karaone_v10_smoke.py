from __future__ import annotations

import csv
import inspect
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

BUNDLE_DIR = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parents[3]
for path in (BUNDLE_DIR, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate_all_waveform_comparisons import generate_for_dir
from scripts.plot_karaone_v10_training import plot_run
from src.karaone_v10.eval import compute_v10_metrics, v10_selection_score
from src.karaone_v10.losses import compute_v10_alignment_losses, compute_v10_pretrain_losses, compute_v10_transport_losses
from src.karaone_v10.model import KaraOneV10ClusteredChannelMoEFlow, KaraOneV10Config
from src.utils import save_wav, write_json


def _batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    semantic = torch.randn(batch_size, 20, 24)
    return {
        "eeg": torch.randn(batch_size, 62, 256),
        "eeg_valid_len": torch.full((batch_size,), 220, dtype=torch.long),
        "stage_idx": torch.randint(0, 2, (batch_size,)),
        "subject_idx": torch.tensor([0, 1, 2, 3])[:batch_size],
        "label_idx": torch.tensor([0, 1, 0, 1])[:batch_size],
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


def _model() -> KaraOneV10ClusteredChannelMoEFlow:
    cfg = KaraOneV10Config(
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
    return KaraOneV10ClusteredChannelMoEFlow(cfg)


def test_v10_forward_alignment_backward():
    model = _model()
    batch = _batch()
    out = model(batch["eeg"], batch["stage_idx"], batch["eeg_valid_len"], lambda_subject_adv=0.1)
    assert out["pred_semantic_summary"].shape == (4, 24)
    assert out["channel_gate"].shape == (4, 62)
    losses = compute_v10_alignment_losses(out, batch)
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert "zero_prior_margin" in losses
    assert "cross_subject_semantic_nce" in losses
    assert "prompt_balanced_ce" in losses


def test_v10_pretrain_transport_and_no_subject_forward_arg():
    model = _model()
    batch = _batch()
    pre = model(batch["eeg"], batch["stage_idx"], batch["eeg_valid_len"], mask_ratio=0.5)
    pre_losses = compute_v10_pretrain_losses(pre, batch)
    pre_losses["total"].backward()
    assert torch.isfinite(pre_losses["total"])

    model.zero_grad(set_to_none=True)
    out = model(batch["eeg"], batch["stage_idx"], batch["eeg_valid_len"], codec_seq=batch["codec_seq"])
    transport_losses = compute_v10_transport_losses(out, batch)
    transport_losses["total"].backward()
    assert torch.isfinite(transport_losses["total"])

    params = inspect.signature(KaraOneV10ClusteredChannelMoEFlow.forward).parameters
    assert "subject_idx" not in params
    assert "speaker_id" not in params


def test_v10_metrics_and_selection_score():
    rng = np.random.default_rng(13)
    outputs = {
        "pred": rng.normal(size=(6, 8)).astype(np.float32),
        "zero": np.zeros((6, 8), dtype=np.float32),
        "target": rng.normal(size=(6, 8)).astype(np.float32),
        "prompt_logits": rng.normal(size=(6, 3)).astype(np.float32),
        "subject_logits": rng.normal(size=(6, 4)).astype(np.float32),
        "label_idx": np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64),
        "subject_idx": np.asarray([0, 1, 2, 3, 0, 1], dtype=np.int64),
        "labels": ["a", "b", "c", "a", "b", "c"],
        "subjects": ["s1", "s2", "s3", "s4", "s1", "s2"],
        "eeg_cluster_id": np.asarray([0, 1, 0, 1, 2, 2], dtype=np.int64),
        "speech_cluster_id": np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64),
        "channel_gate": rng.random(size=(6, 62)).astype(np.float32),
    }
    metrics = compute_v10_metrics(
        outputs,
        train_bank={
            "target": outputs["target"],
            "labels": outputs["labels"],
            "subjects": outputs["subjects"],
            "speech_cluster_id": outputs["speech_cluster_id"],
        },
        prefix="subject_val",
    )
    assert "subject_val_v10_research_gate_pass" in metrics
    assert isinstance(v10_selection_score(metrics, prefix="subject_val"), float)


def test_v10_plot_and_wav_compare_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        history = [
            {
                "epoch": 1,
                "train_total": 2.0,
                "train_zero_prior_margin": 0.1,
                "train_mean_prior_margin": 0.2,
                "subject_val_semantic_over_zero_gain": 0.01,
                "subject_test_semantic_over_zero_gain": 0.0,
                "subject_val_semantic_top3_gain_over_mean": 0.02,
                "subject_test_semantic_top3_gain_over_mean": -0.01,
                "subject_val_same_label_cross_subject_gain": 0.0,
                "subject_test_same_label_cross_subject_gain": -0.02,
                "subject_val_prompt_acc": 0.12,
                "subject_test_prompt_acc": 0.10,
                "subject_val_pred_std_ratio_median": 1.0,
                "subject_test_pred_std_ratio_median": 1.1,
                "subject_val_pred_pairwise_corr_median": 0.5,
                "subject_test_pred_pairwise_corr_median": 0.6,
                "subject_val_channel_gate_entropy_mean": 0.4,
                "subject_test_channel_gate_entropy_mean": 0.4,
                "selection_score": 0.1,
            }
        ]
        write_json(run_dir / "metrics" / "history.json", {"history": history, "best_epoch": 1, "best_score": 0.1})
        channel_dir = run_dir / "channel_reports" / "best_subject_val"
        channel_dir.mkdir(parents=True)
        with (channel_dir / "channel_gate_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["channel", "rank", "mean_gate", "std_gate", "active_rate"])
            writer.writeheader()
            for idx in range(4):
                writer.writerow({"channel": f"Ch{idx + 1:03d}", "rank": idx + 1, "mean_gate": 0.1 * (4 - idx), "std_gate": 0.0, "active_rate": 1.0})
        plot_summary = plot_run(run_dir)
        assert Path(plot_summary["figures"]["training_curves"]).exists()
        assert Path(plot_summary["figures"]["gate_metrics"]).exists()
        assert Path(plot_summary["figures"]["channel_gate_top_channels"]).exists()

        wav_dir = run_dir / "wavs"
        wav_dir.mkdir()
        sr = 16000
        t = np.linspace(0.0, 1.0, sr, endpoint=False)
        original = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        recon = (0.05 * np.sin(2 * np.pi * 230 * t)).astype(np.float32)
        save_wav(wav_dir / "reference_sample.wav", original, sr)
        save_wav(wav_dir / "recon_sample.wav", recon, sr)
        with (wav_dir / "listening_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample_key", "subject", "label", "stage", "trial_index", "wav_type", "file", "rms"],
            )
            writer.writeheader()
            writer.writerow({"sample_key": "sample", "subject": "S1", "label": "a", "stage": "thinking", "trial_index": 1, "wav_type": "original", "file": "reference_sample.wav", "rms": 0.05})
            writer.writerow({"sample_key": "sample", "subject": "S1", "label": "a", "stage": "thinking", "trial_index": 1, "wav_type": "pred_env_scaled", "file": "recon_sample.wav", "rms": 0.05})
        count = generate_for_dir(wav_dir, "original", "pred_env_scaled")
        assert count == 1
        assert (wav_dir / "waveform_compare" / "waveform_compare_manifest.csv").exists()


if __name__ == "__main__":
    test_v10_forward_alignment_backward()
    test_v10_pretrain_transport_and_no_subject_forward_arg()
    test_v10_metrics_and_selection_score()
    test_v10_plot_and_wav_compare_smoke()
    print("KaraOne v10 smoke passed")


from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.karaone_v12.losses import compute_v12_alignment_losses, compute_v12_codec_losses, compute_v12_pretrain_losses, compute_v12_time_losses
from src.karaone_v12.model import KaraOneV12Config, KaraOneV12TokenGenerator
from src.karaone_v12.time_anchor import best_lag_corr, rms_envelope


def main() -> None:
    test_lagaware_corr_recovers_shift()
    test_v12_forward_backward()
    print("karaone_v12_smoke_pass")


def test_lagaware_corr_recovers_shift() -> None:
    sr = 100
    t = np.arange(0, 2.0, 1.0 / sr)
    base = np.exp(-((t - 0.8) ** 2) / 0.01).astype(np.float32)
    shifted = np.zeros_like(base)
    shifted[35:] = base[:-35]
    zero = float(np.corrcoef(base, shifted)[0, 1])
    best, lag, _ = best_lag_corr(base, shifted, sample_rate=sr, max_lag_sec=0.8, min_overlap_sec=0.5)
    assert zero < 0.5
    assert best > 0.95
    assert abs(abs(lag) - 0.35) < 0.03
    env = rms_envelope(base, sample_rate=sr, hop_sec=0.01, win_sec=0.03)
    assert env.ndim == 1 and env.size > 0


def test_v12_forward_backward() -> None:
    torch.manual_seed(7)
    cfg = KaraOneV12Config(
        n_channels_eeg=8,
        eeg_len=128,
        patch_size=16,
        patch_stride=8,
        d_model=32,
        num_layers=1,
        num_heads=4,
        num_stages=2,
        num_subjects=4,
        num_labels=5,
        semantic_dim=24,
        semantic_token_vocab=11,
        semantic_token_steps=12,
        codec_dim=10,
        codec_token_vocab=13,
        codec_token_steps=9,
        channel_experts=4,
        channel_top_k=4,
        channel_clusters=3,
        perceiver_queries=12,
        active_mask_steps=20,
        duration_sec=2.0,
        max_lag_sec=1.0,
    )
    model = KaraOneV12TokenGenerator(cfg, codec_codebook=torch.randn(cfg.codec_token_vocab, cfg.codec_dim))
    batch = synthetic_batch(cfg)
    out = model(batch["eeg"], batch["stage_idx"], batch["eeg_valid_len"], channel_cluster_id=batch["channel_cluster_id"], mask_ratio=0.2)
    assert out["pred_lag_sec"].shape == (batch["eeg"].shape[0],)
    assert out["pred_active_mask_logits"].shape == (batch["eeg"].shape[0], cfg.active_mask_steps)
    pretrain = compute_v12_pretrain_losses(out, batch)
    assert torch.isfinite(pretrain["total"])
    pretrain["total"].backward(retain_graph=True)
    for aligner in ["linear", "mlp", "clip", "ctc", "ot", "perceiver", "hybrid"]:
        model.cfg.aligner = aligner
        model.zero_grad(set_to_none=True)
        out = model(batch["eeg"], batch["stage_idx"], batch["eeg_valid_len"], channel_cluster_id=batch["channel_cluster_id"])
        losses = compute_v12_alignment_losses(out, batch, aligner=aligner)
        assert torch.isfinite(losses["total"]), aligner
        losses["total"].backward()
    model.zero_grad(set_to_none=True)
    out = model(batch["eeg"], batch["stage_idx"], batch["eeg_valid_len"], channel_cluster_id=batch["channel_cluster_id"])
    time_losses = compute_v12_time_losses(out, batch)
    assert torch.isfinite(time_losses["total"])
    time_losses["total"].backward()
    model.zero_grad(set_to_none=True)
    out = model(batch["eeg"], batch["stage_idx"], batch["eeg_valid_len"], channel_cluster_id=batch["channel_cluster_id"])
    codec = compute_v12_codec_losses(out, batch)
    assert torch.isfinite(codec["total"])
    codec["total"].backward()
    with torch.no_grad():
        generated = model.generate_codec_tokens(batch["eeg"], batch["stage_idx"], batch["eeg_valid_len"], channel_cluster_id=batch["channel_cluster_id"])
    assert generated["pred_codec_seq"].shape == (batch["eeg"].shape[0], cfg.codec_token_steps, cfg.codec_dim)
    assert generated["pred_codec_seq_placed"].shape == generated["pred_codec_seq"].shape


def synthetic_batch(cfg: KaraOneV12Config) -> dict[str, torch.Tensor]:
    b = 4
    semantic_tokens = torch.randint(0, cfg.semantic_token_vocab, (b, cfg.semantic_token_steps))
    codec_tokens = torch.randint(0, cfg.codec_token_vocab, (b, cfg.codec_token_steps))
    active = torch.zeros(b, cfg.active_mask_steps)
    active[:, 6:14] = 1.0
    return {
        "eeg": torch.randn(b, cfg.n_channels_eeg, cfg.eeg_len),
        "eeg_valid_len": torch.full((b,), cfg.eeg_len, dtype=torch.long),
        "stage_idx": torch.randint(0, cfg.num_stages, (b,)),
        "subject_idx": torch.arange(b) % cfg.num_subjects,
        "label_idx": torch.arange(b) % cfg.num_labels,
        "semantic_seq": torch.randn(b, cfg.semantic_token_steps, cfg.semantic_dim),
        "semantic_summary": torch.randn(b, cfg.semantic_dim),
        "audio_semantic_tokens": semantic_tokens,
        "audio_semantic_token_mask": torch.ones(b, cfg.semantic_token_steps),
        "codec_seq": torch.randn(b, cfg.codec_token_steps, cfg.codec_dim),
        "codec_token_targets": codec_tokens,
        "codec_token_mask": torch.ones(b, cfg.codec_token_steps),
        "prosody_active": torch.rand(b, cfg.semantic_token_steps),
        "prosody_energy": torch.rand(b, cfg.semantic_token_steps),
        "prosody_duration": torch.rand(b),
        "prosody_onset": torch.rand(b),
        "speech_cluster_id": torch.arange(b) % 2,
        "eeg_cluster_id": torch.arange(b) % 2,
        "channel_cluster_id": torch.randint(0, cfg.channel_clusters, (b, cfg.n_channels_eeg)),
        "time_onset_sec": torch.tensor([0.4, 0.5, 0.6, 0.7]),
        "time_duration_sec": torch.tensor([0.6, 0.5, 0.7, 0.6]),
        "time_center_sec": torch.tensor([0.7, 0.75, 0.95, 1.0]),
        "time_lag_sec": torch.tensor([0.1, -0.2, 0.0, 0.3]),
        "time_confidence": torch.ones(b),
        "time_active_mask": active,
        "time_envelope": active.clone(),
        "time_fit_split": torch.ones(b, dtype=torch.bool),
    }


if __name__ == "__main__":
    main()

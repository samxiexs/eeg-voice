from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.karaone_v11.losses import compute_v11_alignment_losses, compute_v11_codec_losses, compute_v11_pretrain_losses
from src.karaone_v11.model import KaraOneV11Config, KaraOneV11TokenGenerator


def main() -> None:
    torch.manual_seed(3)
    cfg = KaraOneV11Config(
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
    )
    codebook = torch.randn(cfg.codec_token_vocab, cfg.codec_dim)
    model = KaraOneV11TokenGenerator(cfg, codec_codebook=codebook)
    batch = synthetic_batch(cfg)
    out = model(batch["eeg"], batch["stage_idx"], batch["eeg_valid_len"], channel_cluster_id=batch["channel_cluster_id"], mask_ratio=0.2)
    pretrain = compute_v11_pretrain_losses(out, batch)
    pretrain["total"].backward(retain_graph=True)
    for aligner in ["mlp", "clip", "ctc", "ot", "perceiver", "hybrid"]:
        model.zero_grad(set_to_none=True)
        out = model(batch["eeg"], batch["stage_idx"], batch["eeg_valid_len"], channel_cluster_id=batch["channel_cluster_id"])
        losses = compute_v11_alignment_losses(out, batch, aligner=aligner)
        assert torch.isfinite(losses["total"]), aligner
        losses["total"].backward()
    model.zero_grad(set_to_none=True)
    out = model(batch["eeg"], batch["stage_idx"], batch["eeg_valid_len"], channel_cluster_id=batch["channel_cluster_id"])
    codec = compute_v11_codec_losses(out, batch)
    assert torch.isfinite(codec["total"])
    codec["total"].backward()
    with torch.no_grad():
        generated = model.generate_codec_tokens(batch["eeg"], batch["stage_idx"], batch["eeg_valid_len"], channel_cluster_id=batch["channel_cluster_id"])
    assert generated["pred_codec_seq"].shape == (batch["eeg"].shape[0], cfg.codec_token_steps, cfg.codec_dim)
    assert generated["pred_codec_token_ids"].shape == (batch["eeg"].shape[0], cfg.codec_token_steps)
    print("karaone_v11_smoke_pass")


def synthetic_batch(cfg: KaraOneV11Config) -> dict[str, torch.Tensor]:
    b = 4
    eeg = torch.randn(b, cfg.n_channels_eeg, cfg.eeg_len)
    semantic_tokens = torch.randint(0, cfg.semantic_token_vocab, (b, cfg.semantic_token_steps))
    codec_tokens = torch.randint(0, cfg.codec_token_vocab, (b, cfg.codec_token_steps))
    return {
        "eeg": eeg,
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
    }


if __name__ == "__main__":
    main()

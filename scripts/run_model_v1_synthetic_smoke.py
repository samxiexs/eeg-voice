#!/usr/bin/env python3
"""Run a small synthetic EEGVoiceTokenV1 optimization smoke test."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch

from src.eeg_voice_model import EEGVoiceBatch
from src.eeg_voice_model.builders import build_eeg_voice_token_v1


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def make_batch(config, batch_size: int, channels: int, device: torch.device) -> EEGVoiceBatch:
    return EEGVoiceBatch(
        eeg=torch.randn(batch_size, channels, config.window_samples, device=device),
        sensor_pos=torch.randn(batch_size, channels, 3, device=device),
        channel_mask=torch.ones(batch_size, channels, dtype=torch.bool, device=device),
        dataset_id=["synthetic_ds006104"] * batch_size,
        language=["en"] * batch_size,
        domain_group=["english_first_core"] * batch_size,
        speaker_id=[f"synthetic_spk_{idx:03d}" for idx in range(batch_size)],
        audio_embedding=torch.randn(batch_size, config.audio_embedding_dim, device=device),
        sensor_type=torch.ones(batch_size, channels, dtype=torch.long, device=device),
        acquisition_device_id=torch.ones(batch_size, dtype=torch.long, device=device),
        montage_id=torch.ones(batch_size, dtype=torch.long, device=device),
        reference_id=torch.ones(batch_size, dtype=torch.long, device=device),
        sampling_rate_hz=torch.full((batch_size,), float(config.sample_rate), device=device),
        native_channel_count=torch.full((batch_size,), float(channels), device=device),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model_v1.yaml")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--amp-bf16", action="store_true", help="Use CUDA bf16 autocast for the forward pass.")
    parser.add_argument("--checkpoint-out", type=Path)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    model = build_eeg_voice_token_v1(args.config).to(device).train()
    batch = make_batch(model.config, args.batch_size, args.channels, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.amp_bf16 and device.type == "cuda"
        else nullcontext()
    )

    last_out = None
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        with amp_context:
            out = model(batch)
            loss = out["loss"]
        if not torch.isfinite(loss):
            raise SystemExit(f"Non-finite loss at step {step}: {loss.detach().cpu().item()}")
        loss.backward()
        opt.step()
        last_out = out
        print(
            json.dumps(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "tokens_shape": list(out["tokens"].shape),
                    "retrieval_logits_shape": list(out["retrieval_logits"].shape),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = {
        "device": str(device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "model": type(model).__name__,
        "window_samples": model.config.window_samples,
        "batch_size": args.batch_size,
        "channels": args.channels,
        "steps": args.steps,
    }
    if last_out is not None:
        summary["final_loss"] = float(last_out["loss"].detach().cpu())
        summary["final_tokens_shape"] = list(last_out["tokens"].shape)
    if device.type == "cuda":
        summary["gpu_name"] = torch.cuda.get_device_name(device)
        summary["peak_vram_gb"] = round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)

    if args.checkpoint_out:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "summary": summary,
            },
            args.checkpoint_out,
        )
        summary["checkpoint_out"] = str(args.checkpoint_out)

    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Train FEIS v3 tokenized EEG-to-codec generation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(BUNDLE_DIR / "../artifacts/matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.feis_v3.data import (
    FEISV3AudioTokenBank,
    FEISV3ClusterBank,
    FEISV3Dataset,
    FEISV3RepeatAwareBatchSampler,
    assert_v3_model_forward_keys,
)
from src.feis_v3.eval import evaluate_feis_v3
from src.feis_v3.losses import compute_feis_v3_losses
from src.feis_v3.model import FEISV3ModelConfig, FEISV3TokenGenerator
from src.utils import ensure_dir, load_simple_yaml, resolve_bundle_path, set_seed, write_json


def parse_args():
    p = argparse.ArgumentParser(description="Train FEIS v3 tokenized generation.")
    p.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "feis_v3_tokenized_generation.yaml"))
    p.add_argument("--stage", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--run-suffix", default="v3")
    p.add_argument("--aligner", default=None, choices=["mlp", "clip", "ctc", "ot", "perceiver", "hybrid"])
    p.add_argument("--device", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--eval-limit", type=int, default=None)
    p.add_argument("--allow-negative-train", action="store_true")
    p.add_argument("--init-from", default=None)
    return p.parse_args()


def _format_path(raw: str, stage: str) -> str:
    return str(raw).replace("{stage}", stage)


def _move_batch(batch: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def _loss_kwargs(cfg: dict) -> dict[str, float]:
    loss_cfg = cfg.get("loss", {})
    keys = [
        "lambda_semantic_ce",
        "lambda_prompt_ce",
        "lambda_clip",
        "lambda_ctc",
        "lambda_ot",
        "lambda_perceiver",
        "lambda_repeat",
        "lambda_cross_subject",
        "lambda_voice_push",
        "lambda_codec_ce",
        "lambda_prosody",
        "lambda_variant",
        "lambda_subject_confusion",
        "lambda_moe",
        "contrast_temperature",
    ]
    return {key: float(loss_cfg.get(key, 0.0 if key.startswith("lambda_") else 0.07)) for key in keys}


def _write_figures(run_dir: Path, history: list[dict], channel_gate: np.ndarray | None = None) -> None:
    fig_dir = ensure_dir(run_dir / "figures")
    if history:
        epochs = [row["epoch"] for row in history]
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.plot(epochs, [row["train"].get("total", 0.0) for row in history], marker="o", label="train_total")
        ax.plot(epochs, [row["subject_val"].get("prompt_acc", 0.0) for row in history], marker="o", label="val_prompt_acc")
        ax.plot(epochs, [row["subject_val"].get("codec_token_top1", 0.0) for row in history], marker="o", label="val_codec_top1")
        ax.set_xlabel("epoch")
        ax.set_title("FEIS v3 training curves")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.savefig(fig_dir / "training_curves.png", dpi=180)
        plt.close(fig)

        latest = history[-1]["subject_val"]
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        names = ["prompt_acc", "semantic_token_top3_gain_over_prior", "token_retrieval_cross_subject_gain", "codec_token_top1"]
        ax.bar(names, [float(latest.get(name, 0.0)) for name in names])
        ax.tick_params(axis="x", rotation=20)
        ax.set_title("Token alignment metrics")
        fig.savefig(fig_dir / "token_alignment_metrics.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.bar(
            ["same_repeat", "shuffled"],
            [
                float(latest.get("same_subject_label_repeat_consistency", 0.0)),
                float(latest.get("shuffled_repeat_consistency", 0.0)),
            ],
        )
        ax.set_title("Repeat consistency")
        fig.savefig(fig_dir / "repeat_consistency.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.bar(
            ["top1", "chance1", "top3", "chance3"],
            [
                float(latest.get("same_label_audio_variant_top1", 0.0)),
                float(latest.get("same_label_audio_variant_chance_top1", 0.0)),
                float(latest.get("same_label_audio_variant_top3", 0.0)),
                float(latest.get("same_label_audio_variant_chance_top3", 0.0)),
            ],
        )
        ax.set_title("Self-recording variant retrieval")
        fig.savefig(fig_dir / "self_recording_variant_retrieval.png", dpi=180)
        plt.close(fig)

    if channel_gate is not None and channel_gate.size:
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.bar(np.arange(channel_gate.shape[0]), channel_gate)
        ax.set_xlabel("EEG channel")
        ax.set_ylabel("mean gate")
        ax.set_title("Channel gate top channels")
        fig.savefig(fig_dir / "channel_gate_top_channels.png", dpi=180)
        plt.close(fig)


def _selection_score(metrics: dict[str, Any]) -> float:
    return (
        4.0 * float(metrics.get("prompt_acc", 0.0) - metrics.get("prompt_chance", 0.0))
        + float(metrics.get("semantic_token_top3_gain_over_prior", 0.0))
        + float(metrics.get("token_retrieval_cross_subject_gain", 0.0))
        + float(metrics.get("generated_over_labelprior_codec_margin", 0.0))
    )


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    set_seed(int(cfg.get("train", {}).get("seed", 7)))
    stage = args.stage or str(cfg.get("data", {}).get("stage", "stimuli"))
    aligner = args.aligner or os.environ.get("ALIGNER") or str(cfg.get("train", {}).get("aligner", "hybrid"))
    device = args.device or os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    token_cfg = cfg.get("tokens", {})
    data_cfg = cfg.get("data", {})
    model_cfg_raw = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    token_path = resolve_bundle_path(_format_path(token_cfg.get("cache", "../artifacts/audio_targets/feis_v3_tokens_{stage}.npz"), stage), BUNDLE_DIR)
    cluster_path = resolve_bundle_path(_format_path(token_cfg.get("cluster_cache", "../artifacts/audio_targets/feis_v3_clusters_{stage}.npz"), stage), BUNDLE_DIR)
    if not token_path.exists():
        raise FileNotFoundError(f"Missing FEIS v3 token cache: {token_path}. Run build_feis_v3_tokens.py first.")
    if not cluster_path.exists():
        raise FileNotFoundError(f"Missing FEIS v3 cluster cache: {cluster_path}. Run build_feis_v3_clusters.py first.")
    token_bank = FEISV3AudioTokenBank(token_path)
    cluster_bank = FEISV3ClusterBank(cluster_path)
    data_root = resolve_bundle_path(data_cfg.get("root", "../data/feis"), BUNDLE_DIR)
    common = dict(
        data_root=data_root,
        token_bank=token_bank,
        stage=stage,
        cluster_bank=cluster_bank,
        eeg_len=int(data_cfg.get("eeg_len", 1280)),
        include_anomalous=bool(data_cfg.get("include_anomalous", False)),
        subject_val=str(data_cfg.get("subject_val", "20")),
        subject_test=str(data_cfg.get("subject_test", "21")),
        negative_stage=str(data_cfg.get("negative_stage", "resting")),
        allow_negative_train=bool(args.allow_negative_train),
    )
    train_ds = FEISV3Dataset(split="train", **common)
    subject_val_ds = FEISV3Dataset(split="subject_val", **common)
    subject_test_ds = FEISV3Dataset(split="subject_test", **common)
    repeat_ds = FEISV3Dataset(split="repeat_test", **common)
    resting_ds = FEISV3Dataset(split="resting_control", **common)

    model_cfg = FEISV3ModelConfig(
        n_channels_eeg=int(model_cfg_raw.get("n_channels_eeg", 14)),
        d_model=int(model_cfg_raw.get("d_model", 160)),
        num_heads=int(model_cfg_raw.get("num_heads", 4)),
        num_layers=int(model_cfg_raw.get("num_layers", 2)),
        ff_mult=int(model_cfg_raw.get("ff_mult", 4)),
        dropout=float(model_cfg_raw.get("dropout", 0.15)),
        semantic_vocab=token_bank.semantic_vocab_size,
        codec_vocab=token_bank.codec_vocab_size,
        semantic_steps=token_bank.semantic_steps,
        codec_steps=token_bank.codec_steps,
        num_labels=token_bank.num_labels,
        num_stages=5,
        channel_clusters=cluster_bank.num_clusters,
        audio_variant_clusters=token_bank.audio_variant_clusters,
        channel_moe=bool(model_cfg_raw.get("channel_moe", True)),
        channel_top_k=int(model_cfg_raw.get("channel_top_k", 6)),
        channel_experts=int(model_cfg_raw.get("channel_experts", 4)),
        use_stage_token=bool(model_cfg_raw.get("use_stage_token", True)),
        use_subject_id_in_forward=bool(model_cfg_raw.get("use_subject_id_in_forward", False)),
        num_subjects_for_adversary=token_bank.num_subjects,
    )
    model = FEISV3TokenGenerator(model_cfg).to(device)
    assert_v3_model_forward_keys(("eeg", "stage_idx", "eeg_valid_len", "channel_cluster_id"))
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[init] loaded {args.init_from} missing={len(missing)} unexpected={len(unexpected)}")

    epochs = int(args.epochs or train_cfg.get("epochs", 50))
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 64))
    sampler = FEISV3RepeatAwareBatchSampler(train_ds, batch_size=batch_size, seed=int(train_cfg.get("seed", 7)), shuffle=True)
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=int(train_cfg.get("num_workers", 0)))
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-3)),
    )
    run_name = f"feis_v3_tokenized_generation_{stage}_{aligner}_{args.run_suffix}"
    run_dir = resolve_bundle_path(cfg.get("output", {}).get("root", "../artifacts/outputs_feis"), BUNDLE_DIR) / run_name
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "metrics")
    history: list[dict[str, Any]] = []
    best_score = -1e9
    best_path = run_dir / "checkpoints" / "best.pt"
    loss_kwargs = _loss_kwargs(cfg)
    max_steps = args.max_steps if args.max_steps is not None else None
    channel_gate_snapshot: np.ndarray | None = None
    print(f"[data] stage={stage} train={len(train_ds)} repeat={len(repeat_ds)} subject_val={len(subject_val_ds)} subject_test={len(subject_test_ds)} resting={len(resting_ds)}")
    print(f"[model] aligner={aligner} inputs=['eeg','stage_idx','eeg_valid_len','channel_cluster_id'] subject_id_forward=False")
    for epoch in range(epochs):
        model.train()
        agg: dict[str, float] = {}
        seen = 0
        steps = 0
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            out = model(
                batch["eeg"],
                stage_idx=batch["stage_idx"],
                eeg_valid_len=batch["eeg_valid_len"],
                channel_cluster_id=batch["channel_cluster_id"],
            )
            if "channel_gate" in out and channel_gate_snapshot is None:
                channel_gate_snapshot = out["channel_gate"].detach().cpu().numpy().mean(axis=0)
            losses = compute_feis_v3_losses(out, batch, aligner=aligner, **loss_kwargs)
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            opt.step()
            bsz = int(batch["eeg"].shape[0])
            seen += bsz
            steps += 1
            for key, value in losses.items():
                if torch.is_tensor(value) and torch.isfinite(value).all():
                    agg[key] = agg.get(key, 0.0) + float(value.detach().cpu()) * bsz
            if max_steps and steps >= max_steps:
                break
        train_metrics = {key: val / max(seen, 1) for key, val in agg.items()}
        subject_val_metrics = evaluate_feis_v3(
            model,
            subject_val_ds,
            token_bank,
            device=device,
            batch_size=batch_size,
            split_name="subject_val",
            compute_controls=False,
            max_samples=args.eval_limit,
        )
        score = _selection_score(subject_val_metrics)
        row = {"epoch": epoch, "train": train_metrics, "subject_val": subject_val_metrics, "selection_score": score}
        history.append(row)
        write_json(run_dir / "metrics" / "history.json", {"history": history})
        print(
            f"epoch {epoch:03d} total={train_metrics.get('total', 0.0):.3f} "
            f"prompt={train_metrics.get('prompt_acc', 0.0):.3f} sem3={train_metrics.get('semantic_top3', 0.0):.3f} "
            f"codec={train_metrics.get('codec_top1', 0.0):.3f} | val_prompt={subject_val_metrics['prompt_acc']:.3f} "
            f"val_codec={subject_val_metrics['codec_token_top1']:.3f} score={score:+.3f}"
        )
        ckpt_payload = {
            "model_state": model.state_dict(),
            "model_config": vars(model_cfg),
            "stage": stage,
            "aligner": aligner,
            "token_cache": str(token_path),
            "cluster_cache": str(cluster_path),
            "run_dir": str(run_dir),
            "selection_score": float(score),
            "model_forward_inputs": ["eeg", "stage_idx", "eeg_valid_len", "channel_cluster_id"],
            "subject_id_forward_input": False,
            "speaker_id_forward_input": False,
            "retrieval_is_diagnostic_only": True,
            "generated_artifact": "generated_codec",
            "allow_negative_train": bool(args.allow_negative_train),
        }
        torch.save(ckpt_payload, run_dir / "checkpoints" / "last.pt")
        if score > best_score:
            best_score = score
            torch.save(ckpt_payload, best_path)
        if max_steps:
            break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    final_subject_val = evaluate_feis_v3(
        model,
        subject_val_ds,
        token_bank,
        device=device,
        batch_size=batch_size,
        split_name="subject_val",
        max_samples=args.eval_limit,
    )
    final_subject_test = evaluate_feis_v3(
        model,
        subject_test_ds,
        token_bank,
        device=device,
        batch_size=batch_size,
        split_name="subject_test",
        max_samples=args.eval_limit,
    )
    repeat_metrics = evaluate_feis_v3(
        model,
        repeat_ds,
        token_bank,
        device=device,
        batch_size=batch_size,
        split_name="repeat_holdout",
        max_samples=args.eval_limit,
    )
    resting_metrics = evaluate_feis_v3(
        model,
        resting_ds,
        token_bank,
        device=device,
        batch_size=batch_size,
        split_name="resting_negative_control",
        max_samples=args.eval_limit,
    )
    latest = {
        "run_dir": str(run_dir),
        "stage": stage,
        "aligner": aligner,
        "best_checkpoint": str(best_path),
        "generated_artifact": "generated_codec",
        "retrieval_name": "retrieval_diagnostic",
        "retrieval_is_diagnostic_only": True,
        "subject_id_forward_input": False,
        "speaker_id_forward_input": False,
        "allow_negative_train": bool(args.allow_negative_train),
        "subject_holdout": {"subject_val": final_subject_val, "subject_test": final_subject_test},
        "repeat_holdout": repeat_metrics,
        "resting_negative_control": resting_metrics,
        "generation_gate_pass": bool(final_subject_val.get("generation_gate_pass") and final_subject_test.get("generation_gate_pass") and resting_metrics.get("resting_negative_control_pass")),
        "claim_status": "diagnostic generated codec attempt",
    }
    if latest["generation_gate_pass"]:
        latest["claim_status"] = "EEG-to-Speech generation gate passed on subject_val and subject_test"
    write_json(run_dir / "metrics" / "latest_metrics.json", latest)
    write_json(run_dir / "metrics" / "repeat_reliability_metrics.json", repeat_metrics)
    write_json(
        run_dir / "metrics" / "self_recording_specificity_metrics.json",
        {
            "subject_val": {key: val for key, val in final_subject_val.items() if "audio_variant" in key or "self_recording" in key},
            "subject_test": {key: val for key, val in final_subject_test.items() if "audio_variant" in key or "self_recording" in key},
        },
    )
    _write_figures(run_dir, history, channel_gate_snapshot)
    print(f"[done] {run_dir}")


if __name__ == "__main__":
    main()

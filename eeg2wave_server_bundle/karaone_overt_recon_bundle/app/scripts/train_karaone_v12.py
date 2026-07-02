from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from scripts.train_karaone_v11 import (  # noqa: E402
    default_device,
    freeze_except_codec,
    loss_kwargs,
    make_loader,
    move_batch,
    progress,
    value_counts,
)
from src.karaone_v9.data import KaraOneV9TargetBank  # noqa: E402
from src.karaone_v12.data import (  # noqa: E402
    KaraOneV10ClusterBank,
    KaraOneV11TokenBank,
    KaraOneV12Dataset,
    KaraOneV12TimeAnchorBank,
    load_channel_names,
    outputs_to_token_bank,
)
from src.karaone_v12.eval import collect_v12_outputs, compute_v12_metrics, row_gate_summary, v12_selection_score, write_channel_reports  # noqa: E402
from src.karaone_v12.losses import compute_v12_alignment_losses, compute_v12_codec_losses, compute_v12_pretrain_losses, compute_v12_time_losses  # noqa: E402
from src.karaone_v12.model import KaraOneV12Config, KaraOneV12TokenGenerator  # noqa: E402
from src.utils import count_parameters, count_trainable_parameters, ensure_dir, load_simple_yaml, resolve_bundle_path, set_seed, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KaraOne v12 time-aware tokenized neural speech generation stages.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone_v12.yaml"))
    parser.add_argument("--phase", choices=["pretrain", "align", "time", "codec"], default="align")
    parser.add_argument("--stages", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-suffix", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cluster-bank", default=None)
    parser.add_argument("--token-bank", default=None)
    parser.add_argument("--time-anchor-bank", default=None)
    parser.add_argument("--aligner", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    train_cfg = cfg.get("train", {})
    time_cfg = cfg.get("time_train", {})
    codec_cfg = cfg.get("codec", {})
    set_seed(int(train_cfg.get("seed", 11)))
    device = torch.device(args.device or default_device())
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    stages = tuple(item.strip() for item in str(args.stages or cfg["data"].get("stages", "overt_like")).split(",") if item.strip())
    cache_cfg = cfg.get("cache", {})
    targets = KaraOneV9TargetBank(
        resolve_bundle_path(cache_cfg["semantic"], BUNDLE_DIR),
        codec_cache=resolve_bundle_path(cache_cfg["codec"], BUNDLE_DIR) if cache_cfg.get("codec") else None,
        prosody_cache=resolve_bundle_path(cache_cfg["prosody"], BUNDLE_DIR) if cache_cfg.get("prosody") else None,
        semantic_token_cache=resolve_bundle_path(cache_cfg["semantic_tokens"], BUNDLE_DIR) if cache_cfg.get("semantic_tokens") else None,
        data_root=root,
    )
    subject_val = str(cfg["data"].get("subject_val", "P02"))
    subject_test = str(cfg["data"].get("subject_test", "MM21"))
    eeg_len = int(cfg["data"].get("eeg_len", 1280))
    cluster_bank = KaraOneV10ClusterBank(resolve_bundle_path(args.cluster_bank or cache_cfg.get("cluster_bank", ""), BUNDLE_DIR))
    token_bank = KaraOneV11TokenBank(resolve_bundle_path(args.token_bank or cache_cfg.get("v11_token_bank", ""), BUNDLE_DIR))
    time_bank = KaraOneV12TimeAnchorBank(resolve_bundle_path(args.time_anchor_bank or cache_cfg.get("v12_time_anchor_bank", ""), BUNDLE_DIR))
    train_ds = KaraOneV12Dataset(root, targets, str(cfg["data"].get("train_split", "subject_train")), cluster_bank=cluster_bank, token_bank=token_bank, time_anchor_bank=time_bank, stages=stages, subject_val=subject_val, subject_test=subject_test, eeg_len=eeg_len, require_codec=args.phase == "codec")
    val_ds = KaraOneV12Dataset(root, targets, "subject_val", cluster_bank=cluster_bank, token_bank=token_bank, time_anchor_bank=time_bank, stages=stages, subject_val=subject_val, subject_test=subject_test, eeg_len=eeg_len)
    test_ds = KaraOneV12Dataset(root, targets, "subject_test", cluster_bank=cluster_bank, token_bank=token_bank, time_anchor_bank=time_bank, stages=stages, subject_val=subject_val, subject_test=subject_test, eeg_len=eeg_len)
    aligner = str(args.aligner or os.environ.get("ALIGNER") or train_cfg.get("aligner", cfg.get("model", {}).get("aligner", "hybrid"))).lower()
    model_cfg = make_model_config(cfg, targets, train_ds, token_bank, time_bank, eeg_len=eeg_len, stages=stages)
    model_cfg.aligner = aligner
    model = KaraOneV12TokenGenerator(model_cfg, codec_codebook=torch.from_numpy(token_bank.codec_codebook)).to(device)
    if args.checkpoint:
        load_checkpoint(model, resolve_bundle_path(args.checkpoint, BUNDLE_DIR))
    if args.phase == "codec" and args.freeze_encoder:
        freeze_except_codec_and_time(model)
    if args.phase == "time":
        freeze_except_time(model)

    epochs = resolve_epochs(args, train_cfg, time_cfg, codec_cfg)
    if args.smoke:
        epochs = min(epochs, 1)
        args.max_steps = args.max_steps or 2
    batch_size = int(args.batch_size or (4 if args.smoke else train_cfg.get("batch_size", 32)))
    loader = make_loader(train_ds, batch_size=batch_size, cfg=cfg)
    lr, weight_decay = resolve_optim(args.phase, train_cfg, time_cfg, codec_cfg)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)

    suffix = args.run_suffix or f"v12_{args.phase}_{aligner}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    stage_tag = "_".join(stages)
    out_root = resolve_bundle_path(cfg["output"]["root"], BUNDLE_DIR)
    out_dir = ensure_dir(out_root / f"{cfg['output'].get('prefix', 'karaone_v12')}_{args.phase}_{stage_tag}_{aligner}_{suffix}")
    for sub in ["checkpoints", "metrics", "channel_reports"]:
        ensure_dir(out_dir / sub)
    channel_names = load_channel_names(root, n_channels=int(cfg.get("model", {}).get("n_channels_eeg", 62)))
    write_json(out_dir / "run_config.json", {"phase": args.phase, "aligner": aligner, "stages": list(stages), "device": str(device), "train_n": len(train_ds), "subject_val_n": len(val_ds), "subject_test_n": len(test_ds), "cluster_bank": str(cluster_bank.path), "token_bank": str(token_bank.path), "time_anchor_bank": str(time_bank.path), "config": cfg})
    print_run_header(args.phase, aligner, stages, device, out_dir, train_ds, val_ds, test_ds, model, batch_size, epochs, lr, weight_decay, args.max_steps, bool(args.verbose), time_bank)

    train_bank = dataset_target_bank(train_ds, model, device=device, batch_size=batch_size)
    best_score = -1e9
    best_epoch = -1
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running: dict[str, list[float]] = {}
        iterator = progress(loader, total=len(loader), desc=f"v12 {args.phase} epoch {epoch}")
        for step, batch in enumerate(iterator, start=1):
            if args.max_steps and step > int(args.max_steps):
                break
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch["eeg"], batch["stage_idx"], batch["eeg_valid_len"], channel_cluster_id=batch.get("channel_cluster_id"), mask_ratio=float(train_cfg.get("mask_ratio", 0.25)) if args.phase == "pretrain" else 0.0, lambda_subject_adv=float(train_cfg.get("lambda_subject_adv", 0.10)))
            if args.phase == "pretrain":
                losses = compute_v12_pretrain_losses(out, batch, lambda_channel_balance=float(train_cfg.get("lambda_channel_balance", 0.05)), lambda_gate_sparsity=float(train_cfg.get("lambda_gate_sparsity", 0.02)), lambda_gate_entropy=float(train_cfg.get("lambda_gate_entropy", 0.02)))
            elif args.phase == "time":
                losses = compute_v12_time_losses(out, batch, **time_loss_kwargs(time_cfg))
            elif args.phase == "codec":
                losses = compute_v12_codec_losses(out, batch, lambda_codec_token_ce=float(codec_cfg.get("lambda_codec_token_ce", 1.0)), lambda_codec_latent=float(codec_cfg.get("lambda_codec_latent", 0.5)), lambda_semantic_guard=float(codec_cfg.get("lambda_semantic_guard", 0.2)), lambda_boundary_continuity=float(codec_cfg.get("lambda_boundary_continuity", 0.05)), lambda_time_guard=float(codec_cfg.get("lambda_time_guard", 0.05)))
            else:
                losses = compute_v12_alignment_losses(out, batch, aligner=aligner, **loss_kwargs(train_cfg), lambda_boundary_ctc=float(train_cfg.get("lambda_boundary_ctc", 0.05)), lambda_forward_monotonic=float(train_cfg.get("lambda_forward_monotonic", 0.02)))
            losses["total"].backward()
            clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            optimizer.step()
            for key, value in losses.items():
                running.setdefault(key, []).append(float(value.detach().cpu()))
            if int(args.log_interval or 0) > 0 and (step == 1 or step % int(args.log_interval) == 0):
                print_step_log(epoch, step, len(loader), losses, out, batch, bool(args.verbose))

        train_losses = {f"train_{key}": float(np.mean(values)) for key, values in running.items()}
        val_outputs = collect_v12_outputs(model, val_ds, device=device, batch_size=batch_size)
        test_outputs = collect_v12_outputs(model, test_ds, device=device, batch_size=batch_size)
        val_metrics = compute_v12_metrics(val_outputs, train_bank=train_bank, prefix="subject_val")
        test_metrics = compute_v12_metrics(test_outputs, train_bank=train_bank, prefix="subject_test")
        write_channel_reports(out_dir / "channel_reports" / f"epoch_{epoch:03d}_subject_val", val_outputs, channel_names)
        score = selection_score({**val_metrics, **test_metrics})
        row = {"epoch": epoch, **train_losses, **val_metrics, **test_metrics, "selection_score": score}
        history.append(row)
        write_json(out_dir / "metrics" / "latest_metrics.json", row)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(model, out_dir / "checkpoints" / "best.pt", cfg=cfg, epoch=epoch, metrics=row, aligner=aligner)
            write_channel_reports(out_dir / "channel_reports" / "best_subject_val", val_outputs, channel_names)
            write_channel_reports(out_dir / "channel_reports" / "best_subject_test", test_outputs, channel_names)
        save_checkpoint(model, out_dir / "checkpoints" / "last.pt", cfg=cfg, epoch=epoch, metrics=row, aligner=aligner)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
        print(json.dumps({"event": "v12_epoch_gate_summary", **row_gate_summary(row)}, ensure_ascii=False, indent=2), flush=True)

    write_json(out_dir / "metrics" / "history.json", {"history": history, "best_epoch": best_epoch, "best_score": best_score})
    print(json.dumps({"out_dir": str(out_dir), "best_epoch": best_epoch, "best_score": best_score}, ensure_ascii=False, indent=2), flush=True)


def make_model_config(cfg: dict, targets: KaraOneV9TargetBank, train_ds: KaraOneV12Dataset, token_bank: KaraOneV11TokenBank, time_bank: KaraOneV12TimeAnchorBank, *, eeg_len: int, stages: tuple[str, ...]) -> KaraOneV12Config:
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    time_cfg = cfg.get("time_anchor", {})
    max_lag = float(model_cfg.get("max_lag_sec", 1.0))
    if any(str(stage) == "thinking" for stage in stages):
        max_lag = float(time_cfg.get("max_lag_sec_thinking", max_lag))
    return KaraOneV12Config(
        n_channels_eeg=int(model_cfg.get("n_channels_eeg", 62)),
        eeg_len=int(eeg_len),
        eeg_sample_rate=float(model_cfg.get("eeg_sample_rate", data_cfg.get("eeg_sample_rate", 256.0))),
        patch_size=int(model_cfg.get("patch_size", 32)),
        patch_stride=int(model_cfg.get("patch_stride", 16)),
        d_model=int(model_cfg.get("d_model", 128)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.15)),
        channel_dropout=float(model_cfg.get("channel_dropout", 0.12)),
        cond_dim=int(model_cfg.get("cond_dim", 32)),
        num_stages=int(train_ds.num_stages),
        num_subjects=int(train_ds.num_subjects),
        num_labels=int(train_ds.num_labels),
        semantic_dim=int(targets.semantic_dim),
        semantic_token_vocab=max(2, int(targets.semantic_token_vocab)),
        semantic_token_steps=int(targets.semantic_token_steps),
        codec_dim=int(token_bank.codec_dim or targets.codec_dim),
        codec_token_vocab=max(2, int(token_bank.codec_token_vocab)),
        codec_token_steps=int(token_bank.codec_steps),
        channel_experts=int(model_cfg.get("channel_experts", 6)),
        channel_top_k=int(model_cfg.get("channel_top_k", 16)),
        channel_clusters=max(1, int(token_bank.n_channel_clusters)),
        channel_desc_dim=int(model_cfg.get("channel_desc_dim", 32)),
        channel_embedding_dim=int(model_cfg.get("channel_embedding_dim", 16)),
        channel_cluster_embedding_dim=int(model_cfg.get("channel_cluster_embedding_dim", 8)),
        moe_temperature=float(model_cfg.get("moe_temperature", 0.7)),
        shared_dim=int(model_cfg.get("shared_dim", 128)),
        domain_dim=int(model_cfg.get("domain_dim", 64)),
        perceiver_queries=int(model_cfg.get("perceiver_queries", targets.semantic_token_steps)),
        aligner=str(model_cfg.get("aligner", "hybrid")),
        sample_rate=int(time_cfg.get("sample_rate", cfg.get("synthesis", {}).get("sample_rate", 16000))),
        duration_sec=float(time_cfg.get("duration_sec", cfg.get("synthesis", {}).get("duration_sec", 2.0))),
        active_mask_steps=int(model_cfg.get("active_mask_steps", time_bank.active_steps)),
        max_lag_sec=max_lag,
    )


def resolve_epochs(args, train_cfg, time_cfg, codec_cfg) -> int:
    if args.epochs:
        return int(args.epochs)
    if args.phase == "codec":
        return int(codec_cfg.get("epochs", 10))
    if args.phase == "time":
        return int(time_cfg.get("epochs", 8))
    return int(train_cfg.get("epochs", 50))


def resolve_optim(phase: str, train_cfg: dict, time_cfg: dict, codec_cfg: dict) -> tuple[float, float]:
    if phase == "codec":
        return float(codec_cfg.get("lr", train_cfg.get("lr", 3e-4))), float(codec_cfg.get("weight_decay", train_cfg.get("weight_decay", 1e-3)))
    if phase == "time":
        return float(time_cfg.get("lr", train_cfg.get("lr", 3e-4))), float(time_cfg.get("weight_decay", train_cfg.get("weight_decay", 1e-3)))
    return float(train_cfg.get("lr", 3e-4)), float(train_cfg.get("weight_decay", 1e-3))


def time_loss_kwargs(time_cfg: dict[str, Any]) -> dict[str, float]:
    keys = ["lambda_lag", "lambda_onset", "lambda_duration", "lambda_active_mask", "lambda_active_iou", "lambda_shift_envelope"]
    return {key: float(time_cfg[key]) for key in keys if key in time_cfg}


def dataset_target_bank(dataset: KaraOneV12Dataset, model: KaraOneV12TokenGenerator, *, device: torch.device, batch_size: int) -> dict[str, Any]:
    outputs = collect_v12_outputs(model, dataset, device=device, batch_size=batch_size)
    bank = outputs_to_token_bank(outputs)
    bank.update({"target": outputs["target"], "labels": outputs["labels"], "subjects": outputs["subjects"]})
    return bank


def selection_score(metrics: dict[str, Any]) -> float:
    score = v12_selection_score(metrics, prefix="subject_val")
    if float(metrics.get("subject_test_token_retrieval_cross_subject_gain", 0.0)) < 0.0:
        score -= 0.05
    if float(metrics.get("subject_test_active_iou", 0.0)) <= 0.20:
        score -= 0.05
    return float(score)


def print_run_header(phase, aligner, stages, device, out_dir, train_ds, val_ds, test_ds, model, batch_size, epochs, lr, weight_decay, max_steps, verbose, time_bank):
    payload = {
        "event": "v12_run_start",
        "phase": phase,
        "aligner": aligner,
        "stages": list(stages),
        "device": str(device),
        "out_dir": str(out_dir),
        "train_n": len(train_ds),
        "subject_val_n": len(val_ds),
        "subject_test_n": len(test_ds),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "max_steps": int(max_steps) if max_steps else None,
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "parameters": int(count_parameters(model)),
        "trainable_parameters": int(count_trainable_parameters(model)),
        "time_anchor_bank_available": bool(time_bank.available),
        "stage_lag_prior": time_bank.stage_lag_prior,
        "train_subjects": sorted({entry.subject for entry in train_ds.entries}),
        "val_subjects": sorted({entry.subject for entry in val_ds.entries}),
        "test_subjects": sorted({entry.subject for entry in test_ds.entries}),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def print_step_log(epoch: int, step: int, total_steps: int, losses: dict[str, torch.Tensor], out: dict[str, torch.Tensor], batch: dict[str, Any], verbose: bool) -> None:
    payload: dict[str, Any] = {
        "event": "v12_train_step",
        "epoch": int(epoch),
        "step": int(step),
        "total_steps": int(total_steps),
        "losses": {key: float(value.detach().cpu()) for key, value in losses.items() if torch.is_tensor(value)},
        "time_anchor": {
            "pred_lag_mean": float(out["pred_lag_sec"].detach().mean().cpu()),
            "pred_onset_mean": float(out["pred_onset_sec"].detach().mean().cpu()),
            "pred_duration_mean": float(out["pred_duration_sec"].detach().mean().cpu()),
        },
        "channel_moe": {
            "gate_mean": float(out["channel_gate"].detach().mean().cpu()),
            "gate_active_ratio": float((out["channel_gate"].detach() > 1e-4).float().mean().cpu()),
        },
    }
    if verbose:
        payload["batch"] = {"subjects": value_counts(batch.get("subject", [])), "labels": value_counts(batch.get("label", []))}
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def save_checkpoint(model: torch.nn.Module, path: Path, *, cfg: dict, epoch: int, metrics: dict[str, Any], aligner: str) -> None:
    ensure_dir(path.parent)
    torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "metrics": metrics, "aligner": aligner, "model_kind": "karaone_v12_time_aware_tokenized_generation"}, path)


def load_checkpoint(model: torch.nn.Module, path: Path) -> Any:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(json.dumps({"checkpoint": str(path), "missing": missing, "unexpected": unexpected}, ensure_ascii=False, indent=2), flush=True)
    return payload


def freeze_except_time(model: KaraOneV12TokenGenerator) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("time_anchor_head")


def freeze_except_codec_and_time(model: KaraOneV12TokenGenerator) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("codec_token_head") or name.startswith("time_anchor_head")


if __name__ == "__main__":
    main()

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
from torch.utils.data import DataLoader

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.karaone_v9.data import KaraOneV9TargetBank
from src.karaone_v10.data import (
    KaraOneV10ClusterBalancedBatchSampler,
    KaraOneV10ClusterBank,
    KaraOneV10ClusteredDataset,
    load_channel_names,
)
from src.karaone_v10.eval import collect_v10_outputs, compute_v10_metrics, outputs_to_v10_bank, row_gate_summary, v10_selection_score, write_channel_reports
from src.karaone_v10.losses import compute_v10_alignment_losses, compute_v10_pretrain_losses, compute_v10_transport_losses
from src.karaone_v10.model import KaraOneV10ClusteredChannelMoEFlow, KaraOneV10Config
from src.utils import count_parameters, count_trainable_parameters, ensure_dir, load_simple_yaml, resolve_bundle_path, set_seed, write_json

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KaraOne v10 final clustered Channel-MoE semantic-flow stages.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone_v10.yaml"))
    parser.add_argument("--phase", choices=["pretrain", "align", "transport", "flow"], default="align")
    parser.add_argument("--stages", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-suffix", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cluster-bank", default=None)
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
    transport_cfg = cfg.get("transport", {})
    set_seed(int(train_cfg.get("seed", 7)))
    device = torch.device(args.device or default_device())
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    stages = tuple(
        item.strip()
        for item in str(args.stages or cfg["data"].get("stages", "overt_like")).split(",")
        if item.strip()
    )
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
    cluster_path = args.cluster_bank or cache_cfg.get("cluster_bank", "")
    cluster_bank = KaraOneV10ClusterBank(resolve_bundle_path(cluster_path, BUNDLE_DIR) if cluster_path else None)
    if not cluster_bank.available:
        print("[v10] warning: cluster bank missing; cluster ids default to 0. Run scripts/build_karaone_v91_clusters.py first.", flush=True)
    train_split = str(cfg["data"].get("train_split", "subject_train"))
    phase = "transport" if args.phase == "flow" else args.phase
    require_codec = phase == "transport"
    train_ds = KaraOneV10ClusteredDataset(
        root,
        targets,
        train_split,
        cluster_bank=cluster_bank,
        stages=stages,
        subject_val=subject_val,
        subject_test=subject_test,
        eeg_len=eeg_len,
        require_codec=require_codec,
    )
    val_ds = KaraOneV10ClusteredDataset(root, targets, "subject_val", cluster_bank=cluster_bank, stages=stages, subject_val=subject_val, subject_test=subject_test, eeg_len=eeg_len)
    test_ds = KaraOneV10ClusteredDataset(root, targets, "subject_test", cluster_bank=cluster_bank, stages=stages, subject_val=subject_val, subject_test=subject_test, eeg_len=eeg_len)
    model = KaraOneV10ClusteredChannelMoEFlow(make_model_config(cfg, targets, train_ds, eeg_len=eeg_len)).to(device)
    if args.checkpoint:
        load_checkpoint(model, resolve_bundle_path(args.checkpoint, BUNDLE_DIR))
    if phase == "transport" and args.freeze_encoder:
        if not args.checkpoint:
            print("[v10] warning: --freeze-encoder without --checkpoint is only meaningful for smoke tests.", flush=True)
        freeze_except_transport(model)

    epochs = int(args.epochs or (transport_cfg.get("epochs", 10) if phase == "transport" else train_cfg.get("epochs", 20)))
    if args.smoke:
        epochs = min(epochs, 1)
        args.max_steps = args.max_steps or 2
    batch_size = int(args.batch_size or (4 if args.smoke else train_cfg.get("batch_size", 32)))
    loader = make_loader(train_ds, batch_size=batch_size, cfg=cfg)
    lr = float(transport_cfg.get("lr", train_cfg.get("lr", 3e-4)) if phase == "transport" else train_cfg.get("lr", 3e-4))
    weight_decay = float(transport_cfg.get("weight_decay", train_cfg.get("weight_decay", 1e-3)) if phase == "transport" else train_cfg.get("weight_decay", 1e-3))
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=lr, weight_decay=weight_decay)

    suffix = args.run_suffix or f"v10_{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    stage_tag = "_".join(stages)
    out_root = resolve_bundle_path(cfg["output"]["root"], BUNDLE_DIR)
    out_dir = ensure_dir(out_root / f"{cfg['output'].get('prefix', 'karaone_v10_final')}_{phase}_{stage_tag}_{suffix}")
    ensure_dir(out_dir / "checkpoints")
    ensure_dir(out_dir / "metrics")
    ensure_dir(out_dir / "channel_reports")
    channel_names = load_channel_names(root, n_channels=int(cfg.get("model", {}).get("n_channels_eeg", 62)))
    write_json(
        out_dir / "run_config.json",
        {
            "phase": phase,
            "stages": list(stages),
            "device": str(device),
            "train_n": len(train_ds),
            "subject_val_n": len(val_ds),
            "subject_test_n": len(test_ds),
            "cluster_bank": str(cluster_bank.path) if cluster_bank.path else None,
            "config": cfg,
        },
    )
    log_interval = int(args.log_interval or train_cfg.get("log_interval", 0) or 0)
    verbose = bool(args.verbose or train_cfg.get("verbose", False))
    print_run_header(
        phase=phase,
        stages=stages,
        device=device,
        out_dir=out_dir,
        root=root,
        cluster_bank=cluster_bank,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        model=model,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        max_steps=args.max_steps,
        verbose=verbose,
    )

    train_bank = dataset_target_bank(train_ds)
    best_score = -1e9
    best_epoch = -1
    history: list[dict[str, Any]] = []
    patience = int(train_cfg.get("early_stop_patience", 0))
    for epoch in range(1, epochs + 1):
        model.train()
        running: dict[str, list[float]] = {}
        iterator = progress(loader, total=len(loader), desc=f"v10 {phase} epoch {epoch}")
        for step, batch in enumerate(iterator, start=1):
            if args.max_steps and step > int(args.max_steps):
                break
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if phase == "pretrain":
                out = model(batch["eeg"], batch["stage_idx"], batch["eeg_valid_len"], mask_ratio=float(train_cfg.get("mask_ratio", 0.25)))
                losses = compute_v10_pretrain_losses(
                    out,
                    batch,
                    lambda_channel_balance=float(train_cfg.get("lambda_channel_balance", 0.05)),
                    lambda_gate_sparsity=float(train_cfg.get("lambda_gate_sparsity", 0.02)),
                    lambda_gate_entropy=float(train_cfg.get("lambda_gate_entropy", 0.02)),
                )
            elif phase == "transport":
                out = model(
                    batch["eeg"],
                    batch["stage_idx"],
                    batch["eeg_valid_len"],
                    codec_seq=batch["codec_seq"],
                    scheduled_teacher_ratio=float(transport_cfg.get("scheduled_teacher_ratio", 0.0)),
                )
                losses = compute_v10_transport_losses(
                    out,
                    batch,
                    lambda_flow=float(transport_cfg.get("lambda_flow", 1.0)),
                    lambda_condition_semantic=float(transport_cfg.get("lambda_condition_semantic", 0.2)),
                    lambda_codec_consistency=float(transport_cfg.get("lambda_codec_consistency", 0.2)),
                    lambda_boundary_continuity=float(transport_cfg.get("lambda_boundary_continuity", 0.05)),
                )
            else:
                lambda_subject_adv = float(train_cfg.get("lambda_subject_adv", 0.10))
                out = model(batch["eeg"], batch["stage_idx"], batch["eeg_valid_len"], lambda_subject_adv=lambda_subject_adv)
                losses = compute_v10_alignment_losses(
                    out,
                    batch,
                    lambda_seq_ot=float(train_cfg.get("lambda_seq_ot", 1.0)),
                    lambda_seq_cos=float(train_cfg.get("lambda_seq_cos", 0.5)),
                    lambda_global_nce=float(train_cfg.get("lambda_global_nce", 0.5)),
                    lambda_soft_nce=float(train_cfg.get("lambda_soft_nce", 0.5)),
                    lambda_semantic_token=float(train_cfg.get("lambda_semantic_token", 0.3)),
                    lambda_ctc=float(train_cfg.get("lambda_ctc", 0.2)),
                    lambda_prompt=float(train_cfg.get("lambda_prompt", 0.1)),
                    lambda_prosody=float(train_cfg.get("lambda_prosody", 0.5)),
                    lambda_subject_adv=lambda_subject_adv,
                    lambda_coral=float(train_cfg.get("lambda_coral", 0.05)),
                    lambda_group_dro=float(train_cfg.get("lambda_group_dro", 0.2)),
                    lambda_variance=float(train_cfg.get("lambda_variance", 0.1)),
                    nce_temperature=float(train_cfg.get("nce_temperature", 0.07)),
                    soft_target_temperature=float(train_cfg.get("soft_target_temperature", 0.08)),
                    lambda_cluster_nce=float(train_cfg.get("lambda_cluster_nce", 0.25)),
                    lambda_hard_negative=float(train_cfg.get("lambda_hard_negative", 0.15)),
                    lambda_gate_consistency=float(train_cfg.get("lambda_gate_consistency", 0.05)),
                    lambda_channel_balance=float(train_cfg.get("lambda_channel_balance", 0.05)),
                    lambda_gate_sparsity=float(train_cfg.get("lambda_gate_sparsity", 0.02)),
                    lambda_gate_entropy=float(train_cfg.get("lambda_gate_entropy", 0.02)),
                    lambda_domain_subject=float(train_cfg.get("lambda_domain_subject", 0.05)),
                    lambda_content_domain_orth=float(train_cfg.get("lambda_content_domain_orth", 0.02)),
                    lambda_eeg_specific_margin=float(train_cfg.get("lambda_eeg_specific_margin", 0.30)),
                    lambda_mean_prior_margin=float(train_cfg.get("lambda_mean_prior_margin", 0.20)),
                    lambda_cross_subject_semantic=float(train_cfg.get("lambda_cross_subject_semantic", 0.35)),
                    lambda_label_prototype_pull=float(train_cfg.get("lambda_label_prototype_pull", 0.10)),
                    lambda_prompt_balanced=float(train_cfg.get("lambda_prompt_balanced", 0.20)),
                    lambda_pairwise_decorrelation=float(train_cfg.get("lambda_pairwise_decorrelation", 0.05)),
                    semantic_margin=float(train_cfg.get("semantic_margin", 0.04)),
                    cross_subject_temperature=float(train_cfg.get("cross_subject_temperature", 0.06)),
                )
            losses["total"].backward()
            clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            optimizer.step()
            for key, value in losses.items():
                running.setdefault(key, []).append(float(value.detach().cpu()))
            if log_interval > 0 and (step == 1 or step % log_interval == 0):
                print_step_log(
                    epoch=epoch,
                    step=step,
                    total_steps=len(loader),
                    losses=losses,
                    out=out,
                    batch=batch,
                    verbose=verbose,
                )

        train_losses = {f"train_{key}": float(np.mean(values)) for key, values in running.items()}
        val_outputs = collect_v10_outputs(model, val_ds, device=device, batch_size=batch_size)
        test_outputs = collect_v10_outputs(model, test_ds, device=device, batch_size=batch_size)
        val_metrics = compute_v10_metrics(val_outputs, train_bank=train_bank, prefix="subject_val")
        test_metrics = compute_v10_metrics(test_outputs, train_bank=train_bank, prefix="subject_test")
        write_channel_reports(out_dir / "channel_reports" / f"epoch_{epoch:03d}_subject_val", val_outputs, channel_names)
        score = selection_score({**val_metrics, **test_metrics})
        row = {"epoch": epoch, **train_losses, **val_metrics, **test_metrics, "selection_score": score}
        history.append(row)
        write_json(out_dir / "metrics" / "latest_metrics.json", row)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(model, out_dir / "checkpoints" / "best.pt", cfg=cfg, epoch=epoch, metrics=row)
            write_channel_reports(out_dir / "channel_reports" / "best_subject_val", val_outputs, channel_names)
            write_channel_reports(out_dir / "channel_reports" / "best_subject_test", test_outputs, channel_names)
        save_checkpoint(model, out_dir / "checkpoints" / "last.pt", cfg=cfg, epoch=epoch, metrics=row)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
        print_epoch_gate_summary(row)
        if patience > 0 and best_epoch > 0 and epoch - best_epoch >= patience:
            break

    write_json(out_dir / "metrics" / "history.json", {"history": history, "best_epoch": best_epoch, "best_score": best_score})
    print(json.dumps({"out_dir": str(out_dir), "best_epoch": best_epoch, "best_score": best_score}, ensure_ascii=False, indent=2), flush=True)


def make_model_config(cfg: dict, targets: KaraOneV9TargetBank, train_ds: KaraOneV10ClusteredDataset, *, eeg_len: int) -> KaraOneV10Config:
    model_cfg = cfg.get("model", {})
    return KaraOneV10Config(
        n_channels_eeg=int(model_cfg.get("n_channels_eeg", 62)),
        eeg_len=int(eeg_len),
        patch_size=int(model_cfg.get("patch_size", 32)),
        patch_stride=int(model_cfg.get("patch_stride", 16)),
        d_model=int(model_cfg.get("d_model", 128)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.15)),
        channel_dropout=float(model_cfg.get("channel_dropout", 0.10)),
        cond_dim=int(model_cfg.get("cond_dim", 32)),
        num_stages=int(train_ds.num_stages),
        num_subjects=int(train_ds.num_subjects),
        num_labels=int(train_ds.num_labels),
        semantic_dim=int(targets.semantic_dim),
        semantic_token_vocab=max(2, int(targets.semantic_token_vocab)),
        codec_dim=int(targets.codec_dim),
        channel_experts=int(model_cfg.get("channel_experts", 6)),
        channel_top_k=int(model_cfg.get("channel_top_k", 16)),
        channel_desc_dim=int(model_cfg.get("channel_desc_dim", 32)),
        channel_embedding_dim=int(model_cfg.get("channel_embedding_dim", 16)),
        moe_temperature=float(model_cfg.get("moe_temperature", 0.7)),
        shared_dim=int(model_cfg.get("shared_dim", model_cfg.get("d_model", 128))),
        domain_dim=int(model_cfg.get("domain_dim", 64)),
        transport_layers=int(model_cfg.get("transport_layers", 2)),
        transport_heads=int(model_cfg.get("transport_heads", 4)),
        heun_steps=int(model_cfg.get("heun_steps", 32)),
    )


def make_loader(dataset: KaraOneV10ClusteredDataset, *, batch_size: int, cfg: dict) -> DataLoader:
    use_sampler = bool(cfg.get("clusters", {}).get("cluster_balanced_sampler", True)) and dataset.cluster_bank.available
    if use_sampler:
        sampler = KaraOneV10ClusterBalancedBatchSampler(dataset, batch_size=batch_size, seed=int(cfg.get("train", {}).get("seed", 7)))
        return DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)


def print_run_header(
    *,
    phase: str,
    stages: tuple[str, ...],
    device: torch.device,
    out_dir: Path,
    root: Path,
    cluster_bank: KaraOneV10ClusterBank,
    train_ds: KaraOneV10ClusteredDataset,
    val_ds: KaraOneV10ClusteredDataset,
    test_ds: KaraOneV10ClusteredDataset,
    model: torch.nn.Module,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    max_steps: int | None,
    verbose: bool,
) -> None:
    payload = {
        "event": "v10_run_start",
        "phase": phase,
        "stages": list(stages),
        "device": str(device),
        "data_root": str(root),
        "out_dir": str(out_dir),
        "cluster_bank": str(cluster_bank.path) if cluster_bank.path else None,
        "cluster_bank_available": bool(cluster_bank.available),
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
        "train_subjects": sorted({entry.subject for entry in train_ds.entries}),
        "val_subjects": sorted({entry.subject for entry in val_ds.entries}),
        "test_subjects": sorted({entry.subject for entry in test_ds.entries}),
    }
    if verbose:
        payload["train_cluster_counts"] = dataset_cluster_counts(train_ds)
        payload["val_cluster_counts"] = dataset_cluster_counts(val_ds)
        payload["test_cluster_counts"] = dataset_cluster_counts(test_ds)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def print_step_log(
    *,
    epoch: int,
    step: int,
    total_steps: int,
    losses: dict[str, torch.Tensor],
    out: dict[str, torch.Tensor],
    batch: dict[str, Any],
    verbose: bool,
) -> None:
    loss_items = {key: float(value.detach().cpu()) for key, value in losses.items() if torch.is_tensor(value)}
    gate = out.get("channel_gate")
    gate_payload: dict[str, float] = {}
    if gate is not None:
        gate_detached = gate.detach()
        gate_payload = {
            "gate_mean": float(gate_detached.mean().cpu()),
            "gate_active_ratio": float((gate_detached > 1e-4).float().mean().cpu()),
            "gate_top16_mass": float(
                (
                    gate_detached.topk(k=min(16, gate_detached.shape[1]), dim=1).values.sum(dim=1)
                    / gate_detached.sum(dim=1).clamp_min(1e-8)
                )
                .mean()
                .cpu()
            ),
        }
    payload: dict[str, Any] = {
        "event": "v10_train_step",
        "epoch": int(epoch),
        "step": int(step),
        "total_steps": int(total_steps),
        "losses": loss_items,
        "channel_moe": gate_payload,
    }
    if verbose:
        payload["batch"] = {
            "subjects": _value_counts(batch.get("subject", [])),
            "labels": _value_counts(batch.get("label", [])),
            "eeg_clusters": _tensor_counts(batch.get("eeg_cluster_id")),
            "speech_clusters": _tensor_counts(batch.get("speech_cluster_id")),
            "valid_len_min": int(batch["eeg_valid_len"].min().detach().cpu()) if "eeg_valid_len" in batch else None,
            "valid_len_max": int(batch["eeg_valid_len"].max().detach().cpu()) if "eeg_valid_len" in batch else None,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def print_epoch_gate_summary(row: dict[str, Any]) -> None:
    keys = [
        "subject_val_semantic_over_zero_gain",
        "subject_val_semantic_over_mean_gain",
        "subject_val_semantic_top3_gain_over_mean",
        "subject_val_same_label_cross_subject_gain",
        "subject_val_prompt_acc",
        "subject_val_pred_std_ratio_median",
        "subject_val_pred_pairwise_corr_median",
        "subject_val_channel_gate_entropy_mean",
        "subject_val_v10_research_gate_pass",
        "selection_score",
    ]
    print(json.dumps({"event": "v10_epoch_gate_summary", **row_gate_summary(row)}, ensure_ascii=False, indent=2), flush=True)


def dataset_cluster_counts(dataset: KaraOneV10ClusteredDataset) -> dict[str, dict[str, int]]:
    out = {"eeg_cluster": {}, "speech_cluster": {}, "subjects": {}, "labels": {}}
    for entry in dataset.entries:
        cluster = dataset.cluster_bank.lookup(entry.subject, entry.stage, entry.trial_index)
        _inc(out["eeg_cluster"], str(int(cluster["eeg_cluster_id"])))
        _inc(out["speech_cluster"], str(int(cluster["speech_cluster_id"])))
        _inc(out["subjects"], entry.subject)
        _inc(out["labels"], entry.label)
    return out


def _inc(counts: dict[str, int], key: str) -> None:
    counts[key] = int(counts.get(key, 0)) + 1


def _tensor_counts(value: Any) -> dict[str, int]:
    if value is None or not torch.is_tensor(value):
        return {}
    arr = value.detach().cpu().view(-1).tolist()
    return _value_counts([str(int(item)) for item in arr])


def _value_counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if torch.is_tensor(values):
        iterable = [str(int(item)) for item in values.detach().cpu().view(-1).tolist()]
    else:
        iterable = [str(item) for item in values]
    for item in iterable:
        counts[item] = int(counts.get(item, 0)) + 1
    return counts


def dataset_target_bank(dataset: KaraOneV10ClusteredDataset) -> dict[str, Any]:
    rows = [dataset[i] for i in range(len(dataset))]
    return outputs_to_v10_bank(
        {
            "target": np.stack([row["semantic_summary"].numpy() for row in rows], axis=0),
            "labels": [str(row["label"]) for row in rows],
            "subjects": [str(row["subject"]) for row in rows],
            "speech_cluster_id": np.asarray([int(row["speech_cluster_id"].item()) for row in rows], dtype=np.int64),
            "eeg_cluster_id": np.asarray([int(row["eeg_cluster_id"].item()) for row in rows], dtype=np.int64),
        }
    )


def selection_score(metrics: dict[str, Any]) -> float:
    score = v10_selection_score(metrics, prefix="subject_val")
    if float(metrics.get("subject_test_semantic_over_zero_gain", 0.0)) < 0.0:
        score -= 0.05
    if float(metrics.get("subject_test_same_label_cross_subject_gain", 0.0)) < 0.0:
        score -= 0.05
    return float(score)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def save_checkpoint(model: torch.nn.Module, path: Path, *, cfg: dict, epoch: int, metrics: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    torch.save(
        {"model": model.state_dict(), "config": cfg, "epoch": epoch, "metrics": metrics, "model_kind": "karaone_v10_final_clustered_channel_moe_flow"},
        path,
    )


def load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    payload = torch.load(path, map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(json.dumps({"checkpoint": str(path), "missing": missing, "unexpected": unexpected}, ensure_ascii=False, indent=2), flush=True)


def freeze_except_transport(model: KaraOneV10ClusteredChannelMoEFlow) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("transport.")


def progress(iterable, *, total: int, desc: str):
    if tqdm is None or os.environ.get("DISABLE_TQDM", "0") == "1":
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    main()

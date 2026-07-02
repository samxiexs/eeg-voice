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
from src.karaone_v11.data import KaraOneV10ClusterBalancedBatchSampler, KaraOneV10ClusterBank, KaraOneV11Dataset, KaraOneV11TokenBank, load_channel_names, outputs_to_token_bank
from src.karaone_v11.eval import collect_v11_outputs, compute_v11_metrics, row_gate_summary, v11_selection_score, write_channel_reports
from src.karaone_v11.losses import compute_v11_alignment_losses, compute_v11_codec_losses, compute_v11_pretrain_losses
from src.karaone_v11.model import KaraOneV11Config, KaraOneV11TokenGenerator
from src.utils import count_parameters, count_trainable_parameters, ensure_dir, load_simple_yaml, resolve_bundle_path, set_seed, write_json

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KaraOne v11 tokenized neural speech generation stages.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone_v11.yaml"))
    parser.add_argument("--phase", choices=["pretrain", "align", "codec"], default="align")
    parser.add_argument("--stages", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-suffix", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cluster-bank", default=None)
    parser.add_argument("--token-bank", default=None)
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
    if not token_bank.available:
        print("[v11] warning: token bank missing; run scripts/build_karaone_v11_tokens.py first.", flush=True)
    train_ds = KaraOneV11Dataset(root, targets, str(cfg["data"].get("train_split", "subject_train")), cluster_bank=cluster_bank, token_bank=token_bank, stages=stages, subject_val=subject_val, subject_test=subject_test, eeg_len=eeg_len, require_codec=args.phase == "codec")
    val_ds = KaraOneV11Dataset(root, targets, "subject_val", cluster_bank=cluster_bank, token_bank=token_bank, stages=stages, subject_val=subject_val, subject_test=subject_test, eeg_len=eeg_len)
    test_ds = KaraOneV11Dataset(root, targets, "subject_test", cluster_bank=cluster_bank, token_bank=token_bank, stages=stages, subject_val=subject_val, subject_test=subject_test, eeg_len=eeg_len)
    aligner = str(args.aligner or os.environ.get("ALIGNER") or train_cfg.get("aligner", cfg.get("model", {}).get("aligner", "hybrid"))).lower()
    model_cfg = make_model_config(cfg, targets, train_ds, token_bank, eeg_len=eeg_len)
    model_cfg.aligner = aligner
    model = KaraOneV11TokenGenerator(model_cfg, codec_codebook=torch.from_numpy(token_bank.codec_codebook)).to(device)
    if args.checkpoint:
        load_checkpoint(model, resolve_bundle_path(args.checkpoint, BUNDLE_DIR))
    if args.phase == "codec" and args.freeze_encoder:
        freeze_except_codec(model)

    epochs = int(args.epochs or (codec_cfg.get("epochs", 10) if args.phase == "codec" else train_cfg.get("epochs", 50)))
    if args.smoke:
        epochs = min(epochs, 1)
        args.max_steps = args.max_steps or 2
    batch_size = int(args.batch_size or (4 if args.smoke else train_cfg.get("batch_size", 32)))
    loader = make_loader(train_ds, batch_size=batch_size, cfg=cfg)
    lr = float(codec_cfg.get("lr", train_cfg.get("lr", 3e-4)) if args.phase == "codec" else train_cfg.get("lr", 3e-4))
    weight_decay = float(codec_cfg.get("weight_decay", train_cfg.get("weight_decay", 1e-3)) if args.phase == "codec" else train_cfg.get("weight_decay", 1e-3))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)

    suffix = args.run_suffix or f"v11_{args.phase}_{aligner}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    stage_tag = "_".join(stages)
    out_root = resolve_bundle_path(cfg["output"]["root"], BUNDLE_DIR)
    out_dir = ensure_dir(out_root / f"{cfg['output'].get('prefix', 'karaone_v11')}_{args.phase}_{stage_tag}_{aligner}_{suffix}")
    ensure_dir(out_dir / "checkpoints")
    ensure_dir(out_dir / "metrics")
    ensure_dir(out_dir / "channel_reports")
    channel_names = load_channel_names(root, n_channels=int(cfg.get("model", {}).get("n_channels_eeg", 62)))
    write_json(out_dir / "run_config.json", {"phase": args.phase, "aligner": aligner, "stages": list(stages), "device": str(device), "train_n": len(train_ds), "subject_val_n": len(val_ds), "subject_test_n": len(test_ds), "cluster_bank": str(cluster_bank.path), "token_bank": str(token_bank.path), "config": cfg})
    print_run_header(args.phase, aligner, stages, device, out_dir, train_ds, val_ds, test_ds, model, batch_size, epochs, lr, weight_decay, args.max_steps, bool(args.verbose))

    train_bank = dataset_target_bank(train_ds, model, device=device, batch_size=batch_size)
    best_score = -1e9
    best_epoch = -1
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running: dict[str, list[float]] = {}
        iterator = progress(loader, total=len(loader), desc=f"v11 {args.phase} epoch {epoch}")
        for step, batch in enumerate(iterator, start=1):
            if args.max_steps and step > int(args.max_steps):
                break
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch["eeg"], batch["stage_idx"], batch["eeg_valid_len"], channel_cluster_id=batch.get("channel_cluster_id"), mask_ratio=float(train_cfg.get("mask_ratio", 0.25)) if args.phase == "pretrain" else 0.0, lambda_subject_adv=float(train_cfg.get("lambda_subject_adv", 0.10)))
            if args.phase == "pretrain":
                losses = compute_v11_pretrain_losses(out, batch, lambda_channel_balance=float(train_cfg.get("lambda_channel_balance", 0.05)), lambda_gate_sparsity=float(train_cfg.get("lambda_gate_sparsity", 0.02)), lambda_gate_entropy=float(train_cfg.get("lambda_gate_entropy", 0.02)))
            elif args.phase == "codec":
                losses = compute_v11_codec_losses(out, batch, lambda_codec_token_ce=float(codec_cfg.get("lambda_codec_token_ce", 1.0)), lambda_codec_latent=float(codec_cfg.get("lambda_codec_latent", 0.5)), lambda_semantic_guard=float(codec_cfg.get("lambda_semantic_guard", 0.2)), lambda_boundary_continuity=float(codec_cfg.get("lambda_boundary_continuity", 0.05)))
            else:
                losses = compute_v11_alignment_losses(out, batch, aligner=aligner, **loss_kwargs(train_cfg))
            losses["total"].backward()
            clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            optimizer.step()
            for key, value in losses.items():
                running.setdefault(key, []).append(float(value.detach().cpu()))
            if int(args.log_interval or 0) > 0 and (step == 1 or step % int(args.log_interval) == 0):
                print_step_log(epoch, step, len(loader), losses, out, batch, bool(args.verbose))

        train_losses = {f"train_{key}": float(np.mean(values)) for key, values in running.items()}
        val_outputs = collect_v11_outputs(model, val_ds, device=device, batch_size=batch_size)
        test_outputs = collect_v11_outputs(model, test_ds, device=device, batch_size=batch_size)
        val_metrics = compute_v11_metrics(val_outputs, train_bank=train_bank, prefix="subject_val")
        test_metrics = compute_v11_metrics(test_outputs, train_bank=train_bank, prefix="subject_test")
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
        print(json.dumps({"event": "v11_epoch_gate_summary", **row_gate_summary(row)}, ensure_ascii=False, indent=2), flush=True)

    write_json(out_dir / "metrics" / "history.json", {"history": history, "best_epoch": best_epoch, "best_score": best_score})
    print(json.dumps({"out_dir": str(out_dir), "best_epoch": best_epoch, "best_score": best_score}, ensure_ascii=False, indent=2), flush=True)


def make_model_config(cfg: dict, targets: KaraOneV9TargetBank, train_ds: KaraOneV11Dataset, token_bank: KaraOneV11TokenBank, *, eeg_len: int) -> KaraOneV11Config:
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    return KaraOneV11Config(
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
    )


def make_loader(dataset: KaraOneV11Dataset, *, batch_size: int, cfg: dict) -> DataLoader:
    use_sampler = bool(cfg.get("clusters", {}).get("cluster_balanced_sampler", True)) and dataset.cluster_bank.available
    if use_sampler:
        sampler = KaraOneV10ClusterBalancedBatchSampler(dataset.base, batch_size=batch_size, seed=int(cfg.get("train", {}).get("seed", 11)))
        return DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)


def loss_kwargs(train_cfg: dict[str, Any]) -> dict[str, float]:
    keys = [
        "lambda_token_ce", "lambda_token_ctc", "lambda_clip", "lambda_soft_ot", "lambda_perceiver",
        "lambda_prompt", "lambda_prompt_balanced", "lambda_prosody", "lambda_cross_subject",
        "lambda_same_label_pull", "lambda_zero_margin", "lambda_mean_margin", "lambda_pairwise_decorrelation",
        "lambda_subject_adv", "lambda_channel_balance", "lambda_gate_sparsity", "lambda_gate_entropy",
        "lambda_gate_consistency", "semantic_margin", "temperature",
    ]
    return {key: float(train_cfg[key]) for key in keys if key in train_cfg}


def dataset_target_bank(dataset: KaraOneV11Dataset, model: KaraOneV11TokenGenerator, *, device: torch.device, batch_size: int) -> dict[str, Any]:
    outputs = collect_v11_outputs(model, dataset, device=device, batch_size=batch_size)
    bank = outputs_to_token_bank(outputs)
    bank.update({"target": outputs["target"], "labels": outputs["labels"], "subjects": outputs["subjects"]})
    return bank


def selection_score(metrics: dict[str, Any]) -> float:
    score = v11_selection_score(metrics, prefix="subject_val")
    if float(metrics.get("subject_test_semantic_token_top3_gain_over_prior", 0.0)) < 0.0:
        score -= 0.05
    if float(metrics.get("subject_test_token_retrieval_cross_subject_gain", 0.0)) < 0.0:
        score -= 0.05
    return float(score)


def print_run_header(phase, aligner, stages, device, out_dir, train_ds, val_ds, test_ds, model, batch_size, epochs, lr, weight_decay, max_steps, verbose):
    payload = {
        "event": "v11_run_start",
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
        "train_subjects": sorted({entry.subject for entry in train_ds.entries}),
        "val_subjects": sorted({entry.subject for entry in val_ds.entries}),
        "test_subjects": sorted({entry.subject for entry in test_ds.entries}),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def print_step_log(epoch: int, step: int, total_steps: int, losses: dict[str, torch.Tensor], out: dict[str, torch.Tensor], batch: dict[str, Any], verbose: bool) -> None:
    payload: dict[str, Any] = {
        "event": "v11_train_step",
        "epoch": int(epoch),
        "step": int(step),
        "total_steps": int(total_steps),
        "losses": {key: float(value.detach().cpu()) for key, value in losses.items() if torch.is_tensor(value)},
        "channel_moe": {
            "gate_mean": float(out["channel_gate"].detach().mean().cpu()),
            "gate_active_ratio": float((out["channel_gate"].detach() > 1e-4).float().mean().cpu()),
        },
    }
    if verbose:
        payload["batch"] = {"subjects": value_counts(batch.get("subject", [])), "labels": value_counts(batch.get("label", []))}
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def value_counts(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in values:
        out[str(item)] = int(out.get(str(item), 0)) + 1
    return out


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def save_checkpoint(model: torch.nn.Module, path: Path, *, cfg: dict, epoch: int, metrics: dict[str, Any], aligner: str) -> None:
    ensure_dir(path.parent)
    torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "metrics": metrics, "aligner": aligner, "model_kind": "karaone_v11_tokenized_generation"}, path)


def load_checkpoint(model: torch.nn.Module, path: Path) -> Any:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(json.dumps({"checkpoint": str(path), "missing": missing, "unexpected": unexpected}, ensure_ascii=False, indent=2), flush=True)
    return payload


def freeze_except_codec(model: KaraOneV11TokenGenerator) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("codec_token_head")


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

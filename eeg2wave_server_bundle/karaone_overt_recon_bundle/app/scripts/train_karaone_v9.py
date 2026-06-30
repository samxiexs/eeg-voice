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

from src.karaone_v9.data import KaraOneV9Dataset, KaraOneV9TargetBank
from src.karaone_v9.eval import collect_v9_outputs, compute_v9_metrics, outputs_to_bank
from src.karaone_v9.losses import compute_v9_alignment_losses, compute_v9_pretrain_losses, compute_v9_transport_losses
from src.karaone_v9.model import KaraOneV9Config, KaraOneV9NeuralSemanticTransport
from src.utils import ensure_dir, load_simple_yaml, resolve_bundle_path, set_seed, write_json

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KaraOne v9 neural semantic transport stages.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone_v9.yaml"))
    parser.add_argument("--phase", choices=["pretrain", "align", "transport"], default="align")
    parser.add_argument("--stages", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-suffix", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
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
    train_split = str(cfg["data"].get("train_split", "subject_train"))
    require_codec = args.phase == "transport"
    train_ds = KaraOneV9Dataset(
        root,
        targets,
        train_split,
        stages=stages,
        subject_val=subject_val,
        subject_test=subject_test,
        eeg_len=eeg_len,
        require_codec=require_codec,
    )
    val_ds = KaraOneV9Dataset(root, targets, "subject_val", stages=stages, subject_val=subject_val, subject_test=subject_test, eeg_len=eeg_len)
    test_ds = KaraOneV9Dataset(root, targets, "subject_test", stages=stages, subject_val=subject_val, subject_test=subject_test, eeg_len=eeg_len)
    model = KaraOneV9NeuralSemanticTransport(make_model_config(cfg, targets, train_ds, eeg_len=eeg_len)).to(device)
    if args.checkpoint:
        load_checkpoint(model, resolve_bundle_path(args.checkpoint, BUNDLE_DIR))
    if args.phase == "transport" and args.freeze_encoder:
        if not args.checkpoint:
            print("[v9] warning: --freeze-encoder was set for transport without --checkpoint; this is only meaningful for smoke tests.")
        freeze_except_transport(model)

    epochs = int(args.epochs or (transport_cfg.get("epochs", 10) if args.phase == "transport" else train_cfg.get("epochs", 20)))
    if args.smoke:
        epochs = min(epochs, 1)
        args.max_steps = args.max_steps or 2
    batch_size = int(args.batch_size or (4 if args.smoke else train_cfg.get("batch_size", 32)))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    lr = float(transport_cfg.get("lr", train_cfg.get("lr", 3e-4)) if args.phase == "transport" else train_cfg.get("lr", 3e-4))
    weight_decay = float(transport_cfg.get("weight_decay", train_cfg.get("weight_decay", 1e-3)) if args.phase == "transport" else train_cfg.get("weight_decay", 1e-3))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)

    suffix = args.run_suffix or f"v9_{args.phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    stage_tag = "_".join(stages)
    out_root = resolve_bundle_path(cfg["output"]["root"], BUNDLE_DIR)
    out_dir = ensure_dir(out_root / f"{cfg['output'].get('prefix', 'karaone_v9')}_{args.phase}_{stage_tag}_{suffix}")
    ensure_dir(out_dir / "checkpoints")
    ensure_dir(out_dir / "metrics")
    write_json(
        out_dir / "run_config.json",
        {
            "phase": args.phase,
            "stages": list(stages),
            "device": str(device),
            "train_n": len(train_ds),
            "subject_val_n": len(val_ds),
            "subject_test_n": len(test_ds),
            "config": cfg,
        },
    )

    train_bank = dataset_target_bank(train_ds)
    best_score = -1e9
    best_epoch = -1
    history: list[dict[str, Any]] = []
    patience = int(train_cfg.get("early_stop_patience", 0))

    for epoch in range(1, epochs + 1):
        model.train()
        running: dict[str, list[float]] = {}
        iterator = progress(loader, total=len(loader), desc=f"v9 {args.phase} epoch {epoch}")
        for step, batch in enumerate(iterator, start=1):
            if args.max_steps and step > int(args.max_steps):
                break
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if args.phase == "pretrain":
                out = model(batch["eeg"], batch["stage_idx"], batch["eeg_valid_len"], mask_ratio=float(train_cfg.get("mask_ratio", 0.25)))
                losses = compute_v9_pretrain_losses(out)
            elif args.phase == "transport":
                out = model(
                    batch["eeg"],
                    batch["stage_idx"],
                    batch["eeg_valid_len"],
                    mask_ratio=0.0,
                    lambda_subject_adv=0.0,
                    codec_seq=batch["codec_seq"],
                )
                losses = compute_v9_transport_losses(
                    out,
                    batch,
                    lambda_flow=float(transport_cfg.get("lambda_flow", 1.0)),
                    lambda_condition_semantic=float(transport_cfg.get("lambda_condition_semantic", 0.2)),
                )
            else:
                lambda_subject_adv = float(train_cfg.get("lambda_subject_adv", 0.10))
                out = model(
                    batch["eeg"],
                    batch["stage_idx"],
                    batch["eeg_valid_len"],
                    mask_ratio=0.0,
                    lambda_subject_adv=lambda_subject_adv,
                )
                losses = compute_v9_alignment_losses(
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
                )
            losses["total"].backward()
            clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            optimizer.step()
            for key, value in losses.items():
                running.setdefault(key, []).append(float(value.detach().cpu()))

        train_losses = {f"train_{key}": float(np.mean(values)) for key, values in running.items()}
        val_outputs = collect_v9_outputs(model, val_ds, device=device, batch_size=batch_size)
        test_outputs = collect_v9_outputs(model, test_ds, device=device, batch_size=batch_size)
        val_metrics = compute_v9_metrics(val_outputs, train_bank=train_bank, prefix="subject_val")
        test_metrics = compute_v9_metrics(test_outputs, train_bank=train_bank, prefix="subject_test")
        score = selection_score(val_metrics)
        row = {"epoch": epoch, **train_losses, **val_metrics, **test_metrics, "selection_score": score}
        history.append(row)
        write_json(out_dir / "metrics" / "latest_metrics.json", row)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(model, out_dir / "checkpoints" / "best.pt", cfg=cfg, epoch=epoch, metrics=row)
        save_checkpoint(model, out_dir / "checkpoints" / "last.pt", cfg=cfg, epoch=epoch, metrics=row)
        print(json.dumps(row, ensure_ascii=False, indent=2))
        if patience > 0 and best_epoch > 0 and epoch - best_epoch >= patience:
            break

    write_json(out_dir / "metrics" / "history.json", {"history": history, "best_epoch": best_epoch, "best_score": best_score})
    print(json.dumps({"out_dir": str(out_dir), "best_epoch": best_epoch, "best_score": best_score}, ensure_ascii=False, indent=2))


def make_model_config(cfg: dict, targets: KaraOneV9TargetBank, train_ds: KaraOneV9Dataset, *, eeg_len: int) -> KaraOneV9Config:
    model_cfg = cfg.get("model", {})
    return KaraOneV9Config(
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
        transport_layers=int(model_cfg.get("transport_layers", 2)),
        transport_heads=int(model_cfg.get("transport_heads", 4)),
    )


def dataset_target_bank(dataset: KaraOneV9Dataset) -> dict[str, Any]:
    rows = [dataset[i] for i in range(len(dataset))]
    return outputs_to_bank(
        {
            "target": np.stack([row["semantic_summary"].numpy() for row in rows], axis=0),
            "labels": [str(row["label"]) for row in rows],
            "subjects": [str(row["subject"]) for row in rows],
        }
    )


def selection_score(metrics: dict[str, Any]) -> float:
    return float(metrics.get("subject_val_semantic_over_mean_gain", 0.0)) + float(
        metrics.get("subject_val_semantic_top3_gain_over_mean", 0.0)
    ) + 0.25 * float(metrics.get("subject_val_same_label_cross_subject_gain", 0.0)) - 0.05 * float(
        metrics.get("subject_val_subject_leakage_acc", 0.0)
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def save_checkpoint(model: torch.nn.Module, path: Path, *, cfg: dict, epoch: int, metrics: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "metrics": metrics, "model_kind": "karaone_v9_neural_semantic_transport"}, path)


def load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    payload = torch.load(path, map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(json.dumps({"checkpoint": str(path), "missing": missing, "unexpected": unexpected}, ensure_ascii=False, indent=2))


def freeze_except_transport(model: KaraOneV9NeuralSemanticTransport) -> None:
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

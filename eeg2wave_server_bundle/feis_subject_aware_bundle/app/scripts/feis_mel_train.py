"""Train the FEIS EEG-only mel-alignment model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

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

from src.feis_mel.data import FEISMelDataset, assert_mel_identity_free_keys
from src.feis_mel.eval import evaluate_feis_mel
from src.feis_mel.losses import compute_feis_mel_losses
from src.feis_mel.model import FEISEEGToMel, FEISMelConfig
from src.feis_mel.targets import MelLabelTargets
from src.utils import ensure_dir, load_simple_yaml, resolve_bundle_path, set_seed, write_json


def parse_args():
    p = argparse.ArgumentParser(description="Train FEIS EEG-only mel alignment.")
    p.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "feis_mel_align.yaml"))
    p.add_argument("--stage", default=None)
    p.add_argument("--stages", default=None, help="Compatibility alias; pass one stage.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--run-suffix", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    return p.parse_args()


def _resolve_stage(args, cfg) -> str:
    raw = args.stage or args.stages or cfg.get("data", {}).get("stage", "thinking")
    parts = [part.strip() for chunk in str(raw).split(",") for part in chunk.split() if part.strip()]
    if len(parts) != 1:
        raise ValueError(f"FEIS mel training is stage-specific; got {raw!r}")
    return parts[0]


def _selection_score(metrics: dict) -> float:
    return (
        4.0 * (metrics["content_top1"] - metrics["content_chance"])
        + metrics["retrieval_top1"]
        + metrics["mel_PCC"]
        + (metrics["mean_mel_baseline_dtw"] - metrics["pred_to_label_bank_dtw"])
    )


def _write_figures(run_dir: Path) -> None:
    history = run_dir / "metrics" / "history.csv"
    if not history.exists():
        return
    with history.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return
    fig_dir = ensure_dir(run_dir / "figures")
    x = np.asarray([float(row["epoch"]) for row in rows])
    specs = [
        ("training_curves.png", ["train_total", "train_mel_dtw", "train_content_ce", "train_contrastive"], "Training losses"),
        ("content_accuracy.png", ["train_content_acc", "val_content_top1", "val_retrieval_top1", "val_mel_PCC"], "Content and mel metrics"),
    ]
    for name, keys, title in specs:
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        for key in keys:
            if key not in rows[0]:
                continue
            ax.plot(x, [float(row[key]) for row in rows], marker="o" if len(rows) <= 12 else None, label=key)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
        fig.savefig(fig_dir / name, dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    set_seed(int(cfg["train"].get("seed", 7)))
    stage = _resolve_stage(args, cfg)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    target_path = resolve_bundle_path(cfg["data"]["target_cache"], BUNDLE_DIR)
    if not target_path.exists():
        raise FileNotFoundError(f"Missing mel target cache: {target_path}. Run scripts/build_feis_mel_targets.py first.")
    targets = MelLabelTargets(target_path)
    common = dict(
        data_root=resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR),
        targets=targets,
        stage=stage,
        include_anomalous=bool(cfg["data"].get("include_anomalous", False)),
    )
    train_ds = FEISMelDataset(split="train", **common)
    val_ds = FEISMelDataset(split="val_seen", **common)
    test_seen = FEISMelDataset(split="test_seen", **common)
    test_holdout = FEISMelDataset(split="test_holdout", **common)
    print(f"[data] EEG-only mel stage={stage} train={len(train_ds)} val={len(val_ds)} test_holdout={len(test_holdout)}")

    cm = cfg["model"]
    model = FEISEEGToMel(FEISMelConfig(
        n_channels_eeg=int(cm.get("n_channels_eeg", 14)),
        d_model=int(cm.get("d_model", 192)),
        cond_dim=int(cm.get("cond_dim", 16)),
        num_labels=targets.num_labels,
        target_steps=targets.T,
        mel_dim=targets.D,
        num_blocks=int(cm.get("num_blocks", 5)),
        kernel_size=int(cm.get("kernel_size", 5)),
        num_heads=int(cm.get("num_heads", 6)),
        num_cross_layers=int(cm.get("num_cross_layers", 2)),
        ff_mult=int(cm.get("ff_mult", 4)),
        channel_moe=bool(cm.get("channel_moe", True)),
        moe_num_experts=int(cm.get("moe_num_experts", 4)),
        moe_top_k=int(cm.get("moe_top_k", 2)),
        channel_dropout=float(cm.get("channel_dropout", 0.2)),
        dropout=float(cm.get("dropout", 0.2)),
    )).to(device)
    print(f"[model] EEG-only mel cross-attention, channel_moe={model.cfg.channel_moe}, inputs=['eeg']")

    epochs = args.epochs or int(cfg["train"].get("epochs", 40))
    batch_size = int(cfg["train"].get("batch_size", 64))
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        drop_last=True,
    )
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 3e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-3)),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    label_prototypes = torch.from_numpy(targets.label_prototypes).float().to(device)
    loss_cfg = cfg["loss"]
    loss_kwargs = dict(
        use_dtw=bool(loss_cfg.get("dtw", True)),
        dtw_band=int(loss_cfg.get("dtw_band", 10)),
        dtw_top_k=int(loss_cfg.get("dtw_top_k", 3)),
        lambda_dtw=float(loss_cfg.get("lambda_dtw", 1.0)),
        lambda_content_ce=float(loss_cfg.get("lambda_content_ce", 0.75)),
        lambda_contrastive=float(loss_cfg.get("lambda_contrastive", 0.25)),
        lambda_log_rms=float(loss_cfg.get("lambda_log_rms", 0.1)),
        lambda_moe_load_balance=float(loss_cfg.get("lambda_moe_load_balance", 0.05)),
        lambda_moe_sparsity=float(loss_cfg.get("lambda_moe_sparsity", 0.005)),
        lambda_moe_route_entropy=float(loss_cfg.get("lambda_moe_route_entropy", 0.01)),
        lambda_moe_cluster=float(loss_cfg.get("lambda_moe_cluster", 0.05)),
        contrast_temperature=float(loss_cfg.get("contrast_temperature", 0.07)),
    )

    run_name = f"feis_mel_{stage}_{args.run_suffix or 'v1'}"
    run_dir = resolve_bundle_path(cfg["output"]["root"], BUNDLE_DIR) / run_name
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "metrics")
    hist_csv = run_dir / "metrics" / "history.csv"
    hist_jsonl = run_dir / "metrics" / "history.jsonl"
    hist_jsonl.write_text("", encoding="utf-8")
    fields = [
        "epoch", "train_total", "train_mel_dtw", "train_content_ce", "train_content_acc",
        "train_contrastive", "train_retrieval_acc", "train_moe_gate_mean",
        "val_content_top1", "val_retrieval_top1", "val_mel_PCC", "val_pred_to_label_bank_dtw",
        "val_mean_mel_baseline_dtw", "val_score",
    ]
    with hist_csv.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(fields)

    def save_ckpt(path: Path, score: float) -> None:
        torch.save({
            "model_state": model.state_dict(),
            "model_config": vars(model.cfg),
            "label_vocab": targets.label_vocab,
            "target_mean": targets.target_mean,
            "target_std": targets.target_std,
            "label_prototypes": targets.label_prototypes,
            "target_cache": str(target_path),
            "stage": stage,
            "selection_score": float(score),
            "identity_free": True,
            "method": "eeg_only_mel_alignment",
            "model_inputs": ["eeg"],
            "target_kind": "mel",
        }, path)

    best_score = -1e9
    best_path = run_dir / "checkpoints" / "best.pt"
    for epoch in range(epochs):
        model.train()
        agg: dict[str, float] = {}
        seen, steps = 0, 0
        for batch in loader:
            assert_mel_identity_free_keys(tuple(batch.keys()))
            eeg = batch["eeg"].to(device)
            out = model(eeg)
            losses = compute_feis_mel_losses(
                out,
                batch["target_bank"].to(device),
                batch["label_idx"].to(device),
                batch["target_log_rms"].to(device),
                label_prototypes,
                **loss_kwargs,
            )
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 1.0)))
            opt.step()
            bsz = int(eeg.shape[0])
            seen += bsz
            for key, value in losses.items():
                if torch.isfinite(value).all():
                    agg[key] = agg.get(key, 0.0) + float(value.detach()) * bsz
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
        sched.step()
        train_metrics = {key: value / max(seen, 1) for key, value in agg.items()}
        val = evaluate_feis_mel(model, val_ds, targets, device=device, batch_size=batch_size, dtw_band=int(loss_cfg.get("dtw_band", 10)))
        score = _selection_score(val)
        print(
            f"epoch {epoch:03d} | total {train_metrics['total']:.3f} mel {train_metrics['mel_dtw']:.3f} "
            f"content {train_metrics['content_acc']:.3f} retr {train_metrics['retrieval_acc']:.3f} | "
            f"val top1 {val['content_top1']:.3f} retr {val['retrieval_top1']:.3f} "
            f"pcc {val['mel_PCC']:.3f} pred_dtw {val['pred_to_label_bank_dtw']:.3f} "
            f"mean_dtw {val['mean_mel_baseline_dtw']:.3f} score {score:+.3f}"
        )
        with hist_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"epoch": epoch, "train": train_metrics, "val": val, "selection_score": score}) + "\n")
        with hist_csv.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([
                epoch,
                train_metrics["total"],
                train_metrics["mel_dtw"],
                train_metrics["content_ce"],
                train_metrics["content_acc"],
                train_metrics["contrastive"],
                train_metrics["retrieval_acc"],
                train_metrics.get("moe_gate_mean", 0.0),
                val["content_top1"],
                val["retrieval_top1"],
                val["mel_PCC"],
                val["pred_to_label_bank_dtw"],
                val["mean_mel_baseline_dtw"],
                score,
            ])
        if score > best_score:
            best_score = score
            save_ckpt(best_path, score)
        if args.max_steps:
            break
    save_ckpt(run_dir / "checkpoints" / "last.pt", best_score)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    best_model = FEISEEGToMel(FEISMelConfig(**ckpt["model_config"])).to(device)
    best_model.load_state_dict(ckpt["model_state"], strict=True)
    res = {
        "selection": {
            "criterion": "content above chance + retrieval + mel PCC + mean-baseline gain",
            "best_score": best_score,
            "identity_free": True,
            "model_inputs": ["eeg"],
        },
        "test_seen": evaluate_feis_mel(best_model, test_seen, targets, device=device, batch_size=batch_size, dtw_band=int(loss_cfg.get("dtw_band", 10))),
        "test_holdout": evaluate_feis_mel(best_model, test_holdout, targets, device=device, batch_size=batch_size, dtw_band=int(loss_cfg.get("dtw_band", 10))),
    }
    write_json(run_dir / "metrics" / "test_metrics.json", res)
    write_json(run_dir / "mel_alignment_qc.json", {
        "content_gate": "content_top1 must exceed chance before claiming generation",
        "resting_negative_control": "resting should not match speaking/thinking if EEG content is used",
        "test_holdout": res["test_holdout"],
    })
    _write_figures(run_dir)
    print(f"[done] {run_dir}")


if __name__ == "__main__":
    main()


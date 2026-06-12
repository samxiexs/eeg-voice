"""Train EEG-only direct EEG -> EnCodec-latent speech reconstruction.

This is the no-subject-id route. The model receives only EEG and stage index;
subject is used only as dataset metadata and for post-hoc diagnostics.

  python scripts/direct_train.py --config configs/direct_eeg2speech.yaml
  python scripts/direct_train.py --config configs/direct_eeg2speech.yaml --max-steps 2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.direct_eeg2speech.eval import evaluate_direct
from src.direct_eeg2speech.losses import compute_direct_losses
from src.direct_eeg2speech.model import DirectEEG2Speech, DirectEEG2SpeechConfig
from src.feis_factored.data import FactoredFEISDataset
from src.feis_factored.targets import FactoredTargets
from src.utils import ensure_dir, load_simple_yaml, resolve_bundle_path, set_seed, write_json


def parse_args():
    p = argparse.ArgumentParser(description="Train EEG-only direct EEG-to-speech model.")
    p.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "direct_eeg2speech.yaml"))
    p.add_argument("--stages", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--run-suffix", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    return p.parse_args()


def _mean_latent_norm(targets: FactoredTargets) -> np.ndarray:
    return ((targets.global_mean_raw_seq() - targets.target_mean.reshape(1, -1))
            / targets.target_std.reshape(1, -1)).astype(np.float32)


def _selection_score(metrics: dict) -> float:
    # Higher is better: content above chance + latent fidelity + anti-collapse.
    content_gain = metrics["content_top1"] - metrics["content_chance"]
    collapse_penalty = max(0.0, 0.25 - metrics["pred_std_ratio_median"])
    corr_penalty = max(0.0, metrics["pred_pairwise_corr_median"] - 0.25)
    return (
        4.0 * content_gain
        + metrics["latent_recon_cos"]
        + 0.1 * metrics["mean_latent_distance"]
        - collapse_penalty
        - corr_penalty
    )


def main():
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    ct, cm = cfg["train"], cfg["model"]
    set_seed(int(ct.get("seed", 7)))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    stages = tuple(s.strip() for s in (args.stages or cfg["data"].get("stages", "stimuli,thinking")).split(",") if s.strip())
    targets = FactoredTargets(resolve_bundle_path(cfg["data"]["target_cache"], BUNDLE_DIR))
    common = dict(
        data_root=resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR),
        targets=targets,
        stages=stages,
        include_anomalous=bool(cfg["data"].get("include_anomalous", False)),
        holdout_offset=int(cfg["data"].get("holdout_offset", 0)),
        holdout_random=bool(cfg["data"].get("holdout_random", True)),
        seed=int(ct.get("seed", 7)),
    )
    train_ds = FactoredFEISDataset(split="train", **common)
    val_seen = FactoredFEISDataset(split="val_seen", **common)
    test_seen = FactoredFEISDataset(split="test_seen", **common)
    test_holdout = FactoredFEISDataset(split="test_holdout", **common)
    print(f"[data] EEG-only stages={stages} train={len(train_ds)} val={len(val_seen)} "
          f"test_seen={len(test_seen)} test_holdout={len(test_holdout)}")

    model = DirectEEG2Speech(DirectEEG2SpeechConfig(
        n_channels_eeg=int(cm.get("n_channels_eeg", 14)),
        d_model=int(cm.get("d_model", 256)),
        cond_dim=int(cm.get("cond_dim", 32)),
        num_labels=train_ds.num_labels,
        num_stages=train_ds.num_stages,
        target_steps=targets.T,
        target_dim=targets.D,
        num_blocks=int(cm.get("num_blocks", 5)),
        kernel_size=int(cm.get("kernel_size", 5)),
        channel_dropout=float(cm.get("channel_dropout", 0.2)),
        dropout=float(cm.get("dropout", 0.2)),
        num_transformer_layers=int(cm.get("num_transformer_layers", 3)),
        num_heads=int(cm.get("num_heads", 8)),
        ff_mult=int(cm.get("ff_mult", 4)),
    )).to(device)

    epochs = args.epochs or int(ct.get("epochs", 100))
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(ct.get("lr", 3e-4)),
        weight_decay=float(ct.get("weight_decay", 1e-3)),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    bs = int(ct.get("batch_size", 64))
    loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=int(ct.get("num_workers", 0)),
        drop_last=True,
    )
    lam = dict(
        lambda_recon_cos=float(ct.get("lambda_recon_cos", 1.0)),
        lambda_recon_smoothl1=float(ct.get("lambda_recon_smoothl1", 0.5)),
        lambda_delta=float(ct.get("lambda_delta", 0.25)),
        lambda_content_ce=float(ct.get("lambda_content_ce", 0.75)),
        lambda_log_rms=float(ct.get("lambda_log_rms", 0.2)),
        lambda_std=float(ct.get("lambda_std", 0.2)),
        lambda_diversity=float(ct.get("lambda_diversity", 0.2)),
        lambda_mean_margin=float(ct.get("lambda_mean_margin", 0.1)),
        mean_margin=float(ct.get("mean_margin", 0.25)),
    )
    mean_latent = torch.from_numpy(_mean_latent_norm(targets)).to(device)

    run = f"direct_{'_'.join(stages)}_{args.run_suffix or 'v1'}"
    run_dir = resolve_bundle_path(cfg["output"]["root"], BUNDLE_DIR) / run
    ensure_dir(run_dir / "checkpoints"); ensure_dir(run_dir / "metrics")
    hist_jsonl = run_dir / "metrics" / "history.jsonl"
    hist_csv = run_dir / "metrics" / "history.csv"
    hist_jsonl.write_text("", encoding="utf-8")
    csv_fields = ["epoch", "train_total", "train_content_acc", "train_recon_cos",
                  "train_std_ratio", "train_mean_distance", "val_top1", "val_recon_cos",
                  "val_std_ratio", "val_pred_corr", "val_score"]
    with hist_csv.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(csv_fields)

    def save_ckpt(path: Path, score: float):
        torch.save({
            "model_state": model.state_dict(),
            "model_config": vars(model.cfg),
            "label_vocab": train_ds.label_vocab,
            "stages": list(stages),
            "target_mean": targets.target_mean,
            "target_std": targets.target_std,
            "default_decoder_scales": targets.default_decoder_scales,
            "holdout_offset": int(cfg["data"].get("holdout_offset", 0)),
            "holdout_random": bool(cfg["data"].get("holdout_random", True)),
            "selection_score": float(score),
            "no_subject_id": True,
        }, path)

    best_score = -1e9
    best_path = run_dir / "checkpoints" / "best.pt"
    for epoch in range(epochs):
        model.train()
        agg, cnt, steps = {}, 0, 0
        for batch in loader:
            out = model(batch["eeg"].to(device), batch["stage_idx"].to(device))
            losses = compute_direct_losses(
                out,
                batch["target_seq"].to(device),
                batch["label_idx"].to(device),
                target_log_rms=batch["target_log_rms"].to(device),
                mean_latent=mean_latent,
                **lam,
            )
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            clip_grad_norm_(model.parameters(), float(ct.get("grad_clip", 1.0)))
            opt.step()
            b = batch["eeg"].shape[0]
            cnt += b
            for k, v in losses.items():
                if torch.isfinite(v).all():
                    agg[k] = agg.get(k, 0.0) + float(v.detach()) * b
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
        sched.step()
        tr = {k: v / max(cnt, 1) for k, v in agg.items()}
        ev = evaluate_direct(model, val_seen, targets, device=device, batch_size=bs)
        score = _selection_score(ev)
        print(f"epoch {epoch:03d} | total {tr['total']:.3f} content_acc "
              f"{tr['content_acc']:.3f} recon_cos {tr['recon_cos']:.3f} "
              f"std {tr['std_ratio']:.3f} mean_dist {tr.get('mean_distance',0):.3f} | "
              f"val top1 {ev['content_top1']:.3f} recon {ev['latent_recon_cos']:.3f} "
              f"std {ev['pred_std_ratio_median']:.3f} corr {ev['pred_pairwise_corr_median']:.3f} "
              f"score {score:+.3f}")
        with hist_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"epoch": epoch, "train": tr, "val": ev,
                                 "selection_score": score}) + "\n")
        with hist_csv.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([epoch, tr["total"], tr["content_acc"], tr["recon_cos"],
                                     tr["std_ratio"], tr.get("mean_distance", 0.0),
                                     ev["content_top1"], ev["latent_recon_cos"],
                                     ev["pred_std_ratio_median"], ev["pred_pairwise_corr_median"], score])
        if score > best_score:
            best_score = score
            save_ckpt(best_path, score)
        if args.max_steps:
            break

    save_ckpt(run_dir / "checkpoints" / "last.pt", best_score)

    # Final metrics are always from best.pt, not the last in-memory epoch.
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    best_model = DirectEEG2Speech(DirectEEG2SpeechConfig(**ckpt["model_config"])).to(device)
    best_model.load_state_dict(ckpt["model_state"], strict=True)
    res = {
        "selection": {
            "criterion": "composite(content above chance + recon + anti-collapse)",
            "best_score": best_score,
            "no_subject_id": True,
        },
        "test_seen": evaluate_direct(best_model, test_seen, targets, device=device, batch_size=bs),
        "test_holdout": evaluate_direct(best_model, test_holdout, targets, device=device, batch_size=bs),
    }
    write_json(run_dir / "metrics" / "test_metrics.json", res)
    for split, metrics in res.items():
        if split == "selection":
            continue
        print(f"[{split}] top1={metrics['content_top1']:.4f} chance={metrics['content_chance']:.4f} "
              f"recon={metrics['latent_recon_cos']:.3f} std={metrics['pred_std_ratio_median']:.3f} "
              f"corr={metrics['pred_pairwise_corr_median']:.3f} voice_gap="
              f"{metrics['speaker_retrieval_same_subject_gap']:.3f}")
    print(f"[done] {run_dir}")


if __name__ == "__main__":
    main()

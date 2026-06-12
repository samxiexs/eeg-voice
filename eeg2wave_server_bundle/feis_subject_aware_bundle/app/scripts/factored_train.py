"""Train the factored FEIS model (content x speaker grid).

  python scripts/factored_train.py --config configs/factored.yaml
  # stages / holdout offset overridable:
  python scripts/factored_train.py --config configs/factored.yaml --stages stimuli,thinking
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

from src.utils import ensure_dir, load_simple_yaml, resolve_bundle_path, set_seed, write_json
from src.feis_factored.data import FactoredFEISDataset
from src.feis_factored.eval import evaluate
from src.feis_factored.losses import compute_factored_losses
from src.feis_factored.model import FactoredConfig, FactoredEEG2Speech
from src.feis_factored.targets import FactoredTargets


def parse_args():
    p = argparse.ArgumentParser(description="Train factored FEIS content x speaker model.")
    p.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "factored.yaml"))
    p.add_argument("--stages", default=None, help="comma list, e.g. stimuli,thinking")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--run-suffix", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    ct, cm = cfg["train"], cfg["model"]
    set_seed(int(ct.get("seed", 7)))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    stages = tuple((args.stages or cfg["data"].get("stages", "stimuli,thinking")).split(","))
    cache = resolve_bundle_path(cfg["data"]["target_cache"], BUNDLE_DIR)
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    targets = FactoredTargets(cache)

    holdout_random = bool(cfg["data"].get("holdout_random", False))
    common = dict(data_root=root, targets=targets, stages=stages,
                  include_anomalous=bool(cfg["data"].get("include_anomalous", False)),
                  holdout_offset=int(cfg["data"].get("holdout_offset", 0)),
                  holdout_random=holdout_random)
    train_ds = FactoredFEISDataset(split="train", **common)
    val_seen = FactoredFEISDataset(split="val_seen", **common)
    test_seen = FactoredFEISDataset(split="test_seen", **common)
    test_holdout = FactoredFEISDataset(split="test_holdout", **common)
    print(f"[data] stages={stages} | train={len(train_ds)} val_seen={len(val_seen)} "
          f"test_seen={len(test_seen)} test_holdout={len(test_holdout)} | "
          f"subjects={train_ds.num_subjects} labels={train_ds.num_labels} "
          f"holdout_cells={len(train_ds.holdout_cells)} holdout_random={holdout_random}")

    model = FactoredEEG2Speech(FactoredConfig(
        n_channels_eeg=int(cm.get("n_channels_eeg", 14)),
        d_model=int(cm.get("d_model", 256)),
        cond_dim=int(cm.get("cond_dim", 32)),
        num_subjects=train_ds.num_subjects,
        num_labels=train_ds.num_labels,
        num_stages=train_ds.num_stages,
        target_steps=targets.T, target_dim=targets.D,
        content_dim=int(cm.get("content_dim", 128)),
        speaker_dim=int(cm.get("speaker_dim", 64)),
        num_blocks=int(cm.get("num_blocks", 5)),
        kernel_size=int(cm.get("kernel_size", 5)),
        channel_dropout=float(cm.get("channel_dropout", 0.2)),
        dropout=float(cm.get("dropout", 0.2)),
    )).to(device)

    epochs = args.epochs or int(ct.get("epochs", 100))
    opt = torch.optim.AdamW(model.parameters(), lr=float(ct.get("lr", 3e-4)),
                            weight_decay=float(ct.get("weight_decay", 1e-3)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    bs = int(ct.get("batch_size", 64))
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=int(ct.get("num_workers", 0)),
                        drop_last=True)

    lam = dict(lambda_supcon=float(ct.get("lambda_supcon", 1.0)),
               lambda_content_ce=float(ct.get("lambda_content_ce", 0.5)),
               lambda_proto=float(ct.get("lambda_proto", 0.5)),
               lambda_recon_cos=float(ct.get("lambda_recon_cos", 1.0)),
               lambda_recon_mse=float(ct.get("lambda_recon_mse", 0.25)),
               lambda_speaker=float(ct.get("lambda_speaker", 0.5)),
               lambda_adv=float(ct.get("lambda_adv", 0.3)),
               lambda_log_rms=float(ct.get("lambda_log_rms", 0.2)),
               lambda_std=float(ct.get("lambda_std", 0.0)),
               supcon_temperature=float(ct.get("supcon_temperature", 0.1)))

    run = f"factored_{'_'.join(stages)}_{args.run_suffix or 'v2'}"
    run_dir = resolve_bundle_path(cfg["output"]["root"], BUNDLE_DIR) / run
    ensure_dir(run_dir / "checkpoints"); ensure_dir(run_dir / "metrics")
    hist_jsonl = run_dir / "metrics" / "history.jsonl"
    hist_csv = run_dir / "metrics" / "history.csv"
    hist_jsonl.write_text("", encoding="utf-8")
    csv_fields = ["epoch", "train_total", "train_content_acc", "train_recon_cos", "train_log_rms",
                  "train_std_ratio", "train_adv_subj_acc", "val_top1", "val_zeroeeg", "val_gain"]
    with hist_csv.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(csv_fields)

    def save_ckpt(path, val_gain):
        torch.save({"model_state": model.state_dict(), "model_config": vars(model.cfg),
                    "subject_vocab": train_ds.subject_vocab, "label_vocab": train_ds.label_vocab,
                    "stages": list(stages), "target_mean": targets.target_mean,
                    "target_std": targets.target_std,
                    "default_decoder_scales": targets.default_decoder_scales,
                    "holdout_offset": int(cfg["data"].get("holdout_offset", 0)),
                    "holdout_random": holdout_random,
                    "val_content_gain": float(val_gain),
                    "no_eeg_content_gain": bool(val_gain <= 0.0)}, path)

    # selection: maximise val CONTENT GAIN (top1 - zeroeeg), NOT raw top1.
    best_gain = -1e9
    for epoch in range(epochs):
        model.train()
        agg, cnt, steps = {}, 0, 0
        for batch in loader:
            out = model(batch["eeg"].to(device), batch["subject_idx"].to(device),
                        batch["stage_idx"].to(device))
            losses = compute_factored_losses(
                out, batch["target_seq"].to(device), batch["label_idx"].to(device),
                batch["content_proto"].to(device),
                speaker_proto=batch["speaker_proto"].to(device),
                subject_idx=batch["subject_idx"].to(device),
                target_log_rms=batch["target_log_rms"].to(device), **lam)
            opt.zero_grad(); losses["total"].backward()
            clip_grad_norm_(model.parameters(), float(ct.get("grad_clip", 1.0))); opt.step()
            b = batch["eeg"].shape[0]; cnt += b
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v.detach()) * b
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
        sched.step()
        tr = {k: v / max(cnt, 1) for k, v in agg.items()}
        # model selection on VAL (held-out rep of seen cells), gain-based
        val_ds = val_seen if len(val_seen) > 0 else test_seen
        ev = evaluate(model, val_ds, targets, device=device, batch_size=bs)
        gain = ev["content_gain"]
        print(f"epoch {epoch:03d} | total {tr['total']:.3f} content_acc {tr['content_acc']:.3f} "
              f"recon_cos {tr['recon_cos']:.3f} log_rms {tr.get('log_rms_loss',0):.3f} "
              f"std_ratio {tr.get('std_ratio',0):.3f} adv {tr.get('adv_subject_acc',0):.3f} | "
              f"val top1 {ev['within_subject_content_top1']:.3f} zeroeeg "
              f"{ev['within_subject_content_top1_zeroeeg']:.3f} gain {gain:+.3f}")
        with hist_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"epoch": epoch, "train": tr,
                                 "val": {k: ev[k] for k in ("within_subject_content_top1",
                                 "within_subject_content_top1_zeroeeg", "content_gain")}}) + "\n")
        with hist_csv.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([epoch, tr["total"], tr["content_acc"], tr["recon_cos"],
                                     tr.get("log_rms_loss", 0.0), tr.get("std_ratio", 0.0),
                                     tr.get("adv_subject_acc", 0.0), ev["within_subject_content_top1"],
                                     ev["within_subject_content_top1_zeroeeg"], gain])
        if gain > best_gain:
            best_gain = gain
            save_ckpt(run_dir / "checkpoints" / "best.pt", gain)
        if args.max_steps:
            break
    # always keep a last checkpoint too
    save_ckpt(run_dir / "checkpoints" / "last.pt", best_gain)

    # final eval on both test splits (the decisive numbers)
    res = {"selection": {"criterion": "val content_gain (top1 - zeroeeg)",
                         "best_val_gain": best_gain,
                         "no_eeg_content_gain": bool(best_gain <= 0.0)},
           "test_seen": evaluate(model, test_seen, targets, device=device, batch_size=bs),
           "test_holdout": evaluate(model, test_holdout, targets, device=device, batch_size=bs)}
    for k, v in res.items():
        if k == "selection":
            print(f"[selection] best_val_gain={best_gain:+.4f} "
                  f"no_eeg_content_gain={res['selection']['no_eeg_content_gain']}")
            continue
        print(f"[{k}] top1={v['within_subject_content_top1']:.4f} zeroeeg="
              f"{v['within_subject_content_top1_zeroeeg']:.4f} gain={v['content_gain']:+.4f} "
              f"recon_cos={v['recon_cos_to_cell']:.3f} std_ratio={v['pred_std_ratio_median']:.3f} "
              f"pred_corr={v['pred_pairwise_corr_median']:.3f}")
        print(f"   by-stage:", v["content_top1_by_stage"])
    write_json(run_dir / "metrics" / "test_metrics.json", res)
    if res["selection"]["no_eeg_content_gain"]:
        print("[verdict] NO EEG CONTENT GAIN — content not decodable above zero-EEG baseline. "
              "Per V2_PLAN Stage-1 gate: stop FEIS-only model stacking; pivot to auditory dataset / pretraining.")
    print(f"[done] {run_dir}")


if __name__ == "__main__":
    main()

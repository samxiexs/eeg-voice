"""Multi-dataset v4 trainer: shared latent space, single or joint datasets.

  # FEIS only
  python scripts/v3_train_md.py --config configs/v3_multidataset.yaml --datasets feis
  # KaraOne only
  python scripts/v3_train_md.py --config configs/v3_multidataset.yaml --datasets karaone
  # Joint (shared trunk, weighted sampling)
  python scripts/v3_train_md.py --config configs/v3_multidataset.yaml --datasets feis,karaone

Batches are dataset-homogeneous (channel counts differ); when multiple datasets
are active the loop draws batches by sampling weight, all flowing through the
shared trunk + content/contrastive heads.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.utils import ensure_dir, load_simple_yaml, set_seed, write_json
from src.v3.datasets import REGISTRY, UnifiedEEGSpeechDataset, build_global_subjects
from src.v3.eval import evaluate_unified
from src.v3.losses import compute_v3_losses
from src.v3.model import DatasetHead, EEG2SpeechMD, EEG2SpeechMDConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-dataset v4 trainer.")
    p.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "v3_multidataset.yaml"))
    p.add_argument("--datasets", default=None, help="Comma list overriding config, e.g. feis,karaone")
    p.add_argument("--run-suffix", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--init-from", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    return p.parse_args()


def resolve_specs(config: dict, args: argparse.Namespace):
    names = (args.datasets.split(",") if args.datasets else config.get("datasets", ["feis"]))
    names = [n.strip() for n in names if n.strip()]
    specs = []
    for n in names:
        if n not in REGISTRY:
            raise KeyError(f"Dataset '{n}' not in REGISTRY {list(REGISTRY)}")
        specs.append(REGISTRY[n])
    return specs, names


def main() -> None:
    args = parse_args()
    config = load_simple_yaml(args.config)
    cfg_t, cfg_m = config["train"], config["model"]
    set_seed(int(cfg_t.get("seed", 7)))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    target_steps = int(cfg_m.get("target_steps", 150))

    specs, names = resolve_specs(config, args)
    subj_maps, total_subjects = build_global_subjects(specs, BUNDLE_DIR)

    # Build per-dataset splits sharing the global subject index + common T.
    trains, vals, tests = {}, {}, {}
    for spec in specs:
        common = dict(spec=spec, bundle_dir=BUNDLE_DIR, target_steps=target_steps,
                      subject_global_index=subj_maps[spec.name])
        trains[spec.name] = UnifiedEEGSpeechDataset(split="train", **common)
        vals[spec.name] = UnifiedEEGSpeechDataset(split="val", **common)
        tests[spec.name] = UnifiedEEGSpeechDataset(split="test", **common)

    heads = [DatasetHead(name=s.name, n_channels=s.n_channels, num_labels=trains[s.name].num_labels)
             for s in specs]
    model = EEG2SpeechMD(EEG2SpeechMDConfig(
        datasets=heads,
        d_model=int(cfg_m.get("d_model", 256)),
        cond_dim=int(cfg_m.get("cond_dim", 64)),
        num_subjects=total_subjects,
        target_steps=target_steps,
        target_dim=int(trains[specs[0].name].targets.dim),
        num_blocks=int(cfg_m.get("num_blocks", 5)),
        kernel_size=int(cfg_m.get("kernel_size", 5)),
        channel_dropout=float(cfg_m.get("channel_dropout", 0.1)),
        dropout=float(cfg_m.get("dropout", 0.1)),
        embed_dim=int(cfg_m.get("embed_dim", trains[specs[0].name].targets.dim)),
    )).to(device)

    if args.init_from:
        ck = torch.load(args.init_from, map_location=device)
        miss, unexp = model.load_state_dict(ck["model_state"], strict=False)
        print(f"[init-from] {args.init_from} missing={len(miss)} unexpected={len(unexp)}")

    epochs = args.epochs or int(cfg_t.get("epochs", 120))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg_t.get("lr", 2e-4)),
                            weight_decay=float(cfg_t.get("weight_decay", 1e-4)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    bs = int(cfg_t.get("batch_size", 32))
    nw = int(cfg_t.get("num_workers", 0))
    raw_weights = config.get("sampling_weights", [1.0] * len(specs))
    raw_weights = [float(w) for w in raw_weights]
    weight_of = {s.name: (raw_weights[i] if i < len(raw_weights) else 1.0) for i, s in enumerate(specs)}

    train_loaders = {s.name: DataLoader(trains[s.name], batch_size=bs, shuffle=True, num_workers=nw,
                                        drop_last=False) for s in specs}
    val_loaders = {s.name: DataLoader(vals[s.name], batch_size=bs, shuffle=False, num_workers=nw)
                   for s in specs}

    def loss_for(batch, name):
        out = model(batch["eeg"].to(device), batch["subject_index"].to(device), name)
        return compute_v3_losses(
            out,
            target_sequence=batch["target_sequence"].to(device),
            target_summary=batch["target_summary"].to(device),
            label_ids=batch["label_id"].to(device),
            target_mask=batch["target_mask"].to(device),
            lambda_contrastive=float(cfg_t.get("lambda_contrastive", 1.0)),
            lambda_latent_cosine=float(cfg_t.get("lambda_latent_cosine", 1.0)),
            lambda_latent_mse=float(cfg_t.get("lambda_latent_mse", 0.5)),
            lambda_cls=float(cfg_t.get("lambda_cls", 0.5)),
            contrastive_temperature=float(cfg_t.get("contrastive_temperature", 0.07)),
        )

    suffix = args.run_suffix or "md"
    run_name = f"{'_'.join(names)}_{suffix}"
    out_root = (BUNDLE_DIR / config["output"]["root"]).resolve()
    run_dir = out_root / run_name
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "metrics")

    best = -1e9
    history = []
    for epoch in range(epochs):
        # ---- weighted round-robin over dataset batches ----
        model.train()
        iters = {n: iter(dl) for n, dl in train_loaders.items()}
        remaining = {n: len(dl) for n, dl in train_loaders.items()}
        agg, count, steps = {}, 0, 0
        while any(remaining.values()):
            avail = [s.name for s in specs if remaining[s.name] > 0]
            w = [weight_of[n] for n in avail]
            name = random.choices(avail, weights=w, k=1)[0]
            try:
                batch = next(iters[name])
            except StopIteration:
                remaining[name] = 0
                continue
            remaining[name] -= 1
            losses = loss_for(batch, name)
            opt.zero_grad()
            losses["total"].backward()
            clip_grad_norm_(model.parameters(), float(cfg_t.get("grad_clip", 1.0)))
            opt.step()
            b = batch["eeg"].shape[0]
            count += b
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v.detach()) * b
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
        sched.step()
        tr = {k: v / max(count, 1) for k, v in agg.items()}

        # ---- validation (averaged across datasets) ----
        model.eval()
        v_cos, v_contr, v_acc, vc = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for name, dl in val_loaders.items():
                for vb in dl:
                    lv = loss_for(vb, name)
                    bb = vb["eeg"].shape[0]
                    v_cos += float(lv["latent_cosine"]) * bb
                    v_contr += float(lv["contrastive"]) * bb
                    v_acc += float(lv["cls_acc"]) * bb
                    vc += bb
                    if args.max_steps:
                        break
        v_cos, v_contr, v_acc = v_cos / max(vc, 1), v_contr / max(vc, 1), v_acc / max(vc, 1)
        score = v_cos - 0.1 * v_contr
        history.append({"epoch": epoch, "train": tr, "val_cos": v_cos, "val_contr": v_contr,
                        "val_cls_acc": v_acc, "score": score})
        print(f"epoch {epoch:03d} | train total {tr.get('total',0):.3f} cls_acc {tr.get('cls_acc',0):.3f} "
              f"| val cos {v_cos:.3f} contr {v_contr:.3f} cls_acc {v_acc:.3f}")
        if score > best:
            best = score
            torch.save({
                "model_state": model.state_dict(),
                "model_config": {**vars(model.config), "datasets": [vars(h) for h in heads]},
                "datasets": names,
                "subject_maps": subj_maps,
                "label_vocabs": {n: trains[n].label_vocab for n in names},
                "target_steps": target_steps,
                "epoch": epoch,
            }, run_dir / "checkpoints" / "best.pt")

    write_json(run_dir / "metrics" / "history.json", history)

    # ---- per-dataset test retrieval ----
    results = {}
    for s in specs:
        results[s.name] = evaluate_unified(model, trains[s.name], tests[s.name], s.name,
                                           device=device, top_k=int(config.get("eval", {}).get("top_k", 5)),
                                           batch_size=bs)
        print(f"[test:{s.name}]", results[s.name])
    write_json(run_dir / "metrics" / "test_metrics.json", results)
    print(f"[done] run dir: {run_dir}")


if __name__ == "__main__":
    main()

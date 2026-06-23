from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.karaone_recon.data import KaraOneTrialDataset
from src.karaone_recon.eval import evaluate
from src.karaone_recon.losses import compute_losses
from src.karaone_recon.model import KaraOneConfig, KaraOneEEG2Codec
from src.karaone_recon.targets import KaraOneTargets
from src.utils import ensure_dir, load_simple_yaml, resolve_bundle_path, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KaraOne EEG -> EnCodec latent reconstruction.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone.yaml"))
    parser.add_argument("--stages", default=None, help="comma list, e.g. overt_like or overt_like,thinking")
    parser.add_argument("--model", choices=["baseline", "moe"], default="baseline")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-suffix", default=None)
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def _load_common(cfg: dict, stages: tuple[str, ...]):
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    targets = KaraOneTargets(resolve_bundle_path(cfg["data"]["target_cache"], BUNDLE_DIR), data_root=root)
    heldout = cfg["data"].get("heldout_subjects", ["P02", "MM21"])
    common = dict(
        data_root=root,
        targets=targets,
        stages=stages,
        split_protocol=str(cfg["data"].get("split_protocol", "trial")),
        heldout_subjects=heldout,
        eeg_len=int(cfg["data"].get("eeg_len", 1280)),
    )
    return root, targets, common


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    train_cfg, model_cfg = cfg["train"], cfg["model"]
    set_seed(int(train_cfg.get("seed", 7)))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    stages = tuple((args.stages or cfg["data"].get("stages", "overt_like")).split(","))
    _, targets, common = _load_common(cfg, stages)
    train_ds = KaraOneTrialDataset(split="train", **common)
    val_ds = KaraOneTrialDataset(split="val", **common)
    test_ds = KaraOneTrialDataset(split="test", **common)
    subject_test = KaraOneTrialDataset(split="subject_test", split_protocol="subject_holdout", **{k: v for k, v in common.items() if k != "split_protocol"})
    print(
        f"[data] stages={stages} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"subject_test={len(subject_test)} subjects={train_ds.num_subjects} labels={train_ds.num_labels}"
    )

    # The MoE lives in the EEG encoder (channel selection/clustering). The `moe`
    # model turns on the channel-experts front-end; the output head stays a plain
    # MLP (num_experts=1) since channel filtering belongs at the encoder, not here.
    num_channel_experts = (
        1 if args.model == "baseline" else max(4, int(model_cfg.get("num_channel_experts", 4)))
    )
    model = KaraOneEEG2Codec(
        KaraOneConfig(
            n_channels_eeg=int(model_cfg.get("n_channels_eeg", 62)),
            d_model=int(model_cfg.get("d_model", 256)),
            cond_dim=int(model_cfg.get("cond_dim", 64)),
            num_subjects=train_ds.num_subjects,
            num_labels=train_ds.num_labels,
            num_stages=train_ds.num_stages,
            target_steps=targets.T,
            target_dim=targets.D,
            content_dim=int(model_cfg.get("content_dim", 128)),
            speaker_dim=int(model_cfg.get("speaker_dim", 64)),
            num_blocks=int(model_cfg.get("num_blocks", 6)),
            kernel_size=int(model_cfg.get("kernel_size", 5)),
            channel_dropout=float(model_cfg.get("channel_dropout", 0.15)),
            dropout=float(model_cfg.get("dropout", 0.15)),
            num_experts=int(model_cfg.get("num_experts", 1)),
            num_channel_experts=num_channel_experts,
        )
    ).to(device)
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[init] loaded {args.init_from}")

    epochs = int(args.epochs or train_cfg.get("epochs", 120))
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-3)),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    batch_size = int(train_cfg.get("batch_size", 48))
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        drop_last=True,
    )
    loss_kwargs = {
        name: float(train_cfg.get(name, default))
        for name, default in {
            "lambda_recon_cos": 1.0,
            "lambda_recon_mse": 0.5,
            "lambda_content_ce": 0.5,
            "lambda_subject_ce": 0.05,
            "lambda_supcon": 0.5,
            "lambda_proto": 0.25,
            "lambda_subject_proto": 0.25,
            "lambda_log_rms": 0.2,
            "lambda_std": 0.0,
            "lambda_router_balance": 0.01,
            "lambda_channel_balance": 0.01,
            "supcon_temperature": 0.1,
        }.items()
    }

    run = f"karaone_{args.model}_{'_'.join(stages)}_{args.run_suffix or 'v1'}"
    run_dir = resolve_bundle_path(cfg["output"]["root"], BUNDLE_DIR) / run
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "metrics")
    history_jsonl = run_dir / "metrics" / "history.jsonl"
    history_csv = run_dir / "metrics" / "history.csv"
    history_jsonl.write_text("", encoding="utf-8")
    csv_fields = [
        "epoch",
        "train_total",
        "train_recon_cos",
        "train_recon_mse",
        "train_content_acc",
        "train_subject_acc",
        "train_std_ratio",
        "val_pred_cos",
        "val_zero_cos",
        "val_gain",
    ]
    with history_csv.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(csv_fields)

    def save_ckpt(path: Path, val_gain: float) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_config": vars(model.cfg),
                "stages": list(stages),
                "subject_vocab": train_ds.subject_vocab,
                "label_vocab": train_ds.label_vocab,
                "target_mean": targets.target_mean,
                "target_std": targets.target_std,
                "default_decoder_scales": targets.default_decoder_scales,
                "val_pred_over_zero_cos_gain": float(val_gain),
                "model_kind": args.model,
            },
            path,
        )

    best_gain = -1e9
    for epoch in range(epochs):
        model.train()
        agg: dict[str, float] = {}
        seen = 0
        steps = 0
        for batch in loader:
            out = model(batch["eeg"].to(device), batch["subject_idx"].to(device), batch["stage_idx"].to(device))
            losses = compute_losses(
                out,
                batch["target_seq"].to(device),
                batch["label_idx"].to(device),
                batch["subject_idx"].to(device),
                batch["content_proto"].to(device),
                batch["subject_proto"].to(device),
                batch["target_log_rms"].to(device),
                **loss_kwargs,
            )
            opt.zero_grad()
            losses["total"].backward()
            clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            opt.step()
            b = int(batch["eeg"].shape[0])
            seen += b
            for name, value in losses.items():
                agg[name] = agg.get(name, 0.0) + float(value.detach()) * b
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
        sched.step()
        train_metrics = {name: value / max(seen, 1) for name, value in agg.items()}
        val_metrics = evaluate(model, val_ds, targets, device=device, batch_size=batch_size)
        gain = float(val_metrics["pred_over_zero_cos_gain"])
        print(
            f"epoch {epoch:03d} total={train_metrics['total']:.3f} "
            f"recon_cos={train_metrics['recon_cos']:.3f} mse={train_metrics['recon_mse']:.3f} "
            f"content_acc={train_metrics['content_acc']:.3f} subject_acc={train_metrics['subject_acc']:.3f} "
            f"std={train_metrics['std_ratio']:.3f} | val pred={val_metrics['pred_recon_cos']:.3f} "
            f"zero={val_metrics['zeroeeg_recon_cos']:.3f} gain={gain:+.3f}"
        )
        with history_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, "train": train_metrics, "val": val_metrics}) + "\n")
        with history_csv.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    epoch,
                    train_metrics["total"],
                    train_metrics["recon_cos"],
                    train_metrics["recon_mse"],
                    train_metrics["content_acc"],
                    train_metrics["subject_acc"],
                    train_metrics["std_ratio"],
                    val_metrics["pred_recon_cos"],
                    val_metrics["zeroeeg_recon_cos"],
                    gain,
                ]
            )
        if gain > best_gain:
            best_gain = gain
            save_ckpt(run_dir / "checkpoints" / "best.pt", best_gain)
        if args.max_steps:
            break

    save_ckpt(run_dir / "checkpoints" / "last.pt", best_gain)
    final = {
        "selection": {"criterion": "val pred_recon_cos - zeroeeg_recon_cos", "best_val_gain": best_gain},
        "test": evaluate(model, test_ds, targets, device=device, batch_size=batch_size),
        "subject_test": evaluate(model, subject_test, targets, device=device, batch_size=batch_size),
    }
    write_json(run_dir / "metrics" / "test_metrics.json", final)
    print(json.dumps(final["selection"], indent=2))
    print(f"[done] {run_dir}")


if __name__ == "__main__":
    main()

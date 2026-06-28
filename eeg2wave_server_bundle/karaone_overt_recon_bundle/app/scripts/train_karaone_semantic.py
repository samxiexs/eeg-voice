from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
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
from src.utils import ensure_dir, load_simple_yaml, resolve_bundle_path, resolve_target_cache, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train KaraOne EEG -> HuBERT semantic sequence prediction."
    )
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone.yaml"))
    parser.add_argument("--stages", default=None, help="comma list; default config data.stages")
    parser.add_argument("--model", choices=["baseline", "moe"], default="baseline")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-suffix", default=None)
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--lambda-dtw", type=float, default=0.0, help="optional DTW over HuBERT frames")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    train_cfg, model_cfg = cfg["train"], cfg["model"]
    da_cfg = cfg.get("domain_adapt", {})
    set_seed(int(train_cfg.get("seed", 7)))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    stages = tuple((args.stages or cfg["data"].get("stages", "overt_like")).split(","))

    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    target_kind, cache = resolve_target_cache(cfg, BUNDLE_DIR, "hubert_sequence")
    if not cache.exists():
        raise FileNotFoundError(
            f"Missing HuBERT cache: {cache}. Build it with "
            "`python scripts/extract_karaone_targets.py --target hubert_sequence`."
        )
    targets = KaraOneTargets(cache, data_root=root)
    print(f"[target] semantic kind={target_kind} D={targets.D} T={targets.T} cache={cache.name}")

    common = dict(
        data_root=root,
        targets=targets,
        stages=stages,
        split_protocol=str(cfg["data"].get("split_protocol", "trial")),
        heldout_subjects=cfg["data"].get("heldout_subjects", ["P02", "MM21"]),
        eeg_len=int(cfg["data"].get("eeg_len", 1280)),
    )
    train_ds = KaraOneTrialDataset(split="train", **common)
    val_ds = KaraOneTrialDataset(split="val", **common)
    test_ds = KaraOneTrialDataset(split="test", **common)
    subject_test = KaraOneTrialDataset(
        split="subject_test",
        split_protocol="subject_holdout",
        **{k: v for k, v in common.items() if k != "split_protocol"},
    )
    print(
        f"[data] stages={stages} train={len(train_ds)} val={len(val_ds)} "
        f"test={len(test_ds)} subject_test={len(subject_test)}"
    )

    num_channel_experts = 1 if args.model == "baseline" else max(4, int(model_cfg.get("num_channel_experts", 4)))
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
            content_dim=targets.D,
            speaker_dim=int(model_cfg.get("speaker_dim", 64)),
            num_blocks=int(model_cfg.get("num_blocks", 6)),
            kernel_size=int(model_cfg.get("kernel_size", 5)),
            channel_dropout=float(model_cfg.get("channel_dropout", 0.15)),
            dropout=float(model_cfg.get("dropout", 0.15)),
            num_experts=1,
            num_channel_experts=num_channel_experts,
            encoder_kind=str(model_cfg.get("encoder_kind", "cnn")),
            transformer_layers=int(model_cfg.get("transformer_layers", 4)),
            transformer_heads=int(model_cfg.get("transformer_heads", 4)),
            patch_stride=int(model_cfg.get("patch_stride", 4)),
            decoder_scale_dim=int(targets.decoder_scale_dim),
            instance_norm=bool(da_cfg.get("instance_norm", False)),
            use_domain_adv=bool(da_cfg.get("adversarial", False)),
            hubert_dim=0,
            hubert_steps=targets.T,
        )
    ).to(device)
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[init] loaded {args.init_from}")

    use_domain_adv = bool(da_cfg.get("adversarial", False))
    lambda_domain_adv = float(da_cfg.get("lambda_domain_adv", 0.0)) if use_domain_adv else 0.0
    print(
        f"[model] kind={args.model} encoder={model_cfg.get('encoder_kind', 'cnn')} "
        f"channel_experts={num_channel_experts}"
    )
    print(
        f"[domain] instance_norm={bool(da_cfg.get('instance_norm', False))} "
        f"adversarial={use_domain_adv} lambda={lambda_domain_adv}"
    )

    epochs = int(args.epochs or train_cfg.get("epochs", 120))
    batch_size = int(train_cfg.get("batch_size", 48))
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-3)),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        drop_last=True,
    )

    loss_kwargs = {
        "lambda_recon_cos": float(train_cfg.get("lambda_recon_cos", 1.0)),
        "lambda_recon_mse": float(train_cfg.get("lambda_recon_mse", 0.5)),
        "lambda_content_ce": float(train_cfg.get("lambda_content_ce", 0.5)),
        "lambda_supcon": float(train_cfg.get("lambda_supcon", 0.5)),
        "lambda_proto": float(train_cfg.get("lambda_proto", 0.25)),
        "lambda_log_rms": 0.0,
        "lambda_std": float(train_cfg.get("lambda_std", 0.1)),
        "lambda_router_balance": 0.0,
        "lambda_channel_balance": float(train_cfg.get("lambda_channel_balance", 0.0)),
        "lambda_clip": float(train_cfg.get("lambda_clip", 0.5)),
        "lambda_dtw": float(args.lambda_dtw),
        "lambda_energy_env": 0.0,
        "lambda_multiscale_mel": 0.0,
        "lambda_frame_energy": 0.0,
        "lambda_voiced_rms": 0.0,
        "lambda_decoder_scale": 0.0,
        "lambda_ctc": float(train_cfg.get("lambda_ctc", 0.2)),
        "lambda_hubert_aux": 0.0,
        "lambda_hubert_clip": 0.0,
        "supcon_temperature": float(train_cfg.get("supcon_temperature", 0.1)),
        "clip_temperature": float(train_cfg.get("clip_temperature", 0.07)),
        "dtw_band": float(train_cfg.get("dtw_band", 0.2)),
    }
    print(
        "[loss] semantic primary: "
        f"cos={loss_kwargs['lambda_recon_cos']} mse={loss_kwargs['lambda_recon_mse']} "
        f"clip={loss_kwargs['lambda_clip']} ctc={loss_kwargs['lambda_ctc']} "
        f"dtw={loss_kwargs['lambda_dtw']}"
    )

    run = f"karaone_semantic_{args.model}_{'_'.join(stages)}_{args.run_suffix or 'v1'}"
    run_dir = resolve_bundle_path(cfg["output"]["root"], BUNDLE_DIR) / run
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "metrics")
    history_jsonl = run_dir / "metrics" / "history.jsonl"
    history_csv = run_dir / "metrics" / "history.csv"
    history_jsonl.write_text("", encoding="utf-8")
    with history_csv.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(
            [
                "epoch",
                "train_total",
                "train_recon_cos",
                "train_recon_mse",
                "train_content_acc",
                "train_ctc",
                "train_clip_loss",
                "val_pred_cos",
                "val_mean_cos",
                "val_gain",
                "val_label_top1",
                "val_trial_top1",
            ]
        )

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
                "val_pred_over_mean_cos_gain": float(val_gain),
                "model_kind": f"semantic_{args.model}",
                "target_kind": target_kind,
            },
            path,
        )

    best_gain = -1e9
    epochs_no_improve = 0
    patience = int(train_cfg.get("early_stop_patience", 0))
    for epoch in range(epochs):
        model.train()
        agg: dict[str, float] = {}
        seen = 0
        steps = 0
        progress = epoch / max(epochs - 1, 1)
        grl_lambda = lambda_domain_adv * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0) if use_domain_adv else 0.0
        for batch in loader:
            subject_idx = batch["subject_idx"].to(device)
            out = model(
                batch["eeg"].to(device),
                subject_idx,
                batch["stage_idx"].to(device),
                batch["eeg_valid_len"].to(device),
                lambda_domain=grl_lambda,
            )
            losses = compute_losses(
                out,
                batch["target_seq"].to(device),
                batch["label_idx"].to(device),
                batch["content_proto"].to(device),
                batch["target_log_rms"].to(device),
                target_decoder_scale=batch["target_decoder_scale"].to(device),
                **loss_kwargs,
            )
            total = losses["total"]
            if use_domain_adv and "subject_logits" in out:
                domain_adv = F.cross_entropy(out["subject_logits"], subject_idx)
                total = total + domain_adv
                losses = {
                    **losses,
                    "domain_adv": domain_adv.detach(),
                    "grl_lambda": out["pred_latent"].new_tensor(grl_lambda),
                }
            opt.zero_grad()
            total.backward()
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
        gain = float(val_metrics["pred_over_mean_cos_gain"])
        print(
            f"epoch {epoch:03d} total={train_metrics['total']:.3f} "
            f"hubert_cos_loss={train_metrics['recon_cos']:.3f} "
            f"content_acc={train_metrics['content_acc']:.3f} "
            f"ctc={train_metrics['ctc']:.3f} | val pred={val_metrics['pred_recon_cos']:.3f} "
            f"mean={val_metrics['mean_recon_cos']:.3f} gain={gain:+.3f} "
            f"label@1={val_metrics.get('pred_within_subject_label_top1', 0.0):.3f}"
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
                    train_metrics["ctc"],
                    train_metrics["clip_loss"],
                    val_metrics["pred_recon_cos"],
                    val_metrics["mean_recon_cos"],
                    gain,
                    val_metrics.get("pred_within_subject_label_top1", 0.0),
                    val_metrics.get("pred_within_subject_trial_top1", 0.0),
                ]
            )
        if gain > best_gain:
            best_gain = gain
            epochs_no_improve = 0
            save_ckpt(run_dir / "checkpoints" / "best.pt", best_gain)
        else:
            epochs_no_improve += 1
        try:
            from src.karaone_recon.plotting import plot_history

            plot_history(history_jsonl, run_dir / "metrics" / "training_curves.png", title=run)
        except Exception as exc:  # noqa: BLE001
            print(f"[plot] skipped ({exc})")
        if args.max_steps:
            break
        if patience > 0 and epochs_no_improve >= patience:
            print(f"[early-stop] no val semantic-gain improvement for {patience} epochs; best={best_gain:+.4f}")
            break

    save_ckpt(run_dir / "checkpoints" / "last.pt", best_gain)
    final = {
        "target_kind": target_kind,
        "selection": {"criterion": "val semantic pred_over_mean_cos_gain", "best_val_gain": best_gain},
        "test": evaluate(model, test_ds, targets, device=device, batch_size=batch_size),
        "subject_test": evaluate(model, subject_test, targets, device=device, batch_size=batch_size),
    }
    write_json(run_dir / "metrics" / "test_metrics.json", final)
    print(json.dumps(final["selection"], indent=2))
    print(f"[done] {run_dir}")


if __name__ == "__main__":
    main()

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
from src.karaone_recon.semantic_tokens import KaraOneSemanticTokenTargets
from src.karaone_recon.targets import KaraOneTargets
from src.utils import ensure_dir, load_simple_yaml, resolve_bundle_path, resolve_target_cache, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KaraOne EEG -> HuBERT semantic tokens.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone.yaml"))
    parser.add_argument("--stages", default=None)
    parser.add_argument("--model", choices=["baseline", "moe"], default="baseline")
    parser.add_argument("--token-cache", default="../artifacts/audio_targets/karaone_trial_hubert_tokens_k64.npz")
    parser.add_argument("--selection", choices=["legacy", "eeg_audio_generation"], default="legacy")
    parser.add_argument("--speech-token-ctc", action="store_true", help="use CTC over HuBERT semantic-token sequences")
    parser.add_argument("--trial-contrastive", action="store_true", help="enable trial-level EEG-HuBERT InfoNCE")
    parser.add_argument("--label-aux-weight", type=float, default=None, help="weak label CE weight")
    parser.add_argument("--lambda-label-supcon", type=float, default=None, help="override same-label supervised contrastive weight")
    parser.add_argument("--lambda-prompt-ctc", type=float, default=None, help="override prompt-label CTC weight")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-suffix", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def _score(metrics: dict, selection: str) -> float:
    if selection == "eeg_audio_generation":
        trial_gain = metrics.get("pred_within_subject_trial_top1", 0.0) - metrics.get(
            "mean_within_subject_trial_top1", 0.0
        )
        token_gain = max(
            metrics.get("semantic_token_edit_gain", 0.0),
            metrics.get("semantic_token_over_zeroeeg_edit_gain", 0.0),
        )
        return float(metrics.get("pred_over_mean_cos_gain", 0.0) + 0.8 * trial_gain + 0.5 * token_gain)
    return float(
        metrics.get("pred_over_mean_cos_gain", 0.0)
        + 0.5 * metrics.get("pred_within_subject_label_top1", 0.0)
        + 0.5 * metrics.get("pred_within_subject_trial_top1", 0.0)
    )


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    train_cfg, model_cfg = cfg["train"], cfg["model"]
    da_cfg = cfg.get("domain_adapt", {})
    set_seed(int(train_cfg.get("seed", 7)))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    stages = tuple((args.stages or cfg["data"].get("stages", "overt_like")).split(","))
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    _, hubert_cache = resolve_target_cache(cfg, BUNDLE_DIR, "hubert_sequence")
    targets = KaraOneTargets(hubert_cache, data_root=root)
    token_targets = KaraOneSemanticTokenTargets(resolve_bundle_path(args.token_cache, BUNDLE_DIR))
    common = dict(
        data_root=root,
        targets=targets,
        stages=stages,
        split_protocol=str(cfg["data"].get("split_protocol", "trial")),
        heldout_subjects=cfg["data"].get("heldout_subjects", ["P02", "MM21"]),
        eeg_len=int(cfg["data"].get("eeg_len", 1280)),
        semantic_token_targets=token_targets,
    )
    train_ds = KaraOneTrialDataset(split="train", **common)
    val_ds = KaraOneTrialDataset(split="val", **common)
    test_ds = KaraOneTrialDataset(split="test", **common)
    subject_test = KaraOneTrialDataset(
        split="subject_test",
        split_protocol="subject_holdout",
        **{k: v for k, v in common.items() if k != "split_protocol"},
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
            semantic_token_vocab=int(token_targets.vocab_size),
            semantic_token_steps=int(token_targets.T),
        )
    ).to(device)
    use_domain_adv = bool(da_cfg.get("adversarial", False))
    lambda_domain_adv = float(da_cfg.get("lambda_domain_adv", 0.0)) if use_domain_adv else 0.0
    epochs = int(args.epochs or train_cfg.get("epochs", 30))
    batch_size = int(train_cfg.get("batch_size", 48))
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 3e-4)), weight_decay=float(train_cfg.get("weight_decay", 1e-3)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=int(train_cfg.get("num_workers", 0)), drop_last=True)
    loss_kwargs = {
        "lambda_recon_cos": float(train_cfg.get("sem_token_lambda_recon_cos", 1.0)),
        "lambda_recon_mse": float(train_cfg.get("sem_token_lambda_recon_mse", 0.3)),
        "lambda_content_ce": float(train_cfg.get("lambda_content_ce", 0.5)),
        "lambda_supcon": float(train_cfg.get("lambda_supcon", 0.5)),
        "lambda_proto": float(train_cfg.get("lambda_proto", 0.25)),
        "lambda_log_rms": 0.0,
        "lambda_std": float(train_cfg.get("lambda_std", 0.1)),
        "lambda_router_balance": 0.0,
        "lambda_channel_balance": float(train_cfg.get("lambda_channel_balance", 0.0)),
        "lambda_clip": float(train_cfg.get("lambda_clip", 0.5)),
        "lambda_ctc": float(train_cfg.get("lambda_ctc", 0.2)),
        "lambda_semantic_token_ce": float(train_cfg.get("lambda_semantic_token_ce", 1.0)),
        "lambda_speech_token_ctc": 0.5 if args.speech_token_ctc else 0.0,
        "lambda_trial_infonce": 0.5 if args.trial_contrastive else 0.0,
    }
    if args.selection == "eeg_audio_generation":
        loss_kwargs["lambda_content_ce"] = 0.05
        loss_kwargs["lambda_supcon"] = 0.0
        loss_kwargs["lambda_ctc"] = 0.0
    if args.label_aux_weight is not None:
        loss_kwargs["lambda_content_ce"] = float(args.label_aux_weight)
    if args.lambda_label_supcon is not None:
        loss_kwargs["lambda_supcon"] = float(args.lambda_label_supcon)
    if args.lambda_prompt_ctc is not None:
        loss_kwargs["lambda_ctc"] = float(args.lambda_prompt_ctc)
    run = f"karaone_semantic_tokens_{args.model}_{'_'.join(stages)}_{args.run_suffix or 'v3'}"
    run_dir = resolve_bundle_path(cfg["output"]["root"], BUNDLE_DIR) / run
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "metrics")
    history = run_dir / "metrics" / "history.csv"
    with history.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(["epoch", "train_total", "train_token_ce", "train_token_ctc", "val_gain", "val_trial_top1", "val_label_top1", "val_token_edit_gain", "score"])

    def save(path: Path, score: float) -> None:
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
                "target_kind": "hubert_sequence",
                "model_kind": f"semantic_tokens_{args.model}",
                "semantic_token_cache": str(resolve_bundle_path(args.token_cache, BUNDLE_DIR)),
                "selection": str(args.selection),
                "val_selection_score": float(score),
            },
            path,
        )

    best = -1e9
    for epoch in range(epochs):
        model.train()
        agg: dict[str, float] = {}
        seen = 0
        progress = epoch / max(epochs - 1, 1)
        grl_lambda = lambda_domain_adv * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0) if use_domain_adv else 0.0
        for step, batch in enumerate(loader):
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
                semantic_token_targets=batch["semantic_token_targets"].to(device),
                semantic_token_mask=batch["semantic_token_mask"].to(device),
                **loss_kwargs,
            )
            total = losses["total"]
            if use_domain_adv and "subject_logits" in out:
                total = total + F.cross_entropy(out["subject_logits"], subject_idx)
            opt.zero_grad()
            total.backward()
            clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            opt.step()
            b = int(batch["eeg"].shape[0])
            seen += b
            for name, value in losses.items():
                agg[name] = agg.get(name, 0.0) + float(value.detach()) * b
            if args.max_steps and step + 1 >= args.max_steps:
                break
        sched.step()
        train_metrics = {name: value / max(seen, 1) for name, value in agg.items()}
        val_metrics = evaluate(model, val_ds, targets, device=device, batch_size=batch_size, target_kind="hubert_sequence")
        score = _score(val_metrics, args.selection)
        print(
            f"epoch {epoch:03d} total={train_metrics['total']:.3f} "
            f"token_ce={train_metrics.get('semantic_token_ce', 0.0):.3f} "
            f"val_gain={val_metrics['pred_over_mean_cos_gain']:+.3f} "
            f"trial_top1={val_metrics.get('pred_within_subject_trial_top1', 0.0):.3f} "
            f"label_top1={val_metrics.get('pred_within_subject_label_top1', 0.0):.3f} "
            f"token_edit_gain={val_metrics.get('semantic_token_edit_gain', 0.0):+.3f} score={score:+.3f}"
        )
        with history.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    epoch,
                    train_metrics["total"],
                    train_metrics.get("semantic_token_ce", 0.0),
                    train_metrics.get("speech_token_ctc", 0.0),
                    val_metrics["pred_over_mean_cos_gain"],
                    val_metrics.get("pred_within_subject_trial_top1", 0.0),
                    val_metrics.get("pred_within_subject_label_top1", 0.0),
                    val_metrics.get("semantic_token_edit_gain", 0.0),
                    score,
                ]
            )
        if score > best:
            best = score
            save(run_dir / "checkpoints" / "best.pt", best)
        if args.max_steps:
            break
    save(run_dir / "checkpoints" / "last.pt", best)
    best_path = run_dir / "checkpoints" / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[eval] loaded best checkpoint for final metrics: {best_path}")
    final = {
        "selection": {"criterion": str(args.selection), "best_val_score": best},
        "test": evaluate(model, test_ds, targets, device=device, batch_size=batch_size, target_kind="hubert_sequence"),
        "subject_test": evaluate(model, subject_test, targets, device=device, batch_size=batch_size, target_kind="hubert_sequence"),
    }
    write_json(run_dir / "metrics" / "test_metrics.json", final)
    print(json.dumps(final["selection"], indent=2))
    print(f"[done] {run_dir}")


if __name__ == "__main__":
    main()

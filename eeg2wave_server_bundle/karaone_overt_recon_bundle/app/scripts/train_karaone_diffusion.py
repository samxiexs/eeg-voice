from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.karaone_recon.data import KaraOneTrialDataset
from src.karaone_recon.diffusion import DiffusionConfig, EEGLatentDiffusion
from src.karaone_recon.eval import _corr_median, _retrieval_stats
from src.karaone_recon.targets import KaraOneTargets
from src.utils import ensure_dir, load_simple_yaml, resolve_bundle_path, resolve_target_cache, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KaraOne EEG-conditioned latent DIFFUSION (generative, no mean collapse).")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone.yaml"))
    parser.add_argument("--stages", default=None)
    parser.add_argument("--model", choices=["baseline", "moe"], default="moe")
    parser.add_argument("--mode", choices=["diffusion", "flow"], default=None, help="generative head: DDPM diffusion or conditional flow matching (default: config diffusion.mode)")
    parser.add_argument("--target", choices=["mel", "encodec_latent"], default=None, help="acoustic target (default: config target.kind)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--eval-steps", type=int, default=None, help="DDIM steps used during eval")
    parser.add_argument("--run-suffix", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate_diffusion(model, dataset, targets, device, batch_size, steps) -> dict:
    """Sample latents and measure fidelity + the anti-collapse diagnostics."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    gm = torch.from_numpy(targets.global_mean_norm).to(device).float()
    totals: dict[str, float] = defaultdict(float)
    count = 0
    pred_summaries, pred_flat, target_flat = [], [], []
    subjects, labels, trials = [], [], []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        valid = batch["eeg_valid_len"].to(device)
        target = batch["target_seq"].to(device)
        pred = model.sample(eeg, valid, steps=steps)
        zero = model.sample(torch.zeros_like(eeg), valid, steps=steps)
        b = int(eeg.shape[0])
        gm_b = gm.unsqueeze(0).expand_as(target)
        totals["pred_recon_cos"] += float(F.cosine_similarity(pred, target, dim=-1).mean(dim=1).sum())
        totals["zeroeeg_recon_cos"] += float(F.cosine_similarity(zero, target, dim=-1).mean(dim=1).sum())
        totals["mean_recon_cos"] += float(F.cosine_similarity(gm_b, target, dim=-1).mean(dim=1).sum())
        count += b
        pred_summaries.append(pred.mean(dim=1).cpu().numpy())
        pred_flat.append(pred.reshape(b, -1).cpu().numpy())
        target_flat.append(target.reshape(b, -1).cpu().numpy())
        subjects.extend([str(s) for s in batch["subject"]])
        labels.extend([str(s) for s in batch["label"]])
        trials.extend([int(s) for s in batch["trial_index"]])

    out = {name: value / max(count, 1) for name, value in totals.items()}
    pred_summary = np.concatenate(pred_summaries, axis=0)
    pred_matrix = np.concatenate(pred_flat, axis=0)
    target_matrix = np.concatenate(target_flat, axis=0)
    pred_std = pred_matrix.std(axis=0)
    target_std = target_matrix.std(axis=0)
    out.update(
        {
            "n": int(count),
            "pred_over_zero_cos_gain": out["pred_recon_cos"] - out["zeroeeg_recon_cos"],
            "pred_over_mean_cos_gain": out["pred_recon_cos"] - out["mean_recon_cos"],
            "pred_std_ratio_median": float(np.median(pred_std / np.maximum(target_std, 1e-6))),
            "pred_pairwise_corr_median": _corr_median(pred_matrix),
        }
    )
    out.update({f"pred_{k}": v for k, v in _retrieval_stats(pred_summary, subjects, labels, trials, targets).items()})
    return out


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    dcfg = cfg.get("diffusion", {})
    train_cfg = cfg["train"]
    set_seed(int(train_cfg.get("seed", 7)))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    stages = tuple((args.stages or cfg["data"].get("stages", "overt_like")).split(","))

    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    target_kind, cache = resolve_target_cache(cfg, BUNDLE_DIR, args.target)
    targets = KaraOneTargets(cache, data_root=root)
    print(f"[target] kind={target_kind} D={targets.D} T={targets.T}")
    if target_kind == "encodec_latent" and not targets.has_complete_audio_metadata:
        print("[target] WARNING: EnCodec cache is legacy/incomplete; rebuild with scripts/extract_karaone_targets.py --target encodec_latent --force")
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
    print(f"[data] stages={stages} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    num_channel_experts = 1 if args.model == "baseline" else max(4, int(dcfg.get("num_channel_experts", 4)))
    model = EEGLatentDiffusion(
        DiffusionConfig(
            latent_dim=targets.D,
            target_steps=targets.T,
            n_channels_eeg=int(cfg["model"].get("n_channels_eeg", 62)),
            d_model=int(cfg["model"].get("d_model", 256)),
            cond_ch=int(dcfg.get("cond_ch", 128)),
            hidden=int(dcfg.get("hidden", 256)),
            num_blocks=int(dcfg.get("num_blocks", 6)),
            encoder_blocks=int(cfg["model"].get("num_blocks", 6)),
            kernel_size=int(cfg["model"].get("kernel_size", 5)),
            dropout=float(cfg["model"].get("dropout", 0.1)),
            num_channel_experts=num_channel_experts,
            encoder_kind=str(cfg["model"].get("encoder_kind", "cnn")),
            transformer_layers=int(cfg["model"].get("transformer_layers", 4)),
            transformer_heads=int(cfg["model"].get("transformer_heads", 4)),
            patch_stride=int(cfg["model"].get("patch_stride", 4)),
            timesteps=int(dcfg.get("timesteps", 1000)),
            schedule=str(dcfg.get("schedule", "cosine")),
            x0_clip=float(dcfg.get("x0_clip", 8.0)),
            mode=str(args.mode or dcfg.get("mode", "diffusion")),
        )
    ).to(device)
    print(f"[generative] mode={model.cfg.mode} (flow = deterministic ODE, fewer steps, no mean-collapse)")

    epochs = int(args.epochs or dcfg.get("epochs", 60))
    eval_steps = int(args.eval_steps or dcfg.get("eval_steps", 20))
    ddim_steps = int(dcfg.get("ddim_steps", 50))
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(dcfg.get("lr", train_cfg.get("lr", 3e-4))),
        weight_decay=float(dcfg.get("weight_decay", 1e-4)),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    batch_size = int(train_cfg.get("batch_size", 48))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=int(train_cfg.get("num_workers", 0)), drop_last=True)

    run = f"karaone_diffusion_{args.model}_{'_'.join(stages)}_{args.run_suffix or 'v1'}"
    run_dir = resolve_bundle_path(cfg["output"]["root"], BUNDLE_DIR) / run
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "metrics")
    history_jsonl = run_dir / "metrics" / "history.jsonl"
    history_jsonl.write_text("", encoding="utf-8")
    history_csv = run_dir / "metrics" / "history.csv"
    csv_fields = ["epoch", "train_loss", "val_pred_cos", "val_mean_cos", "val_gain_mean", "val_std_ratio", "val_pairwise_corr"]
    with history_csv.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(csv_fields)

    def save_ckpt(path: Path, metric: float) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "diffusion_config": vars(model.cfg),
                "stages": list(stages),
                "subject_vocab": train_ds.subject_vocab,
                "label_vocab": train_ds.label_vocab,
                "target_mean": targets.target_mean,
                "target_std": targets.target_std,
                "default_decoder_scales": targets.default_decoder_scales,
                "ddim_steps": ddim_steps,
                "best_val_pred_over_mean_gain": float(metric),
                "model_kind": f"diffusion_{args.model}",
                "target_kind": target_kind,
            },
            path,
        )

    best_gain = -1e9
    last_val: dict = {}
    for epoch in range(epochs):
        model.train()
        agg = 0.0
        seen = 0
        steps = 0
        for batch in loader:
            loss = model.loss(batch["target_seq"].to(device), batch["eeg"].to(device), batch["eeg_valid_len"].to(device))
            opt.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            opt.step()
            b = int(batch["eeg"].shape[0])
            agg += float(loss.detach()) * b
            seen += b
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
        sched.step()
        train_loss = agg / max(seen, 1)

        do_eval = ((epoch + 1) % max(args.eval_every, 1) == 0) or (epoch == epochs - 1) or bool(args.max_steps)
        if do_eval:
            val = evaluate_diffusion(model, val_ds, targets, device, batch_size, eval_steps)
            last_val = val
            gain = float(val["pred_over_mean_cos_gain"])
            print(
                f"epoch {epoch:03d} loss={train_loss:.4f} | val sample_cos={val['pred_recon_cos']:.3f} "
                f"mean={val['mean_recon_cos']:.3f} gain_mean={gain:+.3f} "
                f"std_ratio={val['pred_std_ratio_median']:.2f} pair_corr={val['pred_pairwise_corr_median']:.2f}"
            )
            with history_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"epoch": epoch, "train": {"loss": train_loss}, "val": val}) + "\n")
            with history_csv.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(
                    [epoch, train_loss, val["pred_recon_cos"], val["mean_recon_cos"], gain,
                     val["pred_std_ratio_median"], val["pred_pairwise_corr_median"]]
                )
            if gain > best_gain:
                best_gain = gain
                save_ckpt(run_dir / "checkpoints" / "best.pt", best_gain)
            try:
                from src.karaone_recon.plotting import plot_diffusion_history

                plot_diffusion_history(history_jsonl, run_dir / "metrics" / "training_curves.png", title=run)
            except Exception as exc:  # noqa: BLE001
                print(f"[plot] skipped ({exc})")
        else:
            print(f"epoch {epoch:03d} loss={train_loss:.4f}")
        if args.max_steps:
            break

    save_ckpt(run_dir / "checkpoints" / "last.pt", best_gain)
    final = {
        "selection": {"criterion": "val pred_over_mean_cos_gain", "best_val_gain": best_gain},
        "last_val": last_val,
        "test": evaluate_diffusion(model, test_ds, targets, device, batch_size, ddim_steps),
    }
    write_json(run_dir / "metrics" / "test_metrics.json", final)
    print(json.dumps(final["selection"], indent=2))
    print(f"[done] {run_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
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
from src.karaone_recon.model import KaraOneConfig, KaraOneEEG2Codec
from src.karaone_recon.refiner import RefinerConfig, ResidualDenoisingRefiner
from src.karaone_recon.targets import KaraOneTargets
from src.utils import ensure_dir, load_simple_yaml, resolve_bundle_path, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train second-stage KaraOne latent residual refiner (a single-step "
            "deterministic post-filter, NOT a diffusion sampler; see refiner.py)."
        )
    )
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone.yaml"))
    parser.add_argument("--checkpoint", required=True, help="Base KaraOne reconstruction checkpoint")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate_refiner(base_model, refiner, dataset, device, batch_size: int) -> dict:
    base_model.eval()
    refiner.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    base_cos = 0.0
    refined_cos = 0.0
    base_mse = 0.0
    refined_mse = 0.0
    n = 0
    for batch in loader:
        eeg = batch["eeg"].to(device)
        subject_idx = batch["subject_idx"].to(device)
        stage_idx = batch["stage_idx"].to(device)
        target = batch["target_seq"].to(device)
        base_pred, _ = base_model.generate_full(eeg, subject_idx, stage_idx)
        level = torch.zeros(base_pred.shape[0], device=device)
        refined = refiner(base_pred, level)
        b = int(eeg.shape[0])
        base_cos += float(F.cosine_similarity(base_pred, target, dim=-1).mean()) * b
        refined_cos += float(F.cosine_similarity(refined, target, dim=-1).mean()) * b
        base_mse += float(F.mse_loss(base_pred, target)) * b
        refined_mse += float(F.mse_loss(refined, target)) * b
        n += b
    return {
        "n": n,
        "base_recon_cos": base_cos / max(n, 1),
        "refined_recon_cos": refined_cos / max(n, 1),
        "cos_gain": refined_cos / max(n, 1) - base_cos / max(n, 1),
        "base_recon_mse": base_mse / max(n, 1),
        "refined_recon_mse": refined_mse / max(n, 1),
    }


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    set_seed(int(cfg["train"].get("seed", 7)))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base_ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    base_model = KaraOneEEG2Codec(KaraOneConfig(**base_ckpt["model_config"])).to(device)
    base_model.load_state_dict(base_ckpt["model_state"], strict=True)
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad_(False)

    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    targets = KaraOneTargets(resolve_bundle_path(cfg["data"]["target_cache"], BUNDLE_DIR), data_root=root)
    common = dict(
        data_root=root,
        targets=targets,
        stages=tuple(base_ckpt["stages"]),
        split_protocol=str(cfg["data"].get("split_protocol", "trial")),
        heldout_subjects=cfg["data"].get("heldout_subjects", ["P02", "MM21"]),
        eeg_len=int(cfg["data"].get("eeg_len", 1280)),
    )
    train_ds = KaraOneTrialDataset(split="train", **common)
    val_ds = KaraOneTrialDataset(split="val", **common)
    test_ds = KaraOneTrialDataset(split="test", **common)
    batch_size = int(cfg["train"].get("batch_size", 48))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    refiner = ResidualDenoisingRefiner(RefinerConfig(target_dim=targets.D)).to(device)
    opt = torch.optim.AdamW(refiner.parameters(), lr=float(cfg["train"].get("lr", 3e-4)), weight_decay=1e-4)
    run_dir = ensure_dir(Path(args.checkpoint).resolve().parents[1] / "refiner")
    ensure_dir(run_dir / "checkpoints")
    best_gain = -1e9
    for epoch in range(int(args.epochs)):
        refiner.train()
        total = 0.0
        seen = 0
        steps = 0
        for batch in loader:
            with torch.no_grad():
                base_pred, _ = base_model.generate_full(
                    batch["eeg"].to(device),
                    batch["subject_idx"].to(device),
                    batch["stage_idx"].to(device),
                )
            target = batch["target_seq"].to(device)
            level = torch.rand(base_pred.shape[0], device=device) * float(args.noise_std)
            noisy_base = base_pred + torch.randn_like(base_pred) * level.view(-1, 1, 1)
            refined = refiner(noisy_base, level)
            loss = F.mse_loss(refined, target) + (1.0 - F.cosine_similarity(refined, target, dim=-1).mean())
            opt.zero_grad()
            loss.backward()
            clip_grad_norm_(refiner.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * int(base_pred.shape[0])
            seen += int(base_pred.shape[0])
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
        val = evaluate_refiner(base_model, refiner, val_ds, device, batch_size)
        print(f"epoch {epoch:03d} loss={total/max(seen,1):.4f} val_gain={val['cos_gain']:+.4f}")
        if val["cos_gain"] > best_gain:
            best_gain = float(val["cos_gain"])
            torch.save(
                {
                    "refiner_state": refiner.state_dict(),
                    "refiner_config": vars(refiner.cfg),
                    "base_checkpoint": str(args.checkpoint),
                    "stages": list(base_ckpt["stages"]),
                    "target_mean": targets.target_mean,
                    "target_std": targets.target_std,
                    "best_val_gain": best_gain,
                },
                run_dir / "checkpoints" / "best_refiner.pt",
            )
        if args.max_steps:
            break
    write_json(run_dir / "test_metrics.json", evaluate_refiner(base_model, refiner, test_ds, device, batch_size))
    print(f"[done] {run_dir}")


if __name__ == "__main__":
    main()

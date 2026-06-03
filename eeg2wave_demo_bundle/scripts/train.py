from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.dataset import FEISThinkingDataset
from src.losses import compute_total_loss
from src.model import EEG2WaveVQModel
from src.utils import ensure_dir, load_simple_yaml, resolve_bundle_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train minimal FEIS EEG2Wave demo.")
    parser.add_argument("--subject", required=True, help="FEIS subject id, e.g. 01")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "config.yaml"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(config: dict, subject_id: str, data_root: str | None) -> tuple[DataLoader, DataLoader, DataLoader]:
    cfg_data = config["data"]
    cfg_audio = config["audio"]
    cfg_train = config["train"]
    root = data_root or cfg_data["root"]
    common = dict(
        subject_id=subject_id,
        data_root=root,
        stage=cfg_data["stage"],
        train_ratio=cfg_data["train_ratio"],
        val_ratio=cfg_data["val_ratio"],
        audio_sr=cfg_audio["sample_rate"],
        audio_dur=cfg_audio["duration_sec"],
    )
    train_ds = FEISThinkingDataset(split="train", **common)
    val_ds = FEISThinkingDataset(split="val", **common)
    test_ds = FEISThinkingDataset(split="test", **common)
    loader_kwargs = dict(batch_size=cfg_train["batch_size"], num_workers=cfg_train["num_workers"])
    return (
        DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs),
        DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs),
        DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs),
    )


def build_model(config: dict) -> EEG2WaveVQModel:
    cfg_model = config["model"]
    cfg_audio = config["audio"]
    return EEG2WaveVQModel(
        n_channels_eeg=cfg_model["n_channels_eeg"],
        hidden_dim=cfg_model["hidden_dim"],
        codebook_size=cfg_model["codebook_size"],
        vq_beta=cfg_model["vq_beta"],
        vq_decay=cfg_model["vq_decay"],
        output_samples=cfg_audio["n_samples"],
    )


def evaluate(model: EEG2WaveVQModel, loader: DataLoader, device: torch.device, config: dict) -> dict[str, float]:
    model.eval()
    sums = {"total": 0.0, "l1": 0.0, "stft": 0.0, "vq": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device)
            target = batch["waveform"].to(device)
            recon, vq_loss, _, _ = model(eeg)
            losses = compute_total_loss(
                recon,
                target,
                vq_loss,
                lambda_stft=config["train"]["lambda_stft"],
                lambda_vq=config["train"]["lambda_vq"],
            )
            batch_size = eeg.shape[0]
            count += batch_size
            for key in sums:
                sums[key] += float(losses[key].item()) * batch_size
    return {key: value / max(count, 1) for key, value in sums.items()}


def main() -> None:
    args = parse_args()
    config = load_simple_yaml(args.config)
    set_seed(int(config["train"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved_data_root = resolve_bundle_path(args.data_root or config["data"]["root"], BUNDLE_DIR)
    train_loader, val_loader, test_loader = build_loaders(config, args.subject, str(resolved_data_root))
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))

    output_root = resolve_bundle_path(args.output_root or config["output"]["root"], BUNDLE_DIR)
    ckpt_dir = ensure_dir(output_root / "checkpoints")
    metrics_dir = ensure_dir(output_root / "metrics")
    best_ckpt = ckpt_dir / f"subject_{args.subject}_best.pt"

    best_val = float("inf")
    epochs = int(config["train"]["epochs"])
    progress = tqdm(range(1, epochs + 1), desc=f"train subject {args.subject}")
    for epoch in progress:
        model.train()
        train_sums = {"total": 0.0, "l1": 0.0, "stft": 0.0, "vq": 0.0}
        seen = 0
        for batch in train_loader:
            eeg = batch["eeg"].to(device)
            target = batch["waveform"].to(device)
            recon, vq_loss, _, perplexity = model(eeg)
            losses = compute_total_loss(
                recon,
                target,
                vq_loss,
                lambda_stft=config["train"]["lambda_stft"],
                lambda_vq=config["train"]["lambda_vq"],
            )

            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip"]))
            optimizer.step()

            batch_size = eeg.shape[0]
            seen += batch_size
            for key in train_sums:
                train_sums[key] += float(losses[key].item()) * batch_size

        train_metrics = {key: value / max(seen, 1) for key, value in train_sums.items()}
        val_metrics = evaluate(model, val_loader, device, config)
        progress.set_postfix(
            train=f"{train_metrics['total']:.4f}",
            val=f"{val_metrics['total']:.4f}",
            px=f"{float(perplexity.item()):.2f}",
        )

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save(
                {
                    "subject_id": args.subject,
                    "config": config,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_total": best_val,
                },
                best_ckpt,
            )

    test_metrics = evaluate(model, test_loader, device, config)
    write_json(
        metrics_dir / f"subject_{args.subject}_train_summary.json",
        {
            "subject_id": args.subject,
            "checkpoint": str(best_ckpt),
            "best_val_total": best_val,
            "final_train": train_metrics,
            "final_val": val_metrics,
            "final_test": test_metrics,
        },
    )
    print(f"Saved best checkpoint to {best_ckpt}")


if __name__ == "__main__":
    main()

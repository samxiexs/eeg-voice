from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.dataset import FEISProtocolDataset
from src.losses import compute_total_loss
from src.model import EEG2WaveVQModel
from src.subject_conditioned_waveform import SubjectConditionedEEG2WaveVQModel
from src.utils import ensure_dir, load_simple_yaml, resolve_bundle_path, set_seed, write_json


LOSS_KEYS = ("total", "l1", "stft", "log_stft", "rms", "envelope", "vq", "cls", "cls_acc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FEIS waveform baseline under Protocol S/G/U.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "waveform_protocol.yaml"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--holdout-subject", default=None)
    parser.add_argument("--stage", default=None)
    parser.add_argument("--ablation-mode", default=None)
    parser.add_argument("--include-anomalous", action="store_true")
    parser.add_argument("--subject-conditioning", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def build_dataset(config: dict, args: argparse.Namespace, split: str) -> FEISProtocolDataset:
    cfg_data = config["data"]
    cfg_audio = config["audio"]
    return FEISProtocolDataset(
        data_root=str(resolve_bundle_path(args.data_root or cfg_data["root"], BUNDLE_DIR)),
        protocol=str(args.protocol or cfg_data["protocol"]).upper(),
        split=split,
        stage=str(args.stage or cfg_data["stage"]),
        subject_id=args.subject or cfg_data.get("subject_id"),
        holdout_subject_id=args.holdout_subject or cfg_data.get("holdout_subject_id"),
        include_anomalous=bool(args.include_anomalous or cfg_data.get("include_anomalous", False)),
        ablation_mode=str(args.ablation_mode or cfg_data.get("ablation_mode", "none")),
        target_cache_path=None,
        require_targets=False,
        audio_sr=int(cfg_audio["sample_rate"]),
        audio_dur=float(cfg_audio["duration_sec"]),
        audio_normalize=str(cfg_audio.get("normalize", "rms")),
        audio_target_rms=float(cfg_audio.get("target_rms", 0.08)),
        audio_max_gain=float(cfg_audio.get("max_gain", 10.0)),
        seed=int(config["train"]["seed"]),
    )


def build_model(config: dict, dataset: FEISProtocolDataset, subject_conditioning: bool) -> torch.nn.Module:
    cfg_model = config["model"]
    common = dict(
        n_channels_eeg=int(cfg_model["n_channels_eeg"]),
        hidden_dim=int(cfg_model["hidden_dim"]),
        codebook_size=int(cfg_model["codebook_size"]),
        vq_beta=float(cfg_model["vq_beta"]),
        vq_decay=float(cfg_model["vq_decay"]),
        output_samples=int(config["audio"]["n_samples"]),
        num_labels=int(dataset.num_labels),
    )
    if subject_conditioning:
        return SubjectConditionedEEG2WaveVQModel(
            **common,
            num_subjects=int(dataset.num_subjects),
            subject_embedding_dim=int(cfg_model.get("subject_embedding_dim", 64)),
        )
    return EEG2WaveVQModel(**common)


def forward_model(model: torch.nn.Module, batch: dict[str, object], device: torch.device, subject_conditioning: bool):
    eeg = batch["eeg"].to(device)
    if subject_conditioning:
        return model(eeg, batch["subject_index"].to(device))
    return model(eeg)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict,
    subject_conditioning: bool,
) -> dict[str, float]:
    model.eval()
    sums = {key: 0.0 for key in LOSS_KEYS}
    count = 0
    with torch.no_grad():
        for batch in loader:
            target = batch["waveform"].to(device)
            label_ids = batch["label_id"].to(device)
            recon, vq_loss, _, _, logits = forward_model(model, batch, device, subject_conditioning)
            losses = compute_total_loss(
                recon,
                target,
                vq_loss,
                lambda_stft=config["train"]["lambda_stft"],
                lambda_log_stft=config["train"]["lambda_log_stft"],
                lambda_rms=config["train"]["lambda_rms"],
                lambda_envelope=config["train"]["lambda_envelope"],
                lambda_vq=config["train"]["lambda_vq"],
                cls_logits=logits,
                label_ids=label_ids,
                lambda_cls=config["train"]["lambda_cls"],
            )
            batch_size = target.shape[0]
            count += batch_size
            for key in sums:
                sums[key] += float(losses[key].item()) * batch_size
    return {key: value / max(count, 1) for key, value in sums.items()}


def main() -> None:
    args = parse_args()
    config = load_simple_yaml(args.config)
    if args.protocol is not None:
        config["data"]["protocol"] = args.protocol
    if args.stage is not None:
        config["data"]["stage"] = args.stage
    if args.ablation_mode is not None:
        config["data"]["ablation_mode"] = args.ablation_mode
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.subject_conditioning:
        config["model"]["use_subject_conditioning"] = True
    set_seed(int(config["train"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = build_dataset(config, args, split="train")
    val_ds = build_dataset(config, args, split="val")
    test_ds = build_dataset(config, args, split="test")
    train_loader = DataLoader(train_ds, batch_size=int(config["train"]["batch_size"]), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(config["train"]["batch_size"]), shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=int(config["train"]["batch_size"]), shuffle=False, num_workers=0)

    subject_conditioning = bool(config["model"].get("use_subject_conditioning", False))
    model = build_model(config, train_ds, subject_conditioning=subject_conditioning).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )

    protocol = str(config["data"]["protocol"]).upper()
    run_name = f"{protocol.lower()}_{config['data']['stage']}_{config['data'].get('ablation_mode', 'none')}"
    if protocol == "S":
        run_name += f"_subject_{args.subject or config['data'].get('subject_id')}"
    if protocol == "U":
        run_name += f"_holdout_{args.holdout_subject or config['data'].get('holdout_subject_id')}"
    if subject_conditioning:
        run_name += "_subject_conditioned"
    run_tag = os.environ.get("FEIS_RUN_TAG", "").strip()
    if run_tag:
        run_name += f"_{run_tag}"
    output_root = resolve_bundle_path(args.output_root or config["output"]["root"], BUNDLE_DIR)
    run_root = ensure_dir(output_root / run_name)
    ckpt_dir = ensure_dir(run_root / "checkpoints")
    metrics_dir = ensure_dir(run_root / "metrics")
    best_ckpt = ckpt_dir / "best.pt"

    best_val = float("inf")
    history = []
    for epoch in tqdm(range(1, int(config["train"]["epochs"]) + 1), desc=f"waveform {run_name}"):
        model.train()
        sums = {key: 0.0 for key in LOSS_KEYS}
        count = 0
        for batch in train_loader:
            target = batch["waveform"].to(device)
            label_ids = batch["label_id"].to(device)
            recon, vq_loss, _, _, logits = forward_model(model, batch, device, subject_conditioning)
            losses = compute_total_loss(
                recon,
                target,
                vq_loss,
                lambda_stft=config["train"]["lambda_stft"],
                lambda_log_stft=config["train"]["lambda_log_stft"],
                lambda_rms=config["train"]["lambda_rms"],
                lambda_envelope=config["train"]["lambda_envelope"],
                lambda_vq=config["train"]["lambda_vq"],
                cls_logits=logits,
                label_ids=label_ids,
                lambda_cls=config["train"]["lambda_cls"],
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            clip_grad_norm_(model.parameters(), float(config["train"].get("grad_clip", 1.0)))
            optimizer.step()
            batch_size = target.shape[0]
            count += batch_size
            for key in sums:
                sums[key] += float(losses[key].item()) * batch_size
        train_metrics = {key: value / max(count, 1) for key, value in sums.items()}
        val_metrics = evaluate(model, val_loader, device, config, subject_conditioning)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save(
                {
                    "config": config,
                    "model_state": model.state_dict(),
                    "subject_conditioning": subject_conditioning,
                },
                best_ckpt,
            )

    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, test_loader, device, config, subject_conditioning)
    write_json(metrics_dir / "history.json", {"history": history, "test": test_metrics})
    print(f"Saved waveform checkpoint to {best_ckpt}")
    print(f"Test total loss: {test_metrics['total']:.4f}")


if __name__ == "__main__":
    main()

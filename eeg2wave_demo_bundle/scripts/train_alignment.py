from __future__ import annotations

import argparse
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

from src.alignment_losses import compute_alignment_losses
from src.alignment_model import EEGSpeechAlignmentModel
from src.dataset import FEISProtocolDataset
from src.eval_utils import retrieval_topk
from src.utils import ensure_dir, load_simple_yaml, resolve_bundle_path, set_seed, write_json


ALIGNMENT_KEYS = (
    "total",
    "embedding_cosine_loss",
    "embedding_cosine",
    "embedding_mse",
    "prosody_loss",
    "cls",
    "cls_acc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train subject-aware FEIS EEG-to-speech alignment model.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "alignment.yaml"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--protocol", default=None, help="S | G | U")
    parser.add_argument("--subject", default=None, help="Required for Protocol S")
    parser.add_argument("--holdout-subject", default=None, help="Required for Protocol U")
    parser.add_argument("--stage", default=None)
    parser.add_argument("--ablation-mode", default=None)
    parser.add_argument("--include-anomalous", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def _valid_steps_from_samples(valid_samples: torch.Tensor) -> torch.Tensor:
    return torch.ceil(valid_samples.float() / 32.0).long().clamp_min(1)


def build_datasets(config: dict, args: argparse.Namespace) -> tuple[FEISProtocolDataset, FEISProtocolDataset, FEISProtocolDataset]:
    cfg_data = config["data"]
    cfg_audio = config["audio"]
    resolved_root = resolve_bundle_path(args.data_root or cfg_data["root"], BUNDLE_DIR)
    protocol = str(args.protocol or cfg_data["protocol"]).upper()
    common = dict(
        data_root=str(resolved_root),
        protocol=protocol,
        stage=str(args.stage or cfg_data["stage"]),
        subject_id=args.subject or cfg_data.get("subject_id"),
        holdout_subject_id=args.holdout_subject or cfg_data.get("holdout_subject_id"),
        include_anomalous=bool(args.include_anomalous or cfg_data.get("include_anomalous", False)),
        ablation_mode=str(args.ablation_mode or cfg_data.get("ablation_mode", "none")),
        target_cache_path=resolve_bundle_path(config["targets"]["cache_path"], BUNDLE_DIR),
        require_targets=True,
        audio_sr=int(cfg_audio["sample_rate"]),
        audio_dur=float(cfg_audio["duration_sec"]),
        audio_normalize=str(cfg_audio.get("normalize", "rms")),
        audio_target_rms=float(cfg_audio.get("target_rms", 0.08)),
        audio_max_gain=float(cfg_audio.get("max_gain", 10.0)),
        seed=int(config["train"]["seed"]),
    )
    train_ds = FEISProtocolDataset(split="train", **common)
    val_ds = FEISProtocolDataset(split="val", **common)
    test_ds = FEISProtocolDataset(split="test", **common)
    return train_ds, val_ds, test_ds


def build_model(config: dict, dataset: FEISProtocolDataset) -> EEGSpeechAlignmentModel:
    cfg_model = config["model"]
    return EEGSpeechAlignmentModel(
        n_channels_eeg=int(cfg_model["n_channels_eeg"]),
        hidden_dim=int(cfg_model["hidden_dim"]),
        speech_embedding_dim=int(dataset.target_embedding_dim),
        prosody_dim=int(dataset.prosody_dim),
        num_labels=int(dataset.num_labels),
        latent_dim=int(cfg_model.get("latent_dim", 192)),
        use_label_head=bool(cfg_model.get("use_label_head", True)),
        use_subject_demo_head=bool(cfg_model.get("use_subject_demo_head", False)),
        num_subjects=int(dataset.num_subjects),
        subject_embedding_dim=int(cfg_model.get("subject_embedding_dim", 64)),
    )


def _build_retrieval_bank(dataset: FEISProtocolDataset) -> tuple[torch.Tensor, list[str]]:
    bank_ids = dataset.unique_template_ids(split="train")
    bank = np.stack([dataset.get_template_target(template_id)["speech_embedding"] for template_id in bank_ids], axis=0)
    return torch.from_numpy(bank).float(), bank_ids


def evaluate(
    model: EEGSpeechAlignmentModel,
    dataset: FEISProtocolDataset,
    loader: DataLoader,
    device: torch.device,
    config: dict,
) -> dict[str, float]:
    model.eval()
    retrieval_bank, bank_ids = _build_retrieval_bank(dataset)
    retrieval_bank = retrieval_bank.to(device)
    sums = {key: 0.0 for key in ALIGNMENT_KEYS}
    total = 0
    all_pred: list[torch.Tensor] = []
    all_target_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device)
            speech_target = batch["speech_embedding"].to(device)
            prosody_target = batch["prosody_target"].to(device)
            label_ids = batch["label_id"].to(device)
            valid_steps = _valid_steps_from_samples(batch["eeg_valid_num_samples"].to(device))
            outputs = model(eeg, subject_indices=batch["subject_index"].to(device), valid_steps=valid_steps)
            losses = compute_alignment_losses(
                pred_embedding=outputs["speech_embedding"],
                target_embedding=speech_target,
                pred_prosody=outputs["prosody"],
                target_prosody=prosody_target,
                lambda_cosine=float(config["train"]["lambda_cosine"]),
                lambda_mse=float(config["train"]["lambda_mse"]),
                lambda_prosody=float(config["train"]["lambda_prosody"]),
                label_logits=outputs.get("label_logits"),
                label_ids=label_ids,
                lambda_cls=float(config["train"].get("lambda_cls", 0.0)),
            )
            batch_size = eeg.shape[0]
            total += batch_size
            for key in ALIGNMENT_KEYS:
                sums[key] += float(losses[key].item()) * batch_size
            all_pred.append(outputs["speech_embedding"].detach())
            all_target_ids.extend(list(batch["template_id"]))
    metrics = {key: value / max(total, 1) for key, value in sums.items()}
    pred_matrix = torch.cat(all_pred, dim=0).cpu().numpy()
    retrieval = retrieval_topk(
        predicted=pred_matrix,
        target_template_ids=all_target_ids,
        bank_embeddings=retrieval_bank.cpu().numpy(),
        bank_template_ids=bank_ids,
        topk=(1, 5),
    )
    metrics.update(retrieval)
    return metrics


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
    set_seed(int(config["train"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds = build_datasets(config, args)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["train"]["num_workers"]),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"]["num_workers"]),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"]["num_workers"]),
    )

    model = build_model(config, train_ds).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )

    protocol = str(config["data"]["protocol"]).upper()
    output_root = resolve_bundle_path(args.output_root or config["output"]["root"], BUNDLE_DIR)
    run_name = f"{protocol.lower()}_{config['data']['stage']}_{config['data'].get('ablation_mode', 'none')}"
    if protocol == "S":
        run_name += f"_subject_{train_ds.subject_vocab[0]}"
    if protocol == "U":
        run_name += f"_holdout_{args.holdout_subject or config['data'].get('holdout_subject_id')}"
    run_root = ensure_dir(output_root / run_name)
    ckpt_dir = ensure_dir(run_root / "checkpoints")
    metrics_dir = ensure_dir(run_root / "metrics")
    best_ckpt = ckpt_dir / "best.pt"

    history: list[dict[str, object]] = []
    best_val = float("inf")
    epochs = int(config["train"]["epochs"])
    progress = tqdm(range(1, epochs + 1), desc=f"alignment {run_name}")
    for epoch in progress:
        model.train()
        sums = {key: 0.0 for key in ALIGNMENT_KEYS}
        total = 0
        for batch in train_loader:
            eeg = batch["eeg"].to(device)
            speech_target = batch["speech_embedding"].to(device)
            prosody_target = batch["prosody_target"].to(device)
            label_ids = batch["label_id"].to(device)
            valid_steps = _valid_steps_from_samples(batch["eeg_valid_num_samples"].to(device))
            outputs = model(eeg, subject_indices=batch["subject_index"].to(device), valid_steps=valid_steps)
            losses = compute_alignment_losses(
                pred_embedding=outputs["speech_embedding"],
                target_embedding=speech_target,
                pred_prosody=outputs["prosody"],
                target_prosody=prosody_target,
                lambda_cosine=float(config["train"]["lambda_cosine"]),
                lambda_mse=float(config["train"]["lambda_mse"]),
                lambda_prosody=float(config["train"]["lambda_prosody"]),
                label_logits=outputs.get("label_logits"),
                label_ids=label_ids,
                lambda_cls=float(config["train"].get("lambda_cls", 0.0)),
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            clip_grad_norm_(model.parameters(), float(config["train"].get("grad_clip", 1.0)))
            optimizer.step()

            batch_size = eeg.shape[0]
            total += batch_size
            for key in ALIGNMENT_KEYS:
                sums[key] += float(losses[key].item()) * batch_size

        train_metrics = {key: value / max(total, 1) for key, value in sums.items()}
        val_metrics = evaluate(model, val_ds, val_loader, device, config)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        progress.set_postfix(
            train=f"{train_metrics['total']:.4f}",
            val=f"{val_metrics['total']:.4f}",
            cos=f"{val_metrics['embedding_cosine']:.3f}",
            r1=f"{val_metrics['retrieval_top1']:.3f}",
        )
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save(
                {
                    "config": config,
                    "model_state": model.state_dict(),
                    "protocol": protocol,
                    "run_name": run_name,
                    "train_subjects": train_ds.subject_vocab,
                    "label_vocab": train_ds.label_vocab,
                },
                best_ckpt,
            )

    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, test_ds, test_loader, device, config)
    write_json(
        metrics_dir / "history.json",
        {
            "run_name": run_name,
            "history": history,
            "test": test_metrics,
        },
    )
    print(f"Saved best checkpoint to {best_ckpt}")
    print(f"Test embedding cosine: {test_metrics['embedding_cosine']:.4f}")
    print(f"Test retrieval@1: {test_metrics['retrieval_top1']:.4f}")


if __name__ == "__main__":
    main()

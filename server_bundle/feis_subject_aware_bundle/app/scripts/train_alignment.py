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
from src.alignment_retrieval import build_retrieval_bank, evaluate_embedding_retrieval
from src.dataset import FEISProtocolDataset
from src.utils import build_protocol_run_name, ensure_dir, load_simple_yaml, resolve_bundle_path, set_seed, write_json


ALIGNMENT_KEYS = (
    "total",
    "sequence_cosine_loss",
    "sequence_cosine",
    "sequence_mse",
    "summary_cosine",
    "embedding_cosine",
    "embedding_mse",
    "contrastive",
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


def _top_k(config: dict) -> int:
    return int(config.get("eval", {}).get("top_k", config["train"].get("retrieval_top_k", 5)))


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
        speech_embedding_dim=int(dataset.target_sequence_dim),
        prosody_dim=int(dataset.prosody_dim) if bool(cfg_model.get("use_prosody_head", dataset.prosody_dim > 0)) else 0,
        num_labels=int(dataset.num_labels),
        latent_dim=int(cfg_model.get("latent_dim", 192)),
        target_steps=int(dataset.target_sequence_steps),
        use_label_head=bool(cfg_model.get("use_label_head", True)),
        use_subject_demo_head=bool(cfg_model.get("use_subject_demo_head", False)),
        num_subjects=int(dataset.num_subjects),
        subject_embedding_dim=int(cfg_model.get("subject_embedding_dim", 64)),
    )


def evaluate(
    model: EEGSpeechAlignmentModel,
    dataset: FEISProtocolDataset,
    loader: DataLoader,
    device: torch.device,
    config: dict,
) -> dict[str, float]:
    model.eval()
    retrieval_bank = build_retrieval_bank(dataset, policy="auto")
    sums = {key: 0.0 for key in ALIGNMENT_KEYS}
    total = 0
    all_pred_sequence: list[np.ndarray] = []
    all_pred_summary: list[np.ndarray] = []
    all_target_ids: list[str] = []
    all_target_labels: list[str] = []
    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device)
            target_sequence = batch["target_sequence"].to(device)
            target_mask = batch["target_mask"].to(device)
            target_summary = batch["target_summary"].to(device)
            label_ids = batch["label_id"].to(device)
            valid_steps = _valid_steps_from_samples(batch["eeg_valid_num_samples"].to(device))
            outputs = model(eeg, subject_indices=batch["subject_index"].to(device), valid_steps=valid_steps)
            losses = compute_alignment_losses(
                pred_sequence=outputs["speech_sequence"],
                target_sequence=target_sequence,
                target_mask=target_mask,
                pred_summary=outputs["speech_embedding"],
                target_summary=target_summary,
                pred_prosody=outputs.get("prosody"),
                target_prosody=batch["prosody_target"].to(device) if "prosody_target" in batch else None,
                lambda_seq_cosine=float(config["train"].get("lambda_seq_cosine", config["train"].get("lambda_cosine", 1.0))),
                lambda_seq_mse=float(config["train"].get("lambda_seq_mse", config["train"].get("lambda_mse", 0.5))),
                lambda_contrastive=float(config["train"].get("lambda_contrastive", 1.0)),
                lambda_prosody=float(config["train"].get("lambda_prosody", 0.0)),
                contrastive_temperature=float(config["train"].get("contrastive_temperature", 0.07)),
                label_logits=outputs.get("label_logits"),
                label_ids=label_ids,
                lambda_cls=float(config["train"].get("lambda_cls", 0.0)),
            )
            batch_size = eeg.shape[0]
            total += batch_size
            for key in ALIGNMENT_KEYS:
                sums[key] += float(losses[key].item()) * batch_size
            all_pred_sequence.append(outputs["speech_sequence"].detach().cpu().numpy())
            all_pred_summary.append(outputs["speech_embedding"].detach().cpu().numpy())
            all_target_ids.extend(list(batch["template_id"]))
            all_target_labels.extend(list(batch["label"]))
    metrics = {key: value / max(total, 1) for key, value in sums.items()}
    retrieval = evaluate_embedding_retrieval(
        bank=retrieval_bank,
        predicted_sequences=np.concatenate(all_pred_sequence, axis=0),
        predicted_summaries=np.concatenate(all_pred_summary, axis=0),
        target_template_ids=all_target_ids,
        target_labels=all_target_labels,
        top_k=_top_k(config),
    )
    metrics.update(retrieval["metrics"])
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
    run_name = build_protocol_run_name(
        config=config,
        protocol=protocol,
        stage=str(config["data"]["stage"]),
        ablation_mode=str(config["data"].get("ablation_mode", "none")),
        subject_id=args.subject or config["data"].get("subject_id"),
        holdout_subject_id=args.holdout_subject or config["data"].get("holdout_subject_id"),
    )
    run_root = ensure_dir(output_root / run_name)
    ckpt_dir = ensure_dir(run_root / "checkpoints")
    metrics_dir = ensure_dir(run_root / "metrics")
    best_ckpt = ckpt_dir / "best.pt"

    history: list[dict[str, object]] = []
    best_val = float("inf")
    epochs = int(config["train"]["epochs"])
    progress = tqdm(range(1, epochs + 1), desc=f"alignment {run_name}")
    top1_key = f"retrieval_top1_{'label' if protocol == 'U' else 'exact'}"
    for epoch in progress:
        model.train()
        sums = {key: 0.0 for key in ALIGNMENT_KEYS}
        total = 0
        for batch in train_loader:
            eeg = batch["eeg"].to(device)
            target_sequence = batch["target_sequence"].to(device)
            target_mask = batch["target_mask"].to(device)
            target_summary = batch["target_summary"].to(device)
            label_ids = batch["label_id"].to(device)
            valid_steps = _valid_steps_from_samples(batch["eeg_valid_num_samples"].to(device))
            outputs = model(eeg, subject_indices=batch["subject_index"].to(device), valid_steps=valid_steps)
            losses = compute_alignment_losses(
                pred_sequence=outputs["speech_sequence"],
                target_sequence=target_sequence,
                target_mask=target_mask,
                pred_summary=outputs["speech_embedding"],
                target_summary=target_summary,
                pred_prosody=outputs.get("prosody"),
                target_prosody=batch["prosody_target"].to(device) if "prosody_target" in batch else None,
                lambda_seq_cosine=float(config["train"].get("lambda_seq_cosine", config["train"].get("lambda_cosine", 1.0))),
                lambda_seq_mse=float(config["train"].get("lambda_seq_mse", config["train"].get("lambda_mse", 0.5))),
                lambda_contrastive=float(config["train"].get("lambda_contrastive", 1.0)),
                lambda_prosody=float(config["train"].get("lambda_prosody", 0.0)),
                contrastive_temperature=float(config["train"].get("contrastive_temperature", 0.07)),
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
            r1=f"{float(val_metrics.get(top1_key, 0.0) or 0.0):.3f}",
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
                    "target_kind": train_ds.target_kind,
                    "target_steps": train_ds.target_sequence_steps,
                    "target_dim": train_ds.target_sequence_dim,
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
            "target_kind": train_ds.target_kind,
            "history": history,
            "test": test_metrics,
        },
    )


if __name__ == "__main__":
    main()

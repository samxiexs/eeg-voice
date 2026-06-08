from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.alignment_losses import compute_alignment_losses
from src.alignment_model import EEGSpeechAlignmentModel
from src.dataset import FEISProtocolDataset
from src.eval_utils import nearest_centroid_subject_probe, retrieval_topk
from src.utils import load_simple_yaml, resolve_bundle_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FEIS EEG-to-speech alignment model and priors-only baselines.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "alignment.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--holdout-subject", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--stage", default=None)
    parser.add_argument("--ablation-mode", default=None)
    parser.add_argument("--include-anomalous", action="store_true")
    parser.add_argument(
        "--retrieval-bank",
        default="auto",
        choices=["auto", "train_templates", "oracle_test_templates"],
    )
    return parser.parse_args()


def _valid_steps_from_samples(valid_samples: torch.Tensor) -> torch.Tensor:
    return torch.ceil(valid_samples.float() / 32.0).long().clamp_min(1)


def build_dataset(config: dict, args: argparse.Namespace, split: str, include_anomalous: bool | None = None) -> FEISProtocolDataset:
    cfg_data = config["data"]
    cfg_audio = config["audio"]
    resolved_root = resolve_bundle_path(args.data_root or cfg_data["root"], BUNDLE_DIR)
    protocol = str(args.protocol or cfg_data["protocol"]).upper()
    if include_anomalous is None:
        include_anomalous = bool(args.include_anomalous or cfg_data.get("include_anomalous", False))
    return FEISProtocolDataset(
        data_root=str(resolved_root),
        protocol=protocol,
        split=split,
        stage=str(args.stage or cfg_data["stage"]),
        subject_id=args.subject or cfg_data.get("subject_id"),
        holdout_subject_id=args.holdout_subject or cfg_data.get("holdout_subject_id"),
        include_anomalous=include_anomalous,
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


def choose_retrieval_bank(dataset: FEISProtocolDataset, policy: str) -> tuple[np.ndarray, list[str], str]:
    policy = str(policy)
    if policy == "auto":
        policy = "oracle_test_templates" if dataset.protocol == "U" else "train_templates"
    split = "train" if policy == "train_templates" else "test"
    template_ids = dataset.unique_template_ids(split=split)
    bank = np.stack([dataset.get_template_target(template_id)["speech_embedding"] for template_id in template_ids], axis=0)
    return bank.astype(np.float32), template_ids, policy


def _masked_metrics(pred: np.ndarray, target: np.ndarray, availability: np.ndarray) -> dict[str, float | None]:
    mask = availability > 0.5
    if mask.sum() == 0:
        return {
            "availability_rate": 0.0,
            "embedding_cosine": None,
            "embedding_mse": None,
        }
    pred_masked = pred[mask]
    target_masked = target[mask]
    pred_norm = pred_masked / np.clip(np.linalg.norm(pred_masked, axis=1, keepdims=True), 1e-8, None)
    target_norm = target_masked / np.clip(np.linalg.norm(target_masked, axis=1, keepdims=True), 1e-8, None)
    cosine = float(np.mean(np.sum(pred_norm * target_norm, axis=1)))
    mse = float(np.mean((pred_masked - target_masked) ** 2))
    return {
        "availability_rate": float(mask.mean()),
        "embedding_cosine": cosine,
        "embedding_mse": mse,
    }


def predict_dataset(
    model: EEGSpeechAlignmentModel,
    dataset: FEISProtocolDataset,
    loader: DataLoader,
    device: torch.device,
    config: dict,
) -> dict[str, object]:
    losses = {
        "total": 0.0,
        "embedding_cosine": 0.0,
        "embedding_mse": 0.0,
        "prosody_loss": 0.0,
        "cls_acc": 0.0,
    }
    total = 0
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    subject_ids: list[str] = []
    template_ids: list[str] = []
    labels: list[str] = []
    trial_indices: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device)
            speech_target = batch["speech_embedding"].to(device)
            prosody_target = batch["prosody_target"].to(device)
            label_ids = batch["label_id"].to(device)
            valid_steps = _valid_steps_from_samples(batch["eeg_valid_num_samples"].to(device))
            outputs = model(eeg, subject_indices=batch["subject_index"].to(device), valid_steps=valid_steps)
            computed = compute_alignment_losses(
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
            losses["total"] += float(computed["total"].item()) * batch_size
            losses["embedding_cosine"] += float(computed["embedding_cosine"].item()) * batch_size
            losses["embedding_mse"] += float(computed["embedding_mse"].item()) * batch_size
            losses["prosody_loss"] += float(computed["prosody_loss"].item()) * batch_size
            losses["cls_acc"] += float(computed["cls_acc"].item()) * batch_size
            preds.append(outputs["speech_embedding"].detach().cpu().numpy())
            targets.append(batch["speech_embedding"].cpu().numpy())
            subject_ids.extend(list(batch["subject_id"]))
            template_ids.extend(list(batch["template_id"]))
            labels.extend(list(batch["label"]))
            trial_indices.extend([int(item) for item in batch["trial_index"].tolist()])
    predictions = np.concatenate(preds, axis=0)
    target_matrix = np.concatenate(targets, axis=0)
    mean_losses = {key: value / max(total, 1) for key, value in losses.items()}
    rows = []
    pred_norm = predictions / np.clip(np.linalg.norm(predictions, axis=1, keepdims=True), 1e-8, None)
    target_norm = target_matrix / np.clip(np.linalg.norm(target_matrix, axis=1, keepdims=True), 1e-8, None)
    per_sample_cos = np.sum(pred_norm * target_norm, axis=1)
    for idx in range(len(template_ids)):
        rows.append(
            {
                "subject_id": subject_ids[idx],
                "label": labels[idx],
                "trial_index": trial_indices[idx],
                "template_id": template_ids[idx],
                "embedding_cosine": float(per_sample_cos[idx]),
            }
        )
    return {
        "metrics": mean_losses,
        "predictions": predictions,
        "targets": target_matrix,
        "subject_ids": subject_ids,
        "template_ids": template_ids,
        "rows": rows,
    }


def evaluate_controls(dataset: FEISProtocolDataset, protocol: str) -> dict[str, object]:
    target_matrix = np.stack(
        [dataset.get_template_target(template_id)["speech_embedding"] for template_id in [entry.template_id for entry in dataset.entries]],
        axis=0,
    )
    controls = {
        "label_only": dataset.build_control_predictions("label_only", use_oracle_for_unseen=False),
    }
    if protocol == "U":
        controls["subject_only_strict"] = dataset.build_control_predictions("subject_only", use_oracle_for_unseen=False)
        controls["subject_only_oracle"] = dataset.build_control_predictions("subject_only", use_oracle_for_unseen=True)
        controls["label_subject_strict"] = dataset.build_control_predictions("label_subject", use_oracle_for_unseen=False)
        controls["label_subject_oracle"] = dataset.build_control_predictions("label_subject", use_oracle_for_unseen=True)
    else:
        controls["subject_only"] = dataset.build_control_predictions("subject_only", use_oracle_for_unseen=False)
        controls["label_subject"] = dataset.build_control_predictions("label_subject", use_oracle_for_unseen=False)

    out: dict[str, object] = {}
    for key, payload in controls.items():
        out[key] = _masked_metrics(
            pred=payload["speech_embeddings"],
            target=target_matrix,
            availability=payload["availability"],
        )
    return out


def run_eval(config: dict, args: argparse.Namespace, include_anomalous: bool | None = None) -> dict[str, object]:
    split = str(args.split)
    train_ds = build_dataset(config, args, split="train", include_anomalous=include_anomalous)
    eval_ds = build_dataset(config, args, split=split, include_anomalous=include_anomalous)
    train_loader = DataLoader(train_ds, batch_size=int(config["train"]["batch_size"]), shuffle=False, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=int(config["train"]["batch_size"]), shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else resolve_bundle_path(config["output"]["root"], BUNDLE_DIR)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "checkpoints" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(config, train_ds).to(device)
    model.load_state_dict(checkpoint["model_state"])

    train_pred = predict_dataset(model, train_ds, train_loader, device, config)
    eval_pred = predict_dataset(model, eval_ds, eval_loader, device, config)
    bank_embeddings, bank_ids, bank_policy = choose_retrieval_bank(eval_ds, args.retrieval_bank)
    retrieval = retrieval_topk(
        predicted=eval_pred["predictions"],
        target_template_ids=eval_pred["template_ids"],
        bank_embeddings=bank_embeddings,
        bank_template_ids=bank_ids,
        topk=(1, 5),
    )
    subject_probe = nearest_centroid_subject_probe(
        train_embeddings=train_pred["predictions"],
        train_subject_ids=train_pred["subject_ids"],
        eval_embeddings=eval_pred["predictions"],
        eval_subject_ids=eval_pred["subject_ids"],
    )
    result = {
        "protocol": eval_ds.protocol,
        "split": split,
        "checkpoint": str(checkpoint_path),
        "retrieval_bank_policy": bank_policy,
        "num_samples": len(eval_ds),
        "include_anomalous": bool(include_anomalous if include_anomalous is not None else args.include_anomalous),
        "model": {
            **eval_pred["metrics"],
            **retrieval,
            "subject_probe_accuracy": subject_probe,
        },
        "controls": evaluate_controls(eval_ds, eval_ds.protocol),
        "predictions": eval_pred["rows"],
    }
    return result


def main() -> None:
    args = parse_args()
    config = load_simple_yaml(args.config)
    if args.protocol is not None:
        config["data"]["protocol"] = args.protocol
    if args.stage is not None:
        config["data"]["stage"] = args.stage
    if args.ablation_mode is not None:
        config["data"]["ablation_mode"] = args.ablation_mode
    result = run_eval(config, args, include_anomalous=None)
    protocol = str(config["data"]["protocol"]).upper()
    sensitivity = None
    if protocol in {"G", "U"} and not bool(args.include_anomalous or config["data"].get("include_anomalous", False)):
        sensitivity = run_eval(config, args, include_anomalous=True)
        sensitivity.pop("predictions", None)

    output_root = resolve_bundle_path(args.output_root or config["output"]["root"], BUNDLE_DIR)
    run_name = f"{protocol.lower()}_{config['data']['stage']}_{config['data'].get('ablation_mode', 'none')}"
    if protocol == "S":
        run_name += f"_subject_{args.subject or config['data'].get('subject_id')}"
    if protocol == "U":
        run_name += f"_holdout_{args.holdout_subject or config['data'].get('holdout_subject_id')}"
    metrics_path = output_root / run_name / "metrics" / f"{args.split}_evaluation.json"
    payload = dict(result)
    if sensitivity is not None:
        payload["sensitivity_including_subject_05"] = sensitivity
    write_json(metrics_path, payload)
    print(f"Saved evaluation to {metrics_path}")
    print(f"Embedding cosine: {payload['model']['embedding_cosine']:.4f}")
    print(f"Retrieval@1: {payload['model']['retrieval_top1']:.4f}")
    if sensitivity is not None and sensitivity["model"]["embedding_cosine"] is not None:
        print(f"Sensitivity w/ subject 05 cosine: {sensitivity['model']['embedding_cosine']:.4f}")


if __name__ == "__main__":
    main()

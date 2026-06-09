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
from src.alignment_retrieval import (
    build_retrieval_bank,
    build_waveform_distance_bank,
    evaluate_embedding_retrieval,
    evaluate_waveform_nta,
    expected_random_retrieval_metrics,
    resolve_retrieval_policy,
    summarize_candidates,
)
from src.dataset import FEISProtocolDataset
from src.losses import stft_distance_single
from src.utils import ensure_dir, load_simple_yaml, resolve_bundle_path, save_wav, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FEIS alignment checkpoints with retrieval-waveform reconstruction.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "alignment_ssl_local.yaml"))
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
        "--retrieval-policy",
        default="auto",
        choices=["auto", "same_subject_train", "pooled_train", "unseen_strict_seen_subjects", "unseen_oracle_holdout"],
    )
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def build_run_name(config: dict, args: argparse.Namespace) -> str:
    protocol = str(config["data"]["protocol"]).upper()
    run_name = f"{protocol.lower()}_{config['data']['stage']}_{config['data'].get('ablation_mode', 'none')}"
    if protocol == "S":
        run_name += f"_subject_{args.subject or config['data'].get('subject_id')}"
    if protocol == "U":
        run_name += f"_holdout_{args.holdout_subject or config['data'].get('holdout_subject_id')}"
    return run_name


def _valid_steps_from_samples(valid_samples: torch.Tensor) -> torch.Tensor:
    return torch.ceil(valid_samples.float() / 32.0).long().clamp_min(1)


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
    target_waveforms: list[np.ndarray] = []
    subject_ids: list[str] = []
    template_ids: list[str] = []
    labels: list[str] = []
    trial_indices: list[int] = []
    audio_paths: list[str] = []
    rows: list[dict[str, object]] = []
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
            pred_np = outputs["speech_embedding"].detach().cpu().numpy()
            target_np = batch["speech_embedding"].cpu().numpy()
            preds.append(pred_np)
            targets.append(target_np)
            target_waveforms.append(batch["waveform"].cpu().numpy())
            subject_ids.extend(list(batch["subject_id"]))
            template_ids.extend(list(batch["template_id"]))
            labels.extend(list(batch["label"]))
            trial_indices.extend([int(item) for item in batch["trial_index"].tolist()])
            audio_paths.extend(list(batch["audio_path"]))
            pred_norm = pred_np / np.clip(np.linalg.norm(pred_np, axis=1, keepdims=True), 1e-8, None)
            target_norm = target_np / np.clip(np.linalg.norm(target_np, axis=1, keepdims=True), 1e-8, None)
            cosine_values = np.sum(pred_norm * target_norm, axis=1)
            for idx in range(batch_size):
                rows.append(
                    {
                        "subject_id": str(batch["subject_id"][idx]),
                        "label": str(batch["label"][idx]),
                        "trial_index": int(batch["trial_index"][idx].item()),
                        "template_id": str(batch["template_id"][idx]),
                        "audio_path": str(batch["audio_path"][idx]),
                        "embedding_cosine": float(cosine_values[idx]),
                    }
                )
    return {
        "metrics": {key: value / max(total, 1) for key, value in losses.items()},
        "predictions": np.concatenate(preds, axis=0),
        "targets": np.concatenate(targets, axis=0),
        "target_waveforms": np.concatenate(target_waveforms, axis=0),
        "subject_ids": subject_ids,
        "template_ids": template_ids,
        "labels": labels,
        "trial_indices": trial_indices,
        "audio_paths": audio_paths,
        "rows": rows,
    }


def _target_metric_key(match_mode: str, prefix: str) -> str:
    suffix = "exact" if match_mode == "exact" else "label"
    return f"{prefix}_{suffix}"


def evaluate_policy(
    predictions: dict[str, object],
    bank,
    output_dir: Path,
    sample_rate: int,
    top_k: int,
    save_retrieved_audio: bool,
) -> dict[str, object]:
    ensure_dir(output_dir)
    embedding_eval = evaluate_embedding_retrieval(
        predicted=predictions["predictions"],
        target_template_ids=predictions["template_ids"],
        target_labels=predictions["labels"],
        bank=bank,
        top_k=top_k,
        evaluation_mask=None,
    )
    bank_index = {template_id: idx for idx, template_id in enumerate(bank.template_ids)}
    top1_template_ids = [candidates[0]["template_id"] for candidates in embedding_eval["ranked_candidates"]]
    retrieved_waveforms = [bank.waveforms[bank_index[template_id]] for template_id in top1_template_ids]
    waveform_bank = build_waveform_distance_bank(bank)
    waveform_eval = evaluate_waveform_nta(
        output_waveforms=retrieved_waveforms,
        target_template_ids=predictions["template_ids"],
        target_labels=predictions["labels"],
        bank_features=waveform_bank,
        evaluation_mask=None,
        cache_keys=top1_template_ids,
    )
    sample_rows = summarize_candidates(
        ranked_candidates=embedding_eval["ranked_candidates"],
        nearest_rows=waveform_eval["nearest_rows"],
        target_template_ids=predictions["template_ids"],
        target_labels=predictions["labels"],
        match_mode=bank.match_mode,
    )
    recon_dir = output_dir / "recon_wavs"
    target_distances: list[float] = []
    for idx, row in enumerate(sample_rows):
        save_path = recon_dir / f"{predictions['subject_ids'][idx]}_{predictions['labels'][idx]}_trial{predictions['trial_indices'][idx]:04d}_retrieved.wav"
        if save_retrieved_audio:
            save_wav(save_path, retrieved_waveforms[idx], sample_rate)
        target_distance = stft_distance_single(
            torch.from_numpy(np.asarray(retrieved_waveforms[idx], dtype=np.float32)),
            torch.from_numpy(np.asarray(predictions["target_waveforms"][idx], dtype=np.float32)),
        )
        target_distances.append(float(target_distance))
        row.update(
            {
                "subject_id": predictions["subject_ids"][idx],
                "label": predictions["labels"][idx],
                "trial_index": predictions["trial_indices"][idx],
                "template_id": predictions["template_ids"][idx],
                "audio_path": predictions["audio_paths"][idx],
                "embedding_cosine_to_target": predictions["rows"][idx]["embedding_cosine"],
                "saved_wav_path": str(save_path) if save_retrieved_audio else None,
                "retrieved_target_stft_distance": float(target_distance),
            }
        )
    np.savez_compressed(
        output_dir / "predicted_embeddings.npz",
        predictions=np.asarray(predictions["predictions"], dtype=np.float32),
        targets=np.asarray(predictions["targets"], dtype=np.float32),
        template_ids=np.asarray(predictions["template_ids"]),
        labels=np.asarray(predictions["labels"]),
        subject_ids=np.asarray(predictions["subject_ids"]),
        trial_indices=np.asarray(predictions["trial_indices"], dtype=np.int32),
    )
    write_json(output_dir / "retrieval_predictions.json", {"predictions": sample_rows})
    summary = {
        **predictions["metrics"],
        **embedding_eval["metrics"],
        **waveform_eval["metrics"],
        "retrieval_policy": bank.policy,
        "candidate_pool_size": bank.size,
        "embedding_dim": int(predictions["predictions"].shape[1]),
        "mean_retrieved_target_stft_distance": float(np.mean(target_distances)) if target_distances else None,
    }
    write_json(output_dir / "summary.json", summary)
    return {
        "summary": summary,
        "predictions": sample_rows,
    }


def evaluate_control_embeddings(
    predicted_embeddings: np.ndarray,
    availability: np.ndarray,
    bank,
    target_template_ids: list[str],
    target_labels: list[str],
    top_k: int,
) -> dict[str, object]:
    mask = np.asarray(availability, dtype=np.float32) > 0.5
    embedding_eval = evaluate_embedding_retrieval(
        predicted=predicted_embeddings,
        target_template_ids=target_template_ids,
        target_labels=target_labels,
        bank=bank,
        top_k=top_k,
        evaluation_mask=mask,
    )
    if int(mask.sum()) == 0:
        return embedding_eval["metrics"]
    bank_index = {template_id: idx for idx, template_id in enumerate(bank.template_ids)}
    top1_template_ids = [
        candidates[0]["template_id"] if mask[idx] else ""
        for idx, candidates in enumerate(embedding_eval["ranked_candidates"])
    ]
    retrieved_waveforms = [
        np.zeros_like(bank.waveforms[0]) if not mask[idx] else bank.waveforms[bank_index[top1_template_ids[idx]]]
        for idx in range(len(top1_template_ids))
    ]
    waveform_eval = evaluate_waveform_nta(
        output_waveforms=retrieved_waveforms,
        target_template_ids=target_template_ids,
        target_labels=target_labels,
        bank_features=build_waveform_distance_bank(bank),
        evaluation_mask=mask,
        cache_keys=top1_template_ids,
    )
    return {
        **embedding_eval["metrics"],
        **waveform_eval["metrics"],
    }


def build_control_summaries(
    dataset: FEISProtocolDataset,
    bank,
    target_template_ids: list[str],
    target_labels: list[str],
    top_k: int,
) -> dict[str, object]:
    controls: dict[str, object] = {
        "random": expected_random_retrieval_metrics(
            bank=bank,
            target_template_ids=target_template_ids,
            target_labels=target_labels,
            top_k=top_k,
        ),
        "label_only": evaluate_control_embeddings(
            predicted_embeddings=dataset.build_control_predictions("label_only", use_oracle_for_unseen=False)["speech_embeddings"],
            availability=np.ones(len(target_template_ids), dtype=np.float32),
            bank=bank,
            target_template_ids=target_template_ids,
            target_labels=target_labels,
            top_k=top_k,
        ),
    }
    if bank.policy == "unseen_strict_seen_subjects":
        strict_subject = dataset.build_control_predictions("subject_only", use_oracle_for_unseen=False)
        strict_label_subject = dataset.build_control_predictions("label_subject", use_oracle_for_unseen=False)
        controls["subject_only_strict"] = evaluate_control_embeddings(
            predicted_embeddings=strict_subject["speech_embeddings"],
            availability=strict_subject["availability"],
            bank=bank,
            target_template_ids=target_template_ids,
            target_labels=target_labels,
            top_k=top_k,
        )
        controls["label_subject_strict"] = evaluate_control_embeddings(
            predicted_embeddings=strict_label_subject["speech_embeddings"],
            availability=strict_label_subject["availability"],
            bank=bank,
            target_template_ids=target_template_ids,
            target_labels=target_labels,
            top_k=top_k,
        )
    elif bank.policy == "unseen_oracle_holdout":
        oracle_subject = dataset.build_control_predictions("subject_only", use_oracle_for_unseen=True)
        oracle_label_subject = dataset.build_control_predictions("label_subject", use_oracle_for_unseen=True)
        controls["subject_only_oracle"] = evaluate_control_embeddings(
            predicted_embeddings=oracle_subject["speech_embeddings"],
            availability=oracle_subject["availability"],
            bank=bank,
            target_template_ids=target_template_ids,
            target_labels=target_labels,
            top_k=top_k,
        )
        controls["label_subject_oracle"] = evaluate_control_embeddings(
            predicted_embeddings=oracle_label_subject["speech_embeddings"],
            availability=oracle_label_subject["availability"],
            bank=bank,
            target_template_ids=target_template_ids,
            target_labels=target_labels,
            top_k=top_k,
        )
    else:
        subject_only = dataset.build_control_predictions("subject_only", use_oracle_for_unseen=False)
        label_subject = dataset.build_control_predictions("label_subject", use_oracle_for_unseen=False)
        controls["subject_only"] = evaluate_control_embeddings(
            predicted_embeddings=subject_only["speech_embeddings"],
            availability=subject_only["availability"],
            bank=bank,
            target_template_ids=target_template_ids,
            target_labels=target_labels,
            top_k=top_k,
        )
        controls["label_subject"] = evaluate_control_embeddings(
            predicted_embeddings=label_subject["speech_embeddings"],
            availability=label_subject["availability"],
            bank=bank,
            target_template_ids=target_template_ids,
            target_labels=target_labels,
            top_k=top_k,
        )
    return controls


def build_phase2_alerts(summary: dict[str, object]) -> dict[str, object]:
    match_mode = str(summary.get("match_mode", "exact"))
    retrieval_key = _target_metric_key(match_mode, "retrieval_top1")
    cosine = summary.get("embedding_cosine")
    retrieval = summary.get(retrieval_key)
    high_cosine_poor_retrieval = (
        isinstance(cosine, (float, int))
        and isinstance(retrieval, (float, int))
        and float(cosine) >= 0.85
        and float(retrieval) <= 0.15
    )
    alerts = {
        "high_cosine_poor_retrieval": bool(high_cosine_poor_retrieval),
    }
    if high_cosine_poor_retrieval:
        alerts["recommended_followup"] = (
            "If pooled HuBERT embeddings exhibit high cosine similarity but poor retrieval discrimination, "
            "investigate sequence-level HuBERT representations and codec-latent targets before investing "
            "further effort into template retrieval optimization."
        )
    return alerts


def run_policy_eval(
    config: dict,
    args: argparse.Namespace,
    predictions: dict[str, object],
    dataset: FEISProtocolDataset,
    output_root: Path,
    policy: str,
    top_k: int,
    save_retrieved_audio: bool,
) -> dict[str, object]:
    bank = build_retrieval_bank(dataset, policy=policy)
    policy_dir = output_root / "retrieval" / str(args.split) / bank.policy
    model_eval = evaluate_policy(
        predictions=predictions,
        bank=bank,
        output_dir=policy_dir,
        sample_rate=int(config["audio"]["sample_rate"]),
        top_k=top_k,
        save_retrieved_audio=save_retrieved_audio,
    )
    controls = build_control_summaries(
        dataset=dataset,
        bank=bank,
        target_template_ids=predictions["template_ids"],
        target_labels=predictions["labels"],
        top_k=top_k,
    )
    result = {
        "retrieval_policy": bank.policy,
        "match_mode": bank.match_mode,
        "candidate_pool_size": bank.size,
        "model": model_eval["summary"],
        "controls": controls,
        "predictions_path": str(policy_dir / "retrieval_predictions.json"),
        "predicted_embeddings_path": str(policy_dir / "predicted_embeddings.npz"),
    }
    result["alerts"] = build_phase2_alerts(model_eval["summary"])
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

    split = str(args.split)
    train_ds = build_dataset(config, args, split="train")
    eval_ds = build_dataset(config, args, split=split)
    loader = DataLoader(eval_ds, batch_size=int(config["train"]["batch_size"]), shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else resolve_bundle_path(config["output"]["root"], BUNDLE_DIR) / build_run_name(config, args)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "checkpoints" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(config, train_ds).to(device)
    model.load_state_dict(checkpoint["model_state"])

    predictions = predict_dataset(model=model, dataset=eval_ds, loader=loader, device=device, config=config)
    output_root = resolve_bundle_path(args.output_root or config["output"]["root"], BUNDLE_DIR) / build_run_name(config, args)
    resolved_policy = resolve_retrieval_policy(eval_ds.protocol, args.retrieval_policy)
    payload = {
        "protocol": eval_ds.protocol,
        "split": split,
        "checkpoint": str(checkpoint_path),
        "num_samples": len(eval_ds),
        "include_anomalous": bool(args.include_anomalous or config["data"].get("include_anomalous", False)),
        "phase2_target_cache": str(eval_ds.target_cache["path"]),
        "phase2_target_backend": str(eval_ds.target_cache["feature_backend"][0]),
        "model": run_policy_eval(
            config=config,
            args=args,
            predictions=predictions,
            dataset=eval_ds,
            output_root=output_root,
            policy=resolved_policy,
            top_k=max(1, int(args.top_k)),
            save_retrieved_audio=True,
        ),
    }
    if str(eval_ds.protocol).upper() == "U" and resolved_policy == "unseen_strict_seen_subjects":
        payload["oracle_ceiling"] = run_policy_eval(
            config=config,
            args=args,
            predictions=predictions,
            dataset=eval_ds,
            output_root=output_root,
            policy="unseen_oracle_holdout",
            top_k=max(1, int(args.top_k)),
            save_retrieved_audio=False,
        )
    metrics_path = output_root / "metrics" / f"{split}_retrieval_evaluation.json"
    write_json(metrics_path, payload)
    print(f"Saved retrieval evaluation to {metrics_path}")
    main_model = payload["model"]["model"]
    match_mode = str(main_model["match_mode"])
    retrieval_key = _target_metric_key(match_mode, "retrieval_top1")
    nta_key = _target_metric_key(match_mode, "NTA")
    print(f"Embedding cosine: {main_model['embedding_cosine']:.4f}")
    print(f"{retrieval_key}: {main_model[retrieval_key]:.4f}")
    print(f"{nta_key}: {main_model[nta_key]:.4f}")


if __name__ == "__main__":
    main()

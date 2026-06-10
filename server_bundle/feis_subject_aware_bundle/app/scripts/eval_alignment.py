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
    compute_first_match_ranks,
    evaluate_embedding_retrieval,
    evaluate_waveform_nta,
    expected_random_retrieval_metrics,
    resolve_retrieval_policy,
    summarize_candidates,
)
from src.audio_features import (
    TARGET_KIND_ENCODEC_LATENT,
    AudioFeatureConfig,
    load_codec_backend,
)
from src.dataset import FEISProtocolDataset
from src.losses import stft_distance_single
from src.utils import (
    build_protocol_run_name,
    ensure_dir,
    load_simple_yaml,
    pad_or_crop_audio,
    resample_audio,
    resolve_bundle_path,
    save_wav,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FEIS alignment checkpoints for sequence and codec targets.")
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
        speech_embedding_dim=int(dataset.target_sequence_dim),
        prosody_dim=int(dataset.prosody_dim) if bool(cfg_model.get("use_prosody_head", dataset.prosody_dim > 0)) else 0,
        num_labels=int(dataset.num_labels),
        latent_dim=int(cfg_model.get("latent_dim", 192)),
        target_steps=int(dataset.target_sequence_steps),
        use_label_head=bool(cfg_model.get("use_label_head", True)),
        use_subject_demo_head=bool(cfg_model.get("use_subject_demo_head", False)),
        num_subjects=int(dataset.num_subjects),
        subject_embedding_dim=int(cfg_model.get("subject_embedding_dim", 64)),
        use_codec_scale_head=bool(cfg_model.get("use_codec_scale_head", dataset.target_kind == TARGET_KIND_ENCODEC_LATENT)),
        use_phoneme_head=bool(cfg_model.get("use_phoneme_head", False)),
        num_phoneme_tokens=int(dataset.phoneme_vocab.size),
        phoneme_steps=int(dataset.phoneme_vocab.max_steps),
    )


def build_run_name(config: dict, args: argparse.Namespace) -> str:
    return build_protocol_run_name(
        config=config,
        protocol=str(config["data"]["protocol"]).upper(),
        stage=str(config["data"]["stage"]),
        ablation_mode=str(config["data"].get("ablation_mode", "none")),
        subject_id=args.subject or config["data"].get("subject_id"),
        holdout_subject_id=args.holdout_subject or config["data"].get("holdout_subject_id"),
    )


def build_feature_config(config: dict) -> AudioFeatureConfig:
    audio_cfg = config["audio"]
    target_cfg = config["targets"]
    return AudioFeatureConfig(
        sample_rate=int(audio_cfg["sample_rate"]),
        duration_sec=float(audio_cfg["duration_sec"]),
        normalize=str(audio_cfg.get("normalize", "rms")),
        target_rms=float(audio_cfg.get("target_rms", 0.08)),
        max_gain=float(audio_cfg.get("max_gain", 10.0)),
        backend=str(target_cfg.get("backend", "auto")),
        target_kind=str(target_cfg.get("target_kind", "hubert_pooled")),
        ssl_model_name_or_path=str(
            resolve_bundle_path(
                target_cfg.get("ssl_model_name_or_path", target_cfg.get("hubert_model_name_or_path", "facebook/hubert-base-ls960")),
                BUNDLE_DIR,
            )
        ),
        codec_model_name_or_path=str(
            resolve_bundle_path(target_cfg.get("codec_model_name_or_path", "facebook/encodec_24khz"), BUNDLE_DIR)
        ),
        local_files_only=bool(target_cfg.get("local_files_only", True)),
        spectral_bins=int(target_cfg.get("spectral_bins", 48)),
        sequence_target_steps=int(target_cfg.get("sequence_target_steps", 16)),
        codec_bandwidth=float(target_cfg.get("codec_bandwidth", 6.0)),
    )


def predict_dataset(
    model: EEGSpeechAlignmentModel,
    loader: DataLoader,
    device: torch.device,
    config: dict,
    use_codec_scale_prediction: bool = True,
) -> dict[str, object]:
    losses = {
        "total": 0.0,
        "sequence_cosine": 0.0,
        "sequence_mse": 0.0,
        "summary_cosine": 0.0,
        "embedding_cosine": 0.0,
        "embedding_mse": 0.0,
        "contrastive": 0.0,
        "prosody_loss": 0.0,
        "codec_scale_loss": 0.0,
        "codec_log_rms_mae": 0.0,
        "phoneme_loss": 0.0,
        "phoneme_acc": 0.0,
        "cls_acc": 0.0,
    }
    total = 0
    pred_sequences: list[np.ndarray] = []
    pred_summaries: list[np.ndarray] = []
    target_sequences: list[np.ndarray] = []
    raw_target_sequences: list[np.ndarray] = []
    target_summaries: list[np.ndarray] = []
    target_masks: list[np.ndarray] = []
    target_waveforms: list[np.ndarray] = []
    decoder_scales: list[np.ndarray] = []
    pred_log_rms_values: list[np.ndarray] = []
    target_log_rms_values: list[np.ndarray] = []
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
            target_sequence = batch["target_sequence"].to(device)
            target_mask = batch["target_mask"].to(device)
            target_summary = batch["target_summary"].to(device)
            label_ids = batch["label_id"].to(device)
            valid_steps = _valid_steps_from_samples(batch["eeg_valid_num_samples"].to(device))
            outputs = model(eeg, subject_indices=batch["subject_index"].to(device), valid_steps=valid_steps)
            computed = compute_alignment_losses(
                pred_sequence=outputs["speech_sequence"],
                target_sequence=target_sequence,
                target_mask=target_mask,
                pred_summary=outputs["speech_embedding"],
                target_summary=target_summary,
                pred_prosody=outputs.get("prosody"),
                target_prosody=batch["prosody_target"].to(device) if "prosody_target" in batch else None,
                pred_codec_log_rms=outputs.get("codec_log_rms") if use_codec_scale_prediction else None,
                target_log_rms=batch["target_log_rms"].to(device) if "target_log_rms" in batch else None,
                lambda_codec_scale=float(config["train"].get("lambda_codec_scale", 0.0)),
                phoneme_logits=outputs.get("phoneme_logits"),
                phoneme_ids=batch["phoneme_ids"].to(device) if "phoneme_ids" in batch else None,
                phoneme_mask=batch["phoneme_mask"].to(device) if "phoneme_mask" in batch else None,
                lambda_phoneme=float(config["train"].get("lambda_phoneme", 0.0)),
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
            for key in losses:
                losses[key] += float(computed[key].item()) * batch_size
            pred_sequence_np = outputs["speech_sequence"].detach().cpu().numpy()
            pred_summary_np = outputs["speech_embedding"].detach().cpu().numpy()
            target_sequence_np = batch["target_sequence"].cpu().numpy()
            target_summary_np = batch["target_summary"].cpu().numpy()
            pred_sequences.append(pred_sequence_np)
            pred_summaries.append(pred_summary_np)
            target_sequences.append(target_sequence_np)
            raw_target_sequences.append(batch["raw_target_sequence"].cpu().numpy())
            target_summaries.append(target_summary_np)
            target_masks.append(batch["target_mask"].cpu().numpy())
            target_waveforms.append(batch["waveform"].cpu().numpy())
            decoder_scales.append(batch["decoder_scale"].cpu().numpy())
            pred_log_rms_values.append(
                outputs["codec_log_rms"].detach().cpu().numpy()
                if "codec_log_rms" in outputs and use_codec_scale_prediction
                else np.full((batch_size,), np.nan, dtype=np.float32)
            )
            target_log_rms_values.append(batch["target_log_rms"].cpu().numpy())
            subject_ids.extend(list(batch["subject_id"]))
            template_ids.extend(list(batch["template_id"]))
            labels.extend(list(batch["label"]))
            trial_indices.extend([int(item) for item in batch["trial_index"].tolist()])
            audio_paths.extend(list(batch["audio_path"]))
            pred_norm = pred_summary_np / np.clip(np.linalg.norm(pred_summary_np, axis=1, keepdims=True), 1e-8, None)
            target_norm = target_summary_np / np.clip(np.linalg.norm(target_summary_np, axis=1, keepdims=True), 1e-8, None)
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
        "predicted_sequences": np.concatenate(pred_sequences, axis=0),
        "predicted_summaries": np.concatenate(pred_summaries, axis=0),
        "target_sequences": np.concatenate(target_sequences, axis=0),
        "raw_target_sequences": np.concatenate(raw_target_sequences, axis=0),
        "target_summaries": np.concatenate(target_summaries, axis=0),
        "target_masks": np.concatenate(target_masks, axis=0),
        "target_waveforms": np.concatenate(target_waveforms, axis=0),
        "decoder_scales": np.concatenate(decoder_scales, axis=0) if decoder_scales else np.zeros((0,), dtype=np.float32),
        "predicted_log_rms": np.concatenate(pred_log_rms_values, axis=0),
        "target_log_rms": np.concatenate(target_log_rms_values, axis=0),
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


def _resample_reconstruction(audio: np.ndarray, source_sr: int, target_sr: int, target_length: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if source_sr != target_sr:
        audio = resample_audio(audio, src_sr=source_sr, dst_sr=target_sr)
    return pad_or_crop_audio(audio, target_len=target_length)


def _inverse_normalize_sequence(
    sequence: np.ndarray,
    target_mean: np.ndarray | None,
    target_std: np.ndarray | None,
) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if target_mean is None or target_std is None:
        return sequence
    mean = np.asarray(target_mean, dtype=np.float32).reshape(1, -1)
    std = np.asarray(target_std, dtype=np.float32).reshape(1, -1)
    return (sequence * std + mean).astype(np.float32)


def _match_log_rms(audio: np.ndarray, log_rms: float | None, max_gain: float = 30.0) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if log_rms is None or not np.isfinite(float(log_rms)):
        return audio
    target_rms = float(np.exp(np.clip(float(log_rms), -12.0, -0.05)))
    current_rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)) + 1e-8)
    gain = float(np.clip(target_rms / current_rms, 1.0 / max_gain, max_gain))
    return (audio * gain).astype(np.float32)


def _decoder_scale_for_sample(predictions: dict[str, object], idx: int, default_decoder_scales: np.ndarray | None) -> np.ndarray | None:
    scales = np.asarray(predictions.get("decoder_scales", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    if scales.ndim >= 2 and scales.shape[0] > idx and scales.shape[1] > 0:
        return scales[idx]
    if scales.ndim == 1 and scales.size > 0:
        return scales
    return default_decoder_scales


def compute_latent_collapse_diagnostics(predicted_sequences: np.ndarray, target_sequences: np.ndarray) -> dict[str, float]:
    predicted_sequences = np.asarray(predicted_sequences, dtype=np.float32)
    target_sequences = np.asarray(target_sequences, dtype=np.float32)
    pred_flat = predicted_sequences.reshape(-1, predicted_sequences.shape[-1])
    target_flat = target_sequences.reshape(-1, target_sequences.shape[-1])
    pred_std = pred_flat.std(axis=0)
    target_std = target_flat.std(axis=0)
    pred_frame_var = predicted_sequences.var(axis=1).mean()
    target_frame_var = target_sequences.var(axis=1).mean()
    pred_centered = pred_flat - pred_flat.mean(axis=0, keepdims=True)
    target_centered = target_flat - target_flat.mean(axis=0, keepdims=True)
    return {
        "pred_sequence_std_mean": float(pred_std.mean()),
        "target_sequence_std_mean": float(target_std.mean()),
        "pred_target_std_ratio": float(pred_std.mean() / max(float(target_std.mean()), 1e-8)),
        "pred_frame_variance_mean": float(pred_frame_var),
        "target_frame_variance_mean": float(target_frame_var),
        "frame_variance_ratio": float(pred_frame_var / max(float(target_frame_var), 1e-8)),
        "pred_cov_trace": float(np.mean(np.square(pred_centered))),
        "target_cov_trace": float(np.mean(np.square(target_centered))),
    }


def evaluate_policy(
    predictions: dict[str, object],
    bank,
    output_dir: Path,
    sample_rate: int,
    top_k: int,
    reconstruction_mode: str,
    codec_backend=None,
    default_decoder_scales: np.ndarray | None = None,
    target_mean: np.ndarray | None = None,
    target_std: np.ndarray | None = None,
) -> dict[str, object]:
    ensure_dir(output_dir)
    embedding_eval = evaluate_embedding_retrieval(
        bank=bank,
        predicted_sequences=predictions["predicted_sequences"],
        predicted_summaries=predictions["predicted_summaries"],
        target_template_ids=predictions["template_ids"],
        target_labels=predictions["labels"],
        top_k=top_k,
        evaluation_mask=None,
    )
    first_match_ranks = compute_first_match_ranks(
        order=embedding_eval["order"],
        bank=bank,
        target_template_ids=predictions["template_ids"],
        target_labels=predictions["labels"],
    )
    bank_index = {template_id: idx for idx, template_id in enumerate(bank.template_ids)}
    top1_template_ids = [candidates[0]["template_id"] for candidates in embedding_eval["ranked_candidates"]]
    target_scale_oracle_waveforms: list[np.ndarray] = []
    target_latent_oracle_waveforms: list[np.ndarray] = []

    if reconstruction_mode == "codec_decode":
        if codec_backend is None:
            raise ValueError("codec_backend is required for codec reconstruction")
        reconstructed_waveforms = []
        for idx in range(len(predictions["template_ids"])):
            decoder_scales = _decoder_scale_for_sample(predictions, idx, default_decoder_scales)
            raw_pred_sequence = _inverse_normalize_sequence(
                predictions["predicted_sequences"][idx],
                target_mean=target_mean,
                target_std=target_std,
            )
            decoded = codec_backend.decode(raw_pred_sequence, decoder_scales=decoder_scales)
            decoded = _resample_reconstruction(
                decoded,
                source_sr=int(codec_backend.sample_rate),
                target_sr=int(sample_rate),
                target_length=int(predictions["target_waveforms"][idx].shape[-1]),
            )
            pred_scaled = _match_log_rms(decoded, float(predictions["predicted_log_rms"][idx]))
            target_scaled = _match_log_rms(decoded, float(predictions["target_log_rms"][idx]))
            raw_target_sequence = predictions.get("raw_target_sequences", predictions["target_sequences"])[idx]
            target_latent_decoded = codec_backend.decode(raw_target_sequence, decoder_scales=decoder_scales)
            target_latent_decoded = _resample_reconstruction(
                target_latent_decoded,
                source_sr=int(codec_backend.sample_rate),
                target_sr=int(sample_rate),
                target_length=int(predictions["target_waveforms"][idx].shape[-1]),
            )
            reconstructed_waveforms.append(pred_scaled)
            target_scale_oracle_waveforms.append(target_scaled)
            target_latent_oracle_waveforms.append(
                _match_log_rms(target_latent_decoded, float(predictions["target_log_rms"][idx]))
            )
    else:
        reconstructed_waveforms = [bank.waveforms[bank_index[template_id]] for template_id in top1_template_ids]

    waveform_bank = build_waveform_distance_bank(bank)
    waveform_eval = evaluate_waveform_nta(
        output_waveforms=reconstructed_waveforms,
        target_template_ids=predictions["template_ids"],
        target_labels=predictions["labels"],
        bank_features=waveform_bank,
        evaluation_mask=None,
        cache_keys=None if reconstruction_mode == "codec_decode" else top1_template_ids,
    )
    sample_rows = summarize_candidates(
        ranked_candidates=embedding_eval["ranked_candidates"],
        nearest_rows=waveform_eval["nearest_rows"],
        target_template_ids=predictions["template_ids"],
        target_labels=predictions["labels"],
        match_mode=bank.match_mode,
    )
    recon_dir = output_dir / "recon_wavs"
    target_scale_dir = output_dir / "target_scale_oracle_wavs"
    target_latent_dir = output_dir / "target_latent_oracle_wavs"
    target_distances: list[float] = []
    for idx, row in enumerate(sample_rows):
        save_path = recon_dir / (
            f"{predictions['subject_ids'][idx]}_{predictions['labels'][idx]}_trial"
            f"{predictions['trial_indices'][idx]:04d}_{reconstruction_mode}.wav"
        )
        save_wav(save_path, reconstructed_waveforms[idx], sample_rate)
        target_scale_path = None
        target_latent_path = None
        if reconstruction_mode == "codec_decode":
            target_scale_path = target_scale_dir / save_path.name.replace("_codec_decode.wav", "_target_scale_oracle.wav")
            target_latent_path = target_latent_dir / save_path.name.replace("_codec_decode.wav", "_target_latent_oracle.wav")
            save_wav(target_scale_path, target_scale_oracle_waveforms[idx], sample_rate)
            save_wav(target_latent_path, target_latent_oracle_waveforms[idx], sample_rate)
        target_distance = stft_distance_single(
            torch.from_numpy(np.asarray(reconstructed_waveforms[idx], dtype=np.float32)),
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
                "saved_wav_path": str(save_path),
                "target_scale_oracle_wav_path": None if target_scale_path is None else str(target_scale_path),
                "target_latent_oracle_wav_path": None if target_latent_path is None else str(target_latent_path),
                "reconstruction_mode": reconstruction_mode,
                "first_match_rank": first_match_ranks[idx],
                "first_match_reciprocal_rank": None
                if first_match_ranks[idx] is None
                else float(1.0 / first_match_ranks[idx]),
                "retrieved_target_stft_distance": float(target_distance),
                "predicted_log_rms": None
                if not np.isfinite(float(predictions["predicted_log_rms"][idx]))
                else float(predictions["predicted_log_rms"][idx]),
                "target_log_rms": float(predictions["target_log_rms"][idx]),
            }
        )
    np.savez_compressed(
        output_dir / "predicted_targets.npz",
        predicted_sequences=np.asarray(predictions["predicted_sequences"], dtype=np.float32),
        predicted_summaries=np.asarray(predictions["predicted_summaries"], dtype=np.float32),
        target_sequences=np.asarray(predictions["target_sequences"], dtype=np.float32),
        raw_target_sequences=np.asarray(predictions["raw_target_sequences"], dtype=np.float32),
        target_summaries=np.asarray(predictions["target_summaries"], dtype=np.float32),
        target_masks=np.asarray(predictions["target_masks"], dtype=np.float32),
        predicted_log_rms=np.asarray(predictions["predicted_log_rms"], dtype=np.float32),
        target_log_rms=np.asarray(predictions["target_log_rms"], dtype=np.float32),
        template_ids=np.asarray(predictions["template_ids"]),
        labels=np.asarray(predictions["labels"]),
        subject_ids=np.asarray(predictions["subject_ids"]),
        trial_indices=np.asarray(predictions["trial_indices"], dtype=np.int32),
    )
    write_json(output_dir / "evaluation_predictions.json", {"predictions": sample_rows})
    summary = {
        **predictions["metrics"],
        **embedding_eval["metrics"],
        **waveform_eval["metrics"],
        **compute_latent_collapse_diagnostics(
            predicted_sequences=np.asarray(predictions["predicted_sequences"], dtype=np.float32),
            target_sequences=np.asarray(predictions["target_sequences"], dtype=np.float32),
        ),
        "retrieval_policy": bank.policy,
        "candidate_pool_size": bank.size,
        "target_kind": bank.target_kind,
        "reconstruction_mode": reconstruction_mode,
        "target_steps": int(predictions["predicted_sequences"].shape[1]),
        "target_dim": int(predictions["predicted_sequences"].shape[2]),
        "mean_retrieved_target_stft_distance": float(np.mean(target_distances)) if target_distances else None,
    }
    write_json(output_dir / "summary.json", summary)
    return {
        "summary": summary,
        "predictions": sample_rows,
    }


def evaluate_control_predictions(
    predicted_sequences: np.ndarray,
    predicted_summaries: np.ndarray,
    availability: np.ndarray,
    bank,
    target_template_ids: list[str],
    target_labels: list[str],
    top_k: int,
) -> dict[str, object]:
    mask = np.asarray(availability, dtype=np.float32) > 0.5
    embedding_eval = evaluate_embedding_retrieval(
        bank=bank,
        predicted_sequences=predicted_sequences,
        predicted_summaries=predicted_summaries,
        target_template_ids=target_template_ids,
        target_labels=target_labels,
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
    }
    control_specs = [("label_only", "label_only", False)]
    if bank.policy == "unseen_strict_seen_subjects":
        control_specs.extend(
            [
                ("subject_only", "subject_only_strict", False),
                ("label_subject", "label_subject_strict", False),
            ]
        )
    elif bank.policy == "unseen_oracle_holdout":
        control_specs.extend(
            [
                ("subject_only", "subject_only_oracle", True),
                ("label_subject", "label_subject_oracle", True),
            ]
        )
    else:
        control_specs.extend(
            [
                ("subject_only", "subject_only", False),
                ("label_subject", "label_subject", False),
            ]
        )
    for mode, key, use_oracle in control_specs:
        payload = dataset.build_control_predictions(mode, use_oracle_for_unseen=use_oracle)
        controls[key] = evaluate_control_predictions(
            predicted_sequences=payload["target_sequences"],
            predicted_summaries=payload["target_summaries"],
            availability=payload["availability"],
            bank=bank,
            target_template_ids=target_template_ids,
            target_labels=target_labels,
            top_k=top_k,
        )
    return controls


def build_phase_alerts(summary: dict[str, object]) -> dict[str, object]:
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
    reconstruction_mode: str,
    codec_backend=None,
    default_decoder_scales: np.ndarray | None = None,
    target_mean: np.ndarray | None = None,
    target_std: np.ndarray | None = None,
) -> dict[str, object]:
    bank = build_retrieval_bank(dataset, policy=policy)
    policy_dir = output_root / "retrieval" / str(args.split) / bank.policy
    model_eval = evaluate_policy(
        predictions=predictions,
        bank=bank,
        output_dir=policy_dir,
        sample_rate=int(config["audio"]["sample_rate"]),
        top_k=top_k,
        reconstruction_mode=reconstruction_mode,
        codec_backend=codec_backend,
        default_decoder_scales=default_decoder_scales,
        target_mean=target_mean,
        target_std=target_std,
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
        "target_kind": bank.target_kind,
        "reconstruction_mode": reconstruction_mode,
        "model": model_eval["summary"],
        "controls": controls,
        "predictions_path": str(policy_dir / "evaluation_predictions.json"),
        "predicted_targets_path": str(policy_dir / "predicted_targets.npz"),
    }
    result["alerts"] = build_phase_alerts(model_eval["summary"])
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
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing_keys or unexpected_keys:
        print(
            "Checkpoint loaded with non-strict compatibility: "
            f"missing={len(missing_keys)} unexpected={len(unexpected_keys)}"
        )
    use_codec_scale_prediction = not any(str(key).startswith("codec_scale_head") for key in missing_keys)
    if not use_codec_scale_prediction:
        print("Codec scale head is not present in this checkpoint; primary codec decode will skip predicted RMS scaling.")

    predictions = predict_dataset(
        model=model,
        loader=loader,
        device=device,
        config=config,
        use_codec_scale_prediction=use_codec_scale_prediction,
    )
    output_root = resolve_bundle_path(args.output_root or config["output"]["root"], BUNDLE_DIR) / build_run_name(config, args)
    resolved_policy = resolve_retrieval_policy(eval_ds.protocol, args.retrieval_policy)
    target_kind = str(eval_ds.target_kind)
    feature_config = build_feature_config(config)
    reconstruction_mode = "codec_decode" if target_kind == TARGET_KIND_ENCODEC_LATENT else "retrieval_waveform"
    codec_backend = load_codec_backend(feature_config) if reconstruction_mode == "codec_decode" else None
    payload = {
        "protocol": eval_ds.protocol,
        "split": split,
        "checkpoint": str(checkpoint_path),
        "num_samples": len(eval_ds),
        "include_anomalous": bool(args.include_anomalous or config["data"].get("include_anomalous", False)),
        "phase_target_cache": str(eval_ds.target_cache["path"]),
        "phase_target_backend": str(eval_ds.target_cache["feature_backend"][0]),
        "target_kind": target_kind,
        "reconstruction_mode": reconstruction_mode,
        "model": run_policy_eval(
            config=config,
            args=args,
            predictions=predictions,
            dataset=eval_ds,
            output_root=output_root,
            policy=resolved_policy,
            top_k=max(1, int(args.top_k)),
            reconstruction_mode=reconstruction_mode,
            codec_backend=codec_backend,
            default_decoder_scales=eval_ds.target_cache.get("default_decoder_scales"),
            target_mean=eval_ds.target_cache.get("target_mean"),
            target_std=eval_ds.target_cache.get("target_std"),
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
            reconstruction_mode=reconstruction_mode,
            codec_backend=codec_backend,
            default_decoder_scales=eval_ds.target_cache.get("default_decoder_scales"),
            target_mean=eval_ds.target_cache.get("target_mean"),
            target_std=eval_ds.target_cache.get("target_std"),
        )
    metrics_path = output_root / "metrics" / f"{split}_evaluation.json"
    write_json(metrics_path, payload)
    print(f"Saved alignment evaluation to {metrics_path}")
    main_model = payload["model"]["model"]
    match_mode = str(main_model["match_mode"])
    retrieval_key = _target_metric_key(match_mode, "retrieval_top1")
    nta_key = _target_metric_key(match_mode, "NTA")
    print(f"Embedding cosine: {float(main_model['embedding_cosine']):.4f}")
    print(f"{retrieval_key}: {float(main_model[retrieval_key]):.4f}")
    print(f"{nta_key}: {float(main_model[nta_key]):.4f}")
    print(f"MRR: {float(main_model['MRR']):.4f}")
    print(f"mean_rank: {float(main_model['mean_rank']):.4f}")


if __name__ == "__main__":
    main()

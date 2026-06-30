from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.karaone_v10.eval import compute_v10_metrics, write_channel_reports
from src.karaone_v11.data import outputs_to_token_bank


@torch.no_grad()
def collect_v11_outputs(model: torch.nn.Module, dataset, *, device: str | torch.device = "cpu", batch_size: int = 32) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    model.eval()
    pred, zero, target = [], [], []
    prompt_logits, subject_logits = [], []
    label_idx, subject_idx = [], []
    eeg_cluster, speech_cluster, cross_cluster = [], [], []
    channel_gate, channel_assign = [], []
    token_logits, token_targets, token_mask = [], [], []
    codec_logits, codec_targets, codec_mask = [], [], []
    labels, subjects, stages, trials = [], [], [], []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        stage = batch["stage_idx"].to(device)
        valid = batch["eeg_valid_len"].to(device)
        channel_clusters = batch.get("channel_cluster_id")
        channel_clusters = channel_clusters.to(device) if torch.is_tensor(channel_clusters) else None
        out = model(eeg, stage, valid, channel_cluster_id=channel_clusters, mask_ratio=0.0, lambda_subject_adv=0.0)
        zero_out = model(torch.zeros_like(eeg), stage, valid, channel_cluster_id=channel_clusters, mask_ratio=0.0, lambda_subject_adv=0.0)
        pred.append(out["pred_semantic_summary"].detach().cpu().numpy())
        zero.append(zero_out["pred_semantic_summary"].detach().cpu().numpy())
        target.append(batch["semantic_summary"].numpy())
        prompt_logits.append(out["prompt_logits"].detach().cpu().numpy())
        subject_logits.append(out["subject_logits"].detach().cpu().numpy())
        label_idx.append(batch["label_idx"].numpy())
        subject_idx.append(batch["subject_idx"].numpy())
        eeg_cluster.append(batch.get("eeg_cluster_id", torch.zeros_like(batch["label_idx"])).numpy())
        speech_cluster.append(batch.get("speech_cluster_id", torch.zeros_like(batch["label_idx"])).numpy())
        cross_cluster.append(batch.get("cross_modal_cluster_id", torch.zeros_like(batch["label_idx"])).numpy())
        channel_gate.append(out["channel_gate"].detach().cpu().numpy())
        channel_assign.append(out["channel_assign"].detach().cpu().numpy())
        token_logits.append(out["semantic_token_logits"].detach().cpu().numpy())
        token_targets.append(batch["audio_semantic_tokens"].numpy())
        token_mask.append(batch["audio_semantic_token_mask"].numpy())
        codec_logits.append(out["codec_token_logits"].detach().cpu().numpy())
        codec_targets.append(batch["codec_token_targets"].numpy())
        codec_mask.append(batch["codec_token_mask"].numpy())
        labels.extend([str(item) for item in batch["label"]])
        subjects.extend([str(item) for item in batch["subject"]])
        stages.extend([str(item) for item in batch["stage"]])
        trials.extend([int(item) for item in batch["trial_index"]])
    sem_logits = np.concatenate(token_logits, axis=0).astype(np.float32)
    sem_targets = np.concatenate(token_targets, axis=0).astype(np.int64)
    sem_mask = np.concatenate(token_mask, axis=0).astype(np.float32)
    pred_tokens = sem_logits.argmax(axis=-1).astype(np.int64)
    vocab = sem_logits.shape[-1]
    return {
        "pred": np.concatenate(pred, axis=0).astype(np.float32),
        "zero": np.concatenate(zero, axis=0).astype(np.float32),
        "target": np.concatenate(target, axis=0).astype(np.float32),
        "prompt_logits": np.concatenate(prompt_logits, axis=0).astype(np.float32),
        "subject_logits": np.concatenate(subject_logits, axis=0).astype(np.float32),
        "label_idx": np.concatenate(label_idx, axis=0).astype(np.int64),
        "subject_idx": np.concatenate(subject_idx, axis=0).astype(np.int64),
        "eeg_cluster_id": np.concatenate(eeg_cluster, axis=0).astype(np.int64),
        "speech_cluster_id": np.concatenate(speech_cluster, axis=0).astype(np.int64),
        "cross_modal_cluster_id": np.concatenate(cross_cluster, axis=0).astype(np.int64),
        "channel_gate": np.concatenate(channel_gate, axis=0).astype(np.float32),
        "channel_assign": np.concatenate(channel_assign, axis=0).astype(np.float32),
        "semantic_token_logits": sem_logits,
        "pred_semantic_tokens": pred_tokens,
        "target_semantic_tokens": sem_targets,
        "target_semantic_token_mask": sem_mask,
        "pred_semantic_token_hist": token_hist(pred_tokens, sem_mask, vocab),
        "target_semantic_token_hist": token_hist(sem_targets, sem_mask, vocab),
        "codec_token_logits": np.concatenate(codec_logits, axis=0).astype(np.float32),
        "codec_token_targets": np.concatenate(codec_targets, axis=0).astype(np.int64),
        "codec_token_mask": np.concatenate(codec_mask, axis=0).astype(np.float32),
        "labels": labels,
        "subjects": subjects,
        "stages": stages,
        "trials": trials,
    }


def compute_v11_metrics(outputs: dict[str, Any], *, train_bank: dict[str, Any] | None = None, prefix: str = "") -> dict[str, float | bool]:
    metrics = compute_v10_metrics(outputs, train_bank=train_bank, prefix="")
    metrics.update(token_metrics(outputs, train_bank=train_bank))
    metrics.update(codec_token_metrics(outputs))
    metrics["v11_alignment_gate_pass"] = v11_alignment_gate_pass(metrics)
    metrics["v11_generation_gate_pass"] = v11_generation_gate_pass(metrics)
    metrics["v11_waveform_claim_allowed"] = bool(metrics["v11_alignment_gate_pass"] and metrics["v11_generation_gate_pass"])
    if prefix:
        return {f"{prefix}_{key}": value for key, value in metrics.items()}
    return metrics


def token_metrics(outputs: dict[str, Any], *, train_bank: dict[str, Any] | None = None) -> dict[str, float]:
    logits = outputs["semantic_token_logits"]
    targets = outputs["target_semantic_tokens"]
    mask = outputs["target_semantic_token_mask"] > 0
    if mask.sum() <= 0:
        return {"semantic_token_acc": 0.0, "semantic_token_top3_acc": 0.0, "semantic_token_top3_gain_over_prior": 0.0, "pred_token_entropy": 0.0, "token_retrieval_cross_subject_gain": 0.0}
    pred = logits.argmax(axis=-1)
    top3 = np.argsort(logits, axis=-1)[..., -min(3, logits.shape[-1]) :]
    acc = float((pred[mask] == targets[mask]).mean())
    top3_acc = float(np.any(top3[mask] == targets[mask, None], axis=-1).mean())
    if train_bank and "target_semantic_tokens" in train_bank:
        bank_tokens = np.asarray(train_bank["target_semantic_tokens"], dtype=np.int64)
        bank_mask = np.asarray(train_bank["target_semantic_token_mask"], dtype=np.float32) > 0
        counts = np.bincount(bank_tokens[bank_mask].reshape(-1), minlength=logits.shape[-1])
    else:
        counts = np.bincount(targets[mask].reshape(-1), minlength=logits.shape[-1])
    prior_top = set(np.argsort(counts)[-min(3, logits.shape[-1]) :].tolist())
    prior_top3 = float(np.asarray([int(tok) in prior_top for tok in targets[mask].reshape(-1)]).mean())
    probs = softmax_np(logits)
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-8)), axis=-1) / np.log(max(logits.shape[-1], 2))
    retrieval_gain = token_retrieval_gain(outputs, train_bank)
    return {
        "semantic_token_acc": acc,
        "semantic_token_top3_acc": top3_acc,
        "semantic_token_top3_gain_over_prior": float(top3_acc - prior_top3),
        "pred_token_entropy": float(entropy[mask].mean()),
        "token_retrieval_cross_subject_gain": retrieval_gain,
    }


def codec_token_metrics(outputs: dict[str, Any]) -> dict[str, float]:
    logits = outputs["codec_token_logits"]
    targets = outputs["codec_token_targets"]
    mask = outputs["codec_token_mask"] > 0
    if mask.sum() <= 0:
        return {"codec_token_acc": 0.0, "codec_token_top3_acc": 0.0}
    pred = logits.argmax(axis=-1)
    top3 = np.argsort(logits, axis=-1)[..., -min(3, logits.shape[-1]) :]
    return {
        "codec_token_acc": float((pred[mask] == targets[mask]).mean()),
        "codec_token_top3_acc": float(np.any(top3[mask] == targets[mask, None], axis=-1).mean()),
    }


def token_retrieval_gain(outputs: dict[str, Any], train_bank: dict[str, Any] | None) -> float:
    if not train_bank or "target_semantic_token_hist" not in train_bank:
        return 0.0
    query = normalize_rows(np.asarray(outputs["pred_semantic_token_hist"], dtype=np.float32))
    bank = normalize_rows(np.asarray(train_bank["target_semantic_token_hist"], dtype=np.float32))
    if query.size == 0 or bank.size == 0:
        return 0.0
    labels = np.asarray(outputs["labels"]).astype(str)
    subjects = np.asarray(outputs["subjects"]).astype(str)
    bank_labels = np.asarray(train_bank["labels"]).astype(str)
    bank_subjects = np.asarray(train_bank["subjects"]).astype(str)
    scores = query @ bank.T
    mean_query = normalize_rows(bank.mean(axis=0, keepdims=True))
    mean_scores = mean_query @ bank.T
    gains = []
    for idx in range(query.shape[0]):
        mask = (bank_labels == labels[idx]) & (bank_subjects != subjects[idx])
        if not mask.any():
            continue
        gains.append(float(scores[idx, mask].max() - mean_scores[0, mask].max()))
    return float(np.mean(gains)) if gains else 0.0


def v11_alignment_gate_pass(metrics: dict[str, Any]) -> bool:
    return bool(
        float(metrics.get("semantic_token_top3_gain_over_prior", 0.0)) > 0.02
        and float(metrics.get("same_label_cross_subject_gain", -1.0)) >= 0.0
        and float(metrics.get("prompt_acc", 0.0)) >= 0.13
        and float(metrics.get("token_retrieval_cross_subject_gain", 0.0)) > 0.0
        and float(metrics.get("pred_token_entropy", 0.0)) > 0.20
        and float(metrics.get("channel_gate_entropy_mean", 0.0)) > 0.20
    )


def v11_generation_gate_pass(metrics: dict[str, Any]) -> bool:
    return bool(float(metrics.get("codec_token_top3_acc", 0.0)) > 0.10 and float(metrics.get("codec_token_acc", 0.0)) > 0.02)


def v11_selection_score(row: dict[str, Any], *, prefix: str = "subject_val") -> float:
    def get(name: str, default: float = 0.0) -> float:
        return float(row.get(f"{prefix}_{name}", default))

    score = (
        1.5 * get("semantic_token_top3_gain_over_prior")
        + 1.0 * get("token_retrieval_cross_subject_gain")
        + 0.6 * get("prompt_acc")
        + 0.5 * get("semantic_over_mean_gain")
        + 0.4 * get("same_label_cross_subject_gain")
        + 0.2 * get("codec_token_top3_acc")
        - 0.2 * max(0.0, 0.20 - get("pred_token_entropy"))
    )
    if get("same_label_cross_subject_gain") < 0.0:
        score -= 0.10
    if get("semantic_token_top3_gain_over_prior") <= 0.0:
        score -= 0.05
    return float(score)


def row_gate_summary(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "subject_val_semantic_token_top3_gain_over_prior",
        "subject_val_token_retrieval_cross_subject_gain",
        "subject_val_same_label_cross_subject_gain",
        "subject_val_prompt_acc",
        "subject_val_pred_token_entropy",
        "subject_val_channel_gate_entropy_mean",
        "subject_val_codec_token_acc",
        "subject_val_v11_alignment_gate_pass",
        "subject_val_v11_generation_gate_pass",
        "subject_test_semantic_token_top3_gain_over_prior",
        "subject_test_token_retrieval_cross_subject_gain",
        "subject_test_same_label_cross_subject_gain",
        "subject_test_prompt_acc",
        "subject_test_v11_alignment_gate_pass",
        "selection_score",
    ]
    return {key: row.get(key) for key in keys}


def token_hist(tokens: np.ndarray, mask: np.ndarray, vocab: int) -> np.ndarray:
    out = np.zeros((tokens.shape[0], int(vocab)), dtype=np.float32)
    for idx in range(tokens.shape[0]):
        valid = mask[idx] > 0
        if valid.any():
            out[idx] = np.bincount(tokens[idx][valid].clip(0, vocab - 1), minlength=vocab).astype(np.float32)
            out[idx] /= max(out[idx].sum(), 1.0)
    return out


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1e-8)


def softmax_np(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-8)


__all__ = [
    "collect_v11_outputs",
    "compute_v11_metrics",
    "outputs_to_token_bank",
    "row_gate_summary",
    "v11_alignment_gate_pass",
    "v11_generation_gate_pass",
    "v11_selection_score",
    "write_channel_reports",
]

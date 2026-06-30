from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_v9_outputs(model: torch.nn.Module, dataset, *, device: str | torch.device = "cpu", batch_size: int = 32) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    model.eval()
    pred, zero, target = [], [], []
    prompt_logits, subject_logits = [], []
    label_idx, subject_idx = [], []
    labels, subjects, trials = [], [], []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        stage = batch["stage_idx"].to(device)
        valid = batch["eeg_valid_len"].to(device)
        out = model(eeg, stage, valid, mask_ratio=0.0, lambda_subject_adv=0.0)
        zero_out = model(torch.zeros_like(eeg), stage, valid, mask_ratio=0.0, lambda_subject_adv=0.0)
        pred.append(out["pred_semantic_summary"].detach().cpu().numpy())
        zero.append(zero_out["pred_semantic_summary"].detach().cpu().numpy())
        target.append(batch["semantic_summary"].numpy())
        prompt_logits.append(out["prompt_logits"].detach().cpu().numpy())
        subject_logits.append(out["subject_logits"].detach().cpu().numpy())
        label_idx.append(batch["label_idx"].numpy())
        subject_idx.append(batch["subject_idx"].numpy())
        labels.extend([str(item) for item in batch["label"]])
        subjects.extend([str(item) for item in batch["subject"]])
        trials.extend([int(item) for item in batch["trial_index"]])
    return {
        "pred": np.concatenate(pred, axis=0).astype(np.float32),
        "zero": np.concatenate(zero, axis=0).astype(np.float32),
        "target": np.concatenate(target, axis=0).astype(np.float32),
        "prompt_logits": np.concatenate(prompt_logits, axis=0).astype(np.float32),
        "subject_logits": np.concatenate(subject_logits, axis=0).astype(np.float32),
        "label_idx": np.concatenate(label_idx, axis=0).astype(np.int64),
        "subject_idx": np.concatenate(subject_idx, axis=0).astype(np.int64),
        "labels": labels,
        "subjects": subjects,
        "trials": trials,
    }


def compute_v9_metrics(
    outputs: dict[str, Any],
    *,
    train_bank: dict[str, Any] | None = None,
    prefix: str = "",
) -> dict[str, float | bool]:
    pred = np.asarray(outputs["pred"], dtype=np.float32)
    target = np.asarray(outputs["target"], dtype=np.float32)
    zero = np.asarray(outputs.get("zero", np.zeros_like(pred)), dtype=np.float32)
    labels = np.asarray(outputs.get("labels", [""] * pred.shape[0])).astype(str)
    subjects = np.asarray(outputs.get("subjects", [""] * pred.shape[0])).astype(str)
    target_bank = np.asarray(train_bank["target"], dtype=np.float32) if train_bank and "target" in train_bank else target
    label_bank = np.asarray(train_bank["labels"]).astype(str) if train_bank and "labels" in train_bank else labels
    subject_bank = np.asarray(train_bank["subjects"]).astype(str) if train_bank and "subjects" in train_bank else subjects
    mean_query = np.repeat(target_bank.mean(axis=0, keepdims=True), pred.shape[0], axis=0)

    pred_cos = paired_cosine(pred, target)
    zero_cos = paired_cosine(zero, target)
    mean_cos = paired_cosine(mean_query, target)
    retrieval = retrieval_metrics(pred, target_bank, labels=labels, label_bank=label_bank, prefix="semantic")
    mean_retrieval = retrieval_metrics(mean_query, target_bank, labels=labels, label_bank=label_bank, prefix="mean")
    same_label_gain = same_label_cross_subject_gain(pred, mean_query, target_bank, labels, label_bank, subjects, subject_bank)
    prompt_acc = _argmax_acc(outputs.get("prompt_logits"), outputs.get("label_idx"))
    subject_acc = _argmax_acc(outputs.get("subject_logits"), outputs.get("subject_idx"))
    metrics: dict[str, float | bool] = {
        "n": float(pred.shape[0]),
        "semantic_cos": float(np.mean(pred_cos)),
        "zero_semantic_cos": float(np.mean(zero_cos)),
        "mean_semantic_cos": float(np.mean(mean_cos)),
        "semantic_over_zero_gain": float(np.mean(pred_cos - zero_cos)),
        "semantic_over_mean_gain": float(np.mean(pred_cos - mean_cos)),
        "prompt_acc": float(prompt_acc),
        "subject_leakage_acc": float(subject_acc),
        "pred_std_ratio_median": std_ratio_median(pred, target),
        "pred_pairwise_corr_median": pairwise_corr_median(pred),
        "same_label_cross_subject_gain": float(same_label_gain),
    }
    metrics.update(retrieval)
    metrics["semantic_top3_gain_over_mean"] = float(
        metrics.get("semantic_label_top3", 0.0) - mean_retrieval.get("mean_label_top3", 0.0)
    )
    metrics["v9_semantic_gate_pass"] = bool(
        metrics["semantic_over_mean_gain"] > 0.0
        and metrics["semantic_top3_gain_over_mean"] > 0.0
        and metrics["same_label_cross_subject_gain"] >= 0.0
    )
    if prefix:
        return {f"{prefix}_{key}": value for key, value in metrics.items()}
    return metrics


def outputs_to_bank(outputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "target": np.asarray(outputs["target"], dtype=np.float32),
        "labels": np.asarray(outputs.get("labels", [])).astype(str),
        "subjects": np.asarray(outputs.get("subjects", [])).astype(str),
    }


def paired_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = normalize_rows(a)
    b_n = normalize_rows(b)
    return np.sum(a_n * b_n, axis=1)


def retrieval_metrics(query: np.ndarray, bank: np.ndarray, *, labels: np.ndarray, label_bank: np.ndarray, prefix: str) -> dict[str, float]:
    if query.shape[0] == 0 or bank.shape[0] == 0:
        return {f"{prefix}_label_top1": 0.0, f"{prefix}_label_top3": 0.0, f"{prefix}_label_mrr": 0.0}
    scores = normalize_rows(query) @ normalize_rows(bank).T
    order = np.argsort(scores, axis=1)[:, ::-1]
    hits1, hits3, reciprocal = [], [], []
    for i in range(query.shape[0]):
        positive = np.flatnonzero(label_bank == labels[i])
        if positive.size == 0:
            hits1.append(0.0)
            hits3.append(0.0)
            reciprocal.append(0.0)
            continue
        rank = min(int(np.where(order[i] == idx)[0][0]) + 1 for idx in positive)
        hits1.append(float(rank <= 1))
        hits3.append(float(rank <= min(3, bank.shape[0])))
        reciprocal.append(1.0 / float(rank))
    return {
        f"{prefix}_label_top1": float(np.mean(hits1)),
        f"{prefix}_label_top3": float(np.mean(hits3)),
        f"{prefix}_label_mrr": float(np.mean(reciprocal)),
    }


def same_label_cross_subject_gain(
    pred: np.ndarray,
    mean_query: np.ndarray,
    bank: np.ndarray,
    labels: np.ndarray,
    label_bank: np.ndarray,
    subjects: np.ndarray,
    subject_bank: np.ndarray,
) -> float:
    pred_n = normalize_rows(pred)
    mean_n = normalize_rows(mean_query)
    bank_n = normalize_rows(bank)
    gains = []
    for i in range(pred.shape[0]):
        mask = (label_bank == labels[i]) & (subject_bank != subjects[i])
        if not bool(mask.any()):
            continue
        target_proto = bank_n[mask].mean(axis=0, keepdims=True)
        target_proto = normalize_rows(target_proto)
        pred_score = float(pred_n[i : i + 1] @ target_proto.T)
        mean_score = float(mean_n[i : i + 1] @ target_proto.T)
        gains.append(pred_score - mean_score)
    return float(np.mean(gains)) if gains else 0.0


def std_ratio_median(pred: np.ndarray, target: np.ndarray) -> float:
    pred_std = np.std(pred, axis=0)
    target_std = np.std(target, axis=0)
    valid = target_std > 1e-8
    if not bool(valid.any()):
        return 0.0
    return float(np.median(pred_std[valid] / np.maximum(target_std[valid], 1e-8)))


def pairwise_corr_median(x: np.ndarray, max_items: int = 256) -> float:
    if x.shape[0] < 2:
        return 0.0
    x = x[: min(max_items, x.shape[0])]
    x = normalize_rows(x - x.mean(axis=1, keepdims=True))
    corr = x @ x.T
    upper = corr[np.triu_indices(corr.shape[0], k=1)]
    return float(np.median(upper)) if upper.size else 0.0


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)


def _argmax_acc(logits, targets) -> float:
    if logits is None or targets is None:
        return 0.0
    logits = np.asarray(logits)
    targets = np.asarray(targets)
    if logits.size == 0 or targets.size == 0:
        return 0.0
    return float(np.mean(np.argmax(logits, axis=1) == targets))

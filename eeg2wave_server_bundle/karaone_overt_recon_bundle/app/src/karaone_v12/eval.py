from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.karaone_v11.eval import (
    collect_v11_outputs,
    compute_v11_metrics,
    outputs_to_token_bank,
    row_gate_summary as v11_row_gate_summary,
    v11_selection_score,
    write_channel_reports,
)


@torch.no_grad()
def collect_v12_outputs(model: torch.nn.Module, dataset, *, device: str | torch.device = "cpu", batch_size: int = 32) -> dict[str, Any]:
    outputs = collect_v11_outputs(model, dataset, device=device, batch_size=batch_size)
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    model.eval()
    pred_onset, pred_duration, pred_lag, pred_active = [], [], [], []
    tgt_onset, tgt_duration, tgt_lag, tgt_active, confidence = [], [], [], [], []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        stage = batch["stage_idx"].to(device)
        valid = batch["eeg_valid_len"].to(device)
        channel_clusters = batch.get("channel_cluster_id")
        channel_clusters = channel_clusters.to(device) if torch.is_tensor(channel_clusters) else None
        out = model(eeg, stage, valid, channel_cluster_id=channel_clusters, mask_ratio=0.0, lambda_subject_adv=0.0)
        pred_onset.append(out["pred_onset_sec"].detach().cpu().numpy())
        pred_duration.append(out["pred_duration_sec"].detach().cpu().numpy())
        pred_lag.append(out["pred_lag_sec"].detach().cpu().numpy())
        pred_active.append(torch.sigmoid(out["pred_active_mask_logits"]).detach().cpu().numpy())
        tgt_onset.append(batch["time_onset_sec"].numpy())
        tgt_duration.append(batch["time_duration_sec"].numpy())
        tgt_lag.append(batch["time_lag_sec"].numpy())
        tgt_active.append(batch["time_active_mask"].numpy())
        confidence.append(batch["time_confidence"].numpy())
    outputs.update(
        {
            "pred_onset_sec": np.concatenate(pred_onset).astype(np.float32),
            "pred_duration_sec": np.concatenate(pred_duration).astype(np.float32),
            "pred_lag_sec": np.concatenate(pred_lag).astype(np.float32),
            "pred_active_mask": np.concatenate(pred_active).astype(np.float32),
            "target_onset_sec": np.concatenate(tgt_onset).astype(np.float32),
            "target_duration_sec": np.concatenate(tgt_duration).astype(np.float32),
            "target_lag_sec": np.concatenate(tgt_lag).astype(np.float32),
            "target_active_mask": np.concatenate(tgt_active).astype(np.float32),
            "time_confidence": np.concatenate(confidence).astype(np.float32),
        }
    )
    return outputs


def compute_v12_metrics(outputs: dict[str, Any], *, train_bank: dict[str, Any] | None = None, prefix: str = "") -> dict[str, float | bool]:
    metrics = compute_v11_metrics(outputs, train_bank=train_bank, prefix="")
    metrics.update(time_anchor_metrics(outputs))
    metrics["time_normalized_token_retrieval_gain"] = float(metrics.get("token_retrieval_cross_subject_gain", 0.0))
    metrics["v12_alignment_gate_pass"] = v12_alignment_gate_pass(metrics)
    metrics["v12_predicted_lag_generation_gate_pass"] = v12_predicted_lag_generation_gate_pass(metrics)
    metrics["v12_waveform_claim_allowed"] = bool(metrics["v12_alignment_gate_pass"] and metrics["v12_predicted_lag_generation_gate_pass"])
    if prefix:
        return {f"{prefix}_{key}": value for key, value in metrics.items()}
    return metrics


def time_anchor_metrics(outputs: dict[str, Any]) -> dict[str, float]:
    conf = np.asarray(outputs.get("time_confidence", []), dtype=np.float32)
    weight = np.where(conf > 0, conf, 1.0).astype(np.float32)
    if weight.size == 0:
        return {"lag_mae_sec": 0.0, "onset_mae_sec": 0.0, "duration_mae_sec": 0.0, "active_iou": 0.0}
    pred_active = np.asarray(outputs["pred_active_mask"], dtype=np.float32)
    target_active = np.asarray(outputs["target_active_mask"], dtype=np.float32)
    if pred_active.shape[1] != target_active.shape[1]:
        target_active = resize_mask_np(target_active, pred_active.shape[1])
    inter = np.minimum(pred_active, target_active).sum(axis=1)
    union = np.maximum(pred_active, target_active).sum(axis=1).clip(min=1e-6)
    iou = inter / union
    return {
        "lag_mae_sec": weighted_mae(outputs["pred_lag_sec"], outputs["target_lag_sec"], weight),
        "onset_mae_sec": weighted_mae(outputs["pred_onset_sec"], outputs["target_onset_sec"], weight),
        "duration_mae_sec": weighted_mae(outputs["pred_duration_sec"], outputs["target_duration_sec"], weight),
        "active_iou": float(np.average(iou, weights=weight)),
    }


def v12_alignment_gate_pass(metrics: dict[str, Any]) -> bool:
    return bool(
        bool(metrics.get("v11_alignment_gate_pass", False))
        and float(metrics.get("time_normalized_token_retrieval_gain", 0.0)) > 0.0
        and float(metrics.get("active_iou", 0.0)) > 0.20
    )


def v12_predicted_lag_generation_gate_pass(metrics: dict[str, Any]) -> bool:
    return bool(
        bool(metrics.get("v11_generation_gate_pass", False))
        and float(metrics.get("lag_mae_sec", 9.0)) <= 0.75
        and float(metrics.get("active_iou", 0.0)) > 0.20
    )


def v12_selection_score(row: dict[str, Any], *, prefix: str = "subject_val") -> float:
    base = v11_selection_score(row, prefix=prefix)

    def get(name: str, default: float = 0.0) -> float:
        return float(row.get(f"{prefix}_{name}", default))

    return float(base + 0.4 * get("active_iou") - 0.2 * get("lag_mae_sec", 1.0) - 0.1 * get("onset_mae_sec", 1.0))


def row_gate_summary(row: dict[str, Any]) -> dict[str, Any]:
    out = v11_row_gate_summary(row)
    for key in [
        "subject_val_lag_mae_sec",
        "subject_val_onset_mae_sec",
        "subject_val_duration_mae_sec",
        "subject_val_active_iou",
        "subject_val_v12_alignment_gate_pass",
        "subject_val_v12_predicted_lag_generation_gate_pass",
        "subject_test_lag_mae_sec",
        "subject_test_active_iou",
        "subject_test_v12_alignment_gate_pass",
    ]:
        out[key] = row.get(key)
    return out


def weighted_mae(pred: np.ndarray, target: np.ndarray, weight: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    return float(np.sum(np.abs(pred - target) * weight) / max(float(weight.sum()), 1e-8))


def resize_mask_np(values: np.ndarray, steps: int) -> np.ndarray:
    out = []
    x_new = np.linspace(0.0, 1.0, int(steps))
    for row in values:
        x_old = np.linspace(0.0, 1.0, row.shape[0])
        out.append(np.interp(x_new, x_old, row))
    return np.asarray(out, dtype=np.float32)


__all__ = [
    "collect_v12_outputs",
    "compute_v12_metrics",
    "outputs_to_token_bank",
    "row_gate_summary",
    "v12_alignment_gate_pass",
    "v12_predicted_lag_generation_gate_pass",
    "v12_selection_score",
    "write_channel_reports",
]

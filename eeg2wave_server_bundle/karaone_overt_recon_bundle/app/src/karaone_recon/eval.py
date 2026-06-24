from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import KaraOneTrialDataset
from .targets import KaraOneTargets


def _corr_median(x: np.ndarray, max_items: int = 256) -> float:
    if x.shape[0] < 2:
        return 0.0
    x = x[:max_items]
    x = x - x.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)
    x = x / denom
    corr = x @ x.T
    upper = corr[np.triu_indices(corr.shape[0], k=1)]
    return float(np.median(upper)) if upper.size else 0.0


def _retrieval_stats(
    pred_summary: np.ndarray,
    subjects: list[str],
    labels: list[str],
    trial_indices: list[int],
    targets: KaraOneTargets,
) -> dict[str, float]:
    target_norm = targets.summary / np.linalg.norm(targets.summary, axis=1, keepdims=True).clip(min=1e-8)
    pred_norm = pred_summary / np.linalg.norm(pred_summary, axis=1, keepdims=True).clip(min=1e-8)
    subject_arr = targets.subject_ids.astype(str)
    trial_arr = targets.trial_indices.astype(np.int32)
    label_arr = targets.labels.astype(str)

    trial_hits = 0
    label_hits = 0
    for i, (subject, label, trial) in enumerate(zip(subjects, labels, trial_indices)):
        mask = subject_arr == subject
        scores = pred_norm[i] @ target_norm[mask].T
        candidate_trials = trial_arr[mask]
        candidate_labels = label_arr[mask]
        best = int(np.argmax(scores))
        trial_hits += int(int(candidate_trials[best]) == int(trial))
        label_hits += int(str(candidate_labels[best]) == str(label))
    n = max(1, len(subjects))
    return {
        "within_subject_trial_top1": float(trial_hits / n),
        "within_subject_label_top1": float(label_hits / n),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: KaraOneTrialDataset,
    targets: KaraOneTargets,
    device: str | torch.device,
    batch_size: int = 64,
) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    totals: dict[str, float] = defaultdict(float)
    by_stage: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    count = 0
    pred_summaries: list[np.ndarray] = []
    zero_summaries: list[np.ndarray] = []
    pred_flat: list[np.ndarray] = []
    target_flat: list[np.ndarray] = []
    subjects: list[str] = []
    labels: list[str] = []
    trials: list[int] = []

    global_mean = torch.from_numpy(targets.global_mean_norm).to(device).float()
    for batch in loader:
        eeg = batch["eeg"].to(device)
        subject_idx = batch["subject_idx"].to(device)
        stage_idx = batch["stage_idx"].to(device)
        valid_len = batch["eeg_valid_len"].to(device)
        target = batch["target_seq"].to(device)
        out = model(eeg, subject_idx, stage_idx, valid_len)
        zero_out = model(torch.zeros_like(eeg), subject_idx, stage_idx, valid_len)
        pred = out["pred_latent"]
        zero_pred = zero_out["pred_latent"]
        b = int(eeg.shape[0])

        pred_cos = F.cosine_similarity(pred, target, dim=-1).mean(dim=1)
        zero_cos = F.cosine_similarity(zero_pred, target, dim=-1).mean(dim=1)
        mean_cos = F.cosine_similarity(global_mean.unsqueeze(0).expand_as(target), target, dim=-1).mean(dim=1)
        pred_mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2))
        zero_mse = F.mse_loss(zero_pred, target, reduction="none").mean(dim=(1, 2))
        mean_mse = F.mse_loss(global_mean.unsqueeze(0).expand_as(target), target, reduction="none").mean(dim=(1, 2))
        content_acc = (out["content_logits"].argmax(dim=-1).cpu() == batch["label_idx"]).float()

        metrics = {
            "pred_recon_cos": pred_cos.detach().cpu().numpy(),
            "zeroeeg_recon_cos": zero_cos.detach().cpu().numpy(),
            "mean_recon_cos": mean_cos.detach().cpu().numpy(),
            "pred_recon_mse": pred_mse.detach().cpu().numpy(),
            "zeroeeg_recon_mse": zero_mse.detach().cpu().numpy(),
            "mean_recon_mse": mean_mse.detach().cpu().numpy(),
            "content_acc": content_acc.numpy(),
        }
        for name, values in metrics.items():
            totals[name] += float(values.sum())
        count += b

        batch_stages = list(batch["stage"])
        for local_idx, stage in enumerate(batch_stages):
            for name, values in metrics.items():
                by_stage[str(stage)][name] += float(values[local_idx])
            by_stage[str(stage)]["n"] += 1.0

        pred_np = pred.detach().cpu().numpy()
        zero_np = zero_pred.detach().cpu().numpy()
        tgt_np = target.detach().cpu().numpy()
        pred_summaries.append(pred_np.mean(axis=1))
        zero_summaries.append(zero_np.mean(axis=1))
        pred_flat.append(pred_np.reshape(pred_np.shape[0], -1))
        target_flat.append(tgt_np.reshape(tgt_np.shape[0], -1))
        subjects.extend([str(item) for item in batch["subject"]])
        labels.extend([str(item) for item in batch["label"]])
        trials.extend([int(item) for item in batch["trial_index"]])

    out_metrics = {name: value / max(count, 1) for name, value in totals.items()}
    pred_summary = np.concatenate(pred_summaries, axis=0)
    zero_summary = np.concatenate(zero_summaries, axis=0)
    pred_matrix = np.concatenate(pred_flat, axis=0)
    target_matrix = np.concatenate(target_flat, axis=0)
    pred_std = pred_matrix.std(axis=0)
    target_std = target_matrix.std(axis=0)
    # Per-sample Pearson correlation over the whole spectrogram/latent (mel-PCC, as in
    # Park 2025 / FESDE). Reflects whole time-freq similarity, not just per-frame cosine.
    pm = pred_matrix - pred_matrix.mean(axis=1, keepdims=True)
    tm = target_matrix - target_matrix.mean(axis=1, keepdims=True)
    pcc = (pm * tm).sum(axis=1) / (
        np.sqrt((pm * pm).sum(axis=1)) * np.sqrt((tm * tm).sum(axis=1)) + 1e-8
    )
    out_metrics.update(
        {
            "n": int(count),
            "pred_over_zero_cos_gain": out_metrics["pred_recon_cos"] - out_metrics["zeroeeg_recon_cos"],
            "pred_over_mean_cos_gain": out_metrics["pred_recon_cos"] - out_metrics["mean_recon_cos"],
            "pred_std_ratio_median": float(np.median(pred_std / np.maximum(target_std, 1e-6))),
            "pred_pairwise_corr_median": _corr_median(pred_matrix),
            "pred_pcc": float(np.mean(pcc)),
        }
    )
    out_metrics.update({f"pred_{k}": v for k, v in _retrieval_stats(pred_summary, subjects, labels, trials, targets).items()})
    out_metrics.update({f"zeroeeg_{k}": v for k, v in _retrieval_stats(zero_summary, subjects, labels, trials, targets).items()})

    stage_payload = {}
    for stage, payload in by_stage.items():
        n = max(payload.pop("n"), 1.0)
        stage_payload[stage] = {name: value / n for name, value in payload.items()}
    out_metrics["by_stage"] = stage_payload
    return out_metrics


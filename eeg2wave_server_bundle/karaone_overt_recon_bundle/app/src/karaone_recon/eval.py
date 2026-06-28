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


def _sample_pcc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    am = a - a.mean(axis=1, keepdims=True)
    bm = b - b.mean(axis=1, keepdims=True)
    return (am * bm).sum(axis=1) / (
        np.sqrt((am * am).sum(axis=1)) * np.sqrt((bm * bm).sum(axis=1)) + 1e-8
    )


def _weighted_pcc_1d(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> np.ndarray:
    out = np.zeros(a.shape[0], dtype=np.float32)
    for i in range(a.shape[0]):
        w = weight[i].astype(np.float64)
        if float(w.sum()) < 2.0:
            out[i] = 0.0
            continue
        x = a[i].astype(np.float64)
        y = b[i].astype(np.float64)
        w = w / max(float(w.sum()), 1e-8)
        xm = float((x * w).sum())
        ym = float((y * w).sum())
        xc = (x - xm) * w
        yc = (y - ym) * w
        den = float(np.sqrt((xc * (x - xm)).sum()) * np.sqrt((yc * (y - ym)).sum())) + 1e-8
        out[i] = float((xc * (y - ym)).sum() / den)
    return out


def _dct_matrix(n_in: int, n_out: int) -> np.ndarray:
    n = np.arange(n_in, dtype=np.float64)[None, :]
    k = np.arange(n_out, dtype=np.float64)[:, None]
    mat = np.cos(np.pi / float(n_in) * (n + 0.5) * k)
    mat[0] *= np.sqrt(1.0 / float(n_in))
    if n_out > 1:
        mat[1:] *= np.sqrt(2.0 / float(n_in))
    return mat.astype(np.float32)


def _mel_metrics(pred_norm: np.ndarray, target_norm: np.ndarray, mean_norm: np.ndarray, targets: KaraOneTargets) -> dict[str, float]:
    """Mel-Corr/PCC and approximate MCD in the cache's raw log-mel domain."""
    t, d = int(target_norm.shape[1]), int(target_norm.shape[2])
    mean = targets.target_mean.reshape(1, 1, d)
    std = targets.target_std.reshape(1, 1, d)
    pred_raw = pred_norm * std + mean
    target_raw = target_norm * std + mean
    mean_raw = mean_norm * std + mean
    pred_flat = pred_raw.reshape(pred_raw.shape[0], -1)
    target_flat = target_raw.reshape(target_raw.shape[0], -1)
    mean_flat = mean_raw.reshape(mean_raw.shape[0], -1)
    n_mfcc = min(13, d)
    dct = _dct_matrix(d, n_mfcc)
    pred_mfcc = pred_raw.reshape(-1, d) @ dct.T
    target_mfcc = target_raw.reshape(-1, d) @ dct.T
    mean_mfcc = mean_raw.reshape(-1, d) @ dct.T
    start = 1 if n_mfcc > 1 else 0
    coef = 10.0 / np.log(10.0) * np.sqrt(2.0)
    pred_mcd = coef * np.sqrt(np.square(pred_mfcc[:, start:] - target_mfcc[:, start:]).sum(axis=1) + 1e-8)
    mean_mcd = coef * np.sqrt(np.square(mean_mfcc[:, start:] - target_mfcc[:, start:]).sum(axis=1) + 1e-8)
    pred_energy = np.exp(np.clip(pred_raw, -12.0, 6.0)).mean(axis=-1).clip(min=1e-8)
    target_energy = np.exp(np.clip(target_raw, -12.0, 6.0)).mean(axis=-1).clip(min=1e-8)
    mean_energy = np.exp(np.clip(mean_raw, -12.0, 6.0)).mean(axis=-1).clip(min=1e-8)
    target_energy_mean = target_energy.mean(axis=1, keepdims=True)
    target_energy_std = target_energy.std(axis=1, keepdims=True)
    target_energy_peak = target_energy.max(axis=1, keepdims=True)
    active = target_energy >= np.maximum(target_energy_mean + 0.5 * target_energy_std, 0.1 * target_energy_peak)
    fallback = np.zeros_like(active, dtype=bool)
    fallback[np.arange(active.shape[0]), target_energy.argmax(axis=1)] = True
    active = np.where(active.sum(axis=1, keepdims=True) > 0, active, fallback).astype(np.float32)
    pred_log_energy = np.log(pred_energy)
    target_log_energy = np.log(target_energy)
    mean_log_energy = np.log(mean_energy)
    active_weight = active[..., None]
    pred_active_mse = np.square(pred_raw - target_raw) * active_weight
    mean_active_mse = np.square(mean_raw - target_raw) * active_weight
    denom = np.maximum(active_weight.sum(axis=(1, 2)) * d, 1.0)
    pred_active_mse = pred_active_mse.sum(axis=(1, 2)) / denom
    mean_active_mse = mean_active_mse.sum(axis=(1, 2)) / denom
    active_peak_target = np.maximum((target_energy * active).max(axis=1), 1e-8)
    active_peak_pred = (pred_energy * active).max(axis=1)
    active_peak_mean = (mean_energy * active).max(axis=1)
    return {
        "pred_mel_corr": float(_sample_pcc(pred_flat, target_flat).mean()),
        "mean_mel_corr": float(_sample_pcc(mean_flat, target_flat).mean()),
        "pred_mel_corr_gain": float(_sample_pcc(pred_flat, target_flat).mean() - _sample_pcc(mean_flat, target_flat).mean()),
        "pred_mcd": float(pred_mcd.reshape(-1, t).mean(axis=1).mean()),
        "mean_mcd": float(mean_mcd.reshape(-1, t).mean(axis=1).mean()),
        "pred_mcd_gain": float(mean_mcd.mean() - pred_mcd.mean()),
        "pred_energy_corr": float(_sample_pcc(pred_log_energy, target_log_energy).mean()),
        "mean_energy_corr": float(_sample_pcc(mean_log_energy, target_log_energy).mean()),
        "pred_energy_corr_gain": float(_sample_pcc(pred_log_energy, target_log_energy).mean() - _sample_pcc(mean_log_energy, target_log_energy).mean()),
        "pred_active_energy_corr": float(_weighted_pcc_1d(pred_log_energy, target_log_energy, active).mean()),
        "mean_active_energy_corr": float(_weighted_pcc_1d(mean_log_energy, target_log_energy, active).mean()),
        "pred_active_recon_mse": float(pred_active_mse.mean()),
        "mean_active_recon_mse": float(mean_active_mse.mean()),
        "pred_active_recon_mse_gain": float(mean_active_mse.mean() - pred_active_mse.mean()),
        "pred_active_peak_ratio": float(np.mean(active_peak_pred / active_peak_target)),
        "mean_active_peak_ratio": float(np.mean(active_peak_mean / active_peak_target)),
        "target_active_frame_rate": float(active.mean()),
    }


def _retrieval_stats(
    pred_summary: np.ndarray,
    subjects: list[str],
    labels: list[str],
    trial_indices: list[int],
    targets: KaraOneTargets,
    topk: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    target_norm = targets.summary / np.linalg.norm(targets.summary, axis=1, keepdims=True).clip(min=1e-8)
    pred_norm = pred_summary / np.linalg.norm(pred_summary, axis=1, keepdims=True).clip(min=1e-8)
    subject_arr = targets.subject_ids.astype(str)
    trial_arr = targets.trial_indices.astype(np.int32)
    label_arr = targets.labels.astype(str)

    trial_hits = {int(k): 0 for k in topk}
    label_hits = {int(k): 0 for k in topk}
    evaluated = 0
    for i, (subject, label, trial) in enumerate(zip(subjects, labels, trial_indices)):
        mask = subject_arr == subject
        if not mask.any():  # subject absent from this (possibly partial) target cache
            continue
        evaluated += 1
        scores = pred_norm[i] @ target_norm[mask].T
        candidate_trials = trial_arr[mask]
        candidate_labels = label_arr[mask]
        order = np.argsort(scores)[::-1]
        for k in topk:
            kk = min(int(k), int(order.shape[0]))
            top = order[:kk]
            trial_hits[int(k)] += int(np.any(candidate_trials[top].astype(np.int32) == int(trial)))
            label_hits[int(k)] += int(np.any(candidate_labels[top].astype(str) == str(label)))
    n = max(1, evaluated)
    out: dict[str, float] = {}
    for k in topk:
        out[f"within_subject_trial_top{int(k)}"] = float(trial_hits[int(k)] / n)
        out[f"within_subject_label_top{int(k)}"] = float(label_hits[int(k)] / n)
    return out


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: KaraOneTrialDataset,
    targets: KaraOneTargets,
    device: str | torch.device,
    batch_size: int = 64,
    aux_targets: KaraOneTargets | None = None,
    residual_mean: bool = False,
    target_kind: str | None = None,
) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    totals: dict[str, float] = defaultdict(float)
    by_stage: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    count = 0
    pred_summaries: list[np.ndarray] = []
    zero_summaries: list[np.ndarray] = []
    mean_summaries: list[np.ndarray] = []
    pred_flat: list[np.ndarray] = []
    target_flat: list[np.ndarray] = []
    pred_seq: list[np.ndarray] = []
    target_seq: list[np.ndarray] = []
    mean_seq: list[np.ndarray] = []
    subjects: list[str] = []
    labels: list[str] = []
    trials: list[int] = []
    # HuBERT aux head (WS3): content-bearing distance + retrieval, when a HuBERT cache
    # is provided and the model has the aux head.
    hubert_pred_summaries: list[np.ndarray] = []
    hubert_cos_sum = 0.0
    hubert_n = 0

    global_mean = torch.from_numpy(targets.global_mean_norm).to(device).float()
    for batch in loader:
        eeg = batch["eeg"].to(device)
        subject_idx = batch["subject_idx"].to(device)
        stage_idx = batch["stage_idx"].to(device)
        valid_len = batch["eeg_valid_len"].to(device)
        target = batch["target_seq"].to(device)
        out = model(eeg, subject_idx, stage_idx, valid_len)
        zero_out = model(torch.zeros_like(eeg), subject_idx, stage_idx, valid_len)
        raw_pred = out["pred_latent"]
        raw_zero_pred = zero_out["pred_latent"]
        if residual_mean:
            mean_batch = global_mean.unsqueeze(0).expand_as(target)
            pred = mean_batch + raw_pred
            zero_pred = mean_batch + raw_zero_pred
        else:
            mean_batch = global_mean.unsqueeze(0).expand_as(target)
            pred = raw_pred
            zero_pred = raw_zero_pred
        b = int(eeg.shape[0])

        pred_cos = F.cosine_similarity(pred, target, dim=-1).mean(dim=1)
        zero_cos = F.cosine_similarity(zero_pred, target, dim=-1).mean(dim=1)
        mean_cos = F.cosine_similarity(mean_batch, target, dim=-1).mean(dim=1)
        pred_mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2))
        zero_mse = F.mse_loss(zero_pred, target, reduction="none").mean(dim=(1, 2))
        mean_mse = F.mse_loss(mean_batch, target, reduction="none").mean(dim=(1, 2))
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
        if residual_mean:
            residual_target = target - mean_batch
            residual_cos = F.cosine_similarity(raw_pred, residual_target, dim=-1).mean(dim=1)
            zero_residual_cos = F.cosine_similarity(raw_zero_pred, residual_target, dim=-1).mean(dim=1)
            residual_mse = F.mse_loss(raw_pred, residual_target, reduction="none").mean(dim=(1, 2))
            zero_residual_mse = F.mse_loss(raw_zero_pred, residual_target, reduction="none").mean(dim=(1, 2))
            metrics.update(
                {
                    "pred_residual_cos": residual_cos.detach().cpu().numpy(),
                    "zeroeeg_residual_cos": zero_residual_cos.detach().cpu().numpy(),
                    "pred_residual_mse": residual_mse.detach().cpu().numpy(),
                    "zeroeeg_residual_mse": zero_residual_mse.detach().cpu().numpy(),
                }
            )
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
        mean_np = global_mean.unsqueeze(0).expand_as(target).detach().cpu().numpy()
        tgt_np = target.detach().cpu().numpy()
        pred_summaries.append(pred_np.mean(axis=1))
        zero_summaries.append(zero_np.mean(axis=1))
        mean_summaries.append(mean_np.mean(axis=1))
        pred_flat.append(pred_np.reshape(pred_np.shape[0], -1))
        target_flat.append(tgt_np.reshape(tgt_np.shape[0], -1))
        pred_seq.append(pred_np)
        target_seq.append(tgt_np)
        mean_seq.append(mean_np)
        subjects.extend([str(item) for item in batch["subject"]])
        labels.extend([str(item) for item in batch["label"]])
        trials.extend([int(item) for item in batch["trial_index"]])

        pred_hub = out.get("pred_hubert")
        if aux_targets is not None and pred_hub is not None and "hubert_seq" in batch:
            hub_t = batch["hubert_seq"].to(device)
            hub_cos = F.cosine_similarity(pred_hub, hub_t, dim=-1).mean(dim=1)  # per-sample
            hubert_cos_sum += float(hub_cos.sum().detach().cpu())
            hubert_n += int(pred_hub.shape[0])
            hubert_pred_summaries.append(pred_hub.mean(dim=1).detach().cpu().numpy())

    out_metrics = {name: value / max(count, 1) for name, value in totals.items()}
    pred_summary = np.concatenate(pred_summaries, axis=0)
    zero_summary = np.concatenate(zero_summaries, axis=0)
    mean_summary = np.concatenate(mean_summaries, axis=0)
    pred_matrix = np.concatenate(pred_flat, axis=0)
    target_matrix = np.concatenate(target_flat, axis=0)
    pred_std = pred_matrix.std(axis=0)
    target_std = target_matrix.std(axis=0)
    # Per-sample Pearson correlation over the whole spectrogram/latent (mel-PCC, as in
    # Park 2025 / FESDE). Reflects whole time-freq similarity, not just per-frame cosine.
    pcc = _sample_pcc(pred_matrix, target_matrix)
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
    if residual_mean:
        out_metrics.update(
            {
                "prediction_mode": "residual_global_mean",
                "pred_over_mean_mse_gain": out_metrics["mean_recon_mse"] - out_metrics["pred_recon_mse"],
                "pred_residual_cos_gain": out_metrics["pred_residual_cos"] - out_metrics["zeroeeg_residual_cos"],
                "pred_residual_mse_gain": out_metrics["zeroeeg_residual_mse"] - out_metrics["pred_residual_mse"],
            }
        )
    if str(target_kind or "").lower() == "mel":
        out_metrics.update(
            _mel_metrics(
                np.concatenate(pred_seq, axis=0),
                np.concatenate(target_seq, axis=0),
                np.concatenate(mean_seq, axis=0),
                targets,
            )
        )
    out_metrics.update({f"pred_{k}": v for k, v in _retrieval_stats(pred_summary, subjects, labels, trials, targets).items()})
    out_metrics.update({f"zeroeeg_{k}": v for k, v in _retrieval_stats(zero_summary, subjects, labels, trials, targets).items()})
    out_metrics.update({f"mean_{k}": v for k, v in _retrieval_stats(mean_summary, subjects, labels, trials, targets).items()})

    # HuBERT-space content metrics (more trustworthy than mel-PCC; WS3).
    if aux_targets is not None and hubert_n > 0:
        out_metrics["pred_hubert_cos"] = hubert_cos_sum / max(hubert_n, 1)
        hub_summary = np.concatenate(hubert_pred_summaries, axis=0)
        out_metrics.update(
            {f"pred_hubert_{k}": v for k, v in _retrieval_stats(hub_summary, subjects, labels, trials, aux_targets).items()}
        )

    stage_payload = {}
    for stage, payload in by_stage.items():
        n = max(payload.pop("n"), 1.0)
        stage_payload[stage] = {name: value / n for name, value in payload.items()}
    out_metrics["by_stage"] = stage_payload
    return out_metrics

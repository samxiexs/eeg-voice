from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .alignment import shift_sequence_np
from .data import KaraOneTrialDataset
from .prototypes import TorchSemanticMelPrototypes
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


def _resample_1d(values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if values.size == 0:
        return np.zeros(n, dtype=np.float64)
    if values.size == n:
        return values.astype(np.float64)
    src = np.linspace(0.0, 1.0, values.size)
    dst = np.linspace(0.0, 1.0, n)
    return np.interp(dst, src, values)


def _dtw_distance_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        return 0.0
    n, m = a.size, b.size
    acc = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            acc[i, j] = cost + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[n, m] / max(n + m, 1))


def _best_shift_corr_1d(a: np.ndarray, b: np.ndarray, max_shift: int = 32) -> tuple[float, int]:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or b.size < 2:
        return 0.0, 0
    best_corr = -1.0
    best_shift = 0
    for shift in range(-int(max_shift), int(max_shift) + 1):
        if shift < 0:
            aa, bb = a[-shift:], b[: a.size + shift]
        elif shift > 0:
            aa, bb = a[: a.size - shift], b[shift:]
        else:
            aa, bb = a, b
        m = min(aa.size, bb.size)
        if m < 2:
            continue
        aa = aa[:m] - aa[:m].mean()
        bb = bb[:m] - bb[:m].mean()
        corr = float((aa * bb).sum() / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-8))
        if corr > best_corr:
            best_corr = corr
            best_shift = int(shift)
    return float(best_corr), int(best_shift)


def _raw_mel_envelope(seq_norm: np.ndarray, targets: KaraOneTargets) -> np.ndarray:
    raw = seq_norm * targets.target_std.reshape(1, 1, -1) + targets.target_mean.reshape(1, 1, -1)
    return np.sqrt(np.exp(np.clip(raw, -12.0, 6.0)).mean(axis=-1).clip(min=1e-12))


def _temporal_elastic_metrics(
    pred_norm: np.ndarray,
    zero_norm: np.ndarray,
    mean_norm: np.ndarray,
    target_norm: np.ndarray,
    targets: KaraOneTargets,
    duration_target: np.ndarray | None,
    duration_pred: np.ndarray | None,
    active_rms: np.ndarray | None,
    pred_log_rms: np.ndarray | None,
    active_peak: np.ndarray | None,
    pred_log_peak: np.ndarray | None,
) -> dict[str, float]:
    pred_flat = pred_norm.reshape(pred_norm.shape[0], -1)
    zero_flat = zero_norm.reshape(zero_norm.shape[0], -1)
    mean_flat = mean_norm.reshape(mean_norm.shape[0], -1)
    tgt_flat = target_norm.reshape(target_norm.shape[0], -1)
    pred_shape = float(_sample_pcc(pred_flat, tgt_flat).mean())
    zero_shape = float(_sample_pcc(zero_flat, tgt_flat).mean())
    mean_shape = float(_sample_pcc(mean_flat, tgt_flat).mean())
    pred_env = _raw_mel_envelope(pred_norm, targets)
    zero_env = _raw_mel_envelope(zero_norm, targets)
    mean_env = _raw_mel_envelope(mean_norm, targets)
    tgt_env = _raw_mel_envelope(target_norm, targets)
    pred_dtw = float(np.mean([_dtw_distance_1d(pred_env[i], tgt_env[i]) for i in range(pred_env.shape[0])]))
    zero_dtw = float(np.mean([_dtw_distance_1d(zero_env[i], tgt_env[i]) for i in range(zero_env.shape[0])]))
    mean_dtw = float(np.mean([_dtw_distance_1d(mean_env[i], tgt_env[i]) for i in range(mean_env.shape[0])]))
    best = [_best_shift_corr_1d(pred_env[i], tgt_env[i], max_shift=max(4, pred_env.shape[1] // 2)) for i in range(pred_env.shape[0])]
    best_corr = float(np.mean([item[0] for item in best])) if best else 0.0
    best_shift = float(np.mean([item[1] for item in best])) if best else 0.0
    metrics = {
        "active_segment_shape_corr": pred_shape,
        "zeroeeg_active_segment_shape_corr": zero_shape,
        "mean_active_segment_shape_corr": mean_shape,
        "pred_over_zero_active_shape_gain": pred_shape - zero_shape,
        "pred_over_mean_active_shape_gain": pred_shape - mean_shape,
        "active_core_mel_corr": pred_shape,
        "active_core_softdtw": pred_dtw,
        "zeroeeg_active_core_softdtw": zero_dtw,
        "mean_active_core_softdtw": mean_dtw,
        "pred_over_zero_softdtw_gain": zero_dtw - pred_dtw,
        "pred_over_mean_softdtw_gain": mean_dtw - pred_dtw,
        "best_shift_full_env_corr": best_corr,
        "best_shift_frames": best_shift,
    }
    if duration_target is not None and duration_pred is not None:
        ratio = duration_pred.astype(np.float64) / np.maximum(duration_target.astype(np.float64), 1.0)
        metrics["active_duration_ratio"] = float(np.mean(ratio))
        metrics["duration_score"] = float(np.mean(np.exp(-np.abs(np.log(np.maximum(ratio, 1e-8))))))
    if active_rms is not None and pred_log_rms is not None:
        pred_rms = np.exp(pred_log_rms.astype(np.float64))
        ratio = pred_rms / np.maximum(active_rms.astype(np.float64), 1e-8)
        metrics["active_rms_ratio"] = float(np.mean(ratio))
        metrics["loudness_score"] = float(np.mean(np.exp(-np.abs(np.log(np.maximum(ratio, 1e-8))))))
    if active_peak is not None and pred_log_peak is not None:
        pred_peak = np.exp(pred_log_peak.astype(np.float64))
        ratio = pred_peak / np.maximum(active_peak.astype(np.float64), 1e-8)
        metrics["active_peak_ratio"] = float(np.mean(ratio))
    return metrics


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


def _collapse_tokens(seq: np.ndarray) -> list[int]:
    out: list[int] = []
    last: int | None = None
    for item in seq.astype(np.int64).tolist():
        value = int(item)
        if last is None or value != last:
            out.append(value)
        last = value
    return out


def _edit_distance(a: list[int], b: list[int]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, av in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, bv in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + int(av != bv))
        prev = cur
    return int(prev[-1])


def _semantic_token_metrics(
    pred_tokens: list[np.ndarray],
    target_tokens: list[np.ndarray],
    masks: list[np.ndarray],
) -> dict[str, float]:
    if not pred_tokens or not target_tokens:
        return {}
    pred = np.concatenate(pred_tokens, axis=0)
    target = np.concatenate(target_tokens, axis=0)
    mask = np.concatenate(masks, axis=0).astype(bool)
    active = int(mask.sum())
    if active <= 0:
        return {}
    correct = (pred == target) & mask
    active_targets = target[mask].astype(np.int64)
    counts = np.bincount(active_targets, minlength=max(int(active_targets.max()) + 1, 1))
    majority = int(counts.argmax())
    pred_edits: list[float] = []
    majority_edits: list[float] = []
    for i in range(pred.shape[0]):
        m = mask[i]
        if not bool(m.any()):
            continue
        tgt_seq = _collapse_tokens(target[i][m])
        pred_seq = _collapse_tokens(pred[i][m])
        majority_seq = [majority]
        denom = float(max(len(tgt_seq), 1))
        pred_edits.append(_edit_distance(pred_seq, tgt_seq) / denom)
        majority_edits.append(_edit_distance(majority_seq, tgt_seq) / denom)
    pred_edit = float(np.mean(pred_edits)) if pred_edits else 1.0
    majority_edit = float(np.mean(majority_edits)) if majority_edits else 1.0
    return {
        "semantic_token_frame_acc": float(correct.sum() / max(active, 1)),
        "semantic_token_edit_distance": pred_edit,
        "semantic_token_majority_edit_distance": majority_edit,
        "semantic_token_edit_gain": majority_edit - pred_edit,
    }


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
    semantic_prototypes: TorchSemanticMelPrototypes | None = None,
    semantic_prototype_residual: bool = False,
) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    totals: dict[str, float] = defaultdict(float)
    by_stage: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    count = 0
    pred_summaries: list[np.ndarray] = []
    zero_summaries: list[np.ndarray] = []
    mean_summaries: list[np.ndarray] = []
    proto_summaries: list[np.ndarray] = []
    pred_flat: list[np.ndarray] = []
    target_flat: list[np.ndarray] = []
    pred_seq: list[np.ndarray] = []
    zero_seq: list[np.ndarray] = []
    target_seq: list[np.ndarray] = []
    mean_seq: list[np.ndarray] = []
    proto_seq: list[np.ndarray] = []
    raw_pred_seq: list[np.ndarray] = []
    raw_zero_seq: list[np.ndarray] = []
    pred_aligned_seq: list[np.ndarray] = []
    oracle_aligned_seq: list[np.ndarray] = []
    pred_lag_values: list[np.ndarray] = []
    target_lag_values: list[np.ndarray] = []
    lag_conf_values: list[np.ndarray] = []
    peak_offset_values: list[np.ndarray] = []
    onset_offset_values: list[np.ndarray] = []
    subjects: list[str] = []
    labels: list[str] = []
    trials: list[int] = []
    # HuBERT aux head (WS3): content-bearing distance + retrieval, when a HuBERT cache
    # is provided and the model has the aux head.
    hubert_pred_summaries: list[np.ndarray] = []
    hubert_cos_sum = 0.0
    hubert_n = 0
    token_pred_values: list[np.ndarray] = []
    zero_token_pred_values: list[np.ndarray] = []
    token_target_values: list[np.ndarray] = []
    token_mask_values: list[np.ndarray] = []
    core_active_masks: list[np.ndarray] = []
    core_silence_masks: list[np.ndarray] = []
    core_pre_noise_masks: list[np.ndarray] = []
    shift_pred_frames: list[np.ndarray] = []
    shift_target_frames: list[np.ndarray] = []
    shift_top1_hits: list[np.ndarray] = []
    shift_top3_hits: list[np.ndarray] = []
    active_duration_targets: list[np.ndarray] = []
    active_duration_preds: list[np.ndarray] = []
    active_rms_targets: list[np.ndarray] = []
    pred_log_rms_values: list[np.ndarray] = []
    active_peak_targets: list[np.ndarray] = []
    pred_log_peak_values: list[np.ndarray] = []

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
        diagnostic_proto_batch = None
        diagnostic_zero_proto_batch = None
        if semantic_prototypes is not None and "semantic_token_logits" in out:
            token_mask = batch.get("semantic_token_mask")
            token_mask_t = token_mask.to(device) if token_mask is not None else None
            diagnostic_proto_batch = semantic_prototypes.prototype_from_logits(out["semantic_token_logits"], token_mask_t)
            diagnostic_zero_proto_batch = semantic_prototypes.prototype_from_logits(zero_out["semantic_token_logits"], token_mask_t)
        if semantic_prototype_residual and diagnostic_proto_batch is not None:
            mean_batch = global_mean.unsqueeze(0).expand_as(target)
            proto_batch = diagnostic_proto_batch
            pred = proto_batch + raw_pred
            zero_pred = diagnostic_zero_proto_batch + raw_zero_pred
        elif residual_mean:
            mean_batch = global_mean.unsqueeze(0).expand_as(target)
            pred = mean_batch + raw_pred
            zero_pred = mean_batch + raw_zero_pred
            proto_batch = diagnostic_proto_batch if diagnostic_proto_batch is not None else mean_batch
        else:
            mean_batch = global_mean.unsqueeze(0).expand_as(target)
            pred = raw_pred
            zero_pred = raw_zero_pred
            proto_batch = diagnostic_proto_batch if diagnostic_proto_batch is not None else mean_batch
        b = int(eeg.shape[0])

        pred_cos = F.cosine_similarity(pred, target, dim=-1).mean(dim=1)
        zero_cos = F.cosine_similarity(zero_pred, target, dim=-1).mean(dim=1)
        mean_cos = F.cosine_similarity(mean_batch, target, dim=-1).mean(dim=1)
        proto_cos = F.cosine_similarity(proto_batch, target, dim=-1).mean(dim=1)
        pred_mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2))
        zero_mse = F.mse_loss(zero_pred, target, reduction="none").mean(dim=(1, 2))
        mean_mse = F.mse_loss(mean_batch, target, reduction="none").mean(dim=(1, 2))
        proto_mse = F.mse_loss(proto_batch, target, reduction="none").mean(dim=(1, 2))
        content_acc = (out["content_logits"].argmax(dim=-1).cpu() == batch["label_idx"]).float()

        metrics = {
            "pred_recon_cos": pred_cos.detach().cpu().numpy(),
            "zeroeeg_recon_cos": zero_cos.detach().cpu().numpy(),
            "mean_recon_cos": mean_cos.detach().cpu().numpy(),
            "semantic_proto_recon_cos": proto_cos.detach().cpu().numpy(),
            "pred_recon_mse": pred_mse.detach().cpu().numpy(),
            "zeroeeg_recon_mse": zero_mse.detach().cpu().numpy(),
            "mean_recon_mse": mean_mse.detach().cpu().numpy(),
            "semantic_proto_recon_mse": proto_mse.detach().cpu().numpy(),
            "content_acc": content_acc.numpy(),
        }
        if residual_mean or semantic_prototype_residual:
            residual_base = proto_batch if semantic_prototype_residual else mean_batch
            residual_target = target - residual_base
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
        raw_pred_np = raw_pred.detach().cpu().numpy()
        raw_zero_np = raw_zero_pred.detach().cpu().numpy()
        mean_np = global_mean.unsqueeze(0).expand_as(target).detach().cpu().numpy()
        proto_np = proto_batch.detach().cpu().numpy()
        tgt_np = target.detach().cpu().numpy()
        if "lag_mel_frames" in batch:
            oracle_frames = batch["lag_mel_frames"].detach().cpu().numpy().astype(np.int32)
            hop_sec = float(batch.get("alignment_mel_hop_sec", torch.tensor([0.016]))[0])
            pred_lag_sec = out.get("pred_lag_mu", torch.zeros((b,), device=device)).detach().cpu().numpy().astype(np.float32)
            pred_frames = np.rint(pred_lag_sec / max(hop_sec, 1e-6)).astype(np.int32)
            max_shift = max(1, int(pred_np.shape[1] * 0.75))
            pred_frames = np.clip(pred_frames, -max_shift, max_shift)
            pred_aligned_seq.append(np.stack([shift_sequence_np(pred_np[i], -int(pred_frames[i])) for i in range(b)], axis=0))
            oracle_aligned_seq.append(np.stack([shift_sequence_np(pred_np[i], -int(oracle_frames[i])) for i in range(b)], axis=0))
            pred_lag_values.append(pred_lag_sec)
            target_lag_values.append(batch["lag_sec"].detach().cpu().numpy().astype(np.float32))
            lag_conf_values.append(batch["lag_confidence"].detach().cpu().numpy().astype(np.float32))
            if "eeg_peak_t" in batch and "audio_peak_t" in batch:
                peak_offset_values.append(
                    (batch["eeg_peak_t"] - batch["audio_peak_t"]).detach().cpu().numpy().astype(np.float32)
                )
            if "eeg_onset_t" in batch and "audio_onset_t" in batch:
                onset_offset_values.append(
                    (batch["eeg_onset_t"] - batch["audio_onset_t"]).detach().cpu().numpy().astype(np.float32)
                )
        pred_summaries.append(pred_np.mean(axis=1))
        zero_summaries.append(zero_np.mean(axis=1))
        mean_summaries.append(mean_np.mean(axis=1))
        proto_summaries.append(proto_np.mean(axis=1))
        pred_flat.append(pred_np.reshape(pred_np.shape[0], -1))
        target_flat.append(tgt_np.reshape(tgt_np.shape[0], -1))
        pred_seq.append(pred_np)
        zero_seq.append(zero_np)
        target_seq.append(tgt_np)
        mean_seq.append(mean_np)
        proto_seq.append(proto_np)
        raw_pred_seq.append(raw_pred_np)
        raw_zero_seq.append(raw_zero_np)
        if "core_active_mask" in batch:
            core_active_masks.append(batch["core_active_mask"].detach().cpu().numpy().astype(np.float32))
        if "core_mask" in batch:
            core_mask_np = batch["core_mask"].detach().cpu().numpy().astype(np.float32)
            core_silence_masks.append(1.0 - core_mask_np)
        if "core_pre_noise_mask" in batch:
            core_pre_noise_masks.append(batch["core_pre_noise_mask"].detach().cpu().numpy().astype(np.float32))
        if "pred_shift_logits" in out and "shift_target_frame" in batch:
            logits = out["pred_shift_logits"].detach().cpu()
            bsz, bins = int(logits.shape[0]), int(logits.shape[1])
            lo, hi = -12, 62
            target_shift = batch["shift_target_frame"].detach().cpu().numpy().astype(np.float32)
            if bins > 1:
                grid = np.linspace(float(lo), float(hi), bins, dtype=np.float32)
                pred_idx = logits.argmax(dim=-1).numpy().astype(np.int64)
                pred_shift = grid[pred_idx]
                target_bin = np.rint((target_shift - float(lo)) / max(float(hi - lo), 1.0) * float(bins - 1)).astype(np.int64)
                target_bin = np.clip(target_bin, 0, bins - 1)
                top3 = torch.topk(logits, k=min(3, bins), dim=-1).indices.numpy()
                shift_top1_hits.append((pred_idx == target_bin).astype(np.float32))
                shift_top3_hits.append(np.asarray([target_bin[i] in set(top3[i].tolist()) for i in range(bsz)], dtype=np.float32))
            else:
                pred_shift = np.zeros_like(target_shift)
                shift_top1_hits.append(np.ones_like(target_shift, dtype=np.float32))
                shift_top3_hits.append(np.ones_like(target_shift, dtype=np.float32))
            shift_pred_frames.append(pred_shift.astype(np.float32))
            shift_target_frames.append(target_shift.astype(np.float32))
        if "active_duration_frames" in batch:
            active_duration_targets.append(batch["active_duration_frames"].detach().cpu().numpy().astype(np.float32))
            if "pred_duration_mu" in out:
                active_duration_preds.append(out["pred_duration_mu"].detach().cpu().numpy().astype(np.float32))
        if "active_rms" in batch:
            active_rms_targets.append(batch["active_rms"].detach().cpu().numpy().astype(np.float32))
            pred_log_rms_values.append(out["pred_log_rms"].detach().cpu().numpy().astype(np.float32))
        if "active_peak" in batch and "pred_log_peak" in out:
            active_peak_targets.append(batch["active_peak"].detach().cpu().numpy().astype(np.float32))
            pred_log_peak_values.append(out["pred_log_peak"].detach().cpu().numpy().astype(np.float32))
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
        if "semantic_token_logits" in out and "semantic_token_targets" in batch:
            token_pred_values.append(out["semantic_token_logits"].argmax(dim=-1).detach().cpu().numpy())
            zero_token_pred_values.append(zero_out["semantic_token_logits"].argmax(dim=-1).detach().cpu().numpy())
            token_target_values.append(batch["semantic_token_targets"].detach().cpu().numpy())
            token_mask_values.append(batch.get("semantic_token_mask", torch.ones_like(batch["semantic_token_targets"]).float()).detach().cpu().numpy())

    out_metrics = {name: value / max(count, 1) for name, value in totals.items()}
    pred_summary = np.concatenate(pred_summaries, axis=0)
    zero_summary = np.concatenate(zero_summaries, axis=0)
    mean_summary = np.concatenate(mean_summaries, axis=0)
    proto_summary = np.concatenate(proto_summaries, axis=0) if proto_summaries else mean_summary
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
            "pred_over_semantic_proto_cos_gain": out_metrics["pred_recon_cos"] - out_metrics["semantic_proto_recon_cos"],
            "pred_std_ratio_median": float(np.median(pred_std / np.maximum(target_std, 1e-6))),
            "pred_pairwise_corr_median": _corr_median(pred_matrix),
            "pred_pcc": float(np.mean(pcc)),
        }
    )
    if residual_mean:
        out_metrics.update(
            {
                "prediction_mode": "speech_core_residual_mean"
                if bool(getattr(targets, "is_speech_core", False))
                else "residual_global_mean",
                "pred_over_mean_mse_gain": out_metrics["mean_recon_mse"] - out_metrics["pred_recon_mse"],
                "pred_over_semantic_proto_mse_gain": out_metrics["semantic_proto_recon_mse"] - out_metrics["pred_recon_mse"],
                "pred_residual_cos_gain": out_metrics["pred_residual_cos"] - out_metrics["zeroeeg_residual_cos"],
                "pred_residual_mse_gain": out_metrics["zeroeeg_residual_mse"] - out_metrics["pred_residual_mse"],
            }
        )
    if semantic_prototype_residual:
        out_metrics["prediction_mode"] = "semantic_prototype_residual"
        out_metrics["pred_over_semantic_proto_mse_gain"] = out_metrics["semantic_proto_recon_mse"] - out_metrics["pred_recon_mse"]
    if str(target_kind or "").lower() == "mel":
        pred_seq_np = np.concatenate(pred_seq, axis=0)
        target_seq_np = np.concatenate(target_seq, axis=0)
        mean_seq_np = np.concatenate(mean_seq, axis=0)
        out_metrics.update(
            _mel_metrics(
                pred_seq_np,
                target_seq_np,
                mean_seq_np,
                targets,
            )
        )
        zero_seq_np = np.concatenate(zero_seq, axis=0)
        zero_metrics = _mel_metrics(zero_seq_np, target_seq_np, mean_seq_np, targets)
        out_metrics.update(
            {
                "zeroeeg_mel_corr": zero_metrics["pred_mel_corr"],
                "zeroeeg_mcd": zero_metrics["pred_mcd"],
                "zeroeeg_energy_corr": zero_metrics["pred_energy_corr"],
                "pred_over_zero_mel_corr_gain": out_metrics["pred_mel_corr"] - zero_metrics["pred_mel_corr"],
                "pred_over_zero_mcd_gain": zero_metrics["pred_mcd"] - out_metrics["pred_mcd"],
                "pred_over_zero_energy_corr_gain": out_metrics["pred_energy_corr"] - zero_metrics["pred_energy_corr"],
            }
        )
        raw_pred_np = np.concatenate(raw_pred_seq, axis=0)
        raw_zero_np = np.concatenate(raw_zero_seq, axis=0)
        residual_base_np = np.concatenate(proto_seq if semantic_prototype_residual else mean_seq, axis=0)
        residual_target_np = target_seq_np - residual_base_np
        residual_std = raw_pred_np.reshape(-1, raw_pred_np.shape[-1]).std(axis=0)
        residual_target_std = residual_target_np.reshape(-1, residual_target_np.shape[-1]).std(axis=0)
        zero_residual_std = raw_zero_np.reshape(-1, raw_zero_np.shape[-1]).std(axis=0)
        out_metrics.update(
            {
                "residual_std_ratio": float(np.median(residual_std / np.maximum(residual_target_std, 1e-6))),
                "zeroeeg_residual_std_ratio": float(np.median(zero_residual_std / np.maximum(residual_target_std, 1e-6))),
                "residual_pairwise_corr": _corr_median(raw_pred_np.reshape(raw_pred_np.shape[0], -1)),
                "zeroeeg_residual_pairwise_corr": _corr_median(raw_zero_np.reshape(raw_zero_np.shape[0], -1)),
            }
        )
        if proto_seq:
            proto_metrics = _mel_metrics(np.concatenate(proto_seq, axis=0), target_seq_np, mean_seq_np, targets)
            out_metrics.update(
                {
                    "semantic_proto_mel_corr": proto_metrics["pred_mel_corr"],
                    "semantic_proto_mcd": proto_metrics["pred_mcd"],
                    "semantic_proto_energy_corr": proto_metrics["pred_energy_corr"],
                    "semantic_proto_active_recon_mse": proto_metrics["pred_active_recon_mse"],
                    "pred_over_semantic_proto_mel_corr_gain": out_metrics["pred_mel_corr"] - proto_metrics["pred_mel_corr"],
                    "pred_over_semantic_proto_mcd_gain": proto_metrics["pred_mcd"] - out_metrics["pred_mcd"],
                    "pred_over_semantic_proto_energy_corr_gain": out_metrics["pred_energy_corr"] - proto_metrics["pred_energy_corr"],
                    "pred_over_semantic_proto_active_recon_mse_gain": proto_metrics["pred_active_recon_mse"] - out_metrics["pred_active_recon_mse"],
                }
            )
        if pred_aligned_seq:
            aligned_metrics = _mel_metrics(
                np.concatenate(pred_aligned_seq, axis=0),
                np.concatenate(target_seq, axis=0),
                np.concatenate(mean_seq, axis=0),
                targets,
            )
            out_metrics.update({f"aligned_{name}": value for name, value in aligned_metrics.items()})
            oracle_aligned_metrics = _mel_metrics(
                np.concatenate(oracle_aligned_seq, axis=0),
                np.concatenate(target_seq, axis=0),
                np.concatenate(mean_seq, axis=0),
                targets,
            )
            out_metrics.update({f"oracle_aligned_{name}": value for name, value in oracle_aligned_metrics.items()})
        if bool(getattr(targets, "is_speech_core", False)):
            for name in (
                "pred_mel_corr",
                "mean_mel_corr",
                "pred_mel_corr_gain",
                "pred_mcd",
                "mean_mcd",
                "pred_mcd_gain",
                "pred_energy_corr",
                "mean_energy_corr",
                "pred_energy_corr_gain",
                "pred_active_recon_mse",
                "mean_active_recon_mse",
                "pred_active_recon_mse_gain",
            ):
                if name in out_metrics:
                    out_metrics[f"core_{name.replace('pred_', '').replace('mean_', 'mean_')}"] = out_metrics[name]
            out_metrics["core_mel_corr_gain"] = out_metrics.get("pred_mel_corr_gain", 0.0)
            out_metrics["core_energy_corr_gain"] = out_metrics.get("pred_energy_corr_gain", 0.0)
            out_metrics["core_mcd_gain"] = out_metrics.get("pred_mcd_gain", 0.0)
            out_metrics["core_active_recon_mse_gain"] = out_metrics.get("pred_active_recon_mse_gain", 0.0)
            if core_active_masks:
                active = np.concatenate(core_active_masks, axis=0).astype(np.float32)
                pred_raw = pred_seq_np * targets.target_std.reshape(1, 1, -1) + targets.target_mean.reshape(1, 1, -1)
                tgt_raw = target_seq_np * targets.target_std.reshape(1, 1, -1) + targets.target_mean.reshape(1, 1, -1)
                pred_energy = np.exp(np.clip(pred_raw, -12.0, 6.0)).mean(axis=-1)
                tgt_energy = np.exp(np.clip(tgt_raw, -12.0, 6.0)).mean(axis=-1)
                pred_active = np.sqrt((pred_energy * active).sum(axis=1) / np.maximum(active.sum(axis=1), 1.0))
                tgt_active = np.sqrt((tgt_energy * active).sum(axis=1) / np.maximum(active.sum(axis=1), 1.0))
                ratio = pred_active / np.maximum(tgt_active, 1e-8)
                out_metrics["active_rms_ratio"] = float(np.mean(ratio))
                out_metrics["active_rms_score"] = float(np.mean(np.exp(-np.abs(np.log(np.maximum(ratio, 1e-8))))))
            if core_silence_masks:
                silence = np.concatenate(core_silence_masks, axis=0).astype(np.float32)
                pred_raw = pred_seq_np * targets.target_std.reshape(1, 1, -1) + targets.target_mean.reshape(1, 1, -1)
                pred_energy = np.exp(np.clip(pred_raw, -12.0, 6.0)).mean(axis=-1)
                out_metrics["silence_leakage"] = float((pred_energy * silence).sum() / max(float(silence.sum()), 1.0))
            if core_pre_noise_masks:
                pre_noise = np.concatenate(core_pre_noise_masks, axis=0).astype(np.float32)
                pred_raw = pred_seq_np * targets.target_std.reshape(1, 1, -1) + targets.target_mean.reshape(1, 1, -1)
                pred_energy = np.exp(np.clip(pred_raw, -12.0, 6.0)).mean(axis=-1)
                out_metrics["pre_noise_energy"] = float((pred_energy * pre_noise).sum() / max(float(pre_noise.sum()), 1.0))
            if bool(getattr(targets, "is_temporal_elastic_core", False)):
                duration_target = np.concatenate(active_duration_targets, axis=0) if active_duration_targets else None
                duration_pred = np.concatenate(active_duration_preds, axis=0) if active_duration_preds else None
                active_rms = np.concatenate(active_rms_targets, axis=0) if active_rms_targets else None
                pred_log_rms = np.concatenate(pred_log_rms_values, axis=0) if pred_log_rms_values else None
                active_peak = np.concatenate(active_peak_targets, axis=0) if active_peak_targets else None
                pred_log_peak = np.concatenate(pred_log_peak_values, axis=0) if pred_log_peak_values else None
                out_metrics.update(
                    _temporal_elastic_metrics(
                        pred_seq_np,
                        zero_seq_np,
                        mean_seq_np,
                        target_seq_np,
                        targets,
                        duration_target=duration_target,
                        duration_pred=duration_pred,
                        active_rms=active_rms,
                        pred_log_rms=pred_log_rms,
                        active_peak=active_peak,
                        pred_log_peak=pred_log_peak,
                    )
                )
    if pred_lag_values:
        pred_lag = np.concatenate(pred_lag_values, axis=0)
        target_lag = np.concatenate(target_lag_values, axis=0)
        conf = np.concatenate(lag_conf_values, axis=0)
        active = conf > 0.0
        if active.any():
            out_metrics["lag_mae_sec"] = float(np.mean(np.abs(pred_lag[active] - target_lag[active])))
            out_metrics["lag_rmse_sec"] = float(np.sqrt(np.mean(np.square(pred_lag[active] - target_lag[active]))))
            out_metrics["lag_target_median_sec"] = float(np.median(target_lag[active]))
            out_metrics["lag_pred_median_sec"] = float(np.median(pred_lag[active]))
            out_metrics["lag_confidence_mean"] = float(np.mean(conf[active]))
            out_metrics["lag_score"] = float(max(0.0, 1.0 - out_metrics["lag_mae_sec"] / 0.75))
            out_metrics["energy_com_error"] = out_metrics["lag_mae_sec"]
            if peak_offset_values:
                peak_offset = np.concatenate(peak_offset_values, axis=0)
                out_metrics["peak_error"] = float(np.mean(np.abs(pred_lag[active] - peak_offset[active])))
            if onset_offset_values:
                onset_offset = np.concatenate(onset_offset_values, axis=0)
                out_metrics["active_onset_error"] = float(np.mean(np.abs(pred_lag[active] - onset_offset[active])))
    if shift_pred_frames:
        pred_shift = np.concatenate(shift_pred_frames, axis=0)
        target_shift = np.concatenate(shift_target_frames, axis=0)
        top1 = np.concatenate(shift_top1_hits, axis=0)
        top3 = np.concatenate(shift_top3_hits, axis=0)
        mae_frames = float(np.mean(np.abs(pred_shift - target_shift)))
        hop_sec = 0.016
        if hasattr(targets, "has_field") and targets.has_field("mel_hop_sec"):
            hop_sec = float(targets.field(str(targets.subject_ids[0]), int(targets.trial_indices[0]), "mel_hop_sec"))
        out_metrics.update(
            {
                "shift_acc_top1": float(np.mean(top1)),
                "shift_acc_top3": float(np.mean(top3)),
                "shift_mae_frames": mae_frames,
                "shift_mae_sec": float(mae_frames * hop_sec),
                "shift_target_median_frame": float(np.median(target_shift)),
                "shift_pred_median_frame": float(np.median(pred_shift)),
                "shift_score": float(max(0.0, 1.0 - mae_frames * hop_sec / 0.75)),
            }
        )
    out_metrics.update({f"pred_{k}": v for k, v in _retrieval_stats(pred_summary, subjects, labels, trials, targets).items()})
    out_metrics.update({f"zeroeeg_{k}": v for k, v in _retrieval_stats(zero_summary, subjects, labels, trials, targets).items()})
    out_metrics.update({f"mean_{k}": v for k, v in _retrieval_stats(mean_summary, subjects, labels, trials, targets).items()})
    out_metrics.update({f"semantic_proto_{k}": v for k, v in _retrieval_stats(proto_summary, subjects, labels, trials, targets).items()})

    # HuBERT-space content metrics (more trustworthy than mel-PCC; WS3).
    if aux_targets is not None and hubert_n > 0:
        out_metrics["pred_hubert_cos"] = hubert_cos_sum / max(hubert_n, 1)
        hub_summary = np.concatenate(hubert_pred_summaries, axis=0)
        out_metrics.update(
            {f"pred_hubert_{k}": v for k, v in _retrieval_stats(hub_summary, subjects, labels, trials, aux_targets).items()}
        )
    token_metrics = _semantic_token_metrics(token_pred_values, token_target_values, token_mask_values)
    out_metrics.update(token_metrics)
    zero_token_metrics = _semantic_token_metrics(zero_token_pred_values, token_target_values, token_mask_values)
    out_metrics.update({f"zeroeeg_{name}": value for name, value in zero_token_metrics.items()})
    if token_metrics and zero_token_metrics:
        out_metrics["semantic_token_over_zeroeeg_edit_gain"] = zero_token_metrics["semantic_token_edit_distance"] - token_metrics["semantic_token_edit_distance"]

    stage_payload = {}
    for stage, payload in by_stage.items():
        n = max(payload.pop("n"), 1.0)
        stage_payload[stage] = {name: value / n for name, value in payload.items()}
    out_metrics["by_stage"] = stage_payload
    return out_metrics

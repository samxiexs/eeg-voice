from __future__ import annotations

import torch
import torch.nn.functional as F

from src.karaone_v11.losses import compute_v11_alignment_losses, compute_v11_codec_losses, compute_v11_pretrain_losses


def compute_v12_pretrain_losses(*args, **kwargs) -> dict[str, torch.Tensor]:
    return compute_v11_pretrain_losses(*args, **kwargs)


def compute_v12_alignment_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    lambda_boundary_ctc: float = 0.05,
    lambda_forward_monotonic: float = 0.02,
    **kwargs,
) -> dict[str, torch.Tensor]:
    losses = compute_v11_alignment_losses(out, batch, **kwargs)
    boundary = ctc_boundary_loss(out, batch)
    monotonic = forward_monotonic_alignment_loss(out)
    losses["ctc_boundary"] = boundary
    losses["forward_monotonic_alignment"] = monotonic
    losses["total"] = losses["total"] + float(lambda_boundary_ctc) * boundary + float(lambda_forward_monotonic) * monotonic
    return losses


def compute_v12_time_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    lambda_lag: float = 1.0,
    lambda_onset: float = 1.0,
    lambda_duration: float = 0.7,
    lambda_active_mask: float = 0.5,
    lambda_active_iou: float = 0.3,
    lambda_shift_envelope: float = 0.2,
) -> dict[str, torch.Tensor]:
    fit_weight = batch.get("time_fit_split")
    if fit_weight is None:
        fit_weight = torch.ones_like(batch["time_onset_sec"], dtype=torch.float32)
    fit_weight = fit_weight.to(out["pred_onset_sec"].device).float()
    lag = weighted_smooth_l1(out["pred_lag_sec"], batch["time_lag_sec"].to(out["pred_lag_sec"].device), fit_weight)
    onset = weighted_smooth_l1(out["pred_onset_sec"], batch["time_onset_sec"].to(out["pred_onset_sec"].device), fit_weight)
    duration = weighted_smooth_l1(out["pred_duration_sec"], batch["time_duration_sec"].to(out["pred_duration_sec"].device), fit_weight)
    target_mask = batch["time_active_mask"].to(out["pred_active_mask_logits"].device, out["pred_active_mask_logits"].dtype)
    if target_mask.shape[1] != out["pred_active_mask_logits"].shape[1]:
        target_mask = F.interpolate(target_mask.unsqueeze(1), size=out["pred_active_mask_logits"].shape[1], mode="nearest").squeeze(1)
    mask_loss = F.binary_cross_entropy_with_logits(out["pred_active_mask_logits"], target_mask, reduction="none")
    mask_loss = (mask_loss.mean(dim=1) * fit_weight).sum() / fit_weight.sum().clamp_min(1.0)
    iou = active_iou_loss(torch.sigmoid(out["pred_active_mask_logits"]), target_mask, fit_weight)
    shift_env = shift_invariant_envelope_loss(torch.sigmoid(out["pred_active_mask_logits"]), target_mask, fit_weight)
    total = (
        float(lambda_lag) * lag
        + float(lambda_onset) * onset
        + float(lambda_duration) * duration
        + float(lambda_active_mask) * mask_loss
        + float(lambda_active_iou) * iou
        + float(lambda_shift_envelope) * shift_env
    )
    return {
        "total": total,
        "lag_huber": lag,
        "onset_huber": onset,
        "duration_huber": duration,
        "active_mask_bce": mask_loss,
        "active_iou_loss": iou,
        "shift_invariant_envelope": shift_env,
    }


def compute_v12_codec_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    lambda_time_guard: float = 0.05,
    **kwargs,
) -> dict[str, torch.Tensor]:
    losses = compute_v11_codec_losses(out, batch, **kwargs)
    time_losses = compute_v12_time_losses(out, batch)
    for key, value in time_losses.items():
        if key != "total":
            losses[f"time_{key}"] = value
    losses["total"] = losses["total"] + float(lambda_time_guard) * time_losses["total"]
    return losses


def weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    target = target.to(pred.device, pred.dtype).view_as(pred)
    weight = weight.to(pred.device, pred.dtype).view_as(pred)
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def active_iou_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    pred = pred.clamp(0.0, 1.0)
    target = target.to(pred.device, pred.dtype).clamp(0.0, 1.0)
    inter = torch.minimum(pred, target).sum(dim=1)
    union = torch.maximum(pred, target).sum(dim=1).clamp_min(1e-6)
    iou = inter / union
    weight = weight.to(pred.device, pred.dtype)
    return ((1.0 - iou) * weight).sum() / weight.sum().clamp_min(1.0)


def shift_invariant_envelope_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, *, max_shift: int = 8) -> torch.Tensor:
    target = target.to(pred.device, pred.dtype)
    losses = []
    for shift in range(-int(max_shift), int(max_shift) + 1):
        if shift >= 0:
            pp = pred[:, shift:]
            tt = target[:, : pp.shape[1]]
        else:
            tt = target[:, -shift:]
            pp = pred[:, : tt.shape[1]]
        if pp.shape[1] > 1:
            losses.append(F.mse_loss(pp, tt, reduction="none").mean(dim=1))
    if not losses:
        return F.mse_loss(pred, target)
    stacked = torch.stack(losses, dim=1)
    best = stacked.min(dim=1).values
    weight = weight.to(pred.device, pred.dtype)
    return (best * weight).sum() / weight.sum().clamp_min(1.0)


def ctc_boundary_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    logits = out["pred_token_boundary_logits"]
    target = batch["time_active_mask"].to(logits.device, logits.dtype)
    if target.shape[1] != logits.shape[1]:
        target = F.interpolate(target.unsqueeze(1), size=logits.shape[1], mode="nearest").squeeze(1)
    boundary = torch.zeros_like(target, dtype=torch.long)
    diff = target[:, 1:] - target[:, :-1]
    boundary[:, 1:] = torch.where(diff > 0, 1, boundary[:, 1:])
    boundary[:, 1:] = torch.where(diff < 0, 0, boundary[:, 1:])
    return F.cross_entropy(logits.transpose(1, 2), boundary.clamp(0, logits.shape[-1] - 1), reduction="mean")


def forward_monotonic_alignment_loss(out: dict[str, torch.Tensor]) -> torch.Tensor:
    active = torch.sigmoid(out["pred_active_mask_logits"])
    # Penalize highly fragmented active masks; monotonic speech activity should
    # normally have one dominant contiguous region in KaraOne tokens.
    return active[:, 1:].sub(active[:, :-1]).abs().mean()


__all__ = [
    "active_iou_loss",
    "compute_v12_alignment_losses",
    "compute_v12_codec_losses",
    "compute_v12_pretrain_losses",
    "compute_v12_time_losses",
    "ctc_boundary_loss",
    "forward_monotonic_alignment_loss",
    "shift_invariant_envelope_loss",
]

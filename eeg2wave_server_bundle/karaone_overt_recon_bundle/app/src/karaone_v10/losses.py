from __future__ import annotations

import torch
import torch.nn.functional as F

from src.karaone_v91.losses import (
    compute_v91_alignment_losses,
    compute_v91_pretrain_losses,
    compute_v91_transport_losses,
)


def compute_v10_pretrain_losses(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor] | None = None, **kwargs) -> dict[str, torch.Tensor]:
    return compute_v91_pretrain_losses(out, batch, **kwargs)


def compute_v10_transport_losses(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
    return compute_v91_transport_losses(out, batch, **kwargs)


def compute_v10_alignment_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    lambda_eeg_specific_margin: float = 0.30,
    lambda_mean_prior_margin: float = 0.20,
    lambda_cross_subject_semantic: float = 0.35,
    lambda_label_prototype_pull: float = 0.10,
    lambda_prompt_balanced: float = 0.20,
    lambda_pairwise_decorrelation: float = 0.05,
    semantic_margin: float = 0.04,
    cross_subject_temperature: float = 0.06,
    **base_kwargs,
) -> dict[str, torch.Tensor]:
    """v10 alignment objective.

    v9.1 could improve global train loss while still losing to zero/mean
    priors and same-label cross-subject prototypes.  These additions make that
    failure mode directly visible to SGD instead of only to evaluation.
    """

    losses = compute_v91_alignment_losses(out, batch, **base_kwargs)
    pred = out["pred_semantic_summary"]
    target = batch["semantic_summary"].to(pred.device, pred.dtype)
    label_idx = _batch_ids(batch, "label_idx", pred.device)
    subject_idx = _batch_ids(batch, "subject_idx", pred.device)
    speech_cluster = _batch_ids(batch, "speech_cluster_id", pred.device)

    zero_margin = zero_prior_margin_loss(pred, target, margin=float(semantic_margin))
    mean_margin = mean_prior_margin_loss(pred, target, margin=float(semantic_margin))
    cross_subject = cross_subject_semantic_infonce(
        pred,
        target,
        label_idx,
        subject_idx,
        speech_cluster,
        temperature=float(cross_subject_temperature),
    )
    prototype_pull = same_label_cross_subject_prototype_pull(pred, target, label_idx, subject_idx)
    prompt_balanced = balanced_prompt_ce(out["prompt_logits"], label_idx)
    decorrelation = pairwise_decorrelation_loss(pred)

    total = (
        losses["total"]
        + float(lambda_eeg_specific_margin) * zero_margin
        + float(lambda_mean_prior_margin) * mean_margin
        + float(lambda_cross_subject_semantic) * cross_subject
        + float(lambda_label_prototype_pull) * prototype_pull
        + float(lambda_prompt_balanced) * prompt_balanced
        + float(lambda_pairwise_decorrelation) * decorrelation
    )
    losses.update(
        {
            "total": total,
            "zero_prior_margin": zero_margin,
            "mean_prior_margin": mean_margin,
            "cross_subject_semantic_nce": cross_subject,
            "same_label_prototype_pull": prototype_pull,
            "prompt_balanced_ce": prompt_balanced,
            "pairwise_decorrelation": decorrelation,
        }
    )
    return losses


def zero_prior_margin_loss(pred: torch.Tensor, target: torch.Tensor, *, margin: float = 0.04) -> torch.Tensor:
    pos = F.cosine_similarity(pred, target, dim=-1)
    zero = torch.zeros_like(pred)
    baseline = F.cosine_similarity(zero, target, dim=-1)
    return F.relu(float(margin) + baseline - pos).mean()


def mean_prior_margin_loss(pred: torch.Tensor, target: torch.Tensor, *, margin: float = 0.04) -> torch.Tensor:
    if pred.shape[0] < 2:
        return pred.new_tensor(0.0)
    pos = F.cosine_similarity(pred, target, dim=-1)
    mean_query = target.mean(dim=0, keepdim=True).expand_as(target)
    baseline = F.cosine_similarity(mean_query, target, dim=-1)
    return F.relu(float(margin) + baseline - pos).mean()


def cross_subject_semantic_infonce(
    pred: torch.Tensor,
    target: torch.Tensor,
    label_idx: torch.Tensor,
    subject_idx: torch.Tensor,
    speech_cluster_id: torch.Tensor,
    *,
    temperature: float = 0.06,
) -> torch.Tensor:
    if pred.shape[0] < 2:
        return pred.new_tensor(0.0)
    e = F.normalize(pred, dim=-1)
    s = F.normalize(target, dim=-1)
    logits = e @ s.T / max(float(temperature), 1e-4)
    eye = torch.eye(pred.shape[0], device=pred.device, dtype=pred.dtype)
    diff_subject = subject_idx[:, None].ne(subject_idx[None, :])
    same_label = label_idx[:, None].eq(label_idx[None, :])
    same_speech_cluster = speech_cluster_id[:, None].eq(speech_cluster_id[None, :])
    cross_positive = (same_label | same_speech_cluster) & diff_subject
    positive = eye + 0.75 * cross_positive.to(pred.dtype)
    positive = positive / positive.sum(dim=1, keepdim=True).clamp_min(1e-6)
    e2s = -(positive * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    s2e = -(positive.T * torch.log_softmax(logits.T, dim=-1)).sum(dim=-1).mean()
    return 0.5 * (e2s + s2e)


def same_label_cross_subject_prototype_pull(
    pred: torch.Tensor,
    target: torch.Tensor,
    label_idx: torch.Tensor,
    subject_idx: torch.Tensor,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for idx in range(pred.shape[0]):
        mask = label_idx.eq(label_idx[idx]) & subject_idx.ne(subject_idx[idx])
        if not bool(mask.any()):
            continue
        proto = F.normalize(target[mask].mean(dim=0, keepdim=True), dim=-1)
        score = (F.normalize(pred[idx : idx + 1], dim=-1) * proto).sum(dim=-1)
        losses.append(1.0 - score.squeeze(0))
    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def balanced_prompt_ce(logits: torch.Tensor, label_idx: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[0] == 0:
        return logits.new_tensor(0.0)
    num_classes = int(logits.shape[-1])
    counts = torch.bincount(label_idx.clamp(0, num_classes - 1), minlength=num_classes).to(logits.dtype)
    weights = torch.where(counts > 0, counts.sum() / counts.clamp_min(1.0), torch.zeros_like(counts))
    if float(weights.sum().detach().cpu()) <= 0.0:
        return F.cross_entropy(logits, label_idx)
    weights = weights / weights[weights > 0].mean().clamp_min(1e-6)
    return F.cross_entropy(logits, label_idx, weight=weights)


def pairwise_decorrelation_loss(pred: torch.Tensor, *, corr_ceiling: float = 0.70) -> torch.Tensor:
    if pred.shape[0] < 3:
        return pred.new_tensor(0.0)
    centered = pred - pred.mean(dim=1, keepdim=True)
    normed = F.normalize(centered, dim=-1)
    corr = normed @ normed.T
    upper = corr[torch.triu_indices(corr.shape[0], corr.shape[1], offset=1, device=pred.device).unbind()]
    if upper.numel() == 0:
        return pred.new_tensor(0.0)
    return F.relu(upper - float(corr_ceiling)).pow(2).mean()


def _batch_ids(batch: dict[str, torch.Tensor], key: str, device: torch.device) -> torch.Tensor:
    if key not in batch:
        return torch.zeros(1, device=device, dtype=torch.long)
    return batch[key].to(device).long()


from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch
import torch.nn.functional as F


def _normalise(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=1e-8)


def symmetric_view_contrastive(raw_embed: torch.Tensor, topo_embed: torch.Tensor, temperature: float = 0.1) -> dict[str, torch.Tensor]:
    """Same trial is the sole positive for raw-versus-topographic EEG SSL."""
    if raw_embed.shape[0] < 2:
        zero = raw_embed.new_zeros(())
        return {"total": zero, "raw_to_topo": zero, "topo_to_raw": zero}
    logits = _normalise(raw_embed) @ _normalise(topo_embed).T / float(temperature)
    targets = torch.arange(logits.shape[0], device=logits.device)
    a = F.cross_entropy(logits, targets)
    b = F.cross_entropy(logits.T, targets)
    return {"total": 0.5 * (a + b), "raw_to_topo": a.detach(), "topo_to_raw": b.detach()}


def _multi_positive_targets(labels: torch.Tensor, subjects: Iterable[str], alpha: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    labels = labels.detach().view(-1)
    subject_list = [str(item) for item in subjects]
    n = labels.numel()
    target = torch.zeros((n, n), dtype=torch.float32, device=device)
    allowed = torch.ones((n, n), dtype=torch.bool, device=device)
    for row in range(n):
        target[row, row] = 1.0
        for col in range(n):
            if row == col:
                continue
            same_label = bool(labels[row].item() == labels[col].item())
            same_subject = subject_list[row] == subject_list[col]
            if same_label and not same_subject:
                target[row, col] = float(alpha)
            elif same_label and same_subject:
                # Repeated labels from one speaker are neither positives nor negatives.
                allowed[row, col] = False
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return target, allowed


def _soft_ce(logits: torch.Tensor, targets: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    logits = logits.masked_fill(~allowed, -torch.inf)
    log_probs = F.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1)


def multi_positive_clip_loss(
    eeg_embed: torch.Tensor,
    audio_embed: torch.Tensor,
    labels: torch.Tensor,
    subjects: Iterable[str],
    *,
    temperature: float = 0.07,
    cross_subject_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    """CLIP with exact trial positives plus low-weight cross-subject label positives."""
    if eeg_embed.shape[0] < 2:
        zero = eeg_embed.new_zeros(())
        return {"total": zero, "per_example": eeg_embed.new_zeros((eeg_embed.shape[0],)), "eeg_to_audio": zero, "audio_to_eeg": zero}
    logits = _normalise(eeg_embed) @ _normalise(audio_embed).T / float(temperature)
    targets, allowed = _multi_positive_targets(labels, subjects, cross_subject_weight, logits.device)
    eeg_to_audio = _soft_ce(logits, targets, allowed)
    audio_to_eeg = _soft_ce(logits.T, targets.T, allowed.T)
    per_example = 0.5 * (eeg_to_audio + audio_to_eeg)
    return {
        "total": per_example.mean(),
        "per_example": per_example,
        "eeg_to_audio": eeg_to_audio.mean().detach(),
        "audio_to_eeg": audio_to_eeg.mean().detach(),
    }


def group_dro(per_example: torch.Tensor, subjects: Iterable[str], eta: float = 0.1) -> torch.Tensor:
    """Subject-group robust aggregate; no subject representation is used by the model."""
    grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
    for loss, subject in zip(per_example, subjects):
        grouped[str(subject)].append(loss)
    if not grouped:
        return per_example.mean()
    group_losses = torch.stack([torch.stack(values).mean() for values in grouped.values()])
    weights = torch.softmax(float(eta) * group_losses.detach(), dim=0)
    return (weights * group_losses).sum()


def variance_covariance_regularizer(embed: torch.Tensor, variance_floor: float = 1.0) -> dict[str, torch.Tensor]:
    if embed.shape[0] < 2:
        zero = embed.new_zeros(())
        return {"total": zero, "variance": zero, "covariance": zero}
    centered = embed - embed.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    variance = F.relu(float(variance_floor) - std).mean()
    covariance = (centered.T @ centered) / max(1, embed.shape[0] - 1)
    covariance = covariance - torch.diag(torch.diag(covariance))
    covariance_loss = covariance.pow(2).sum() / max(1, embed.shape[1])
    return {"total": variance + covariance_loss, "variance": variance.detach(), "covariance": covariance_loss.detach()}


def masked_token_cross_entropy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if logits.ndim != 3 or target.ndim != 2:
        raise ValueError("Expected token logits [B,T,V] and targets [B,T]")
    loss = F.cross_entropy(logits.transpose(1, 2), target.long(), reduction="none")
    if mask is None:
        return loss.mean()
    return (loss * mask.to(loss.dtype)).sum() / mask.sum().clamp_min(1.0)


def flow_matching_loss(velocity: torch.Tensor, target_velocity: torch.Tensor, active_mask: torch.Tensor | None = None) -> torch.Tensor:
    loss = (velocity - target_velocity).pow(2).mean(dim=-1)
    if active_mask is None:
        return loss.mean()
    if active_mask.shape[1] != loss.shape[1]:
        active_mask = F.interpolate(active_mask.unsqueeze(1), size=loss.shape[1], mode="nearest").squeeze(1)
    return (loss * active_mask.to(loss.dtype)).sum() / active_mask.sum().clamp_min(1.0)

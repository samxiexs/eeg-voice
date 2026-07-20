from __future__ import annotations

import torch
import torch.nn.functional as F


def code_cross_entropy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, codebook_weights: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    loss = F.cross_entropy(logits.permute(0, 3, 1, 2), target.long(), reduction="none")
    active = torch.ones_like(loss, dtype=loss.dtype) if mask is None else mask.to(loss.dtype)
    weights = torch.ones(logits.shape[1], device=logits.device, dtype=logits.dtype) if codebook_weights is None else codebook_weights.to(logits)
    weighted = active * weights.view(1, -1, 1)
    total = (loss * weighted).sum() / weighted.sum().clamp_min(1.0)
    prediction = logits.argmax(dim=-1)
    metrics = {"total": total}
    for codebook in range(logits.shape[1]):
        selected = active[:, codebook] > 0
        metrics[f"q{codebook}_accuracy"] = (prediction[:, codebook][selected] == target[:, codebook][selected]).float().mean().detach() if selected.any() else total.detach() * 0.0
    return metrics


def condition_alignment_loss(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    smooth_l1 = F.smooth_l1_loss(prediction, target)
    cosine = 1.0 - F.cosine_similarity(prediction, target, dim=-1).mean()
    delta = F.smooth_l1_loss(prediction[:, 1:] - prediction[:, :-1], target[:, 1:] - target[:, :-1])
    return {"total": smooth_l1 + 0.5 * cosine + 0.25 * delta, "smooth_l1": smooth_l1, "cosine": cosine, "delta": delta}


def multi_positive_contrastive_loss(
    eeg: torch.Tensor,
    audio: torch.Tensor,
    labels: torch.Tensor,
    subjects: torch.Tensor,
    audio_ids: torch.Tensor,
    *,
    temperature: float = 0.08,
    cross_subject_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Symmetric InfoNCE with strong canonical and soft cross-subject positives.

    Exact audio-key matches are strong positives.  Same-label examples from a
    different subject receive ``cross_subject_weight``; remaining same-label
    pairs are neutral rather than false negatives.
    """

    if eeg.shape != audio.shape or eeg.ndim != 2:
        raise ValueError("eeg and audio embeddings must share shape [B,D]")
    if any(value.shape != (len(eeg),) for value in (labels, subjects, audio_ids)):
        raise ValueError("labels, subjects, and audio_ids must be [B]")
    if not 0.0 <= float(cross_subject_weight) <= 1.0:
        raise ValueError("cross_subject_weight must be in [0,1]")
    zero = eeg.sum() * 0.0
    if len(eeg) < 2:
        return {
            "total": zero,
            "eeg_to_audio": zero.detach(),
            "audio_to_eeg": zero.detach(),
            "extra_positive_fraction": zero.detach(),
            "mean_positive_count": zero.detach() + 1.0,
        }
    eeg = F.normalize(eeg, dim=-1)
    audio = F.normalize(audio, dim=-1)
    logits = eeg @ audio.T / float(temperature)
    same_label = labels[:, None] == labels[None, :]
    same_audio = audio_ids[:, None] == audio_ids[None, :]
    different_subject = subjects[:, None] != subjects[None, :]
    weights = same_audio.to(logits.dtype)
    soft_positive = same_label & different_subject & ~same_audio
    weights = weights + soft_positive.to(logits.dtype) * float(cross_subject_weight)
    positive = weights > 0
    allowed = (~same_label) | positive
    floor = torch.finfo(logits.dtype).min

    def direction(value: torch.Tensor, positive_weights: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        target = positive_weights / positive_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        log_probabilities = F.log_softmax(value.masked_fill(~valid, floor), dim=1)
        return -(target * log_probabilities).sum(dim=1).mean()

    eeg_to_audio = direction(logits, weights, allowed)
    audio_to_eeg = direction(logits.T, weights.T, allowed.T)
    diagonal = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    extra = (positive & ~diagonal).sum().to(logits.dtype)
    all_positive = positive.sum().clamp_min(1).to(logits.dtype)
    return {
        "total": 0.5 * (eeg_to_audio + audio_to_eeg),
        "eeg_to_audio": eeg_to_audio.detach(),
        "audio_to_eeg": audio_to_eeg.detach(),
        "extra_positive_fraction": (extra / all_positive).detach(),
        "mean_positive_count": positive.sum(dim=1).float().mean().detach(),
    }


def paired_contrastive_loss(eeg: torch.Tensor, audio: torch.Tensor, labels: torch.Tensor, temperature: float = 0.08) -> torch.Tensor:
    """Backward-compatible diagonal-pair wrapper."""

    ids = torch.arange(len(labels), device=labels.device)
    return multi_positive_contrastive_loss(
        eeg,
        audio,
        labels,
        ids,
        ids,
        temperature=temperature,
        cross_subject_weight=0.0,
    )["total"]


def soft_label_distillation(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    teacher = torch.softmax(teacher_logits.detach() / temperature, dim=-1)
    student = torch.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(student, teacher, reduction="batchmean") * temperature**2


def variance_regularizer(value: torch.Tensor, floor: float = 0.5) -> torch.Tensor:
    flat = value.reshape(-1, value.shape[-1])
    return F.relu(float(floor) - torch.sqrt(flat.var(dim=0, unbiased=False) + 1e-4)).mean()

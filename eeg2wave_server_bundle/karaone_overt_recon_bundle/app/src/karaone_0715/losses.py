from __future__ import annotations

import torch
import torch.nn.functional as F


def code_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    codebook_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if logits.ndim != 4 or target.ndim != 3 or logits.shape[:3] != target.shape:
        raise ValueError("Expected logits [B,Q,T,V] and targets [B,Q,T]")
    loss = F.cross_entropy(logits.permute(0, 3, 1, 2), target.long(), reduction="none")
    active = torch.ones_like(loss, dtype=loss.dtype) if mask is None else mask.to(loss.dtype)
    if codebook_weights is None:
        weights = torch.ones(logits.shape[1], device=logits.device, dtype=logits.dtype)
    else:
        weights = codebook_weights.to(device=logits.device, dtype=logits.dtype)
    weighted = active * weights.view(1, -1, 1)
    total = (loss * weighted).sum() / weighted.sum().clamp_min(1.0)
    prediction = logits.argmax(dim=-1)
    metrics: dict[str, torch.Tensor] = {"total": total}
    for codebook in range(logits.shape[1]):
        selected = active[:, codebook] > 0
        accuracy = (prediction[:, codebook][selected] == target[:, codebook][selected]).float().mean() if selected.any() else total.detach() * 0.0
        metrics[f"q{codebook}_accuracy"] = accuracy.detach()
    return metrics


def paired_contrastive_loss(eeg: torch.Tensor, audio: torch.Tensor, labels: torch.Tensor, temperature: float = 0.08) -> torch.Tensor:
    """Trial-paired InfoNCE with same-label non-pairs removed as false negatives."""

    if eeg.shape != audio.shape or eeg.ndim != 2:
        raise ValueError("paired embeddings must have matching [B,D] shapes")
    eeg = F.normalize(eeg, dim=-1)
    audio = F.normalize(audio, dim=-1)
    logits = eeg @ audio.T / float(temperature)
    same_label = labels[:, None] == labels[None, :]
    diagonal = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    allowed = (~same_label) | diagonal
    floor = torch.finfo(logits.dtype).min
    eeg_to_audio = F.cross_entropy(logits.masked_fill(~allowed, floor), torch.arange(len(labels), device=labels.device))
    audio_to_eeg = F.cross_entropy(logits.T.masked_fill(~allowed.T, floor), torch.arange(len(labels), device=labels.device))
    return 0.5 * (eeg_to_audio + audio_to_eeg)


def condition_alignment_loss(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    if prediction.shape != target.shape:
        raise ValueError("condition tensors must share [B,T,D]")
    smooth_l1 = F.smooth_l1_loss(prediction, target)
    cosine = 1.0 - F.cosine_similarity(prediction, target, dim=-1).mean()
    pred_delta = prediction[:, 1:] - prediction[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    delta = F.smooth_l1_loss(pred_delta, target_delta)
    return {"total": smooth_l1 + 0.5 * cosine + 0.25 * delta, "smooth_l1": smooth_l1, "cosine": cosine, "delta": delta}


def variance_regularizer(value: torch.Tensor, floor: float = 0.5) -> torch.Tensor:
    flat = value.reshape(-1, value.shape[-1])
    std = torch.sqrt(flat.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(float(floor) - std).mean()


def soft_label_distillation(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    temperature = float(temperature)
    teacher = torch.softmax(teacher_logits.detach() / temperature, dim=-1)
    student = torch.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(student, teacher, reduction="batchmean") * temperature**2

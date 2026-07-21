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


def _masked_correlation_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape != mask.shape or prediction.ndim != 2:
        raise ValueError("prediction, target, and mask must share shape [B,T]")
    weight = mask.to(prediction.dtype)
    count = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    prediction_mean = (prediction * weight).sum(dim=1, keepdim=True) / count
    target_mean = (target * weight).sum(dim=1, keepdim=True) / count
    prediction_centered = (prediction - prediction_mean) * weight
    target_centered = (target - target_mean) * weight
    numerator = (prediction_centered * target_centered).sum(dim=1)
    denominator = torch.sqrt(
        prediction_centered.square().sum(dim=1).clamp_min(1e-8)
        * target_centered.square().sum(dim=1).clamp_min(1e-8)
    )
    return (numerator / denominator).clamp(-1.0, 1.0)


def envelope_correlation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Directly optimize temporal envelope shape, independently of scale."""

    correlation = _masked_correlation_per_sample(prediction, target, mask)
    return {
        "total": 1.0 - correlation.mean(),
        "correlation": correlation.mean().detach(),
    }


def multi_scale_envelope_correlation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    kernel_sizes: tuple[int, ...] = (1, 5, 9),
) -> dict[str, torch.Tensor]:
    """Optimize envelope morphology at frame, medium, and coarse scales."""

    if not kernel_sizes or any(size < 1 or size % 2 == 0 for size in kernel_sizes):
        raise ValueError("kernel_sizes must be a non-empty tuple of positive odd integers")
    if prediction.shape != target.shape or prediction.shape != mask.shape or prediction.ndim != 2:
        raise ValueError("prediction, target, and mask must share shape [B,T]")
    correlations: list[torch.Tensor] = []
    metrics: dict[str, torch.Tensor] = {}
    weight = mask.to(prediction.dtype)
    for size in kernel_sizes:
        if size == 1:
            prediction_scale = prediction
            target_scale = target
            mask_scale = weight
        else:
            prediction_scale = F.avg_pool1d(
                (prediction * weight).unsqueeze(1), size, stride=1, padding=size // 2
            ).squeeze(1)
            target_scale = F.avg_pool1d(
                (target * weight).unsqueeze(1), size, stride=1, padding=size // 2
            ).squeeze(1)
            mask_density = F.avg_pool1d(
                weight.unsqueeze(1), size, stride=1, padding=size // 2
            ).squeeze(1)
            # Exclude windows contaminated by right-padding rather than
            # rewarding predictions for padded silence.
            mask_scale = (mask_density >= 1.0 - 1e-6).to(prediction.dtype)
        correlation = _masked_correlation_per_sample(
            prediction_scale,
            target_scale,
            mask_scale,
        )
        correlations.append(correlation)
        metrics[f"correlation_k{size}"] = correlation.mean().detach()
    stacked = torch.stack(correlations, dim=0)
    metrics["correlation"] = stacked.mean().detach()
    metrics["total"] = 1.0 - stacked.mean()
    return metrics


def soft_activity_dice_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float = 0.10,
    temperature: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Differentiable activity-region overlap on peak-normalized envelopes."""

    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if prediction.shape != target.shape or prediction.shape != mask.shape or prediction.ndim != 2:
        raise ValueError("prediction, target, and mask must share shape [B,T]")
    weight = mask.to(prediction.dtype)
    prediction_masked = prediction * weight
    target_masked = target * weight
    prediction_peak = prediction_masked.max(dim=1, keepdim=True).values.clamp_min(1e-6)
    target_peak = target_masked.max(dim=1, keepdim=True).values.clamp_min(1e-6)
    prediction_normalized = prediction_masked / prediction_peak
    target_normalized = target_masked / target_peak
    prediction_activity = torch.sigmoid(
        (prediction_normalized - float(threshold)) / float(temperature)
    ) * weight
    target_activity = (target_normalized >= float(threshold)).to(prediction.dtype) * weight
    intersection = (prediction_activity * target_activity).sum(dim=1)
    denominator = prediction_activity.sum(dim=1) + target_activity.sum(dim=1)
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    return {"total": 1.0 - dice.mean(), "dice": dice.mean().detach()}


def same_label_morphology_ranking_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    audio_ids: torch.Tensor,
    *,
    margin: float = 0.03,
) -> dict[str, torch.Tensor]:
    """Rank the correct EEG envelope above a wrong same-label EEG envelope.

    The wrong prediction is chosen deterministically within the current batch,
    must have the same label and a different audio key, and therefore tests
    trial-specific morphology rather than easy label discrimination.
    """

    if any(value.shape != (len(prediction),) for value in (labels, audio_ids)):
        raise ValueError("labels and audio_ids must be [B]")
    zero = prediction.sum() * 0.0
    source_indices = torch.full((len(prediction),), -1, dtype=torch.long, device=prediction.device)
    label_values = labels.detach().cpu().tolist()
    audio_values = audio_ids.detach().cpu().tolist()
    for target_index in range(len(prediction)):
        for offset in range(1, len(prediction)):
            source_index = (target_index + offset) % len(prediction)
            if (
                label_values[source_index] == label_values[target_index]
                and audio_values[source_index] != audio_values[target_index]
            ):
                source_indices[target_index] = source_index
                break
    selected = source_indices >= 0
    if not selected.any():
        return {
            "total": zero,
            "active_fraction": zero.detach(),
            "correct_correlation": zero.detach(),
            "shuffled_correlation": zero.detach(),
        }
    correct = _masked_correlation_per_sample(prediction, target, mask)
    wrong_prediction = prediction[source_indices[selected]]
    shuffled = _masked_correlation_per_sample(
        wrong_prediction,
        target[selected],
        mask[selected],
    )
    correct_selected = correct[selected]
    loss = F.relu(float(margin) - correct_selected + shuffled).mean()
    return {
        "total": loss,
        "active_fraction": selected.float().mean().detach(),
        "correct_correlation": correct_selected.mean().detach(),
        "shuffled_correlation": shuffled.mean().detach(),
    }

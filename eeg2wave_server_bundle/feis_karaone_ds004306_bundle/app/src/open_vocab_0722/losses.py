from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossEligibility:
    exact_acoustic: torch.Tensor
    weak_semantic: torch.Tensor
    eeg_self_supervised: torch.Tensor


def loss_eligibility(
    datasets: list[str] | tuple[str, ...],
    pairing_confidence: list[str] | tuple[str, ...],
    *,
    device: torch.device | None = None,
) -> LossEligibility:
    if len(datasets) != len(pairing_confidence):
        raise ValueError("dataset and pairing vectors must have the same length")
    exact = torch.tensor(
        [dataset == "karaone" and pairing == "karaone_same_trial_overt" for dataset, pairing in zip(datasets, pairing_confidence)],
        dtype=torch.bool,
        device=device,
    )
    weak = torch.tensor(
        [
            dataset == "karaone"
            or (dataset == "feis" and pairing == "feis_subject_label")
            for dataset, pairing in zip(datasets, pairing_confidence)
        ],
        dtype=torch.bool,
        device=device,
    )
    self_supervised = torch.ones(len(datasets), dtype=torch.bool, device=device)
    return LossEligibility(exact, weak, self_supervised)


def code_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    codebook_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if logits.ndim != 4 or target.shape != logits.shape[:3] or mask.shape != target.shape:
        raise ValueError("logits/target/mask must be [B,Q,T,V], [B,Q,T], [B,Q,T]")
    per_token = F.cross_entropy(logits.permute(0, 3, 1, 2), target.long(), reduction="none")
    active = mask.to(per_token.dtype)
    weights = (
        torch.ones(logits.shape[1], device=logits.device, dtype=logits.dtype)
        if codebook_weights is None
        else codebook_weights.to(logits)
    )
    combined = active * weights.view(1, -1, 1)
    total = (per_token * combined).sum() / combined.sum().clamp_min(1.0)
    prediction = logits.argmax(dim=-1)
    output = {"total": total}
    for index in range(logits.shape[1]):
        selected = mask[:, index].bool()
        output[f"q{index}_accuracy"] = (
            (prediction[:, index][selected] == target[:, index][selected]).float().mean().detach()
            if selected.any()
            else total.detach() * 0.0
        )
    return output


def _weighted_direction(
    logits: torch.Tensor,
    positive_weights: torch.Tensor,
    allowed: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != positive_weights.shape or logits.shape != allowed.shape:
        raise ValueError("contrastive matrices must have the same shape")
    rows = positive_weights.sum(dim=1) > 0
    if not rows.any():
        return logits.sum() * 0.0
    floor = torch.finfo(logits.dtype).min
    target = positive_weights[rows] / positive_weights[rows].sum(dim=1, keepdim=True).clamp_min(1e-8)
    log_probability = F.log_softmax(logits[rows].masked_fill(~allowed[rows], floor), dim=1)
    return -(target * log_probability).sum(dim=1).mean()


def symmetric_contrastive_loss(
    eeg: torch.Tensor,
    audio: torch.Tensor,
    positive_weights: torch.Tensor,
    *,
    allowed: torch.Tensor | None = None,
    temperature: float = 0.08,
) -> dict[str, torch.Tensor]:
    if eeg.shape != audio.shape or eeg.ndim != 2:
        raise ValueError("EEG and audio embeddings must share [B,D]")
    if positive_weights.shape != (len(eeg), len(eeg)):
        raise ValueError("positive_weights must be [B,B]")
    eeg = F.normalize(eeg, dim=-1)
    audio = F.normalize(audio, dim=-1)
    logits = eeg @ audio.T / float(temperature)
    if allowed is None:
        allowed = torch.ones_like(positive_weights, dtype=torch.bool)
    first = _weighted_direction(logits, positive_weights.to(logits), allowed.bool())
    second = _weighted_direction(logits.T, positive_weights.T.to(logits), allowed.T.bool())
    return {"total": 0.5 * (first + second), "eeg_to_audio": first.detach(), "audio_to_eeg": second.detach()}


def exact_pair_contrastive_loss(
    eeg: torch.Tensor,
    audio: torch.Tensor,
    exact_mask: torch.Tensor,
    *,
    temperature: float = 0.08,
) -> dict[str, torch.Tensor]:
    if exact_mask.shape != (len(eeg),):
        raise ValueError("exact_mask must be [B]")
    positive = torch.diag(exact_mask.to(eeg.dtype))
    # All non-paired samples, including same-label different trials, remain
    # hard negatives in the acoustic space.
    allowed = exact_mask[:, None] & exact_mask[None, :]
    return symmetric_contrastive_loss(eeg, audio, positive, allowed=allowed, temperature=temperature)


def semantic_positive_weights(
    labels: torch.Tensor,
    exact_mask: torch.Tensor,
    semantic_mask: torch.Tensor,
    *,
    weak_weight: float = 0.15,
) -> torch.Tensor:
    if any(value.shape != (len(labels),) for value in (exact_mask, semantic_mask)):
        raise ValueError("label and eligibility vectors must be [B]")
    same_label = labels[:, None] == labels[None, :]
    eligible = semantic_mask[:, None] & semantic_mask[None, :]
    weights = (same_label & eligible).to(torch.float32) * float(weak_weight)
    diagonal = torch.arange(len(labels), device=labels.device)
    weights[diagonal, diagonal] = torch.where(
        exact_mask,
        torch.ones_like(exact_mask, dtype=weights.dtype),
        semantic_mask.to(weights.dtype) * float(weak_weight),
    )
    return weights


def monotonic_local_alignment_loss(
    eeg_tokens: torch.Tensor,
    audio_tokens: torch.Tensor,
    sample_mask: torch.Tensor,
    *,
    positional_sigma: float = 0.20,
    temperature: float = 0.08,
) -> dict[str, torch.Tensor]:
    if eeg_tokens.shape != audio_tokens.shape or eeg_tokens.ndim != 3:
        raise ValueError("local EEG/audio tokens must share [B,T,D]")
    if sample_mask.shape != (len(eeg_tokens),):
        raise ValueError("sample_mask must be [B]")
    if not sample_mask.any():
        zero = eeg_tokens.sum() * 0.0
        return {"total": zero, "cosine": zero.detach()}
    eeg = F.normalize(eeg_tokens[sample_mask], dim=-1)
    audio = F.normalize(audio_tokens[sample_mask], dim=-1)
    steps = eeg.shape[1]
    positions = torch.linspace(0.0, 1.0, steps, device=eeg.device, dtype=eeg.dtype)
    distance = (positions[:, None] - positions[None, :]).square()
    bias = -distance / max(float(positional_sigma) ** 2, 1e-6)
    similarity = torch.einsum("btd,bsd->bts", eeg, audio) / float(temperature)
    forward_weight = torch.softmax(similarity + bias, dim=-1)
    backward_weight = torch.softmax(similarity.transpose(1, 2) + bias, dim=-1)
    aligned_audio = torch.einsum("bts,bsd->btd", forward_weight, audio)
    aligned_eeg = torch.einsum("bst,btd->bsd", backward_weight, eeg)
    cosine = 0.5 * (
        F.cosine_similarity(eeg, aligned_audio, dim=-1).mean()
        + F.cosine_similarity(audio, aligned_eeg, dim=-1).mean()
    )
    return {"total": 1.0 - cosine, "cosine": cosine.detach()}


def lag_tolerant_envelope_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    max_lag_steps: int = 19,
) -> dict[str, torch.Tensor]:
    if prediction.shape != target.shape or target.shape != valid_mask.shape or prediction.ndim != 2:
        raise ValueError("envelope tensors must share [B,T]")

    def correlation(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(first.dtype)
        count = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        first = (first - (first * weight).sum(dim=1, keepdim=True) / count) * weight
        second = (second - (second * weight).sum(dim=1, keepdim=True) / count) * weight
        return (first * second).sum(dim=1) / torch.sqrt(
            first.square().sum(dim=1).clamp_min(1e-8) * second.square().sum(dim=1).clamp_min(1e-8)
        )

    candidates = []
    for lag in range(-int(max_lag_steps), int(max_lag_steps) + 1):
        shifted = torch.roll(prediction, shifts=lag, dims=1)
        mask = valid_mask.clone()
        if lag > 0:
            mask[:, :lag] = False
        elif lag < 0:
            mask[:, lag:] = False
        candidates.append(correlation(shifted, target, mask))
    best = torch.stack(candidates, dim=1).max(dim=1).values
    return {"total": 1.0 - best.mean(), "correlation": best.mean().detach()}


def modulation_spectrum_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    weight = valid_mask.to(prediction.dtype)
    prediction_spectrum = torch.log1p(torch.fft.rfft(prediction * weight, dim=-1).abs())
    target_spectrum = torch.log1p(torch.fft.rfft(target * weight, dim=-1).abs())
    return F.smooth_l1_loss(prediction_spectrum, target_spectrum)


def structure_loss(
    prediction_envelope: torch.Tensor,
    target_envelope: torch.Tensor,
    envelope_mask: torch.Tensor,
    prediction_onset: torch.Tensor,
    target_onset: torch.Tensor,
    prediction_duration: torch.Tensor,
    target_duration: torch.Tensor,
    sample_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if not sample_mask.any():
        zero = prediction_envelope.sum() * 0.0
        return {"total": zero, "envelope": zero.detach(), "modulation": zero.detach(), "timing": zero.detach()}
    selected_envelope = prediction_envelope[sample_mask]
    selected_target = target_envelope[sample_mask]
    selected_valid = envelope_mask[sample_mask]
    envelope = lag_tolerant_envelope_loss(selected_envelope, selected_target, selected_valid)
    modulation = modulation_spectrum_loss(selected_envelope, selected_target, selected_valid)
    timing = F.smooth_l1_loss(prediction_onset[sample_mask], target_onset[sample_mask]) + F.smooth_l1_loss(
        prediction_duration[sample_mask], target_duration[sample_mask]
    )
    return {
        "total": envelope["total"] + 0.5 * modulation + 0.5 * timing,
        "envelope": envelope["correlation"],
        "modulation": modulation.detach(),
        "timing": timing.detach(),
    }


def masked_patch_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    patch_mask: torch.Tensor,
) -> torch.Tensor:
    if reconstruction.shape != target.shape or patch_mask.shape != target.shape[:3]:
        raise ValueError("patch reconstruction tensors are inconsistent")
    if not patch_mask.any():
        return reconstruction.sum() * 0.0
    return F.smooth_l1_loss(reconstruction[patch_mask], target[patch_mask])


def condition_consistency_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.shape != second.shape:
        raise ValueError("condition views must have the same shape")
    return (1.0 - F.cosine_similarity(first, second, dim=-1)).mean()


def text_semantic_loss(
    eeg_semantic: torch.Tensor,
    text_semantic: torch.Tensor,
    sample_mask: torch.Tensor,
) -> torch.Tensor:
    if not sample_mask.any():
        return eeg_semantic.sum() * 0.0
    return (1.0 - F.cosine_similarity(eeg_semantic[sample_mask], text_semantic[sample_mask], dim=-1)).mean()


def moe_regularization(router: dict[str, torch.Tensor], *, z_weight: float = 0.1) -> torch.Tensor:
    return router["balance_loss"] + float(z_weight) * router["z_loss"]


def router_collapse_flags(
    mass: torch.Tensor,
    *,
    dying_threshold: float = 0.05,
    collapse_threshold: float = 0.60,
) -> dict[str, bool]:
    value = mass.detach().float().cpu()
    return {
        "expert_dying": bool((value < float(dying_threshold)).any()),
        "routing_collapse": bool((value > float(collapse_threshold)).any()),
    }


__all__ = [
    "LossEligibility",
    "code_cross_entropy",
    "condition_consistency_loss",
    "exact_pair_contrastive_loss",
    "lag_tolerant_envelope_loss",
    "loss_eligibility",
    "masked_patch_reconstruction_loss",
    "moe_regularization",
    "monotonic_local_alignment_loss",
    "router_collapse_flags",
    "semantic_positive_weights",
    "structure_loss",
    "symmetric_contrastive_loss",
    "text_semantic_loss",
]

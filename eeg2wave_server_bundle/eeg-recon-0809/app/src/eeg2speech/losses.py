"""Supervision-strength-aware content losses and counterfactual controls."""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F

from .model import JointState


def weighted_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # Normalize by the number of supervised examples, not by sum(weight).
    # Otherwise a homogeneous batch of 0.35-weight weak pairs is accidentally
    # promoted back to unit-strength supervision.
    return (values * weight).sum() / (weight > 0).sum().clamp_min(1)


def masked_mfcc_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                     sample_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    element = F.smooth_l1_loss(prediction, target, reduction="none").mean(1)
    per_sample = (element * mask.to(element.dtype)).sum(1) / mask.sum(1).clamp_min(1)
    mfcc = weighted_mean(per_sample, sample_weight)
    delta_prediction = prediction[..., 1:] - prediction[..., :-1]
    delta_target = target[..., 1:] - target[..., :-1]
    delta_mask = mask[:, 1:] & mask[:, :-1]
    delta_element = F.smooth_l1_loss(delta_prediction, delta_target, reduction="none").mean(1)
    delta_per_sample = (delta_element * delta_mask.to(element.dtype)).sum(1) / delta_mask.sum(1).clamp_min(1)
    return mfcc, weighted_mean(delta_per_sample, sample_weight)


def soft_dtw_token_loss(left: torch.Tensor, right: torch.Tensor, left_mask: torch.Tensor,
                        right_mask: torch.Tensor, window_fraction: float = 0.20,
                        temperature: float = 0.15) -> torch.Tensor:
    """Numerically stable, banded local temporal alignment.

    This preserves the intended diagonal/SoftDTW-style local matching while
    avoiding Sinkhorn's alternating divisions over a near-zero kernel. Those
    divisions were unstable on MPS and could poison the optimizer after a few
    updates. Each valid EEG token attends only to valid audio tokens in its
    temporal band through a stable softmax.
    """
    left, right = F.normalize(left, dim=-1, eps=1e-6), F.normalize(right, dim=-1, eps=1e-6)
    if right.shape[1] != left.shape[1]:
        right = F.interpolate(right.transpose(1, 2), size=left.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        right_mask = F.interpolate(right_mask.float().unsqueeze(1), size=left.shape[1], mode="nearest").squeeze(1).bool()
    _, steps, _ = left.shape
    radius = max(1, int(round(steps * window_fraction)))
    grid = torch.arange(steps, device=left.device)
    allowed = (grid[:, None] - grid[None, :]).abs() <= radius
    valid = allowed.unsqueeze(0) & left_mask[:, :, None] & right_mask[:, None, :]
    cost = (1.0 - torch.einsum("btd,bsd->bts", left, right)).clamp(0.0, 2.0)
    logits = (-cost / max(temperature, 1e-4)).masked_fill(~valid, -1e4)
    alignment = torch.softmax(logits, dim=-1) * valid.to(cost.dtype)
    alignment = torch.nan_to_num(alignment, nan=0.0, posinf=0.0, neginf=0.0)
    relative = grid.to(cost.dtype) / max(steps - 1, 1)
    monotonic = (relative[:, None] - relative[None, :]).abs().unsqueeze(0)
    row_valid = valid.any(-1) & left_mask
    per_row = ((cost + 0.10 * monotonic) * alignment).sum(-1)
    return (per_row * row_valid.to(cost.dtype)).sum(1) / row_valid.sum(1).clamp_min(1)


def _multi_positive(logits: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    denominator = torch.logsumexp(logits, dim=1)
    numerator = torch.logsumexp(logits.masked_fill(~positive, -1e4), dim=1)
    return denominator - numerator


def global_clip_loss(left: torch.Tensor, right: torch.Tensor, labels: Iterable[str], scale: torch.Tensor,
                     sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    # Clamp before exp: clamping an already-infinite exp can still yield a
    # nonfinite backward path on some accelerators.
    logit_scale = scale.clamp(max=math.log(100.0)).exp()
    logits = F.normalize(left, dim=-1, eps=1e-6) @ F.normalize(right, dim=-1, eps=1e-6).T * logit_scale
    names = [str(value).strip().lower() for value in labels]
    positive = torch.tensor([[a == b for b in names] for a in names], dtype=torch.bool, device=logits.device)
    weight = torch.ones(len(names), device=logits.device, dtype=logits.dtype) if sample_weight is None else sample_weight.to(logits.dtype)
    return 0.5 * (weighted_mean(_multi_positive(logits, positive), weight) +
                  weighted_mean(_multi_positive(logits.T, positive.T), weight))


def _masked_cosine(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor,
                   weight: torch.Tensor) -> torch.Tensor:
    support = mask[:, None, :].to(left.dtype)
    left = (left * support).flatten(1)
    right = (right * support).flatten(1)
    value = 1.0 - F.cosine_similarity(left, right, dim=1, eps=1e-6)
    return weighted_mean(value, weight)


def residual_distribution_loss(prediction: torch.Tensor, target: torch.Tensor,
                               mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Match residual variation without allocating a 6k x 6k covariance.

    A 39x16 pooled representation preserves coefficient/time structure while
    keeping this diagnostic stable on MPS.  This is deliberately a *match to
    the teacher*, not a generic incentive to inject arbitrary variation.
    """
    if prediction.shape[0] < 2:
        zero = prediction.new_zeros(())
        return zero, zero
    support = mask[:, None, :].to(prediction.dtype)

    # `adaptive_avg_pool1d(161, 16)` is not implemented by PyTorch's MPS
    # backend because 161 is not evenly divisible by 16.  Pool manually into
    # sixteen contiguous bins instead.  Padding is *only* to form the final
    # bin; its zero support is excluded from each bin mean, so this preserves
    # the intended masked-statistics contract for arbitrary valid lengths.
    def masked_temporal_pool(value: torch.Tensor, valid: torch.Tensor, bins: int = 16) -> torch.Tensor:
        batch, channels, frames = value.shape
        width = max(1, math.ceil(frames / bins))
        padded_frames = bins * width
        pad = padded_frames - frames
        if pad:
            value = F.pad(value, (0, pad))
            valid = F.pad(valid, (0, pad))
        value = value.reshape(batch, channels, bins, width)
        valid = valid.reshape(batch, 1, bins, width)
        return (value * valid).sum(-1) / valid.sum(-1).clamp_min(1.0)

    left = masked_temporal_pool(prediction, support).flatten(1)
    right = masked_temporal_pool(target, support).flatten(1)
    left = left - left.mean(0, keepdim=True)
    right = right - right.mean(0, keepdim=True)
    variance = F.smooth_l1_loss(left.std(0, unbiased=False), right.std(0, unbiased=False).detach())
    # The batch is intentionally tiny on laptop hardware.  Matching a compact
    # Gram matrix is more meaningful than a scalar global-standard-deviation
    # floor and remains finite for two-example M0 batches.
    scale = max(left.shape[0], 1)
    covariance = F.smooth_l1_loss((left.T @ left) / scale, ((right.T @ right) / scale).detach())
    return variance, covariance


def teacher_bank_loss(prediction: torch.Tensor, teacher: torch.Tensor, labels: list[str], model,
                      sample_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Frozen full-content-bank InfoNCE plus direct target matching."""
    zero = prediction.new_zeros(())
    if not len(labels) or model.teacher_dimension <= 0:
        return zero, zero, zero
    if not model.teacher_labels or model.teacher_bank.numel() == 0:
        raise RuntimeError("v3 teacher loss requires a frozen train-fold content bank")
    indices = []
    lookup = {label: index for index, label in enumerate(model.teacher_labels)}
    for label in labels:
        if label not in lookup:
            # Validation/test labels are intentionally absent from the train
            # bank. Their direct teacher regression remains valid, while the
            # bank loss is omitted by the caller during evaluation.
            raise RuntimeError(f"teacher bank is missing training content {label}")
        indices.append(lookup[label])
    target = F.normalize(teacher, dim=-1, eps=1e-6)
    source = F.normalize(prediction, dim=-1, eps=1e-6)
    huber = weighted_mean(F.smooth_l1_loss(prediction, teacher, reduction="none").mean(1), sample_weight)
    cosine = weighted_mean(1.0 - (source * target).sum(1), sample_weight)
    bank = F.normalize(model.teacher_bank, dim=-1, eps=1e-6)
    scale = model.clip_logit_scale.clamp(max=math.log(100.0)).exp()
    logits = source @ bank.T * scale
    info_nce = weighted_mean(F.cross_entropy(logits, torch.tensor(indices, device=logits.device), reduction="none"), sample_weight)
    return huber, cosine, info_nce


def joint_content_loss(state: JointState, batch: dict, model, weights: dict) -> tuple[torch.Tensor, dict[str, float]]:
    exact = batch["pairing_weight"].to(state.mfcc.dtype)
    eligible = exact > 0
    zero = state.mfcc.new_zeros(())
    standardized = bool(weights.get("standardized_residual", False))
    if standardized:
        if state.residual_z is None or state.baseline_mfcc is None:
            raise RuntimeError("standardized residual loss requires a zero-centered v3 model")
        target_z = ((batch["content_mfcc"] - state.baseline_mfcc) /
                    model.target_mfcc_scale.unsqueeze(0).clamp_min(1e-4))
        mfcc, delta = masked_mfcc_loss(state.residual_z, target_z, batch["content_mask"], exact)
        residual_cosine = _masked_cosine(state.residual_z, target_z, batch["content_mask"], exact)
        variance, covariance = residual_distribution_loss(state.residual_z[eligible], target_z[eligible],
                                                          batch["content_mask"][eligible]) if eligible.sum() >= 2 else (zero, zero)
    else:
        mfcc, delta = masked_mfcc_loss(state.mfcc, batch["content_mfcc"], batch["content_mask"], exact)
        residual_cosine = zero
        variance = zero
        covariance = zero
    local = zero
    global_ = zero
    teacher_huber = zero
    teacher_cosine = zero
    teacher_info_nce = zero
    hubert_eligible = eligible & batch["hubert_mask"].any(1) if "hubert_mask" in batch else torch.zeros_like(eligible)
    if hubert_eligible.any() and "hubert_local" in batch:
        audio = model.centered_audio(batch["hubert_local"][hubert_eligible])
        local_per_sample = soft_dtw_token_loss(state.local[hubert_eligible], audio, state.token_mask[hubert_eligible], batch["hubert_mask"][hubert_eligible])
        local = weighted_mean(local_per_sample, exact[hubert_eligible])
        if model.teacher_dimension > 0 and "hubert_global" in batch:
            labels = [batch["linguistic_content_id"][i] for i in hubert_eligible.nonzero(as_tuple=False).flatten().tolist()]
            teacher_target = model.teacher_target(batch["hubert_global"][hubert_eligible]).detach()
            teacher_huber, teacher_cosine, teacher_info_nce = teacher_bank_loss(
                state.global_embedding[hubert_eligible], teacher_target, labels, model, exact[hubert_eligible],
            )
            global_ = teacher_info_nce
        else:
            audio_global = F.normalize(audio.mean(1), dim=-1)
            global_ = global_clip_loss(state.global_embedding[hubert_eligible], audio_global,
                                       [batch["linguistic_content_id"][i] for i in hubert_eligible.nonzero(as_tuple=False).flatten().tolist()],
                                       model.clip_logit_scale, exact[hubert_eligible])
    label_mask = batch["phoneme_index"] >= 0
    phoneme = F.cross_entropy(state.phoneme_logits[label_mask], batch["phoneme_index"][label_mask]) if label_mask.any() else zero
    duration = zero
    activity = zero
    if eligible.any() and state.predicted_duration is not None and "audio_duration_frames" in batch:
        target_duration = batch["audio_duration_frames"][eligible].to(state.predicted_duration.dtype).clamp_min(1).log()
        if model.duration_standardized:
            target_duration = ((target_duration - model.duration_log_mean) /
                               model.duration_log_scale.clamp_min(1e-4))
            if state.duration_z is None:
                raise RuntimeError("standardized duration model did not return duration_z")
            duration = F.smooth_l1_loss(state.duration_z[eligible], target_duration)
        else:
            duration = F.smooth_l1_loss(state.predicted_duration[eligible].log(), target_duration)
    if eligible.any() and state.activity_logits is not None and "acoustic_activity" in batch:
        activity = F.binary_cross_entropy_with_logits(state.activity_logits[eligible], batch["acoustic_activity"][eligible].float())
    if not standardized and eligible.sum() >= 2 and state.residual_mfcc is not None:
        target_residual = batch["content_mfcc"][eligible] - state.baseline_mfcc[eligible]
        target_std = target_residual.flatten(1).std(0, unbiased=False).mean().detach()
        predicted_std = state.residual_mfcc[eligible].flatten(1).std(0, unbiased=False).mean()
        floor = float(weights.get("variance_floor_fraction", 0.25)) * target_std
        variance = F.relu(floor - predicted_std)
    rank = zero
    latent_rank = zero
    if eligible.sum() >= 2 and float(weights.get("counterfactual_rank", 0.0)) > 0:
        correct_error = (state.mfcc[eligible] - batch["content_mfcc"][eligible]).abs().mean((1, 2))
        controls = []
        latent_controls = []
        # Block shuffle preserves local EEG spectrum; a full sample permutation
        # can be detected trivially rather than testing temporal decoding.
        model_mask = batch.get("model_time_mask", batch["time_mask"])
        # Counterfactual outputs are fixed comparison baselines.  Their role is
        # to push the *correct* EEG prediction below a detached control error;
        # backpropagating through two additional full EEG encoders merely lets
        # the model manipulate the controls and, on MPS, exhausts memory before
        # the first optimizer step.  This keeps the ranking gradient on the
        # correct branch while making the control branch scientifically stable
        # and O(1) in autograd memory.
        with torch.no_grad():
            shuffled = counterfactual_eeg(batch["eeg"], "time_block_shuffle", time_mask=model_mask,
                                           channel_mask=batch["channel_mask"])
            shuffled_state = model(shuffled, batch["channel_xyz"], batch["channel_mask"], model_mask, batch["dataset_id"])
            controls.append(shuffled_state.mfcc.detach())
            latent_controls.append(shuffled_state.global_embedding.detach())
            labels = list(batch["linguistic_content_id"])
            subjects = list(batch.get("subject", [""] * len(labels)))
            if len(set(labels)) < 2:
                # Weighted M0 sampling can rarely draw one content repeatedly.
                # Keep the valid time-block control, but omit the unavailable
                # wrong-trial comparator rather than aborting training.
                pass
            else:
                order = []
                for index, label in enumerate(labels):
                    candidate = next((other for other, other_label in enumerate(labels)
                                      if other_label != label and subjects[other] == subjects[index]), None)
                    if candidate is None:
                        candidate = next((other for other, other_label in enumerate(labels) if other_label != label), None)
                    if candidate is None:
                        raise RuntimeError("wrong-trial ranking requires at least two contents in a batch")
                    order.append(candidate)
                swapped = batch["eeg"][torch.tensor(order, device=batch["eeg"].device)]
                swapped_state = model(swapped, batch["channel_xyz"], batch["channel_mask"], model_mask, batch["dataset_id"])
                controls.append(swapped_state.mfcc.detach())
                latent_controls.append(swapped_state.global_embedding.detach())
        margin = float(weights.get("counterfactual_margin", 0.02))
        values = []
        for prediction in controls:
            control_error = (prediction[eligible] - batch["content_mfcc"][eligible]).abs().mean((1, 2))
            values.append(F.relu(margin + correct_error - control_error).mean())
        if values:
            rank = torch.stack(values).mean()
        if model.teacher_dimension > 0 and "hubert_global" in batch:
            target = model.teacher_target(batch["hubert_global"][eligible]).detach()
            correct_latent_error = (state.global_embedding[eligible] - target).square().mean(1)
            latent_values = []
            for prediction in latent_controls:
                control_error = (prediction[eligible] - target).square().mean(1)
                latent_values.append(F.relu(margin + correct_latent_error - control_error).mean())
            if latent_values:
                latent_rank = torch.stack(latent_values).mean()
    subject_adversary = zero
    if state.subject_logits is not None and "subject_index" in batch:
        valid = batch["subject_index"] >= 0
        if valid.any():
            subject_adversary = F.cross_entropy(state.subject_logits[valid], batch["subject_index"][valid])
    total = (float(weights["mfcc"]) * mfcc + float(weights["delta"]) * delta +
             float(weights["local_alignment"]) * local + float(weights["global_clip"]) * global_ +
             float(weights["phoneme_auxiliary"]) * phoneme +
             float(weights.get("duration", 0.0)) * duration + float(weights.get("activity", 0.0)) * activity +
             float(weights.get("counterfactual_rank", 0.0)) * rank + float(weights.get("variance_retention", 0.0)) * variance +
             float(weights.get("residual_cosine", 0.0)) * residual_cosine +
             float(weights.get("residual_covariance", 0.0)) * covariance +
             float(weights.get("teacher_huber", 0.0)) * teacher_huber +
             float(weights.get("teacher_cosine", 0.0)) * teacher_cosine +
             float(weights.get("teacher_info_nce", 0.0)) * teacher_info_nce +
             float(weights.get("latent_counterfactual_rank", 0.0)) * latent_rank +
             float(weights.get("subject_adversary", 0.0)) * subject_adversary)
    metrics = {"total": float(total.detach()), "mfcc": float(mfcc.detach()), "delta": float(delta.detach()),
               "local_alignment": float(local.detach()), "global_clip": float(global_.detach()),
               "phoneme_auxiliary": float(phoneme.detach()), "duration": float(duration.detach()),
               "activity": float(activity.detach()), "counterfactual_rank": float(rank.detach()),
               "variance_retention": float(variance.detach()), "residual_cosine": float(residual_cosine.detach()),
               "residual_covariance": float(covariance.detach()), "teacher_huber": float(teacher_huber.detach()),
               "teacher_cosine": float(teacher_cosine.detach()), "teacher_info_nce": float(teacher_info_nce.detach()),
               "latent_counterfactual_rank": float(latent_rank.detach()),
               "subject_adversary": float(subject_adversary.detach())}
    return total, metrics


def counterfactual_eeg(eeg: torch.Tensor, control: str, generator: torch.Generator | None = None,
                       time_mask: torch.Tensor | None = None,
                       channel_mask: torch.Tensor | None = None) -> torch.Tensor:
    if control == "zero":
        return torch.zeros_like(eeg)
    if control == "time_shuffle":
        output = eeg.clone()
        time_mask = torch.ones(eeg.shape[0], eeg.shape[-1], dtype=torch.bool, device=eeg.device) if time_mask is None else time_mask
        for batch in range(eeg.shape[0]):
            valid = time_mask[batch].nonzero(as_tuple=False).flatten()
            if generator is None:
                stride = next((value for value in range(2, len(valid)) if math.gcd(value, len(valid)) == 1), 1)
                permutation = (torch.arange(len(valid), device=eeg.device) * stride + 1) % max(len(valid), 1)
            else:
                permutation = torch.randperm(len(valid), generator=generator, device=eeg.device)
            order = valid[permutation]
            output[batch, :, valid] = eeg[batch, :, order]
        return output
    if control == "time_block_shuffle":
        output = eeg.clone()
        time_mask = torch.ones(eeg.shape[0], eeg.shape[-1], dtype=torch.bool, device=eeg.device) if time_mask is None else time_mask
        block = 64
        for batch in range(eeg.shape[0]):
            valid = time_mask[batch].nonzero(as_tuple=False).flatten()
            if len(valid) <= block:
                continue
            chunks = [valid[offset:offset + block] for offset in range(0, len(valid), block)]
            order = torch.randperm(len(chunks), generator=generator, device=eeg.device) if generator is not None else torch.arange(len(chunks) - 1, -1, -1, device=eeg.device)
            source = torch.cat([chunks[int(index)] for index in order])
            output[batch, :, valid] = eeg[batch, :, source]
        return output
    if control == "channel_shuffle":
        output = eeg.clone()
        channel_mask = torch.ones(eeg.shape[0], eeg.shape[1], dtype=torch.bool, device=eeg.device) if channel_mask is None else channel_mask
        for batch in range(eeg.shape[0]):
            valid = channel_mask[batch].nonzero(as_tuple=False).flatten()
            if generator is None:
                stride = next((value for value in range(2, len(valid)) if math.gcd(value, len(valid)) == 1), 1)
                permutation = (torch.arange(len(valid), device=eeg.device) * stride + 1) % max(len(valid), 1)
            else:
                permutation = torch.randperm(len(valid), generator=generator, device=eeg.device)
            order = valid[permutation]
            output[batch, valid] = eeg[batch, order]
        return output
    raise ValueError(f"unknown counterfactual control {control}")

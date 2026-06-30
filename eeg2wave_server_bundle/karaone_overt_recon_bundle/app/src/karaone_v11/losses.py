from __future__ import annotations

import torch
import torch.nn.functional as F

from src.karaone_v10.losses import (
    balanced_prompt_ce,
    cross_subject_semantic_infonce,
    mean_prior_margin_loss,
    pairwise_decorrelation_loss,
    same_label_cross_subject_prototype_pull,
    zero_prior_margin_loss,
)
from src.karaone_v9.losses import (
    monotonic_soft_ot_loss,
    prompt_ctc_loss,
    prosody_loss,
    semantic_token_ce_loss,
    sequence_cosine_loss_per_sample,
    symmetric_infonce,
    vicreg_variance_covariance,
)
from src.karaone_v9.model import resize_sequence
from src.karaone_v91.losses import (
    channel_entropy_floor_loss,
    channel_gate_consistency_loss,
    channel_load_balance_loss,
    channel_sparsity_loss,
)


def compute_v11_pretrain_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor] | None = None,
    *,
    lambda_masked_recon: float = 1.0,
    lambda_variance: float = 0.05,
    lambda_channel_balance: float = 0.05,
    lambda_gate_sparsity: float = 0.02,
    lambda_gate_entropy: float = 0.02,
) -> dict[str, torch.Tensor]:
    mask = out["patch_mask"].bool()
    pred = out["patch_recon"]
    target = out["patch_tokens_target"]
    recon = F.smooth_l1_loss(pred[mask], target[mask]) if bool(mask.any()) else F.smooth_l1_loss(pred, target)
    variance, covariance = vicreg_variance_covariance(out["pooled"])
    balance = channel_load_balance_loss(out)
    sparsity = channel_sparsity_loss(out)
    entropy = channel_entropy_floor_loss(out)
    total = (
        float(lambda_masked_recon) * recon
        + float(lambda_variance) * variance
        + float(lambda_channel_balance) * balance
        + float(lambda_gate_sparsity) * sparsity
        + float(lambda_gate_entropy) * entropy
    )
    return {
        "total": total,
        "masked_recon": recon,
        "vicreg_variance": variance,
        "vicreg_covariance": covariance,
        "channel_balance": balance,
        "channel_sparsity": sparsity,
        "channel_entropy_floor": entropy,
    }


def compute_v11_alignment_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    aligner: str = "hybrid",
    lambda_token_ce: float = 0.7,
    lambda_token_ctc: float = 0.35,
    lambda_clip: float = 0.5,
    lambda_soft_ot: float = 0.5,
    lambda_perceiver: float = 0.25,
    lambda_prompt: float = 0.2,
    lambda_prompt_balanced: float = 0.2,
    lambda_prosody: float = 0.4,
    lambda_cross_subject: float = 0.35,
    lambda_same_label_pull: float = 0.10,
    lambda_zero_margin: float = 0.25,
    lambda_mean_margin: float = 0.20,
    lambda_pairwise_decorrelation: float = 0.05,
    lambda_subject_adv: float = 0.10,
    lambda_channel_balance: float = 0.05,
    lambda_gate_sparsity: float = 0.02,
    lambda_gate_entropy: float = 0.02,
    lambda_gate_consistency: float = 0.05,
    semantic_margin: float = 0.04,
    temperature: float = 0.07,
) -> dict[str, torch.Tensor]:
    aligner = str(aligner or "hybrid").lower()
    pred_summary = out["pred_semantic_summary"]
    target_summary = batch["semantic_summary"].to(pred_summary.device, pred_summary.dtype)
    target_seq = batch["semantic_seq"].to(pred_summary.device, pred_summary.dtype)
    label_idx = _ids(batch, "label_idx", pred_summary.device)
    subject_idx = _ids(batch, "subject_idx", pred_summary.device)
    speech_cluster = _ids(batch, "speech_cluster_id", pred_summary.device)
    token_targets = batch["audio_semantic_tokens"].to(pred_summary.device)
    token_mask = batch["audio_semantic_token_mask"].to(pred_summary.device)

    token_ce = semantic_token_ce_loss(out["semantic_token_logits"], token_targets, token_mask)
    token_ctc = semantic_token_ctc_loss(out["semantic_token_logits"], token_targets, token_mask)
    clip = symmetric_infonce(pred_summary, target_summary, temperature=temperature)
    soft_ot = monotonic_soft_ot_loss(out["pred_semantic_seq"], target_seq, temperature=0.08, monotonic_sigma=0.25)
    perceiver = semantic_token_ce_loss(out["semantic_token_logits_perceiver"], token_targets, token_mask)
    prompt = F.cross_entropy(out["prompt_logits"], label_idx)
    prompt_balanced = balanced_prompt_ce(out["prompt_logits"], label_idx)
    prompt_ctc = prompt_ctc_loss(out["prompt_ctc_logits"], label_idx, out["token_valid_mask"])
    prosody = prosody_loss(out, batch)
    cross_subject = cross_subject_semantic_infonce(pred_summary, target_summary, label_idx, subject_idx, speech_cluster, temperature=0.06)
    same_label_pull = same_label_cross_subject_prototype_pull(pred_summary, target_summary, label_idx, subject_idx)
    zero_margin = zero_prior_margin_loss(pred_summary, target_summary, margin=semantic_margin)
    mean_margin = mean_prior_margin_loss(pred_summary, target_summary, margin=semantic_margin)
    decorrelation = pairwise_decorrelation_loss(pred_summary)
    subject_adv = F.cross_entropy(out["subject_logits"], subject_idx)
    balance = channel_load_balance_loss(out)
    sparsity = channel_sparsity_loss(out)
    entropy = channel_entropy_floor_loss(out)
    gate_consistency = channel_gate_consistency_loss(out, speech_cluster, subject_idx)

    enabled = _aligner_weights(aligner)
    total = (
        enabled["token_ce"] * float(lambda_token_ce) * token_ce
        + enabled["token_ctc"] * float(lambda_token_ctc) * token_ctc
        + enabled["clip"] * float(lambda_clip) * clip
        + enabled["ot"] * float(lambda_soft_ot) * soft_ot
        + enabled["perceiver"] * float(lambda_perceiver) * perceiver
        + float(lambda_prompt) * (prompt + 0.5 * prompt_ctc)
        + float(lambda_prompt_balanced) * prompt_balanced
        + float(lambda_prosody) * prosody
        + float(lambda_cross_subject) * cross_subject
        + float(lambda_same_label_pull) * same_label_pull
        + float(lambda_zero_margin) * zero_margin
        + float(lambda_mean_margin) * mean_margin
        + float(lambda_pairwise_decorrelation) * decorrelation
        + float(lambda_subject_adv) * subject_adv
        + float(lambda_channel_balance) * balance
        + float(lambda_gate_sparsity) * sparsity
        + float(lambda_gate_entropy) * entropy
        + float(lambda_gate_consistency) * gate_consistency
    )
    return {
        "total": total,
        "semantic_token_ce": token_ce,
        "semantic_token_ctc": token_ctc,
        "clip_nce": clip,
        "token_soft_ot": soft_ot,
        "perceiver_token_ce": perceiver,
        "prompt_ce": prompt,
        "prompt_balanced_ce": prompt_balanced,
        "prompt_ctc": prompt_ctc,
        "prosody": prosody,
        "cross_subject_semantic_nce": cross_subject,
        "same_label_prototype_pull": same_label_pull,
        "zero_prior_margin": zero_margin,
        "mean_prior_margin": mean_margin,
        "pairwise_decorrelation": decorrelation,
        "subject_adv": subject_adv,
        "channel_balance": balance,
        "channel_sparsity": sparsity,
        "channel_entropy_floor": entropy,
        "channel_gate_consistency": gate_consistency,
    }


def compute_v11_codec_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    lambda_codec_token_ce: float = 1.0,
    lambda_codec_latent: float = 0.5,
    lambda_semantic_guard: float = 0.2,
    lambda_boundary_continuity: float = 0.05,
) -> dict[str, torch.Tensor]:
    codec_tokens = batch["codec_token_targets"].to(out["codec_token_logits"].device)
    codec_mask = batch["codec_token_mask"].to(out["codec_token_logits"].device)
    token_ce = masked_token_ce(out["codec_token_logits"], codec_tokens, codec_mask)
    target_codec = resize_sequence(batch["codec_seq"].to(out["pred_codec_seq"].device, out["pred_codec_seq"].dtype), out["pred_codec_seq"].shape[1])
    if codec_mask.shape[1] != out["pred_codec_seq"].shape[1]:
        codec_mask = F.interpolate(codec_mask.unsqueeze(1).float(), size=out["pred_codec_seq"].shape[1], mode="nearest").squeeze(1)
    latent = (F.smooth_l1_loss(out["pred_codec_seq"], target_codec, reduction="none").mean(dim=-1) * codec_mask).sum() / codec_mask.sum().clamp_min(1.0)
    semantic_guard = sequence_cosine_loss_per_sample(
        out["pred_semantic_seq"],
        resize_sequence(batch["semantic_seq"].to(out["pred_semantic_seq"].device, out["pred_semantic_seq"].dtype), out["pred_semantic_seq"].shape[1]),
    ).mean()
    boundary = out["pred_codec_seq"][:, 1:].sub(out["pred_codec_seq"][:, :-1]).abs().mean() if out["pred_codec_seq"].shape[1] > 1 else latent.new_tensor(0.0)
    total = (
        float(lambda_codec_token_ce) * token_ce
        + float(lambda_codec_latent) * latent
        + float(lambda_semantic_guard) * semantic_guard
        + float(lambda_boundary_continuity) * boundary
    )
    return {
        "total": total,
        "codec_token_ce": token_ce,
        "codec_latent": latent,
        "condition_semantic": semantic_guard,
        "chunk_boundary_continuity": boundary,
    }


def semantic_token_ctc_loss(logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] <= 1:
        return logits.new_tensor(0.0)
    blank = logits.shape[-1]
    blank_logits = logits.new_zeros((*logits.shape[:2], 1))
    ctc_logits = torch.cat([logits, blank_logits], dim=-1)
    log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
    input_lengths = torch.full((logits.shape[0],), logits.shape[1], device=logits.device, dtype=torch.long)
    target_rows = []
    lengths = []
    for idx in range(tokens.shape[0]):
        row = tokens[idx].to(logits.device).long().clamp(0, logits.shape[-1] - 1)
        row = row[mask[idx].to(logits.device) > 0]
        if row.numel() == 0:
            row = torch.full((1,), blank, device=logits.device, dtype=torch.long)
        target_rows.append(_collapse_repeats(row))
        lengths.append(int(target_rows[-1].numel()))
    targets = torch.cat(target_rows, dim=0)
    target_lengths = torch.tensor(lengths, device=logits.device, dtype=torch.long)
    if logits.device.type == "mps":
        return F.ctc_loss(log_probs.cpu(), targets.cpu(), input_lengths.cpu(), target_lengths.cpu(), blank=blank, zero_infinity=True).to(logits.device)
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=blank, zero_infinity=True)


def masked_token_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] <= 1 or mask.sum() <= 0:
        return logits.new_tensor(0.0)
    if targets.shape[1] != logits.shape[1]:
        targets = F.interpolate(targets.float().unsqueeze(1), size=logits.shape[1], mode="nearest").squeeze(1).long()
        mask = F.interpolate(mask.float().unsqueeze(1), size=logits.shape[1], mode="nearest").squeeze(1)
    loss = F.cross_entropy(logits.transpose(1, 2), targets.clamp(0, logits.shape[-1] - 1), reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def _collapse_repeats(row: torch.Tensor) -> torch.Tensor:
    if row.numel() <= 1:
        return row
    keep = torch.ones_like(row, dtype=torch.bool)
    keep[1:] = row[1:] != row[:-1]
    return row[keep]


def _aligner_weights(aligner: str) -> dict[str, float]:
    if aligner == "mlp":
        return {"token_ce": 1.0, "token_ctc": 0.0, "clip": 0.0, "ot": 0.0, "perceiver": 0.0}
    if aligner == "clip":
        return {"token_ce": 0.3, "token_ctc": 0.0, "clip": 1.0, "ot": 0.0, "perceiver": 0.0}
    if aligner == "ctc":
        return {"token_ce": 0.5, "token_ctc": 1.0, "clip": 0.0, "ot": 0.0, "perceiver": 0.0}
    if aligner == "ot":
        return {"token_ce": 0.3, "token_ctc": 0.0, "clip": 0.3, "ot": 1.0, "perceiver": 0.0}
    if aligner == "perceiver":
        return {"token_ce": 0.3, "token_ctc": 0.0, "clip": 0.3, "ot": 0.0, "perceiver": 1.0}
    return {"token_ce": 1.0, "token_ctc": 1.0, "clip": 1.0, "ot": 1.0, "perceiver": 1.0}


def _ids(batch: dict[str, torch.Tensor], key: str, device: torch.device) -> torch.Tensor:
    if key not in batch:
        fallback = "label_idx" if key.endswith("cluster_id") else "subject_idx"
        if fallback in batch:
            return torch.zeros_like(batch[fallback], device=device).long()
        return torch.zeros(1, device=device, dtype=torch.long)
    return batch[key].to(device).long()

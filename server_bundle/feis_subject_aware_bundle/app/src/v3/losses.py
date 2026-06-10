"""v3 training objective: InfoNCE + latent (cosine/MSE) + class CE (+ optional KD).

None of these terms is minimised by "predict the mean", which is exactly the
failure mode of the old waveform-regression path. The contrastive term in
particular pushes representations apart, so the model cannot collapse all
inputs to one output.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def symmetric_infonce(
    eeg_embedding: torch.Tensor,
    audio_embedding: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """CLIP-style symmetric contrastive loss; diagonal = matching pairs."""
    a = F.normalize(eeg_embedding, dim=-1)
    b = F.normalize(audio_embedding, dim=-1)
    logits = (a @ b.transpose(0, 1)) / max(float(temperature), 1e-6)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))


def compute_v3_losses(
    outputs: dict[str, torch.Tensor],
    target_sequence: torch.Tensor,
    target_summary: torch.Tensor,
    label_ids: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    lambda_contrastive: float = 1.0,
    lambda_latent_cosine: float = 1.0,
    lambda_latent_mse: float = 0.5,
    lambda_cls: float = 0.5,
    contrastive_temperature: float = 0.07,
    teacher_outputs: dict[str, torch.Tensor] | None = None,
    lambda_kd_latent: float = 0.0,
    lambda_kd_logits: float = 0.0,
    kd_temperature: float = 2.0,
) -> dict[str, torch.Tensor]:
    pred_seq = outputs["speech_sequence"]
    b, t, d = pred_seq.shape
    if target_mask is None:
        mask = torch.ones((b, t), device=pred_seq.device, dtype=pred_seq.dtype)
    else:
        mask = target_mask.to(device=pred_seq.device, dtype=pred_seq.dtype)

    # Latent cosine (per frame) -> drives directional match in EnCodec space.
    cos_per_step = F.cosine_similarity(pred_seq, target_sequence, dim=-1)      # [B, T]
    latent_cosine = _masked_mean(cos_per_step, mask)
    latent_cosine_loss = 1.0 - latent_cosine

    # Latent MSE (per frame).
    mse_per_step = (pred_seq - target_sequence).pow(2).mean(dim=-1)            # [B, T]
    latent_mse = _masked_mean(mse_per_step, mask)

    # Contrastive retrieval (the anti-collapse workhorse).
    contrastive = symmetric_infonce(
        outputs["contrastive_embedding"], target_summary, temperature=contrastive_temperature
    )

    # 16-way classification.
    cls_loss = F.cross_entropy(outputs["label_logits"], label_ids.long())
    cls_acc = (outputs["label_logits"].argmax(dim=-1) == label_ids.long()).float().mean()

    total = (
        lambda_contrastive * contrastive
        + lambda_latent_cosine * latent_cosine_loss
        + lambda_latent_mse * latent_mse
        + lambda_cls * cls_loss
    )

    kd_latent = pred_seq.new_tensor(0.0)
    kd_logits = pred_seq.new_tensor(0.0)
    if teacher_outputs is not None:
        if lambda_kd_latent > 0.0:
            kd_latent = F.mse_loss(pred_seq, teacher_outputs["speech_sequence"].detach())
            total = total + lambda_kd_latent * kd_latent
        if lambda_kd_logits > 0.0:
            t_logits = teacher_outputs["label_logits"].detach() / kd_temperature
            s_logsm = F.log_softmax(outputs["label_logits"] / kd_temperature, dim=-1)
            kd_logits = F.kl_div(s_logsm, F.softmax(t_logits, dim=-1), reduction="batchmean") * (
                kd_temperature ** 2
            )
            total = total + lambda_kd_logits * kd_logits

    return {
        "total": total,
        "contrastive": contrastive.detach(),
        "latent_cosine": latent_cosine.detach(),
        "latent_cosine_loss": latent_cosine_loss.detach(),
        "latent_mse": latent_mse.detach(),
        "cls": cls_loss.detach(),
        "cls_acc": cls_acc.detach(),
        "kd_latent": kd_latent.detach(),
        "kd_logits": kd_logits.detach(),
    }

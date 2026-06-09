from __future__ import annotations

import torch
import torch.nn.functional as F

from .utils import cosine_similarity_batch


def _ensure_mask(mask: torch.Tensor | None, batch: int, steps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if mask is None:
        return torch.ones((batch, steps), device=device, dtype=dtype)
    return mask.to(device=device, dtype=dtype)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _masked_sequence_summary(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (sequence * weights).sum(dim=1) / denom


def _symmetric_contrastive_loss(
    pred_summary: torch.Tensor,
    target_summary: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    pred_norm = F.normalize(pred_summary, dim=-1)
    target_norm = F.normalize(target_summary, dim=-1)
    logits = pred_norm @ target_norm.transpose(0, 1)
    logits = logits / max(float(temperature), 1e-6)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.transpose(0, 1), labels)
    return 0.5 * (loss_a + loss_b)


def compute_alignment_losses(
    pred_sequence: torch.Tensor,
    target_sequence: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    pred_summary: torch.Tensor | None = None,
    target_summary: torch.Tensor | None = None,
    pred_prosody: torch.Tensor | None = None,
    target_prosody: torch.Tensor | None = None,
    lambda_seq_cosine: float = 1.0,
    lambda_seq_mse: float = 0.5,
    lambda_contrastive: float = 1.0,
    lambda_prosody: float = 0.0,
    contrastive_temperature: float = 0.07,
    label_logits: torch.Tensor | None = None,
    label_ids: torch.Tensor | None = None,
    lambda_cls: float = 0.0,
) -> dict[str, torch.Tensor]:
    if pred_sequence.shape != target_sequence.shape:
        raise ValueError(
            f"Sequence shape mismatch: pred={tuple(pred_sequence.shape)} target={tuple(target_sequence.shape)}"
        )
    batch_size, target_steps, target_dim = pred_sequence.shape
    mask = _ensure_mask(target_mask, batch=batch_size, steps=target_steps, device=pred_sequence.device, dtype=pred_sequence.dtype)
    cosine_per_step = cosine_similarity_batch(
        pred_sequence.reshape(-1, target_dim),
        target_sequence.reshape(-1, target_dim),
    ).view(batch_size, target_steps)
    sequence_cosine = _masked_mean(cosine_per_step, mask)
    sequence_cosine_loss = 1.0 - sequence_cosine

    mse_per_step = (pred_sequence - target_sequence).pow(2).mean(dim=-1)
    sequence_mse = _masked_mean(mse_per_step, mask)

    pred_summary = _masked_sequence_summary(pred_sequence, mask) if pred_summary is None else pred_summary
    target_summary = _masked_sequence_summary(target_sequence, mask) if target_summary is None else target_summary
    summary_cosine = cosine_similarity_batch(pred_summary, target_summary).mean()
    contrastive = _symmetric_contrastive_loss(
        pred_summary=pred_summary,
        target_summary=target_summary,
        temperature=contrastive_temperature,
    )

    total = (
        float(lambda_seq_cosine) * sequence_cosine_loss
        + float(lambda_seq_mse) * sequence_mse
        + float(lambda_contrastive) * contrastive
    )

    prosody_loss = pred_sequence.new_tensor(0.0)
    if pred_prosody is not None and target_prosody is not None and target_prosody.numel() > 0:
        prosody_loss = F.smooth_l1_loss(pred_prosody, target_prosody)
        total = total + float(lambda_prosody) * prosody_loss

    cls_loss = pred_sequence.new_tensor(0.0)
    cls_acc = pred_sequence.new_tensor(0.0)
    if label_logits is not None and label_ids is not None:
        cls_loss = F.cross_entropy(label_logits, label_ids.long())
        cls_pred = torch.argmax(label_logits, dim=-1)
        cls_acc = (cls_pred == label_ids.long()).float().mean()
        total = total + float(lambda_cls) * cls_loss

    return {
        "total": total,
        "sequence_cosine_loss": sequence_cosine_loss,
        "sequence_cosine": sequence_cosine,
        "sequence_mse": sequence_mse,
        "summary_cosine": summary_cosine,
        "embedding_cosine": summary_cosine,
        "embedding_mse": F.mse_loss(pred_summary, target_summary),
        "contrastive": contrastive,
        "prosody_loss": prosody_loss,
        "cls": cls_loss,
        "cls_acc": cls_acc,
    }

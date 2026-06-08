from __future__ import annotations

import torch
import torch.nn.functional as F

from .utils import cosine_similarity_batch


def compute_alignment_losses(
    pred_embedding: torch.Tensor,
    target_embedding: torch.Tensor,
    pred_prosody: torch.Tensor,
    target_prosody: torch.Tensor,
    lambda_cosine: float = 1.0,
    lambda_mse: float = 0.5,
    lambda_prosody: float = 0.25,
    label_logits: torch.Tensor | None = None,
    label_ids: torch.Tensor | None = None,
    lambda_cls: float = 0.0,
) -> dict[str, torch.Tensor]:
    embedding_cosine = cosine_similarity_batch(pred_embedding, target_embedding).mean()
    embedding_cosine_loss = 1.0 - embedding_cosine
    embedding_mse = F.mse_loss(pred_embedding, target_embedding)
    prosody_loss = F.smooth_l1_loss(pred_prosody, target_prosody)
    total = (
        float(lambda_cosine) * embedding_cosine_loss
        + float(lambda_mse) * embedding_mse
        + float(lambda_prosody) * prosody_loss
    )
    cls_loss = pred_embedding.new_tensor(0.0)
    cls_acc = pred_embedding.new_tensor(0.0)
    if label_logits is not None and label_ids is not None:
        cls_loss = F.cross_entropy(label_logits, label_ids.long())
        cls_pred = torch.argmax(label_logits, dim=-1)
        cls_acc = (cls_pred == label_ids.long()).float().mean()
        total = total + float(lambda_cls) * cls_loss
    return {
        "total": total,
        "embedding_cosine_loss": embedding_cosine_loss,
        "embedding_cosine": embedding_cosine,
        "embedding_mse": embedding_mse,
        "prosody_loss": prosody_loss,
        "cls": cls_loss,
        "cls_acc": cls_acc,
    }

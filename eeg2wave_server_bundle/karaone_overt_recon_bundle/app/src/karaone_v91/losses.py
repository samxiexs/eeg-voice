from __future__ import annotations

import torch
import torch.nn.functional as F

from src.karaone_v9.losses import (
    compute_v9_alignment_losses,
    compute_v9_pretrain_losses,
    compute_v9_transport_losses,
    sequence_cosine_loss_per_sample,
    vicreg_variance_covariance,
)
from src.karaone_v9.model import resize_sequence


def compute_v91_pretrain_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor] | None = None,
    *,
    lambda_masked_recon: float = 1.0,
    lambda_variance: float = 0.05,
    lambda_channel_balance: float = 0.05,
    lambda_gate_sparsity: float = 0.02,
    lambda_gate_entropy: float = 0.02,
) -> dict[str, torch.Tensor]:
    losses = compute_v9_pretrain_losses(out, lambda_masked_recon=lambda_masked_recon, lambda_variance=lambda_variance)
    balance = channel_load_balance_loss(out)
    sparsity = channel_sparsity_loss(out)
    entropy = channel_entropy_floor_loss(out)
    total = losses["total"] + float(lambda_channel_balance) * balance + float(lambda_gate_sparsity) * sparsity + float(lambda_gate_entropy) * entropy
    losses.update(
        {
            "total": total,
            "channel_balance": balance,
            "channel_sparsity": sparsity,
            "channel_entropy_floor": entropy,
        }
    )
    return losses


def compute_v91_alignment_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    lambda_cluster_nce: float = 0.25,
    lambda_hard_negative: float = 0.15,
    lambda_gate_consistency: float = 0.05,
    lambda_channel_balance: float = 0.05,
    lambda_gate_sparsity: float = 0.02,
    lambda_gate_entropy: float = 0.02,
    lambda_domain_subject: float = 0.05,
    lambda_content_domain_orth: float = 0.02,
    **base_kwargs,
) -> dict[str, torch.Tensor]:
    losses = compute_v9_alignment_losses(out, batch, **base_kwargs)
    cluster_nce = cluster_soft_positive_infonce(
        out["pred_semantic_summary"],
        batch["semantic_summary"].to(out["pred_semantic_summary"].device, out["pred_semantic_summary"].dtype),
        _batch_ids(batch, "speech_cluster_id", out["pred_semantic_summary"].device),
        _batch_ids(batch, "subject_idx", out["pred_semantic_summary"].device),
    )
    hard_negative = cluster_hard_negative_margin(
        out["pred_semantic_summary"],
        batch["semantic_summary"].to(out["pred_semantic_summary"].device, out["pred_semantic_summary"].dtype),
        _batch_ids(batch, "eeg_cluster_id", out["pred_semantic_summary"].device),
        _batch_ids(batch, "label_idx", out["pred_semantic_summary"].device),
    )
    gate_consistency = channel_gate_consistency_loss(
        out,
        _batch_ids(batch, "speech_cluster_id", out["pred_semantic_summary"].device),
        _batch_ids(batch, "subject_idx", out["pred_semantic_summary"].device),
    )
    balance = channel_load_balance_loss(out)
    sparsity = channel_sparsity_loss(out)
    entropy = channel_entropy_floor_loss(out)
    domain_subject = (
        F.cross_entropy(out["domain_subject_logits"], _batch_ids(batch, "subject_idx", out["domain_subject_logits"].device))
        if "domain_subject_logits" in out
        else out["pred_semantic_summary"].new_tensor(0.0)
    )
    orth = out.get("content_domain_dot", out["pred_semantic_summary"].new_zeros(out["pred_semantic_summary"].shape[0])).abs().mean()
    total = (
        losses["total"]
        + float(lambda_cluster_nce) * cluster_nce
        + float(lambda_hard_negative) * hard_negative
        + float(lambda_gate_consistency) * gate_consistency
        + float(lambda_channel_balance) * balance
        + float(lambda_gate_sparsity) * sparsity
        + float(lambda_gate_entropy) * entropy
        + float(lambda_domain_subject) * domain_subject
        + float(lambda_content_domain_orth) * orth
    )
    losses.update(
        {
            "total": total,
            "cluster_nce": cluster_nce,
            "hard_negative_margin": hard_negative,
            "channel_gate_consistency": gate_consistency,
            "channel_balance": balance,
            "channel_sparsity": sparsity,
            "channel_entropy_floor": entropy,
            "domain_subject": domain_subject,
            "content_domain_orth": orth,
        }
    )
    return losses


def compute_v91_transport_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    lambda_flow: float = 1.0,
    lambda_condition_semantic: float = 0.2,
    lambda_codec_consistency: float = 0.2,
    lambda_boundary_continuity: float = 0.05,
) -> dict[str, torch.Tensor]:
    losses = compute_v9_transport_losses(out, batch, lambda_flow=lambda_flow, lambda_condition_semantic=lambda_condition_semantic)
    codec = out.get("transport_codec_consistency_loss", losses["total"].new_tensor(0.0))
    boundary = out.get("transport_boundary_continuity_loss", losses["total"].new_tensor(0.0))
    total = losses["total"] + float(lambda_codec_consistency) * codec + float(lambda_boundary_continuity) * boundary
    losses.update({"total": total, "codec_consistency": codec, "chunk_boundary_continuity": boundary})
    return losses


def cluster_soft_positive_infonce(
    eeg: torch.Tensor,
    speech: torch.Tensor,
    speech_cluster_id: torch.Tensor,
    subject_idx: torch.Tensor,
    *,
    temperature: float = 0.07,
    same_cluster_weight: float = 0.5,
) -> torch.Tensor:
    if eeg.shape[0] < 2:
        return eeg.new_tensor(0.0)
    e = F.normalize(eeg, dim=-1)
    s = F.normalize(speech, dim=-1)
    logits = e @ s.T / max(float(temperature), 1e-4)
    eye = torch.eye(eeg.shape[0], device=eeg.device, dtype=eeg.dtype)
    same_cluster = speech_cluster_id[:, None].eq(speech_cluster_id[None, :])
    diff_subject = subject_idx[:, None].ne(subject_idx[None, :])
    positives = eye + float(same_cluster_weight) * (same_cluster & diff_subject).to(eeg.dtype)
    positives = positives / positives.sum(dim=1, keepdim=True).clamp_min(1e-6)
    e2s = -(positives * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    s2e = -(positives.T * torch.log_softmax(logits.T, dim=-1)).sum(dim=-1).mean()
    return 0.5 * (e2s + s2e)


def cluster_hard_negative_margin(
    eeg: torch.Tensor,
    speech: torch.Tensor,
    eeg_cluster_id: torch.Tensor,
    label_idx: torch.Tensor,
    *,
    margin: float = 0.10,
) -> torch.Tensor:
    if eeg.shape[0] < 2:
        return eeg.new_tensor(0.0)
    sim = F.normalize(eeg, dim=-1) @ F.normalize(speech, dim=-1).T
    pos = sim.diag()
    same_eeg_diff_label = eeg_cluster_id[:, None].eq(eeg_cluster_id[None, :]) & label_idx[:, None].ne(label_idx[None, :])
    same_label_diff_eeg = label_idx[:, None].eq(label_idx[None, :]) & eeg_cluster_id[:, None].ne(eeg_cluster_id[None, :])
    mask = same_eeg_diff_label | same_label_diff_eeg
    mask.fill_diagonal_(False)
    if not bool(mask.any()):
        return eeg.new_tensor(0.0)
    neg = sim.masked_fill(~mask, -1e4).max(dim=1).values
    valid = neg > -1e3
    if not bool(valid.any()):
        return eeg.new_tensor(0.0)
    return F.relu(float(margin) + neg[valid] - pos[valid]).mean()


def channel_gate_consistency_loss(out: dict[str, torch.Tensor], speech_cluster_id: torch.Tensor, subject_idx: torch.Tensor) -> torch.Tensor:
    gate = out.get("channel_gate")
    if gate is None or gate.shape[0] < 2:
        ref = next(iter(out.values()))
        return ref.new_tensor(0.0) if torch.is_tensor(ref) else torch.tensor(0.0)
    same_cluster = speech_cluster_id[:, None].eq(speech_cluster_id[None, :])
    diff_subject = subject_idx[:, None].ne(subject_idx[None, :])
    mask = same_cluster & diff_subject
    mask.fill_diagonal_(False)
    if not bool(mask.any()):
        return gate.new_tensor(0.0)
    dist = torch.cdist(gate, gate, p=1) / float(gate.shape[1])
    return dist[mask].mean()


def channel_load_balance_loss(out: dict[str, torch.Tensor]) -> torch.Tensor:
    load = out.get("channel_load")
    if load is None:
        ref = next(iter(out.values()))
        return ref.new_tensor(0.0) if torch.is_tensor(ref) else torch.tensor(0.0)
    target = torch.full_like(load, 1.0 / max(int(load.numel()), 1))
    return F.mse_loss(load, target)


def channel_sparsity_loss(out: dict[str, torch.Tensor], target_active_ratio: float = 16.0 / 62.0) -> torch.Tensor:
    gate = out.get("channel_gate")
    if gate is None:
        ref = next(iter(out.values()))
        return ref.new_tensor(0.0) if torch.is_tensor(ref) else torch.tensor(0.0)
    active_ratio = (gate > 1e-4).to(gate.dtype).mean(dim=1)
    gate_mass = gate.mean(dim=1)
    return (active_ratio - float(target_active_ratio)).abs().mean() + 0.25 * gate_mass.mean()


def channel_entropy_floor_loss(out: dict[str, torch.Tensor], floor: float = 0.35) -> torch.Tensor:
    entropy = out.get("channel_gate_entropy")
    if entropy is None:
        ref = next(iter(out.values()))
        return ref.new_tensor(0.0) if torch.is_tensor(ref) else torch.tensor(0.0)
    return F.relu(float(floor) - entropy).mean()


def semantic_gate_losses(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    target_sem = batch["semantic_seq"].to(out["pred_semantic_seq"].device, out["pred_semantic_seq"].dtype)
    seq_cos = sequence_cosine_loss_per_sample(out["pred_semantic_seq"], resize_sequence(target_sem, out["pred_semantic_seq"].shape[1])).mean()
    variance, covariance = vicreg_variance_covariance(out["pred_semantic_summary"])
    return {"semantic_seq_cos": seq_cos, "semantic_variance": variance, "semantic_covariance": covariance}


def _batch_ids(batch: dict[str, torch.Tensor], key: str, device: torch.device) -> torch.Tensor:
    if key not in batch:
        fallback = "label_idx" if key.endswith("cluster_id") else "subject_idx"
        if fallback in batch:
            return torch.zeros_like(batch[fallback], device=device).long()
        return torch.zeros(1, device=device, dtype=torch.long)
    return batch[key].to(device).long()

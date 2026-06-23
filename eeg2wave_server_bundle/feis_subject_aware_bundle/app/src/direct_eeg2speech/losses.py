"""Training losses for the EEG-only speech reconstruction path."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _offdiag(mat: torch.Tensor) -> torch.Tensor:
    if mat.shape[0] < 2:
        return mat.new_zeros(0)
    mask = ~torch.eye(mat.shape[0], dtype=torch.bool, device=mat.device)
    return mat[mask]


def _pairwise_corr_loss(pred_summary: torch.Tensor, target_summary: torch.Tensor) -> torch.Tensor:
    if pred_summary.shape[0] < 2:
        return pred_summary.new_tensor(0.0)
    p = F.normalize(pred_summary - pred_summary.mean(dim=1, keepdim=True), dim=1)
    t = F.normalize(target_summary - target_summary.mean(dim=1, keepdim=True), dim=1)
    return F.mse_loss(_offdiag(p @ p.T), _offdiag(t @ t.T))


def compute_direct_losses(
    out: dict[str, torch.Tensor],
    target_seq: torch.Tensor,
    label_idx: torch.Tensor,
    target_log_rms: torch.Tensor | None = None,
    mean_latent: torch.Tensor | None = None,
    lambda_recon_cos: float = 1.0,
    lambda_recon_smoothl1: float = 0.5,
    lambda_delta: float = 0.25,
    lambda_delta2: float = 0.1,
    lambda_temporal_envelope: float = 0.15,
    lambda_content_ce: float = 0.5,
    lambda_log_rms: float = 0.2,
    lambda_std: float = 0.2,
    lambda_diversity: float = 0.2,
    lambda_mean_margin: float = 0.1,
    lambda_moe_load_balance: float = 0.0,
    lambda_moe_sparsity: float = 0.0,
    lambda_moe_route_entropy: float = 0.0,
    lambda_moe_cluster: float = 0.0,
    lambda_latent_diffusion: float = 0.0,
    mean_margin: float = 0.25,
) -> dict[str, torch.Tensor]:
    pred = out["pred_latent"]
    recon_cos = 1.0 - F.cosine_similarity(pred, target_seq, dim=-1).mean()
    recon_smoothl1 = F.smooth_l1_loss(pred, target_seq)

    if pred.shape[1] > 1:
        pred_delta = pred[:, 1:] - pred[:, :-1]
        tgt_delta = target_seq[:, 1:] - target_seq[:, :-1]
        delta = F.smooth_l1_loss(pred_delta, tgt_delta)
    else:
        delta = pred.new_tensor(0.0)

    if pred.shape[1] > 2:
        pred_delta2 = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
        tgt_delta2 = target_seq[:, 2:] - 2.0 * target_seq[:, 1:-1] + target_seq[:, :-2]
        delta2 = F.smooth_l1_loss(pred_delta2, tgt_delta2)
    else:
        delta2 = pred.new_tensor(0.0)

    pred_envelope = torch.sqrt(pred.pow(2).mean(dim=-1) + 1e-8)
    tgt_envelope = torch.sqrt(target_seq.pow(2).mean(dim=-1) + 1e-8)
    temporal_envelope = F.smooth_l1_loss(pred_envelope, tgt_envelope)

    content_ce = F.cross_entropy(out["content_logits"], label_idx.long())
    content_acc = (out["content_logits"].argmax(-1) == label_idx.long()).float().mean()

    pred_std = pred.reshape(-1, pred.shape[-1]).std(dim=0)
    tgt_std = target_seq.reshape(-1, target_seq.shape[-1]).std(dim=0)
    std_match = F.l1_loss(pred_std, tgt_std)
    std_ratio = (pred_std / tgt_std.clamp_min(1e-6)).median().detach()

    pred_summary = pred.mean(dim=1)
    tgt_summary = target_seq.mean(dim=1)
    diversity = _pairwise_corr_loss(pred_summary, tgt_summary)

    mean_margin_loss = pred.new_tensor(0.0)
    mean_distance = pred.new_tensor(float("nan"))
    if mean_latent is not None:
        if mean_latent.ndim == 2:
            mean_latent = mean_latent.unsqueeze(0)
        dist = torch.sqrt(torch.mean((pred - mean_latent.to(pred.device)) ** 2, dim=(1, 2)) + 1e-8)
        mean_distance = dist.mean().detach()
        mean_margin_loss = F.relu(float(mean_margin) - dist).mean()

    log_rms_loss = pred.new_tensor(0.0)
    if target_log_rms is not None and "pred_log_rms" in out:
        log_rms_loss = F.mse_loss(out["pred_log_rms"], target_log_rms.float())

    moe_load_balance = out.get("moe_load_balance", pred.new_tensor(0.0))
    moe_sparsity = out.get("moe_channel_sparsity", pred.new_tensor(0.0))
    moe_route_entropy = out.get("moe_route_entropy", pred.new_tensor(0.0))
    moe_cluster = out.get("moe_cluster_cohesion", pred.new_tensor(0.0))
    diffusion_loss = out.get("diffusion_loss", pred.new_tensor(0.0))

    total = (
        lambda_recon_cos * recon_cos
        + lambda_recon_smoothl1 * recon_smoothl1
        + lambda_delta * delta
        + lambda_delta2 * delta2
        + lambda_temporal_envelope * temporal_envelope
        + lambda_content_ce * content_ce
        + lambda_log_rms * log_rms_loss
        + lambda_std * std_match
        + lambda_diversity * diversity
        + lambda_mean_margin * mean_margin_loss
        + lambda_moe_load_balance * moe_load_balance
        + lambda_moe_sparsity * moe_sparsity
        + lambda_moe_route_entropy * moe_route_entropy
        + lambda_moe_cluster * moe_cluster
        + lambda_latent_diffusion * diffusion_loss
    )
    return {
        "total": total,
        "recon_cos": recon_cos.detach(),
        "recon_smoothl1": recon_smoothl1.detach(),
        "delta": delta.detach(),
        "delta2": delta2.detach(),
        "temporal_envelope": temporal_envelope.detach(),
        "content_ce": content_ce.detach(),
        "content_acc": content_acc.detach(),
        "log_rms_loss": log_rms_loss.detach(),
        "std_match": std_match.detach(),
        "std_ratio": std_ratio,
        "diversity": diversity.detach(),
        "mean_margin": mean_margin_loss.detach(),
        "mean_distance": mean_distance,
        "moe_load_balance": moe_load_balance.detach(),
        "moe_sparsity": moe_sparsity.detach(),
        "moe_route_entropy": moe_route_entropy.detach(),
        "moe_cluster": moe_cluster.detach(),
        "moe_gate_mean": out.get("moe_channel_gate_mean", pred.new_tensor(0.0)).detach(),
        "moe_usage_min": out.get("moe_usage_min", pred.new_tensor(0.0)).detach(),
        "moe_usage_max": out.get("moe_usage_max", pred.new_tensor(0.0)).detach(),
        "moe_active_channels": out.get("moe_active_channels", pred.new_tensor(0.0)).detach(),
        "diffusion_loss": diffusion_loss.detach(),
        "diffusion_eps_mse": out.get("diffusion_eps_mse", pred.new_tensor(0.0)).detach(),
        "diffusion_x0_mse": out.get("diffusion_x0_mse", pred.new_tensor(0.0)).detach(),
        "diffusion_t_mean": out.get("diffusion_t_mean", pred.new_tensor(0.0)).detach(),
    }

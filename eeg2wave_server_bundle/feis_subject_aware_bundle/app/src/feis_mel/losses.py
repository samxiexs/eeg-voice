from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _dtw_path(cost: np.ndarray, band: int) -> tuple[np.ndarray, np.ndarray, float]:
    t_pred, t_tgt = cost.shape
    band = max(int(band), abs(t_pred - t_tgt))
    inf = 1e18
    dp = np.full((t_pred + 1, t_tgt + 1), inf, dtype=np.float64)
    ptr = np.zeros((t_pred, t_tgt), dtype=np.int8)
    dp[0, 0] = 0.0
    for i in range(1, t_pred + 1):
        j0 = max(1, i - band)
        j1 = min(t_tgt, i + band) + 1
        for j in range(j0, j1):
            choices = (dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
            step = int(np.argmin(choices))
            dp[i, j] = cost[i - 1, j - 1] + choices[step]
            ptr[i - 1, j - 1] = step
    i, j = t_pred - 1, t_tgt - 1
    path_i, path_j = [], []
    while i >= 0 and j >= 0:
        path_i.append(i)
        path_j.append(j)
        step = int(ptr[i, j])
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    path_i.reverse()
    path_j.reverse()
    denom = max(len(path_i), 1)
    return np.asarray(path_i, dtype=np.int64), np.asarray(path_j, dtype=np.int64), float(dp[t_pred, t_tgt] / denom)


def softmin_dtw_mel_loss(
    pred_mel: torch.Tensor,
    target_bank: torch.Tensor,
    *,
    band: int = 10,
    top_k: int = 3,
    temperature: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DTW-aligned L1 against the closest references in a label bank."""
    if pred_mel.ndim != 3 or target_bank.ndim != 4:
        raise ValueError(f"Expected pred [B,T,D], bank [B,K,T,D], got {tuple(pred_mel.shape)} and {tuple(target_bank.shape)}")
    bsz, refs = target_bank.shape[0], target_bank.shape[1]
    top_k = max(1, min(int(top_k), refs))
    naive = torch.mean(torch.abs(pred_mel.unsqueeze(1) - target_bank), dim=(2, 3))
    top_idx = torch.topk(naive, k=top_k, dim=1, largest=False).indices
    losses = []
    costs_out = []
    pred_np = pred_mel.detach().float().cpu().numpy()
    bank_np = target_bank.detach().float().cpu().numpy()
    for b in range(bsz):
        item_losses = []
        item_costs = []
        for ref_idx in top_idx[b].detach().cpu().tolist():
            cost = np.mean(np.abs(pred_np[b, :, None, :] - bank_np[b, ref_idx, None, :, :]), axis=-1)
            path_i, path_j, path_cost = _dtw_path(cost, band=band)
            pi = torch.as_tensor(path_i, dtype=torch.long, device=pred_mel.device)
            pj = torch.as_tensor(path_j, dtype=torch.long, device=pred_mel.device)
            item_losses.append(F.l1_loss(pred_mel[b, pi], target_bank[b, ref_idx, pj]))
            item_costs.append(path_cost)
        cost_tensor = torch.as_tensor(item_costs, dtype=pred_mel.dtype, device=pred_mel.device)
        weights = torch.softmax(-cost_tensor / max(float(temperature), 1e-6), dim=0).detach()
        loss_tensor = torch.stack(item_losses)
        losses.append((weights * loss_tensor).sum())
        costs_out.append(float(np.min(item_costs)))
    return torch.stack(losses).mean(), pred_mel.new_tensor(float(np.mean(costs_out)))


def bank_mean_l1_loss(pred_mel: torch.Tensor, target_bank: torch.Tensor) -> torch.Tensor:
    target = target_bank.mean(dim=1)
    return F.l1_loss(pred_mel, target)


def label_contrastive_loss(contrast_embed: torch.Tensor, label_idx: torch.Tensor, label_prototypes: torch.Tensor, temperature: float = 0.07) -> tuple[torch.Tensor, torch.Tensor]:
    embed = F.normalize(contrast_embed, dim=-1)
    proto = F.normalize(label_prototypes.to(contrast_embed.device), dim=-1)
    logits = embed @ proto.T / max(float(temperature), 1e-6)
    loss = F.cross_entropy(logits, label_idx.long())
    acc = (logits.argmax(dim=-1) == label_idx.long()).float().mean()
    return loss, acc


def compute_feis_mel_losses(
    out: dict[str, torch.Tensor],
    target_bank: torch.Tensor,
    label_idx: torch.Tensor,
    target_log_rms: torch.Tensor,
    label_prototypes: torch.Tensor,
    *,
    use_dtw: bool = True,
    dtw_band: int = 10,
    dtw_top_k: int = 3,
    lambda_dtw: float = 1.0,
    lambda_content_ce: float = 0.75,
    lambda_contrastive: float = 0.25,
    lambda_log_rms: float = 0.1,
    lambda_moe_load_balance: float = 0.0,
    lambda_moe_sparsity: float = 0.0,
    lambda_moe_route_entropy: float = 0.0,
    lambda_moe_cluster: float = 0.0,
    contrast_temperature: float = 0.07,
) -> dict[str, torch.Tensor]:
    pred = out["pred_mel"]
    if use_dtw:
        recon, dtw_cost = softmin_dtw_mel_loss(pred, target_bank, band=dtw_band, top_k=dtw_top_k)
    else:
        recon = bank_mean_l1_loss(pred, target_bank)
        dtw_cost = recon.detach()
    content_ce = F.cross_entropy(out["content_logits"], label_idx.long())
    content_acc = (out["content_logits"].argmax(dim=-1) == label_idx.long()).float().mean()
    contrastive, retrieval_acc = label_contrastive_loss(
        out["contrast_embed"],
        label_idx,
        label_prototypes,
        temperature=contrast_temperature,
    )
    log_rms_loss = F.mse_loss(out["pred_log_rms"], target_log_rms.float())
    moe_load_balance = out.get("moe_load_balance", pred.new_tensor(0.0))
    moe_sparsity = out.get("moe_channel_sparsity", pred.new_tensor(0.0))
    moe_route_entropy = out.get("moe_route_entropy", pred.new_tensor(0.0))
    moe_cluster = out.get("moe_cluster_cohesion", pred.new_tensor(0.0))
    total = (
        lambda_dtw * recon
        + lambda_content_ce * content_ce
        + lambda_contrastive * contrastive
        + lambda_log_rms * log_rms_loss
        + lambda_moe_load_balance * moe_load_balance
        + lambda_moe_sparsity * moe_sparsity
        + lambda_moe_route_entropy * moe_route_entropy
        + lambda_moe_cluster * moe_cluster
    )
    return {
        "total": total,
        "mel_dtw": recon.detach(),
        "dtw_cost": dtw_cost.detach(),
        "content_ce": content_ce.detach(),
        "content_acc": content_acc.detach(),
        "contrastive": contrastive.detach(),
        "retrieval_acc": retrieval_acc.detach(),
        "log_rms_loss": log_rms_loss.detach(),
        "moe_load_balance": moe_load_balance.detach(),
        "moe_sparsity": moe_sparsity.detach(),
        "moe_route_entropy": moe_route_entropy.detach(),
        "moe_cluster": moe_cluster.detach(),
        "moe_gate_mean": out.get("moe_channel_gate_mean", pred.new_tensor(0.0)).detach(),
        "moe_usage_min": out.get("moe_usage_min", pred.new_tensor(0.0)).detach(),
        "moe_usage_max": out.get("moe_usage_max", pred.new_tensor(0.0)).detach(),
        "moe_active_channels": out.get("moe_active_channels", pred.new_tensor(0.0)).detach(),
    }


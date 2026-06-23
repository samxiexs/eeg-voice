from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_contrastive(embed: torch.Tensor, label_idx: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    z = F.normalize(embed, dim=-1)
    sim = (z @ z.transpose(0, 1)) / max(float(temperature), 1e-6)
    b = z.shape[0]
    eye = torch.eye(b, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)
    pos = (label_idx.view(-1, 1) == label_idx.view(1, -1)) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_count = pos.sum(dim=1)
    valid = pos_count > 0
    if valid.sum() == 0:
        return embed.new_tensor(0.0)
    return -((log_prob * pos.float()).sum(dim=1)[valid] / pos_count[valid].clamp_min(1)).mean()


def compute_losses(
    out: dict[str, torch.Tensor],
    target_seq: torch.Tensor,
    label_idx: torch.Tensor,
    subject_idx: torch.Tensor,
    content_proto: torch.Tensor,
    subject_proto: torch.Tensor,
    target_log_rms: torch.Tensor,
    lambda_recon_cos: float = 1.0,
    lambda_recon_mse: float = 0.5,
    lambda_content_ce: float = 0.5,
    lambda_subject_ce: float = 0.05,
    lambda_supcon: float = 0.5,
    lambda_proto: float = 0.25,
    lambda_subject_proto: float = 0.25,
    lambda_log_rms: float = 0.2,
    lambda_std: float = 0.0,
    lambda_router_balance: float = 0.01,
    supcon_temperature: float = 0.1,
) -> dict[str, torch.Tensor]:
    pred = out["pred_latent"]
    recon_cos = 1.0 - F.cosine_similarity(pred, target_seq, dim=-1).mean()
    recon_mse = F.mse_loss(pred, target_seq)
    content_ce = F.cross_entropy(out["content_logits"], label_idx.long())
    subject_ce = F.cross_entropy(out["subject_logits"], subject_idx.long())
    content_acc = (out["content_logits"].argmax(dim=-1) == label_idx.long()).float().mean()
    subject_acc = (out["subject_logits"].argmax(dim=-1) == subject_idx.long()).float().mean()
    supcon = supervised_contrastive(out["content_embed"], label_idx, temperature=supcon_temperature)
    proto_cos = 1.0 - F.cosine_similarity(
        F.normalize(out["content_embed"], dim=-1),
        F.normalize(content_proto, dim=-1),
        dim=-1,
    ).mean()
    subject_proto_loss = 1.0 - F.cosine_similarity(
        F.normalize(out["speaker_proto_pred"], dim=-1),
        F.normalize(subject_proto, dim=-1),
        dim=-1,
    ).mean()
    log_rms_loss = F.mse_loss(out["pred_log_rms"], target_log_rms.float())

    pred_std = pred.reshape(-1, pred.shape[-1]).std(dim=0)
    tgt_std = target_seq.reshape(-1, target_seq.shape[-1]).std(dim=0)
    std_ratio = (pred_std / tgt_std.clamp_min(1e-6)).median().detach()
    std_match = F.l1_loss(pred_std, tgt_std) if lambda_std > 0.0 else pred.new_tensor(0.0)

    router_balance = pred.new_tensor(0.0)
    if out["router_probs"].shape[-1] > 1:
        mean_probs = out["router_probs"].mean(dim=0)
        uniform = torch.full_like(mean_probs, 1.0 / mean_probs.numel())
        router_balance = F.mse_loss(mean_probs, uniform)

    total = (
        lambda_recon_cos * recon_cos
        + lambda_recon_mse * recon_mse
        + lambda_content_ce * content_ce
        + lambda_subject_ce * subject_ce
        + lambda_supcon * supcon
        + lambda_proto * proto_cos
        + lambda_subject_proto * subject_proto_loss
        + lambda_log_rms * log_rms_loss
        + lambda_std * std_match
        + lambda_router_balance * router_balance
    )
    return {
        "total": total,
        "recon_cos": recon_cos.detach(),
        "recon_mse": recon_mse.detach(),
        "content_ce": content_ce.detach(),
        "subject_ce": subject_ce.detach(),
        "content_acc": content_acc.detach(),
        "subject_acc": subject_acc.detach(),
        "supcon": supcon.detach(),
        "proto_cos": proto_cos.detach(),
        "subject_proto": subject_proto_loss.detach(),
        "log_rms_loss": log_rms_loss.detach(),
        "std_match": std_match.detach(),
        "std_ratio": std_ratio,
        "router_balance": router_balance.detach(),
    }


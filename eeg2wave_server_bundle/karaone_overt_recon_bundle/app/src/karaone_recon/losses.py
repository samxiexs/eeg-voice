from __future__ import annotations

import torch
import torch.nn.functional as F


def clip_alignment(eeg_embed: torch.Tensor, audio_embed: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric cross-modal InfoNCE between EEG and audio embeddings.

    The recipe from Defossez et al. 2022 (non-invasive speech decoding) and the
    EEG visual-decoding line (NICE/ATM/UBP): instead of only regressing the
    target, pull each trial's EEG embedding toward its own audio embedding and
    push it away from other trials' audio in the batch (in-batch negatives). This
    teaches an EEG<->speech correspondence and fights the mean-seeking blur of
    pure MSE. Only the diagonal (same trial) counts as positive.
    """
    if eeg_embed.shape[0] < 2:
        return eeg_embed.new_tensor(0.0)
    e = F.normalize(eeg_embed, dim=-1)
    a = F.normalize(audio_embed, dim=-1)
    logits = (e @ a.t()) / max(float(temperature), 1e-6)  # [B, B]
    targets = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


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
    content_proto: torch.Tensor,
    target_log_rms: torch.Tensor,
    lambda_recon_cos: float = 1.0,
    lambda_recon_mse: float = 0.5,
    lambda_content_ce: float = 0.5,
    lambda_supcon: float = 0.5,
    lambda_proto: float = 0.25,
    lambda_log_rms: float = 0.2,
    lambda_std: float = 0.0,
    lambda_router_balance: float = 0.01,
    lambda_channel_balance: float = 0.01,
    lambda_clip: float = 0.5,
    supcon_temperature: float = 0.1,
    clip_temperature: float = 0.07,
) -> dict[str, torch.Tensor]:
    # All supervision here is EEG-derived or keyed by the spoken phoneme `label`
    # (the content we want to decode). No subject-ID supervision exists.
    pred = out["pred_latent"]
    recon_cos = 1.0 - F.cosine_similarity(pred, target_seq, dim=-1).mean()
    recon_mse = F.mse_loss(pred, target_seq)
    content_ce = F.cross_entropy(out["content_logits"], label_idx.long())
    content_acc = (out["content_logits"].argmax(dim=-1) == label_idx.long()).float().mean()
    supcon = supervised_contrastive(out["content_embed"], label_idx, temperature=supcon_temperature)
    proto_cos = 1.0 - F.cosine_similarity(
        F.normalize(out["content_embed"], dim=-1),
        F.normalize(content_proto, dim=-1),
        dim=-1,
    ).mean()
    log_rms_loss = F.mse_loss(out["pred_log_rms"], target_log_rms.float())

    # Cross-modal alignment: EEG utterance embedding vs the audio-latent summary
    # (mean over time of the normalized EnCodec target). Audio side is frozen.
    clip_loss = clip_alignment(out["clip_embed"], target_seq.mean(dim=1), temperature=clip_temperature)

    pred_std = pred.reshape(-1, pred.shape[-1]).std(dim=0)
    tgt_std = target_seq.reshape(-1, target_seq.shape[-1]).std(dim=0)
    std_ratio = (pred_std / tgt_std.clamp_min(1e-6)).median().detach()
    std_match = F.l1_loss(pred_std, tgt_std) if lambda_std > 0.0 else pred.new_tensor(0.0)

    router_balance = pred.new_tensor(0.0)
    if out["router_probs"].shape[-1] > 1:
        mean_probs = out["router_probs"].mean(dim=0)
        uniform = torch.full_like(mean_probs, 1.0 / mean_probs.numel())
        router_balance = F.mse_loss(mean_probs, uniform)

    # Encoder channel-MoE load balance (keeps channel clusters from collapsing).
    channel_balance = out.get("channel_balance", pred.new_tensor(0.0))

    total = (
        lambda_recon_cos * recon_cos
        + lambda_recon_mse * recon_mse
        + lambda_content_ce * content_ce
        + lambda_supcon * supcon
        + lambda_proto * proto_cos
        + lambda_log_rms * log_rms_loss
        + lambda_std * std_match
        + lambda_router_balance * router_balance
        + lambda_channel_balance * channel_balance
        + lambda_clip * clip_loss
    )
    return {
        "total": total,
        "recon_cos": recon_cos.detach(),
        "recon_mse": recon_mse.detach(),
        "content_ce": content_ce.detach(),
        "content_acc": content_acc.detach(),
        "supcon": supcon.detach(),
        "proto_cos": proto_cos.detach(),
        "clip_loss": clip_loss.detach(),
        "log_rms_loss": log_rms_loss.detach(),
        "std_match": std_match.detach(),
        "std_ratio": std_ratio,
        "router_balance": router_balance.detach(),
        "channel_balance": channel_balance.detach() if torch.is_tensor(channel_balance) else pred.new_tensor(0.0),
    }


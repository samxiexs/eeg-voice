"""Factored training objective.

  - supervised contrastive on content_embed: positives = SAME label in the batch
    (correctly uses "10 reps / 21 speakers share one content"; no false negatives);
  - content classification CE (16-way, read-out + class separation);
  - content-prototype matching (content_embed -> speaker-independent label prototype);
  - reconstruction: predicted EnCodec latent -> cell target latent (cosine + MSE).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_contrastive(embed: torch.Tensor, label_idx: torch.Tensor,
                           temperature: float = 0.1) -> torch.Tensor:
    """SupCon: positives = same label_idx. Pulls all same-content samples together."""
    z = F.normalize(embed, dim=-1)
    sim = (z @ z.transpose(0, 1)) / max(float(temperature), 1e-6)        # [B, B]
    b = z.shape[0]
    eye = torch.eye(b, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)                                     # drop self
    pos = (label_idx.view(-1, 1) == label_idx.view(1, -1)) & ~eye        # [B, B] positives
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_cnt = pos.sum(1)
    valid = pos_cnt > 0
    if valid.sum() == 0:
        return embed.new_tensor(0.0)
    mean_pos = (log_prob * pos.float()).sum(1)[valid] / pos_cnt[valid].clamp_min(1)
    return -mean_pos.mean()


def compute_factored_losses(
    out: dict[str, torch.Tensor],
    target_seq: torch.Tensor,          # [B, T, D] normalised cell latent
    label_idx: torch.Tensor,
    content_proto: torch.Tensor,       # [B, D] speaker-independent label prototype
    speaker_proto: torch.Tensor | None = None,   # [B, D] audio voice prototype
    subject_idx: torch.Tensor | None = None,     # [B] for the disentanglement adversary
    target_log_rms: torch.Tensor | None = None,  # [B] decoded-wav target log-RMS (energy)
    lambda_supcon: float = 1.0,
    lambda_content_ce: float = 0.5,
    lambda_proto: float = 0.5,
    lambda_recon_cos: float = 1.0,
    lambda_recon_mse: float = 0.25,    # v2: lowered from 1.0 to reduce mean-averaging
    lambda_speaker: float = 0.5,
    lambda_adv: float = 0.3,
    lambda_log_rms: float = 0.2,       # v2: energy/loudness supervision
    lambda_std: float = 0.0,           # v2: optional anti-collapse std-match (OFF by default)
    supcon_temperature: float = 0.1,
) -> dict[str, torch.Tensor]:
    pred = out["pred_latent"]

    # reconstruction (frame-wise)
    recon_cos = 1.0 - F.cosine_similarity(pred, target_seq, dim=-1).mean()
    recon_mse = F.mse_loss(pred, target_seq)

    # --- collapse diagnostics (per-dim std over B,T) ---
    pred_std = pred.reshape(-1, pred.shape[-1]).std(dim=0)               # [D]
    tgt_std = target_seq.reshape(-1, target_seq.shape[-1]).std(dim=0)    # [D]
    std_ratio = (pred_std / tgt_std.clamp_min(1e-6)).median().detach()

    # content
    supcon = supervised_contrastive(out["content_embed"], label_idx, supcon_temperature)
    content_ce = F.cross_entropy(out["content_logits"], label_idx.long())
    content_acc = (out["content_logits"].argmax(-1) == label_idx.long()).float().mean()
    proto_cos = 1.0 - F.cosine_similarity(
        F.normalize(out["content_embed"], dim=-1), F.normalize(content_proto, dim=-1), dim=-1
    ).mean()

    total = (lambda_recon_cos * recon_cos + lambda_recon_mse * recon_mse
             + lambda_supcon * supcon + lambda_content_ce * content_ce + lambda_proto * proto_cos)

    # SPEAKER grounding: learned speaker embedding -> audio voice prototype
    speaker_loss = pred.new_tensor(0.0)
    if speaker_proto is not None and "speaker_proto_pred" in out:
        speaker_loss = 1.0 - F.cosine_similarity(
            F.normalize(out["speaker_proto_pred"], dim=-1), F.normalize(speaker_proto, dim=-1), dim=-1
        ).mean()
        total = total + lambda_speaker * speaker_loss

    # ADVERSARY: content should NOT predict subject (GRL flips the gradient).
    adv_ce = pred.new_tensor(0.0); adv_acc = pred.new_tensor(0.0)
    if subject_idx is not None and "subject_adv_logits" in out:
        adv_ce = F.cross_entropy(out["subject_adv_logits"], subject_idx.long())
        adv_acc = (out["subject_adv_logits"].argmax(-1) == subject_idx.long()).float().mean()
        total = total + lambda_adv * adv_ce

    # ENERGY: supervise predicted decoded-wav log-RMS (loudness).
    log_rms_loss = pred.new_tensor(0.0)
    if target_log_rms is not None and "pred_log_rms" in out:
        log_rms_loss = F.mse_loss(out["pred_log_rms"], target_log_rms.float())
        total = total + lambda_log_rms * log_rms_loss

    # OPTIONAL anti-collapse: match per-dim std of pred to target (off unless lambda_std>0).
    # NOTE: this can game the std-ratio metric, so it is diagnostic-secondary, not a success gate.
    std_match = pred.new_tensor(0.0)
    if lambda_std > 0.0:
        std_match = F.l1_loss(pred_std, tgt_std)
        total = total + lambda_std * std_match

    return {
        "total": total,
        "recon_cos": recon_cos.detach(),
        "recon_mse": recon_mse.detach(),
        "supcon": supcon.detach(),
        "content_ce": content_ce.detach(),
        "content_acc": content_acc.detach(),
        "proto_cos": proto_cos.detach(),
        "speaker_loss": speaker_loss.detach(),
        "adv_ce": adv_ce.detach(),
        "adv_subject_acc": adv_acc.detach(),  # ↓ over training = content losing subject info
        "log_rms_loss": log_rms_loss.detach(),
        "std_match": std_match.detach(),
        "std_ratio": std_ratio,               # diagnostic: pred/target std (→1 = no collapse)
    }

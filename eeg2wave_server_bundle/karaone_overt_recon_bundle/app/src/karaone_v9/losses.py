from __future__ import annotations

import torch
import torch.nn.functional as F

from src.karaone_v9.model import resize_sequence


def compute_v9_pretrain_losses(
    out: dict[str, torch.Tensor],
    *,
    lambda_masked_recon: float = 1.0,
    lambda_variance: float = 0.05,
) -> dict[str, torch.Tensor]:
    mask = out["patch_mask"].bool()
    pred = out["patch_recon"]
    target = out["patch_tokens_target"]
    if bool(mask.any()):
        recon = F.smooth_l1_loss(pred[mask], target[mask])
    else:
        recon = F.smooth_l1_loss(pred, target)
    variance, covariance = vicreg_variance_covariance(out["pooled"])
    total = float(lambda_masked_recon) * recon + float(lambda_variance) * variance
    return {
        "total": total,
        "masked_recon": recon,
        "vicreg_variance": variance,
        "vicreg_covariance": covariance,
    }


def compute_v9_alignment_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    lambda_seq_ot: float = 1.0,
    lambda_seq_cos: float = 0.5,
    lambda_global_nce: float = 0.5,
    lambda_soft_nce: float = 0.5,
    lambda_semantic_token: float = 0.3,
    lambda_ctc: float = 0.2,
    lambda_prompt: float = 0.1,
    lambda_prosody: float = 0.5,
    lambda_subject_adv: float = 0.1,
    lambda_coral: float = 0.05,
    lambda_group_dro: float = 0.2,
    lambda_variance: float = 0.1,
    nce_temperature: float = 0.07,
    soft_target_temperature: float = 0.08,
) -> dict[str, torch.Tensor]:
    pred_seq = out["pred_semantic_seq"]
    target_seq = batch["semantic_seq"].to(pred_seq.device, pred_seq.dtype)
    target_summary = batch["semantic_summary"].to(pred_seq.device, pred_seq.dtype)
    label_idx = batch["label_idx"].to(pred_seq.device)
    subject_idx = batch["subject_idx"].to(pred_seq.device)

    target_for_pred = resize_sequence(target_seq, pred_seq.shape[1])
    seq_cos_per = sequence_cosine_loss_per_sample(pred_seq, target_for_pred)
    seq_cos = seq_cos_per.mean()
    seq_ot = monotonic_soft_ot_loss(pred_seq, target_seq, temperature=0.08, monotonic_sigma=0.25)
    global_nce = symmetric_infonce(out["pred_semantic_summary"], target_summary, temperature=nce_temperature)
    soft_nce = soft_positive_infonce(
        out["pred_semantic_summary"],
        target_summary,
        pred_temperature=nce_temperature,
        target_temperature=soft_target_temperature,
    )
    semantic_token = semantic_token_ce_loss(
        out["semantic_token_logits"],
        batch["semantic_token_targets"].to(pred_seq.device),
        batch["semantic_token_mask"].to(pred_seq.device),
    )
    prompt_ctc = prompt_ctc_loss(out["prompt_ctc_logits"], label_idx, out["token_valid_mask"])
    prompt_ce = F.cross_entropy(out["prompt_logits"], label_idx)
    prosody = prosody_loss(out, batch)
    subject_adv = F.cross_entropy(out["subject_logits"], subject_idx)
    coral = coral_subject_loss(out["pooled"], subject_idx)
    group_dro = group_dro_loss(seq_cos_per, subject_idx)
    variance, covariance = vicreg_variance_covariance(out["pred_semantic_summary"])

    total = (
        float(lambda_seq_ot) * seq_ot
        + float(lambda_seq_cos) * seq_cos
        + float(lambda_global_nce) * global_nce
        + float(lambda_soft_nce) * soft_nce
        + float(lambda_semantic_token) * semantic_token
        + float(lambda_ctc) * prompt_ctc
        + float(lambda_prompt) * prompt_ce
        + float(lambda_prosody) * prosody
        + float(lambda_subject_adv) * subject_adv
        + float(lambda_coral) * coral
        + float(lambda_group_dro) * group_dro
        + float(lambda_variance) * variance
    )
    return {
        "total": total,
        "seq_ot": seq_ot,
        "seq_cos": seq_cos,
        "global_nce": global_nce,
        "soft_nce": soft_nce,
        "semantic_token_ce": semantic_token,
        "prompt_ctc": prompt_ctc,
        "prompt_ce": prompt_ce,
        "prosody": prosody,
        "subject_adv": subject_adv,
        "coral": coral,
        "group_dro": group_dro,
        "vicreg_variance": variance,
        "vicreg_covariance": covariance,
    }


def compute_v9_transport_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    lambda_flow: float = 1.0,
    lambda_condition_semantic: float = 0.2,
) -> dict[str, torch.Tensor]:
    flow = out.get("transport_flow_loss")
    if flow is None:
        raise KeyError("Model output does not contain transport_flow_loss; pass codec_seq to the model")
    target_sem = batch["semantic_seq"].to(out["pred_semantic_seq"].device, out["pred_semantic_seq"].dtype)
    semantic_guard = sequence_cosine_loss_per_sample(out["pred_semantic_seq"], resize_sequence(target_sem, out["pred_semantic_seq"].shape[1])).mean()
    total = float(lambda_flow) * flow + float(lambda_condition_semantic) * semantic_guard
    return {
        "total": total,
        "flow": flow,
        "condition_semantic": semantic_guard,
    }


def sequence_cosine_loss_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.shape[1] != pred.shape[1]:
        target = resize_sequence(target, pred.shape[1])
    pred_n = F.normalize(pred, dim=-1)
    tgt_n = F.normalize(target, dim=-1)
    return 1.0 - (pred_n * tgt_n).sum(dim=-1).mean(dim=-1)


def monotonic_soft_ot_loss(pred: torch.Tensor, target: torch.Tensor, *, temperature: float = 0.08, monotonic_sigma: float = 0.25) -> torch.Tensor:
    pred_n = F.normalize(pred, dim=-1)
    tgt_n = F.normalize(target, dim=-1)
    cost = 1.0 - torch.einsum("btd,bsd->bts", pred_n, tgt_n)
    t_pos = torch.linspace(0.0, 1.0, pred.shape[1], device=pred.device, dtype=pred.dtype).view(1, -1, 1)
    s_pos = torch.linspace(0.0, 1.0, target.shape[1], device=pred.device, dtype=pred.dtype).view(1, 1, -1)
    penalty = (t_pos - s_pos).pow(2) / (2.0 * max(float(monotonic_sigma), 1e-3) ** 2)
    logits = -cost / max(float(temperature), 1e-4) - penalty
    e2s = (torch.softmax(logits, dim=-1) * cost).sum(dim=-1).mean()
    s2e = (torch.softmax(logits.transpose(1, 2), dim=-1) * cost.transpose(1, 2)).sum(dim=-1).mean()
    return 0.5 * (e2s + s2e)


def symmetric_infonce(eeg: torch.Tensor, speech: torch.Tensor, *, temperature: float = 0.07) -> torch.Tensor:
    if eeg.shape[0] < 2:
        return eeg.new_tensor(0.0)
    e = F.normalize(eeg, dim=-1)
    s = F.normalize(speech, dim=-1)
    logits = e @ s.T / max(float(temperature), 1e-4)
    target = torch.arange(logits.shape[0], device=eeg.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def soft_positive_infonce(
    eeg: torch.Tensor,
    speech: torch.Tensor,
    *,
    pred_temperature: float = 0.07,
    target_temperature: float = 0.08,
) -> torch.Tensor:
    if eeg.shape[0] < 2:
        return eeg.new_tensor(0.0)
    e = F.normalize(eeg, dim=-1)
    s = F.normalize(speech, dim=-1)
    with torch.no_grad():
        target_dist = torch.softmax((s @ s.T) / max(float(target_temperature), 1e-4), dim=-1)
    logits_e2s = (e @ s.T) / max(float(pred_temperature), 1e-4)
    logits_s2e = (s @ e.T) / max(float(pred_temperature), 1e-4)
    e2s = -(target_dist * torch.log_softmax(logits_e2s, dim=-1)).sum(dim=-1).mean()
    s2e = -(target_dist.T * torch.log_softmax(logits_s2e, dim=-1)).sum(dim=-1).mean()
    return 0.5 * (e2s + s2e)


def semantic_token_ce_loss(logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] <= 1 or mask.sum() <= 0:
        return logits.new_tensor(0.0)
    target_tokens, target_mask = _resample_tokens(tokens.to(logits.device), mask.to(logits.device), logits.shape[1])
    loss = F.cross_entropy(logits.transpose(1, 2), target_tokens.clamp(0, logits.shape[-1] - 1), reduction="none")
    return (loss * target_mask).sum() / target_mask.sum().clamp_min(1.0)


def prompt_ctc_loss(ctc_logits: torch.Tensor, label_idx: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if ctc_logits.shape[1] < 1:
        return ctc_logits.new_tensor(0.0)
    log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
    input_lengths = valid_mask.long().sum(dim=1).clamp_min(1).to(torch.long)
    targets = label_idx.to(torch.long).view(-1)
    target_lengths = torch.ones_like(targets, dtype=torch.long)
    blank = ctc_logits.shape[-1] - 1
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=blank, zero_infinity=True)


def prosody_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    active = _resize_1d(batch["prosody_active"].to(out["prosody_active_logits"].device), out["prosody_active_logits"].shape[1])
    energy = _resize_1d(batch["prosody_energy"].to(out["prosody_energy"].device), out["prosody_energy"].shape[1])
    active_loss = F.binary_cross_entropy_with_logits(out["prosody_active_logits"], active)
    energy_l1 = F.smooth_l1_loss(out["prosody_energy"], energy)
    duration = batch["prosody_duration"].to(out["prosody_duration"].device, out["prosody_duration"].dtype)
    onset = batch["prosody_onset"].to(out["prosody_onset"].device, out["prosody_onset"].dtype)
    duration_l1 = F.smooth_l1_loss(out["prosody_duration"], duration)
    onset_l1 = F.smooth_l1_loss(out["prosody_onset"], onset)
    corr = correlation_loss_1d(torch.sigmoid(out["prosody_active_logits"]) * out["prosody_energy"], active * energy)
    return active_loss + energy_l1 + duration_l1 + onset_l1 + 0.2 * corr


def correlation_loss_1d(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    corr = (pred * target).sum(dim=1) / (pred.norm(dim=1) * target.norm(dim=1) + 1e-8)
    return (1.0 - corr).mean()


def vicreg_variance_covariance(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
    var_loss = F.relu(1.0 - std).mean()
    if z.shape[0] < 2:
        cov_loss = z.new_tensor(0.0)
    else:
        cov = (z.T @ z) / float(z.shape[0] - 1)
        cov_loss = _off_diagonal(cov).pow(2).sum() / max(float(z.shape[1]), 1.0)
    return var_loss, cov_loss


def coral_subject_loss(z: torch.Tensor, subject_idx: torch.Tensor) -> torch.Tensor:
    unique = torch.unique(subject_idx)
    if unique.numel() < 2 or z.shape[0] < 3:
        return z.new_tensor(0.0)
    zc = z - z.mean(dim=0, keepdim=True)
    global_cov = (zc.T @ zc) / float(max(z.shape[0] - 1, 1))
    losses = []
    for subject in unique:
        mask = subject_idx == subject
        if int(mask.sum().item()) < 2:
            continue
        group = z[mask]
        gc = group - group.mean(dim=0, keepdim=True)
        cov = (gc.T @ gc) / float(max(group.shape[0] - 1, 1))
        mean_loss = F.mse_loss(group.mean(dim=0), z.mean(dim=0))
        cov_loss = F.mse_loss(cov, global_cov)
        losses.append(mean_loss + cov_loss)
    if not losses:
        return z.new_tensor(0.0)
    return torch.stack(losses).mean()


def group_dro_loss(per_sample_loss: torch.Tensor, subject_idx: torch.Tensor, eta: float = 8.0) -> torch.Tensor:
    groups = []
    for subject in torch.unique(subject_idx):
        mask = subject_idx == subject
        if bool(mask.any()):
            groups.append(per_sample_loss[mask].mean())
    if not groups:
        return per_sample_loss.mean()
    group_losses = torch.stack(groups)
    weights = torch.softmax(float(eta) * group_losses.detach(), dim=0)
    return (weights * group_losses).sum()


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n < 2 or m < 2:
        return x.new_zeros(0)
    return x.flatten()[:-1].view(n - 1, m + 1)[:, 1:].flatten()


def _resize_1d(x: torch.Tensor, steps: int) -> torch.Tensor:
    if x.shape[1] == int(steps):
        return x.float()
    return F.interpolate(x.float().unsqueeze(1), size=int(steps), mode="linear", align_corners=False).squeeze(1)


def _resample_tokens(tokens: torch.Tensor, mask: torch.Tensor, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.shape[1] == int(steps):
        return tokens.long(), mask.float()
    idx = torch.linspace(0, tokens.shape[1] - 1, int(steps), device=tokens.device).round().long()
    return tokens[:, idx].long(), mask[:, idx].float()

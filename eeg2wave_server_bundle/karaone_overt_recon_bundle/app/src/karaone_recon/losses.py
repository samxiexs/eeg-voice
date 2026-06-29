from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .alignment import shift_sequence_torch


def _dtw_path(cost: np.ndarray, band: float) -> tuple[np.ndarray, np.ndarray]:
    """Banded (Sakoe-Chiba) DTW backtrace. cost[T,T] -> aligned index arrays (pi, pj)."""
    n, m = cost.shape
    w = max(int(band * max(n, m)), abs(n - m) + 1)
    inf = np.float32(1e18)
    acc = np.full((n + 1, m + 1), inf, dtype=np.float32)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        j0 = max(1, i - w)
        j1 = min(m, i + w)
        for j in range(j0, j1 + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    # backtrace
    i, j = n, m
    pi, pj = [], []
    while i > 0 and j > 0:
        pi.append(i - 1)
        pj.append(j - 1)
        step = min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
        if step == acc[i - 1, j - 1]:
            i, j = i - 1, j - 1
        elif step == acc[i - 1, j]:
            i -= 1
        else:
            j -= 1
    return np.asarray(pi[::-1], dtype=np.int64), np.asarray(pj[::-1], dtype=np.int64)


def dtw_recon_loss(pred: torch.Tensor, target: torch.Tensor, band: float = 0.2) -> torch.Tensor:
    """DTW-aligned L1 between pred and target sequences [B,T,D].

    The warping path is found on a *detached* L2 cost (so it is not differentiated),
    then L1 is computed on the path-aligned frames with gradient flowing through
    `pred`. This makes the loss invariant to the cross-trial onset/rate jitter that
    breaks naive frame-wise regression (NeuroTalk-style alignment)."""
    b = pred.shape[0]
    pred_np = pred.detach().cpu().numpy()
    tgt_np = target.detach().cpu().numpy()
    total = pred.new_tensor(0.0)
    for k in range(b):
        # L2 cost matrix between frames (detached)
        diff = pred_np[k][:, None, :] - tgt_np[k][None, :, :]
        cost = np.sqrt((diff * diff).sum(-1) + 1e-8).astype(np.float32)
        pi, pj = _dtw_path(cost, band)
        pi_t = torch.from_numpy(pi).to(pred.device)
        pj_t = torch.from_numpy(pj).to(pred.device)
        total = total + (pred[k][pi_t] - target[k][pj_t]).abs().mean()
    return total / max(b, 1)


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


def energy_envelope_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - Pearson correlation of the per-frame energy envelope (averaged over batch).

    Directly fights the documented energy collapse (recon RMS ~= half of original,
    decoupled from content): the predicted temporal magnitude contour is pushed to
    track the target's, rather than flattening to a constant low-energy hum. Energy
    is the mean square over the feature axis at each frame, so it is a magnitude
    envelope in the (z-scored) mel/latent target space."""
    pe = pred.pow(2).mean(dim=-1)  # [B, T]
    te = target.pow(2).mean(dim=-1)
    pe = pe - pe.mean(dim=1, keepdim=True)
    te = te - te.mean(dim=1, keepdim=True)
    num = (pe * te).sum(dim=1)
    den = pe.norm(dim=1) * te.norm(dim=1) + 1e-8
    corr = num / den
    return (1.0 - corr).mean()


def multiscale_temporal_l1(pred: torch.Tensor, target: torch.Tensor, scales: tuple[int, ...] = (1, 2, 4)) -> torch.Tensor:
    """Multi-resolution L1 in the target domain (time-averaged at {1,2,4}x).

    A literal multi-resolution STFT loss needs a rendered waveform (deferred to the
    flow/vocoder-in-loop round). In the mel/latent target domain this is the
    equivalent anti-oversmoothing term: matching coarse (downsampled) contours as
    well as fine frames keeps the prediction from blurring into the per-frame mean."""
    total = pred.new_tensor(0.0)
    for s in scales:
        if s <= 1:
            p, t = pred, target
        else:
            p = F.avg_pool1d(pred.transpose(1, 2), kernel_size=int(s), ceil_mode=True).transpose(1, 2)
            t = F.avg_pool1d(target.transpose(1, 2), kernel_size=int(s), ceil_mode=True).transpose(1, 2)
        total = total + F.l1_loss(p, t)
    return total / max(len(scales), 1)


def hubert_aux_loss(pred_hubert: torch.Tensor, hubert_seq: torch.Tensor) -> torch.Tensor:
    """SmoothL1 + (1 - cos) between EEG-predicted HuBERT features and the GT HuBERT
    sequence. A content-bearing, low-SNR-friendly auxiliary target (wav2vec2/HuBERT
    semantic space, Defossez 2022 / AudioLM line) that complements the acoustic
    (mel/latent) regression which the model actually renders from."""
    smooth = F.smooth_l1_loss(pred_hubert, hubert_seq)
    cos = 1.0 - F.cosine_similarity(pred_hubert, hubert_seq, dim=-1).mean()
    return smooth + cos


def prompt_ctc_loss(ctc_logits: torch.Tensor, label_idx: torch.Tensor) -> torch.Tensor:
    """CTC over the prompt label as a one-token sequence.

    KaraOne labels are short prompted phoneme/word classes rather than full
    transcripts. This still gives the sequence encoder a duration-agnostic
    content objective: emit the prompt token somewhere in the acoustic frame
    sequence, with blank elsewhere.
    """
    if ctc_logits.shape[1] < 1:
        return ctc_logits.new_tensor(0.0)
    log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # [T, B, V]
    b, t, _ = ctc_logits.shape
    targets = label_idx.long().clamp_min(0) + 1  # blank=0, labels start at 1
    input_lengths = torch.full((b,), int(t), device=ctc_logits.device, dtype=torch.long)
    target_lengths = torch.ones((b,), device=ctc_logits.device, dtype=torch.long)
    if ctc_logits.device.type == "mps":
        loss = F.ctc_loss(
            log_probs.cpu(),
            targets.cpu(),
            input_lengths.cpu(),
            target_lengths.cpu(),
            blank=0,
            zero_infinity=True,
        )
        return loss.to(ctc_logits.device)
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, zero_infinity=True)


def speech_token_ctc_loss(
    token_logits: torch.Tensor,
    token_targets: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """CTC over waveform-derived HuBERT token sequences, not KaraOne prompt labels.

    The semantic-token head emits K HuBERT k-means token logits. CTC needs an
    extra blank class, so we append a learned-neutral zero logit column and use
    blank=K. Targets are the masked HuBERT token sequence with adjacent repeats
    collapsed, which keeps the target length feasible for CTC.
    """
    if token_logits.ndim != 3 or token_logits.shape[0] == 0:
        return token_logits.new_tensor(0.0)
    b, t, k = token_logits.shape
    blank = int(k)
    blank_logits = token_logits.new_zeros((b, t, 1))
    logits = torch.cat([token_logits, blank_logits], dim=-1)
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
    input_lengths = torch.full((b,), int(t), device=token_logits.device, dtype=torch.long)
    pieces: list[torch.Tensor] = []
    lengths: list[int] = []
    for i in range(b):
        row = token_targets[i].long().clamp(min=0, max=max(k - 1, 0))
        if token_mask is not None:
            row = row[token_mask[i].to(device=row.device) > 0]
        if row.numel() > 1:
            keep = torch.ones_like(row, dtype=torch.bool)
            keep[1:] = row[1:] != row[:-1]
            row = row[keep]
        row = row[:t]
        if row.numel() == 0:
            row = torch.full((1,), blank, device=token_logits.device, dtype=torch.long)
        pieces.append(row)
        lengths.append(int(row.numel()))
    targets = torch.cat(pieces, dim=0)
    target_lengths = torch.tensor(lengths, device=token_logits.device, dtype=torch.long)
    if token_logits.device.type == "mps":
        loss = F.ctc_loss(
            log_probs.cpu(),
            targets.cpu(),
            input_lengths.cpu(),
            target_lengths.cpu(),
            blank=blank,
            zero_infinity=True,
        )
        return loss.to(token_logits.device)
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=blank, zero_infinity=True)


def frame_log_energy_loss(pred_frame_log_energy: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_log_energy = torch.log(target.pow(2).mean(dim=-1).clamp_min(1e-8))
    return F.smooth_l1_loss(pred_frame_log_energy, target_log_energy)


def voiced_region_rms_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match target-active-frame RMS, not only whole-utterance RMS."""
    target_energy = target.pow(2).mean(dim=-1)
    mean = target_energy.mean(dim=1, keepdim=True)
    std = target_energy.std(dim=1, keepdim=True, unbiased=False)
    peak = target_energy.max(dim=1, keepdim=True).values
    thresh = torch.maximum(mean + 0.5 * std, 0.1 * peak)
    mask = (target_energy >= thresh).detach().to(pred.dtype)
    mask = torch.where(mask.sum(dim=1, keepdim=True) > 0, mask, torch.ones_like(mask))
    pred_rms = torch.sqrt(((pred.pow(2).mean(dim=-1) * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)).clamp_min(1e-8))
    target_rms = torch.sqrt(((target_energy * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)).clamp_min(1e-8))
    return F.smooth_l1_loss(torch.log(pred_rms), torch.log(target_rms))


def _feature_stats(
    seq: torch.Tensor,
    target_mean: torch.Tensor | None,
    target_std: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if target_mean is None or target_std is None:
        return None
    mean = target_mean.to(device=seq.device, dtype=seq.dtype).reshape(1, 1, -1)
    std = target_std.to(device=seq.device, dtype=seq.dtype).reshape(1, 1, -1).clamp_min(1e-6)
    if mean.shape[-1] != seq.shape[-1] or std.shape[-1] != seq.shape[-1]:
        return None
    return mean, std


def raw_mel_energy(
    seq: torch.Tensor,
    target_mean: torch.Tensor | None,
    target_std: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    stats = _feature_stats(seq, target_mean, target_std)
    if stats is None:
        return None
    mean, std = stats
    raw_mel = seq * std + mean
    energy = torch.exp(raw_mel.clamp(min=-12.0, max=6.0)).mean(dim=-1).clamp_min(1e-8)
    return energy, torch.log(energy)


def active_mask_from_energy(energy: torch.Tensor) -> torch.Tensor:
    mean = energy.mean(dim=1, keepdim=True)
    std = energy.std(dim=1, keepdim=True, unbiased=False)
    peak = energy.max(dim=1, keepdim=True).values
    threshold = torch.maximum(mean + 0.5 * std, 0.1 * peak)
    mask = (energy >= threshold).to(energy.dtype)
    fallback = torch.zeros_like(mask).scatter_(1, energy.argmax(dim=1, keepdim=True), 1.0)
    return torch.where(mask.sum(dim=1, keepdim=True) > 0, mask, fallback).detach()


def _corr_loss_1d(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    if weight is not None:
        weight = weight.to(pred.dtype)
        denom = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        pred_mean = (pred * weight).sum(dim=1, keepdim=True) / denom
        target_mean = (target * weight).sum(dim=1, keepdim=True) / denom
        pred = (pred - pred_mean) * weight
        target = (target - target_mean) * weight
    else:
        pred = pred - pred.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=1, keepdim=True)
    corr = (pred * target).sum(dim=1) / (pred.norm(dim=1) * target.norm(dim=1) + 1e-8)
    return (1.0 - corr).mean()


def _corr_loss_flat(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    corr = (p * t).sum(dim=1) / (p.norm(dim=1) * t.norm(dim=1) + 1e-8)
    return (1.0 - corr).mean()


def _soft_dtw_1d(pred: torch.Tensor, target: torch.Tensor, gamma: float = 0.08) -> torch.Tensor:
    """Small differentiable Soft-DTW over active-core envelopes [B,T].

    v5 uses it as a *local speed tolerance* inside the already extracted active
    speech core. It is intentionally not applied to the original 2s timeline.
    """
    if pred.ndim != 2 or target.ndim != 2:
        return pred.new_tensor(0.0)
    b, n = pred.shape
    m = target.shape[1]
    if n <= 0 or m <= 0:
        return pred.new_tensor(0.0)
    gamma = max(float(gamma), 1e-4)
    cost = (pred.unsqueeze(2) - target.unsqueeze(1)).pow(2)
    inf = pred.new_tensor(1e8)
    r = pred.new_full((b, n + 1, m + 1), float("inf"))
    r[:, 0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            prev = torch.stack((r[:, i - 1, j], r[:, i, j - 1], r[:, i - 1, j - 1]), dim=-1)
            softmin = -gamma * torch.logsumexp(-prev / gamma, dim=-1)
            r[:, i, j] = cost[:, i - 1, j - 1] + softmin
    out = r[:, n, m]
    out = torch.where(torch.isfinite(out), out, inf)
    return (out / float(max(n + m, 1))).mean()


def _predicted_mel_envelope(
    pred: torch.Tensor,
    target_mean: torch.Tensor | None,
    target_std: torch.Tensor | None,
) -> torch.Tensor:
    payload = raw_mel_energy(pred, target_mean, target_std)
    if payload is None:
        return torch.sqrt(pred.pow(2).mean(dim=-1).clamp_min(1e-8))
    energy, _ = payload
    return torch.sqrt(energy.clamp_min(1e-8))


def temporal_elastic_core_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    out: dict[str, torch.Tensor],
    target_mean: torch.Tensor | None,
    target_std: torch.Tensor | None,
    active_envelope: torch.Tensor | None = None,
    active_duration_frames: torch.Tensor | None = None,
    active_rms: torch.Tensor | None = None,
    active_peak: torch.Tensor | None = None,
    softdtw_gamma: float = 0.08,
) -> dict[str, torch.Tensor]:
    core_shape = _corr_loss_flat(pred, target)
    core_cos = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    pred_env = _predicted_mel_envelope(pred, target_mean, target_std)
    if active_envelope is not None and active_envelope.shape[-1] == pred_env.shape[-1]:
        target_env = active_envelope.to(device=pred.device, dtype=pred.dtype)
    else:
        target_env = _predicted_mel_envelope(target, target_mean, target_std).detach()
    pred_env_z = (pred_env - pred_env.mean(dim=1, keepdim=True)) / pred_env.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    target_env_z = (target_env - target_env.mean(dim=1, keepdim=True)) / target_env.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    envelope_shape = _corr_loss_1d(pred_env_z, target_env_z)
    core_softdtw = _soft_dtw_1d(pred_env_z, target_env_z, gamma=softdtw_gamma)

    duration = pred.new_tensor(0.0)
    if active_duration_frames is not None:
        tgt_dur = active_duration_frames.to(device=pred.device, dtype=torch.float32).view(pred.shape[0])
        if "pred_duration_logits" in out:
            logits = out["pred_duration_logits"]
            bins = int(logits.shape[-1])
            tgt_bin = (tgt_dur.round().long() - 1).clamp(0, max(bins - 1, 0))
            duration = duration + F.cross_entropy(logits, tgt_bin)
        if "pred_duration_mu" in out:
            duration = duration + F.smooth_l1_loss(out["pred_duration_mu"].view_as(tgt_dur), tgt_dur)

    loudness = pred.new_tensor(0.0)
    if active_rms is not None and "pred_log_rms" in out:
        target_log_rms = torch.log(active_rms.to(device=pred.device, dtype=pred.dtype).view(-1).clamp_min(1e-8))
        loudness = loudness + F.smooth_l1_loss(out["pred_log_rms"].view_as(target_log_rms), target_log_rms)
    if active_peak is not None and "pred_log_peak" in out:
        target_log_peak = torch.log(active_peak.to(device=pred.device, dtype=pred.dtype).view(-1).clamp_min(1e-8))
        loudness = loudness + F.smooth_l1_loss(out["pred_log_peak"].view_as(target_log_peak), target_log_peak)

    return {
        "core_shape_corr": core_shape,
        "core_cos": core_cos,
        "core_softdtw": core_softdtw,
        "envelope_shape": envelope_shape,
        "duration": duration,
        "loudness": loudness,
    }


def active_bce_loss(pred_active_logits: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    pos_rate = active_mask.mean().clamp(min=1e-4, max=1.0 - 1e-4)
    pos_weight = ((1.0 - pos_rate) / pos_rate).clamp(min=1.0, max=10.0)
    return F.binary_cross_entropy_with_logits(pred_active_logits, active_mask, pos_weight=pos_weight)


def active_recon_loss(pred: torch.Tensor, target: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    weight = active_mask.unsqueeze(-1).to(pred.dtype)
    per_dim = F.smooth_l1_loss(pred, target, reduction="none")
    denom = (weight.sum() * pred.shape[-1]).clamp_min(1.0)
    return (per_dim * weight).sum() / denom


def masked_recon_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    per_dim = F.smooth_l1_loss(pred, target, reduction="none")
    if mask is None:
        return per_dim.mean()
    weight = mask.to(device=pred.device, dtype=pred.dtype).unsqueeze(-1)
    denom = (weight.sum() * pred.shape[-1]).clamp_min(1.0)
    return (per_dim * weight).sum() / denom


def soft_shift_recon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    min_frames: int = -12,
    max_frames: int = 62,
    step_frames: int = 1,
    temperature: float = 0.08,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Shift-tolerant core reconstruction loss using zero-padded candidate shifts.

    `shift_sequence_torch` is deliberately non-circular: content shifted outside the
    speech-core window is dropped, so the loss does not reward wrapping speech from
    the end back to the beginning.
    """
    t = int(pred.shape[1])
    step = max(1, int(step_frames))
    lo = max(int(min_frames), -t + 1)
    hi = min(int(max_frames), t - 1)
    if lo > hi:
        lo, hi = 0, 0
    losses = []
    for shift in range(lo, hi + 1, step):
        shifted = shift_sequence_torch(pred, int(shift))
        l1 = masked_recon_loss(shifted, target, mask)
        cos = 1.0 - F.cosine_similarity(shifted, target, dim=-1)
        if mask is not None:
            w = mask.to(device=pred.device, dtype=pred.dtype)
            cos_loss = (cos * w).sum() / w.sum().clamp_min(1.0)
        else:
            cos_loss = cos.mean()
        losses.append(l1 + 0.2 * cos_loss)
    stacked = torch.stack(losses)
    tau = max(float(temperature), 1e-4)
    return -tau * torch.logsumexp(-stacked / tau, dim=0)


def silence_energy_loss(
    pred: torch.Tensor,
    mask: torch.Tensor | None,
    target_mean: torch.Tensor | None,
    target_std: torch.Tensor | None,
) -> torch.Tensor:
    if mask is None:
        return pred.new_tensor(0.0)
    payload = raw_mel_energy(pred, target_mean, target_std)
    if payload is None:
        energy = pred.pow(2).mean(dim=-1)
        log_energy = torch.log(energy.clamp_min(1e-8))
    else:
        _, log_energy = payload
    w = mask.to(device=pred.device, dtype=pred.dtype)
    if w.ndim != 2 or w.shape[1] != pred.shape[1]:
        return pred.new_tensor(0.0)
    return (log_energy.exp() * w).sum() / w.sum().clamp_min(1.0)


def shift_supervision_losses(
    pred_shift_logits: torch.Tensor | None,
    target_shift_frame: torch.Tensor | None,
    min_frames: int,
    max_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred_shift_logits is None or target_shift_frame is None:
        zero = pred_shift_logits.new_tensor(0.0) if pred_shift_logits is not None else torch.tensor(0.0)
        return zero, zero
    logits = pred_shift_logits
    b, bins = int(logits.shape[0]), int(logits.shape[-1])
    if bins <= 0:
        zero = logits.new_tensor(0.0)
        return zero, zero
    lo, hi = int(min_frames), int(max_frames)
    target = target_shift_frame.to(device=logits.device, dtype=torch.float32).view(b).clamp(float(lo), float(hi))
    if bins == 1 or hi <= lo:
        ce = logits.new_tensor(0.0)
        pred_frame = logits.new_zeros(b) + float(lo)
    else:
        target_bin = torch.round((target - float(lo)) / float(hi - lo) * float(bins - 1)).long().clamp(0, bins - 1)
        ce = F.cross_entropy(logits, target_bin)
        grid = torch.linspace(float(lo), float(hi), steps=bins, device=logits.device, dtype=logits.dtype)
        pred_frame = (torch.softmax(logits, dim=-1) * grid.unsqueeze(0)).sum(dim=-1)
    reg = F.smooth_l1_loss(pred_frame, target.to(dtype=logits.dtype))
    return ce, reg


def zeroeeg_margin_loss(pred: torch.Tensor, target: torch.Tensor, zero_pred: torch.Tensor | None, margin: float = 0.02) -> torch.Tensor:
    if zero_pred is None:
        return pred.new_tensor(0.0)
    pred_cos = F.cosine_similarity(pred, target, dim=-1).mean(dim=1)
    zero_cos = F.cosine_similarity(zero_pred.detach(), target, dim=-1).mean(dim=1)
    return F.relu(float(margin) - (pred_cos - zero_cos)).mean()


def residual_variance_loss(pred_residual: torch.Tensor | None, residual_target: torch.Tensor | None) -> torch.Tensor:
    if pred_residual is None or residual_target is None:
        ref = pred_residual if pred_residual is not None else residual_target
        return ref.new_tensor(0.0) if ref is not None else torch.tensor(0.0)
    pred_std = pred_residual.reshape(-1, pred_residual.shape[-1]).std(dim=0)
    target_std = residual_target.reshape(-1, residual_target.shape[-1]).std(dim=0).detach()
    floor = 0.25 * target_std
    return F.relu(floor - pred_std).mean()


def pairwise_decorrelation_loss(pred: torch.Tensor, threshold: float = 0.85) -> torch.Tensor:
    flat = pred.reshape(pred.shape[0], -1)
    if flat.shape[0] < 2:
        return pred.new_tensor(0.0)
    flat = flat - flat.mean(dim=1, keepdim=True)
    flat = F.normalize(flat, dim=1)
    corr = flat @ flat.t()
    mask = ~torch.eye(corr.shape[0], device=pred.device, dtype=torch.bool)
    vals = corr[mask]
    return F.relu(vals.abs() - float(threshold)).pow(2).mean()


def peak_energy_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_mean: torch.Tensor | None,
    target_std: torch.Tensor | None,
    top_frac: float = 0.15,
) -> torch.Tensor:
    pred_payload = raw_mel_energy(pred, target_mean, target_std)
    target_payload = raw_mel_energy(target, target_mean, target_std)
    if pred_payload is None or target_payload is None:
        return pred.new_tensor(0.0)
    _, pred_log_energy = pred_payload
    target_energy, target_log_energy = target_payload
    k = max(1, int(round(float(top_frac) * target.shape[1])))
    idx = target_energy.topk(k=min(k, target.shape[1]), dim=1).indices
    pred_top = pred_log_energy.gather(1, idx)
    target_top = target_log_energy.gather(1, idx)
    return F.smooth_l1_loss(pred_top, target_top)


def lag_regression_loss(
    pred_lag_mu: torch.Tensor,
    pred_lag_log_sigma: torch.Tensor,
    target_lag_sec: torch.Tensor,
    lag_confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    target = target_lag_sec.to(device=pred_lag_mu.device, dtype=pred_lag_mu.dtype).view_as(pred_lag_mu)
    conf = (
        torch.ones_like(target)
        if lag_confidence is None
        else lag_confidence.to(device=pred_lag_mu.device, dtype=pred_lag_mu.dtype).view_as(pred_lag_mu).clamp(0.0, 1.0)
    )
    sigma = pred_lag_log_sigma.exp().clamp_min(1e-4)
    nll = F.smooth_l1_loss(pred_lag_mu / sigma, target / sigma, reduction="none") + 0.05 * pred_lag_log_sigma
    denom = conf.sum().clamp_min(1.0)
    return (nll * conf).sum() / denom


def semantic_token_ce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if logits.ndim != 3:
        return logits.new_tensor(0.0)
    tgt = targets.to(device=logits.device, dtype=torch.long)
    if tgt.ndim != 2:
        return logits.new_tensor(0.0)
    if logits.shape[1] != tgt.shape[1]:
        logits = F.interpolate(logits.transpose(1, 2), size=tgt.shape[1], mode="linear", align_corners=False).transpose(1, 2)
    if mask is None:
        active = tgt >= 0
    else:
        active = mask.to(device=logits.device).bool() & (tgt >= 0)
    if not bool(active.any()):
        return logits.new_tensor(0.0)
    vocab = logits.shape[-1]
    tgt = tgt.clamp(min=0, max=max(vocab - 1, 0))
    return F.cross_entropy(logits[active], tgt[active])


def raw_energy_corr_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_mean: torch.Tensor | None,
    target_std: torch.Tensor | None,
) -> torch.Tensor:
    pred_payload = raw_mel_energy(pred, target_mean, target_std)
    target_payload = raw_mel_energy(target, target_mean, target_std)
    if pred_payload is None or target_payload is None:
        return pred.new_tensor(0.0)
    _, pred_log_energy = pred_payload
    _, target_log_energy = target_payload
    return _corr_loss_1d(pred_log_energy, target_log_energy)


def decoder_scale_loss(pred_log_scale: torch.Tensor, target_decoder_scale: torch.Tensor | None) -> torch.Tensor:
    if target_decoder_scale is None:
        return pred_log_scale.new_tensor(0.0)
    target = torch.log(target_decoder_scale.float().clamp_min(1e-6))
    if target.shape[-1] != pred_log_scale.shape[-1]:
        if target.shape[-1] == 1:
            target = target.expand_as(pred_log_scale)
        else:
            target = target[..., : pred_log_scale.shape[-1]]
    return F.smooth_l1_loss(pred_log_scale, target)


class GradientReversal(torch.autograd.Function):
    """Identity forward, sign-flipped (and scaled) gradient backward (DANN).

    Used for subject-adversarial domain adaptation: the subject classifier learns
    to identify the subject from the pooled EEG embedding, while the reversed
    gradient pushes the encoder to make that embedding subject-INVARIANT. This uses
    subject ids at TRAIN time only to *remove* subject information; inference never
    sees subject id, so the model stays subject-agnostic (consistent with the
    existing `del subject_idx` design)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, lambd)


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
    lambda_dtw: float = 0.0,
    lambda_energy_env: float = 0.0,
    lambda_multiscale_mel: float = 0.0,
    lambda_frame_energy: float = 0.0,
    lambda_voiced_rms: float = 0.0,
    lambda_decoder_scale: float = 0.0,
    lambda_ctc: float = 0.0,
    lambda_hubert_aux: float = 0.0,
    lambda_hubert_clip: float = 0.0,
    lambda_residual_l1: float = 0.0,
    lambda_residual_mse: float = 0.0,
    lambda_residual_cos: float = 0.0,
    lambda_active_bce: float = 0.0,
    lambda_raw_energy_corr: float = 0.0,
    lambda_active_recon: float = 0.0,
    lambda_peak_energy: float = 0.0,
    lambda_aligned_recon_cos: float = 0.0,
    lambda_aligned_recon_mse: float = 0.0,
    lambda_aligned_raw_energy_corr: float = 0.0,
    lambda_aligned_active_recon: float = 0.0,
    lambda_aligned_peak_energy: float = 0.0,
    lambda_lag: float = 0.0,
    lambda_semantic_token_ce: float = 0.0,
    lambda_speech_token_ctc: float = 0.0,
    lambda_trial_infonce: float = 0.0,
    lambda_soft_shift: float = 0.0,
    lambda_silence_suppression: float = 0.0,
    lambda_pre_noise_suppression: float = 0.0,
    lambda_shift_ce: float = 0.0,
    lambda_shift_reg: float = 0.0,
    lambda_zeroeeg_margin: float = 0.0,
    lambda_residual_variance: float = 0.0,
    lambda_pairwise_decorrelation: float = 0.0,
    lambda_core_shape_corr: float = 0.0,
    lambda_core_cos: float = 0.0,
    lambda_core_softdtw: float = 0.0,
    lambda_envelope_shape: float = 0.0,
    lambda_duration: float = 0.0,
    lambda_loudness: float = 0.0,
    hubert_seq: torch.Tensor | None = None,
    hubert_summary: torch.Tensor | None = None,
    residual_target: torch.Tensor | None = None,
    zero_pred_seq: torch.Tensor | None = None,
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
    target_decoder_scale: torch.Tensor | None = None,
    lag_sec: torch.Tensor | None = None,
    lag_mel_frames: torch.Tensor | None = None,
    lag_confidence: torch.Tensor | None = None,
    semantic_token_targets: torch.Tensor | None = None,
    semantic_token_mask: torch.Tensor | None = None,
    target_active_mask: torch.Tensor | None = None,
    target_silence_mask: torch.Tensor | None = None,
    target_pre_noise_mask: torch.Tensor | None = None,
    shift_target_frame: torch.Tensor | None = None,
    active_envelope: torch.Tensor | None = None,
    active_duration_frames: torch.Tensor | None = None,
    active_rms: torch.Tensor | None = None,
    active_peak: torch.Tensor | None = None,
    supcon_temperature: float = 0.1,
    clip_temperature: float = 0.07,
    dtw_band: float = 0.2,
    soft_shift_min_frames: int = -12,
    soft_shift_max_frames: int = 62,
    soft_shift_step_frames: int = 1,
    soft_shift_temperature: float = 0.08,
    temporal_elastic_softdtw_gamma: float = 0.08,
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
    trial_infonce = (
        clip_alignment(out["clip_embed"], target_seq.mean(dim=1), temperature=clip_temperature)
        if lambda_trial_infonce > 0.0
        else pred.new_tensor(0.0)
    )

    # DTW-aligned reconstruction: invariant to cross-trial onset/rate jitter.
    dtw_loss = dtw_recon_loss(pred, target_seq, band=dtw_band) if lambda_dtw > 0.0 else pred.new_tensor(0.0)

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

    # Anti-collapse group (WS2a): energy-envelope correlation + multi-resolution L1.
    # Both operate in the (z-scored) acoustic target domain, no waveform render needed.
    energy_env = energy_envelope_loss(pred, target_seq) if lambda_energy_env > 0.0 else pred.new_tensor(0.0)
    multiscale_mel = multiscale_temporal_l1(pred, target_seq) if lambda_multiscale_mel > 0.0 else pred.new_tensor(0.0)
    frame_energy = (
        frame_log_energy_loss(out["pred_frame_log_energy"], target_seq)
        if lambda_frame_energy > 0.0 and "pred_frame_log_energy" in out
        else pred.new_tensor(0.0)
    )
    voiced_rms = voiced_region_rms_loss(pred, target_seq) if lambda_voiced_rms > 0.0 else pred.new_tensor(0.0)
    dec_scale = (
        decoder_scale_loss(out["pred_log_decoder_scale"], target_decoder_scale)
        if lambda_decoder_scale > 0.0 and "pred_log_decoder_scale" in out
        else pred.new_tensor(0.0)
    )
    ctc = prompt_ctc_loss(out["ctc_logits"], label_idx) if lambda_ctc > 0.0 and "ctc_logits" in out else pred.new_tensor(0.0)

    # HuBERT auxiliary content target (WS3): regression to GT HuBERT features, plus a
    # content-bearing symmetric InfoNCE (EEG-predicted HuBERT summary <-> GT summary).
    pred_hubert = out.get("pred_hubert")
    hubert_aux = pred.new_tensor(0.0)
    hubert_clip = pred.new_tensor(0.0)
    if pred_hubert is not None and hubert_seq is not None and lambda_hubert_aux > 0.0:
        hubert_aux = hubert_aux_loss(pred_hubert, hubert_seq)
    if pred_hubert is not None and hubert_summary is not None and lambda_hubert_clip > 0.0:
        hubert_clip = clip_alignment(pred_hubert.mean(dim=1), hubert_summary, temperature=clip_temperature)

    pred_residual = out.get("pred_residual")
    residual_l1 = pred.new_tensor(0.0)
    residual_mse = pred.new_tensor(0.0)
    residual_cos = pred.new_tensor(0.0)
    if pred_residual is not None and residual_target is not None:
        if lambda_residual_l1 > 0.0:
            residual_l1 = F.smooth_l1_loss(pred_residual, residual_target)
        if lambda_residual_mse > 0.0:
            residual_mse = F.mse_loss(pred_residual, residual_target)
        if lambda_residual_cos > 0.0:
            residual_cos = 1.0 - F.cosine_similarity(pred_residual, residual_target, dim=-1).mean()

    active_bce = pred.new_tensor(0.0)
    raw_energy_corr = pred.new_tensor(0.0)
    active_recon = pred.new_tensor(0.0)
    peak_energy = pred.new_tensor(0.0)
    target_energy_payload = raw_mel_energy(target_seq, target_mean, target_std)
    if target_energy_payload is not None and (
        lambda_active_bce > 0.0 or lambda_raw_energy_corr > 0.0 or lambda_active_recon > 0.0 or lambda_peak_energy > 0.0
        or lambda_aligned_raw_energy_corr > 0.0 or lambda_aligned_active_recon > 0.0 or lambda_aligned_peak_energy > 0.0
    ):
        target_energy, _ = target_energy_payload
        active_mask = (
            target_active_mask.to(device=pred.device, dtype=pred.dtype)
            if target_active_mask is not None and target_active_mask.shape[-1] == pred.shape[1]
            else active_mask_from_energy(target_energy)
        )
        if lambda_active_bce > 0.0 and "pred_active_logits" in out:
            active_bce = active_bce_loss(out["pred_active_logits"], active_mask)
        if lambda_raw_energy_corr > 0.0:
            raw_energy_corr = raw_energy_corr_loss(pred, target_seq, target_mean, target_std)
        if lambda_active_recon > 0.0:
            active_recon = active_recon_loss(pred, target_seq, active_mask)
        if lambda_peak_energy > 0.0:
            peak_energy = peak_energy_loss(pred, target_seq, target_mean, target_std)

    aligned_recon_cos = pred.new_tensor(0.0)
    aligned_recon_mse = pred.new_tensor(0.0)
    aligned_raw_energy_corr = pred.new_tensor(0.0)
    aligned_active_recon = pred.new_tensor(0.0)
    aligned_peak_energy = pred.new_tensor(0.0)
    aligned_pred = None
    if lag_mel_frames is not None and (
        lambda_aligned_recon_cos > 0.0
        or lambda_aligned_recon_mse > 0.0
        or lambda_aligned_raw_energy_corr > 0.0
        or lambda_aligned_active_recon > 0.0
        or lambda_aligned_peak_energy > 0.0
    ):
        aligned_pred = shift_sequence_torch(pred, -lag_mel_frames.to(pred.device))
        if lambda_aligned_recon_cos > 0.0:
            aligned_recon_cos = 1.0 - F.cosine_similarity(aligned_pred, target_seq, dim=-1).mean()
        if lambda_aligned_recon_mse > 0.0:
            aligned_recon_mse = F.mse_loss(aligned_pred, target_seq)
        if lambda_aligned_raw_energy_corr > 0.0:
            aligned_raw_energy_corr = raw_energy_corr_loss(aligned_pred, target_seq, target_mean, target_std)
        if target_energy_payload is not None:
            target_energy, _ = target_energy_payload
            active_mask = active_mask_from_energy(target_energy)
            if lambda_aligned_active_recon > 0.0:
                aligned_active_recon = active_recon_loss(aligned_pred, target_seq, active_mask)
            if lambda_aligned_peak_energy > 0.0:
                aligned_peak_energy = peak_energy_loss(aligned_pred, target_seq, target_mean, target_std)

    lag_loss = pred.new_tensor(0.0)
    if lambda_lag > 0.0 and lag_sec is not None and "pred_lag_mu" in out:
        lag_loss = lag_regression_loss(
            out["pred_lag_mu"],
            out.get("pred_lag_log_sigma", torch.zeros_like(out["pred_lag_mu"])),
            lag_sec,
            lag_confidence,
        )

    soft_shift = pred.new_tensor(0.0)
    if lambda_soft_shift > 0.0:
        shift_mask = (
            target_active_mask.to(device=pred.device, dtype=pred.dtype)
            if target_active_mask is not None and target_active_mask.shape[-1] == pred.shape[1]
            else None
        )
        soft_shift = soft_shift_recon_loss(
            pred,
            target_seq,
            min_frames=int(soft_shift_min_frames),
            max_frames=int(soft_shift_max_frames),
            step_frames=int(soft_shift_step_frames),
            temperature=float(soft_shift_temperature),
            mask=shift_mask,
        )

    silence_suppression = (
        silence_energy_loss(pred, target_silence_mask, target_mean, target_std)
        if lambda_silence_suppression > 0.0
        else pred.new_tensor(0.0)
    )
    pre_noise_suppression = (
        silence_energy_loss(pred, target_pre_noise_mask, target_mean, target_std)
        if lambda_pre_noise_suppression > 0.0
        else pred.new_tensor(0.0)
    )

    semantic_token_ce = pred.new_tensor(0.0)
    if lambda_semantic_token_ce > 0.0 and semantic_token_targets is not None and "semantic_token_logits" in out:
        semantic_token_ce = semantic_token_ce_loss(out["semantic_token_logits"], semantic_token_targets, semantic_token_mask)
    speech_token_ctc = pred.new_tensor(0.0)
    if lambda_speech_token_ctc > 0.0 and semantic_token_targets is not None and "semantic_token_logits" in out:
        speech_token_ctc = speech_token_ctc_loss(out["semantic_token_logits"], semantic_token_targets, semantic_token_mask)

    shift_ce, shift_reg = shift_supervision_losses(
        out.get("pred_shift_logits"),
        shift_target_frame,
        int(soft_shift_min_frames),
        int(soft_shift_max_frames),
    )
    if lambda_shift_ce <= 0.0:
        shift_ce = pred.new_tensor(0.0)
    if lambda_shift_reg <= 0.0:
        shift_reg = pred.new_tensor(0.0)
    zero_margin = (
        zeroeeg_margin_loss(pred, target_seq, zero_pred_seq)
        if lambda_zeroeeg_margin > 0.0
        else pred.new_tensor(0.0)
    )
    residual_variance = (
        residual_variance_loss(pred_residual, residual_target)
        if lambda_residual_variance > 0.0
        else pred.new_tensor(0.0)
    )
    pairwise_decorrelation = (
        pairwise_decorrelation_loss(pred)
        if lambda_pairwise_decorrelation > 0.0
        else pred.new_tensor(0.0)
    )
    temporal_losses = temporal_elastic_core_losses(
        pred,
        target_seq,
        out,
        target_mean,
        target_std,
        active_envelope=active_envelope,
        active_duration_frames=active_duration_frames,
        active_rms=active_rms,
        active_peak=active_peak,
        softdtw_gamma=temporal_elastic_softdtw_gamma,
    ) if (
        lambda_core_shape_corr > 0.0
        or lambda_core_cos > 0.0
        or lambda_core_softdtw > 0.0
        or lambda_envelope_shape > 0.0
        or lambda_duration > 0.0
        or lambda_loudness > 0.0
    ) else {
        "core_shape_corr": pred.new_tensor(0.0),
        "core_cos": pred.new_tensor(0.0),
        "core_softdtw": pred.new_tensor(0.0),
        "envelope_shape": pred.new_tensor(0.0),
        "duration": pred.new_tensor(0.0),
        "loudness": pred.new_tensor(0.0),
    }

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
        + lambda_trial_infonce * trial_infonce
        + lambda_dtw * dtw_loss
        + lambda_energy_env * energy_env
        + lambda_multiscale_mel * multiscale_mel
        + lambda_frame_energy * frame_energy
        + lambda_voiced_rms * voiced_rms
        + lambda_decoder_scale * dec_scale
        + lambda_ctc * ctc
        + lambda_hubert_aux * hubert_aux
        + lambda_hubert_clip * hubert_clip
        + lambda_residual_l1 * residual_l1
        + lambda_residual_mse * residual_mse
        + lambda_residual_cos * residual_cos
        + lambda_active_bce * active_bce
        + lambda_raw_energy_corr * raw_energy_corr
        + lambda_active_recon * active_recon
        + lambda_peak_energy * peak_energy
        + lambda_aligned_recon_cos * aligned_recon_cos
        + lambda_aligned_recon_mse * aligned_recon_mse
        + lambda_aligned_raw_energy_corr * aligned_raw_energy_corr
        + lambda_aligned_active_recon * aligned_active_recon
        + lambda_aligned_peak_energy * aligned_peak_energy
        + lambda_lag * lag_loss
        + lambda_semantic_token_ce * semantic_token_ce
        + lambda_speech_token_ctc * speech_token_ctc
        + lambda_soft_shift * soft_shift
        + lambda_silence_suppression * silence_suppression
        + lambda_pre_noise_suppression * pre_noise_suppression
        + lambda_shift_ce * shift_ce
        + lambda_shift_reg * shift_reg
        + lambda_zeroeeg_margin * zero_margin
        + lambda_residual_variance * residual_variance
        + lambda_pairwise_decorrelation * pairwise_decorrelation
        + lambda_core_shape_corr * temporal_losses["core_shape_corr"]
        + lambda_core_cos * temporal_losses["core_cos"]
        + lambda_core_softdtw * temporal_losses["core_softdtw"]
        + lambda_envelope_shape * temporal_losses["envelope_shape"]
        + lambda_duration * temporal_losses["duration"]
        + lambda_loudness * temporal_losses["loudness"]
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
        "trial_infonce": trial_infonce.detach(),
        "dtw_loss": dtw_loss.detach(),
        "log_rms_loss": log_rms_loss.detach(),
        "std_match": std_match.detach(),
        "std_ratio": std_ratio,
        "router_balance": router_balance.detach(),
        "channel_balance": channel_balance.detach() if torch.is_tensor(channel_balance) else pred.new_tensor(0.0),
        "energy_env": energy_env.detach(),
        "multiscale_mel": multiscale_mel.detach(),
        "frame_energy": frame_energy.detach(),
        "voiced_rms": voiced_rms.detach(),
        "decoder_scale": dec_scale.detach(),
        "ctc": ctc.detach(),
        "hubert_aux": hubert_aux.detach(),
        "hubert_clip": hubert_clip.detach(),
        "residual_l1": residual_l1.detach(),
        "residual_mse": residual_mse.detach(),
        "residual_cos": residual_cos.detach(),
        "active_bce": active_bce.detach(),
        "raw_energy_corr": raw_energy_corr.detach(),
        "active_recon": active_recon.detach(),
        "peak_energy": peak_energy.detach(),
        "aligned_recon_cos": aligned_recon_cos.detach(),
        "aligned_recon_mse": aligned_recon_mse.detach(),
        "aligned_raw_energy_corr": aligned_raw_energy_corr.detach(),
        "aligned_active_recon": aligned_active_recon.detach(),
        "aligned_peak_energy": aligned_peak_energy.detach(),
        "lag_loss": lag_loss.detach(),
        "semantic_token_ce": semantic_token_ce.detach(),
        "speech_token_ctc": speech_token_ctc.detach(),
        "soft_shift": soft_shift.detach(),
        "silence_suppression": silence_suppression.detach(),
        "pre_noise_suppression": pre_noise_suppression.detach(),
        "shift_ce": shift_ce.detach(),
        "shift_reg": shift_reg.detach(),
        "zeroeeg_margin": zero_margin.detach(),
        "residual_variance": residual_variance.detach(),
        "pairwise_decorrelation": pairwise_decorrelation.detach(),
        "core_shape_corr": temporal_losses["core_shape_corr"].detach(),
        "core_cos": temporal_losses["core_cos"].detach(),
        "core_softdtw": temporal_losses["core_softdtw"].detach(),
        "envelope_shape": temporal_losses["envelope_shape"].detach(),
        "duration_loss": temporal_losses["duration"].detach(),
        "loudness_loss": temporal_losses["loudness"].detach(),
    }

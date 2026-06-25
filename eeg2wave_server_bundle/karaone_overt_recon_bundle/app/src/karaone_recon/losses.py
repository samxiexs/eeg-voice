from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


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
    lambda_hubert_aux: float = 0.0,
    lambda_hubert_clip: float = 0.0,
    hubert_seq: torch.Tensor | None = None,
    hubert_summary: torch.Tensor | None = None,
    supcon_temperature: float = 0.1,
    clip_temperature: float = 0.07,
    dtw_band: float = 0.2,
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

    # HuBERT auxiliary content target (WS3): regression to GT HuBERT features, plus a
    # content-bearing symmetric InfoNCE (EEG-predicted HuBERT summary <-> GT summary).
    pred_hubert = out.get("pred_hubert")
    hubert_aux = pred.new_tensor(0.0)
    hubert_clip = pred.new_tensor(0.0)
    if pred_hubert is not None and hubert_seq is not None and lambda_hubert_aux > 0.0:
        hubert_aux = hubert_aux_loss(pred_hubert, hubert_seq)
    if pred_hubert is not None and hubert_summary is not None and lambda_hubert_clip > 0.0:
        hubert_clip = clip_alignment(pred_hubert.mean(dim=1), hubert_summary, temperature=clip_temperature)

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
        + lambda_dtw * dtw_loss
        + lambda_energy_env * energy_env
        + lambda_multiscale_mel * multiscale_mel
        + lambda_hubert_aux * hubert_aux
        + lambda_hubert_clip * hubert_clip
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
        "dtw_loss": dtw_loss.detach(),
        "log_rms_loss": log_rms_loss.detach(),
        "std_match": std_match.detach(),
        "std_ratio": std_ratio,
        "router_balance": router_balance.detach(),
        "channel_balance": channel_balance.detach() if torch.is_tensor(channel_balance) else pred.new_tensor(0.0),
        "energy_env": energy_env.detach(),
        "multiscale_mel": multiscale_mel.detach(),
        "hubert_aux": hubert_aux.detach(),
        "hubert_clip": hubert_clip.detach(),
    }


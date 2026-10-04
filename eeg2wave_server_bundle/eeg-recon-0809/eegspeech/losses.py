"""Training objectives.

Listening: ``match_mismatch`` (which of 1 + K speech segments is the one heard,
the Auditory EEG Challenge task, as an InfoNCE over candidates) and
``pair_infonce`` (two people hearing the same stimulus at the same time agree:
a subject-invariant, stimulus-driven code without choosing a speech feature).
Items: ``supcon`` (same item, any modality - heard, spoken, mouthed, imagined - is
a positive) and cross-entropy on the dataset's item head.  Personalisation:
``person_contrastive``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def frame_similarity(a, b, mask):
    """Mean cosine over valid frames: a (B, T, W), b (B, K, T, W) -> (B, K)."""
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    sim = torch.einsum('btw,bktw->bkt', a, b)
    weight = mask[:, None, :].to(sim.dtype)
    return (sim * weight).sum(-1) / weight.sum(-1).clamp_min(1)


def match_mismatch(eeg_frames, speech_frames, mask, temperature=.1):
    """Candidate 0 of ``speech_frames`` (B, 1 + K, T, W) is the matched segment."""
    t = min(eeg_frames.shape[1], speech_frames.shape[2])
    logits = frame_similarity(eeg_frames[:, :t], speech_frames[:, :, :t], mask[:, :t]) / temperature
    target = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target), (logits.argmax(1) == 0).float().mean()


def pair_infonce(a, b, mask, temperature=.1):
    """Symmetric InfoNCE between frame sequences of two people for the same stimulus window (row i <-> row i)."""
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    weight = mask.to(a.dtype)
    sim = torch.einsum('itw,jtw->ijt', a * weight[..., None], b * weight[..., None]) / weight.sum(1).clamp_min(1)[:, None, None]
    logits = sim.sum(-1) / temperature
    labels = torch.arange(len(a), device=a.device)
    loss = .5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    return loss, (logits.argmax(1) == labels).float().mean()


def supcon(z, labels, temperature=.1, cross_only=None):
    """Supervised contrastive loss (Khosla et al. 2020) on unit embeddings z (B, D).

    ``cross_only`` (B,) group ids, e.g. modality: when given, positives must come
    from a different group, so the loss pulls an imagined trial towards spoken or
    heard trials of the same item rather than towards its own modality.
    """
    logits = z @ z.T / temperature
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positive = (labels[:, None] == labels[None, :]) & ~eye
    if cross_only is not None:
        positive &= cross_only[:, None] != cross_only[None, :]
    logits = logits.masked_fill(eye, float('-inf'))
    log_prob = logits - torch.logsumexp(logits, 1, keepdim=True)
    count = positive.sum(1)
    keep = count > 0
    if not keep.any():
        return z.sum() * 0.
    loss = -(log_prob.masked_fill(~positive, 0).sum(1)[keep] / count[keep])
    return loss.mean()


def person_contrastive(a, b, temperature=.1):
    """Symmetric InfoNCE over people: row i of both views is one person (two disjoint window sets)."""
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    logits = a @ b.T / temperature
    labels = torch.arange(len(a), device=a.device)
    return .5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

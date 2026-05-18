"""Losses and small metrics for EEGVoiceTokenV1."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F


def time_l1_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(predicted - target))


def pearson_corr(predicted: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = predicted.flatten(0, -2)
    y = target.flatten(0, -2)
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    numerator = torch.sum(x * y, dim=-1)
    denominator = torch.sqrt(torch.sum(x * x, dim=-1) * torch.sum(y * y, dim=-1) + eps)
    return torch.mean(numerator / (denominator + eps))


def frequency_domain_loss(predicted: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    window = torch.hamming_window(target.shape[-1], device=target.device, dtype=target.dtype)
    predicted_fft = torch.fft.rfft(predicted * window, dim=-1, norm="ortho")
    target_fft = torch.fft.rfft(target * window, dim=-1, norm="ortho")
    amp_loss = F.l1_loss(torch.abs(predicted_fft), torch.abs(target_fft))
    phase_loss = F.l1_loss(torch.angle(predicted_fft), torch.angle(target_fft))
    return amp_loss, phase_loss


def eeg_reconstruction_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    phase_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    """EEG reconstruction loss used as a token anti-collapse constraint."""
    time_loss = time_l1_loss(predicted, target)
    amp_loss, phase_loss = frequency_domain_loss(predicted, target)
    pcc = pearson_corr(predicted, target)
    loss = time_loss + amp_loss + phase_weight * phase_loss + torch.exp(-pcc)
    return {
        "loss": loss,
        "time_loss": time_loss.detach(),
        "freq_amp_loss": amp_loss.detach(),
        "freq_phase_loss": phase_loss.detach(),
        "pcc": pcc.detach(),
    }


def info_nce_logits(
    query_embedding: torch.Tensor,
    positive_embedding: torch.Tensor,
    extra_negative_embedding: torch.Tensor | None = None,
    temperature: torch.Tensor | float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return InfoNCE loss and logits with positives on the first B columns."""
    query = F.normalize(query_embedding, dim=-1)
    positive = F.normalize(positive_embedding, dim=-1)
    candidates = positive
    if extra_negative_embedding is not None and extra_negative_embedding.numel() > 0:
        candidates = torch.cat([candidates, F.normalize(extra_negative_embedding, dim=-1)], dim=0)
    logits = query @ candidates.T
    logits = logits / temperature
    labels = torch.arange(query.shape[0], device=query.device)
    return F.cross_entropy(logits, labels), logits


def token_usage_metrics(tokens: torch.Tensor, codebook_size: int) -> dict[str, torch.Tensor]:
    """Report usage metrics for token indices with shape `[..., num_quantizers]`."""
    if tokens.ndim < 1:
        raise ValueError("Expected token tensor with at least one dimension")
    flat = tokens.reshape(-1, tokens.shape[-1]) if tokens.ndim > 1 else tokens.reshape(-1, 1)
    usage = []
    perplexity = []
    dead_ratio = []
    unique = []
    for idx in range(flat.shape[-1]):
        counts = torch.bincount(flat[:, idx].long(), minlength=codebook_size).float()
        probs = counts / counts.sum().clamp_min(1.0)
        nonzero = probs > 0
        entropy = -(probs[nonzero] * torch.log(probs[nonzero])).sum()
        used = (counts > 0).float().sum()
        usage.append(used / codebook_size)
        perplexity.append(torch.exp(entropy))
        dead_ratio.append(1.0 - used / codebook_size)
        unique.append(used)
    return {
        "usage": torch.stack(usage).mean(),
        "perplexity": torch.stack(perplexity).mean(),
        "dead_code_ratio": torch.stack(dead_ratio).mean(),
        "unique_codes": torch.stack(unique).mean(),
    }


def dataset_predictability_proxy(embedding: torch.Tensor, dataset_id: list[str] | tuple[str, ...]) -> torch.Tensor:
    """Nearest-centroid proxy for whether an embedding exposes dataset identity.

    This is not a trained probe. It is a cheap batch-level warning metric for
    nuisance absorption, especially for q7 residual tokens.
    """
    if len(dataset_id) != embedding.shape[0] or embedding.shape[0] < 3:
        return embedding.new_tensor(0.0)
    pooled = F.normalize(embedding.float(), dim=-1)
    by_dataset: dict[str, list[int]] = defaultdict(list)
    for idx, name in enumerate(dataset_id):
        by_dataset[str(name)].append(idx)
    if len(by_dataset) < 2:
        return embedding.new_tensor(0.0)

    correct = 0
    valid = 0
    for idx, name in enumerate(dataset_id):
        centroids = []
        labels = []
        for candidate_name, candidate_indices in by_dataset.items():
            members = [j for j in candidate_indices if j != idx]
            if not members:
                continue
            centroids.append(pooled[members].mean(dim=0))
            labels.append(candidate_name)
        if not centroids:
            continue
        centroid_tensor = F.normalize(torch.stack(centroids, dim=0), dim=-1)
        pred = labels[int(torch.argmax(pooled[idx] @ centroid_tensor.T))]
        correct += int(pred == str(name))
        valid += 1
    if valid == 0:
        return embedding.new_tensor(0.0)
    return embedding.new_tensor(correct / valid)

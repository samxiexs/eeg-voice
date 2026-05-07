"""Downstream heads for EEG token probes and voice-representation alignment."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import info_nce_loss


def pool_tokens(z: torch.Tensor) -> torch.Tensor:
    """Pool `[B, Q, W, D]` tokens to `[B, D]`."""
    if z.ndim != 4:
        raise ValueError(f"Expected z with shape [B,Q,W,D], got {tuple(z.shape)}")
    return z.mean(dim=(1, 2))


class ProbeHead(nn.Module):
    """Classification probe for ds006104 labels."""

    def __init__(self, dim: int, num_classes: int, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        logits = self.net(pool_tokens(z))
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels.long())
        return out


class AudioContrastiveHead(nn.Module):
    """InfoNCE alignment between EEG token embeddings and audio stream embeddings."""

    def __init__(self, eeg_dim: int, audio_dim: int, proj_dim: int = 256, temperature: float = 0.07):
        super().__init__()
        self.eeg_proj = nn.Sequential(nn.LayerNorm(eeg_dim), nn.Linear(eeg_dim, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))
        self.audio_proj = nn.Sequential(nn.LayerNorm(audio_dim), nn.Linear(audio_dim, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))))

    def forward(self, z: torch.Tensor, audio_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        eeg = self.eeg_proj(pool_tokens(z))
        audio = self.audio_proj(audio_embedding.float())
        loss, logits = info_nce_loss(eeg, audio, temperature=torch.exp(self.log_temperature))
        return {"loss": loss, "logits": logits, "eeg_embedding": eeg, "audio_embedding": audio}


class SegmentContrastiveHead(AudioContrastiveHead):
    """Defossez-style segment retrieval from EEG tokens to audio/content embeddings."""


class PhonemeSequenceHead(nn.Module):
    """Parallel phoneme sequence probe over token time steps.

    The head pools latent queries but keeps the token time axis, returning
    logits with shape `[B, S, num_phoneme_classes]`.
    """

    def __init__(self, dim: int, num_phoneme_classes: int, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_phoneme_classes),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if z.ndim != 4:
            raise ValueError(f"Expected z with shape [B,Q,S,D], got {tuple(z.shape)}")
        sequence = z.mean(dim=1)
        logits = self.net(sequence)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.long().reshape(-1), ignore_index=-100)
        return out


class VoiceProfileHead(nn.Module):
    """Token downstream head for pitch, timbre, speaker, and style attributes."""

    def __init__(
        self,
        dim: int,
        pitch_dim: int = 1,
        timbre_dim: int = 8,
        speaker_dim: int = 128,
        style_classes: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        self.pitch = nn.Linear(dim, pitch_dim)
        self.timbre = nn.Linear(dim, timbre_dim)
        self.speaker = nn.Linear(dim, speaker_dim)
        self.style = nn.Linear(dim, style_classes)

    def forward(self, z: torch.Tensor, targets: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        pooled = self.shared(pool_tokens(z))
        pitch_pred = self.pitch(pooled)
        timbre_pred = self.timbre(pooled)
        speaker_embedding = self.speaker(pooled)
        style_logits = self.style(pooled)
        out = {
            "pitch_pred": pitch_pred,
            "timbre_pred": timbre_pred,
            "speaker_embedding": speaker_embedding,
            "style_logits": style_logits,
        }
        if targets:
            losses = []
            if "pitch" in targets:
                losses.append(F.smooth_l1_loss(pitch_pred, targets["pitch"].float()))
            if "timbre" in targets:
                losses.append(F.smooth_l1_loss(timbre_pred, targets["timbre"].float()))
            if "speaker" in targets:
                target = targets["speaker"].float()
                losses.append(F.mse_loss(F.normalize(speaker_embedding, dim=-1), F.normalize(target, dim=-1)))
            if "style" in targets:
                losses.append(F.cross_entropy(style_logits, targets["style"].long()))
            if losses:
                out["loss"] = torch.stack(losses).mean()
        return out


class TokenMetrics(nn.Module):
    """Report discrete token usage, perplexity, and dead-code ratio."""

    def __init__(self, codebook_size: int):
        super().__init__()
        self.codebook_size = int(codebook_size)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if tokens.ndim < 1:
            raise ValueError("Expected token indices with at least one dimension")
        flat = tokens.reshape(-1, tokens.shape[-1]) if tokens.ndim > 1 else tokens.reshape(-1, 1)
        usage = []
        perplexity = []
        dead_ratio = []
        unique = []
        for q in range(flat.shape[-1]):
            counts = torch.bincount(flat[:, q].long(), minlength=self.codebook_size).float()
            probs = counts / counts.sum().clamp_min(1.0)
            entropy = -(probs[probs > 0] * torch.log(probs[probs > 0])).sum()
            used = (counts > 0).float().sum()
            usage.append(used / self.codebook_size)
            perplexity.append(torch.exp(entropy))
            dead_ratio.append(1.0 - used / self.codebook_size)
            unique.append(used)
        return {
            "codebook_usage": torch.stack(usage).mean(),
            "token_perplexity": torch.stack(perplexity).mean(),
            "dead_code_ratio": torch.stack(dead_ratio).mean(),
            "unique_codes": torch.stack(unique).mean(),
        }


class VoiceAttributeHead(nn.Module):
    """Small head for pitch/intensity/timbre attribute probes."""

    def __init__(self, dim: int, output_dim: int, task: str = "classification", dropout: float = 0.1):
        super().__init__()
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        self.task = task
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, output_dim),
        )

    def forward(self, z: torch.Tensor, target: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        pred = self.net(pool_tokens(z))
        out = {"pred": pred}
        if target is not None:
            if self.task == "classification":
                out["loss"] = F.cross_entropy(pred, target.long())
            else:
                out["loss"] = F.smooth_l1_loss(pred, target.float())
        return out

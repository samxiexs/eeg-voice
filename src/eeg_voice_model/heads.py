"""Alignment, mode, and retrieval heads for EEGVoiceTokenV1."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import info_nce_logits, token_usage_metrics


def pool_tokens(z: torch.Tensor) -> torch.Tensor:
    if z.ndim != 4:
        raise ValueError(f"Expected token latent with shape [B,Q,S,D], got {tuple(z.shape)}")
    return z.mean(dim=(1, 2))


class SequenceClassificationHead(nn.Module):
    """Per-token sequence classifier for phoneme/content labels."""

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
        sequence = z.mean(dim=1)
        logits = self.net(sequence)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.long().reshape(-1), ignore_index=-100)
        return out


class RegressionHead(nn.Module):
    """Pooled token regression head."""

    def __init__(self, dim: int, output_dim: int, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor, target: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        pred = self.net(pool_tokens(z))
        out = {"pred": pred}
        if target is not None:
            out["loss"] = F.smooth_l1_loss(pred, target.float())
        return out


class ClassificationHead(nn.Module):
    """Pooled token classifier."""

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


class DatasetFiLMAdapter(nn.Module):
    """Small dataset adapter for speaking-mode alignment."""

    def __init__(self, dim: int, dataset_adapter_count: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.film = nn.Embedding(dataset_adapter_count, dim * 2)
        self.dataset_adapter_count = int(dataset_adapter_count)
        self.dataset_to_index: dict[str, int] = {}
        nn.init.zeros_(self.film.weight)

    def _indices(self, dataset_id: list[str] | tuple[str, ...], device: torch.device) -> torch.Tensor:
        values = []
        for name in dataset_id:
            key = str(name)
            if key not in self.dataset_to_index:
                next_index = len(self.dataset_to_index) + 1
                self.dataset_to_index[key] = next_index if next_index < self.dataset_adapter_count else 0
            values.append(self.dataset_to_index[key])
        return torch.tensor(values, dtype=torch.long, device=device)

    def forward(self, pooled: torch.Tensor, dataset_id: list[str] | tuple[str, ...]) -> torch.Tensor:
        x = self.norm(pooled)
        gamma, beta = self.film(self._indices(dataset_id, pooled.device)).chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta


class SpeakingModeHead(nn.Module):
    """Shared speaking-mode classifier with per-dataset FiLM adaptation."""

    def __init__(self, dim: int, mode_count: int, dataset_adapter_count: int = 128, dropout: float = 0.1):
        super().__init__()
        self.adapter = DatasetFiLMAdapter(dim, dataset_adapter_count)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, mode_count),
        )

    def forward(
        self,
        z: torch.Tensor,
        dataset_id: list[str] | tuple[str, ...],
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        adapted = self.adapter(pool_tokens(z), dataset_id)
        logits = self.classifier(adapted)
        out = {"logits": logits, "adapted_embedding": adapted}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels.long())
        return out


class RetrievalHead(nn.Module):
    """EEG/audio retrieval with memory-queue hard negatives."""

    def __init__(
        self,
        eeg_dim: int,
        audio_dim: int,
        proj_dim: int = 256,
        queue_size: int = 4096,
        queue_negatives: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.eeg_proj = nn.Sequential(nn.LayerNorm(eeg_dim), nn.Linear(eeg_dim, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))
        self.audio_proj = nn.Sequential(nn.LayerNorm(audio_dim), nn.Linear(audio_dim, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))))
        self.queue_size = int(queue_size)
        self.queue_negatives = int(queue_negatives)
        self.register_buffer("queue", torch.zeros(self.queue_size, proj_dim), persistent=False)
        self.register_buffer("queue_ptr", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("queue_filled", torch.zeros((), dtype=torch.long), persistent=False)

    @torch.no_grad()
    def _enqueue(self, audio: torch.Tensor) -> None:
        if self.queue_size <= 0 or audio.numel() == 0:
            return
        audio = audio.detach()
        batch = min(audio.shape[0], self.queue_size)
        audio = audio[-batch:]
        ptr = int(self.queue_ptr.item())
        first = min(batch, self.queue_size - ptr)
        self.queue[ptr : ptr + first] = audio[:first]
        if first < batch:
            self.queue[: batch - first] = audio[first:]
        self.queue_ptr.fill_((ptr + batch) % self.queue_size)
        self.queue_filled.fill_(min(self.queue_size, int(self.queue_filled.item()) + batch))

    def _hard_negatives(self, eeg: torch.Tensor) -> torch.Tensor | None:
        filled = int(self.queue_filled.item())
        if filled <= 0 or self.queue_negatives <= 0:
            return None
        queue = self.queue[:filled]
        k = min(self.queue_negatives, filled)
        scores = F.normalize(eeg.detach(), dim=-1) @ F.normalize(queue, dim=-1).T
        candidate_score = scores.max(dim=0).values
        indices = torch.topk(candidate_score, k=k, dim=0).indices
        return queue[indices]

    def forward(self, z: torch.Tensor, audio_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        eeg = F.normalize(self.eeg_proj(pool_tokens(z)), dim=-1)
        audio = F.normalize(self.audio_proj(audio_embedding.float()), dim=-1)
        hard = self._hard_negatives(eeg)
        extra = hard
        loss, logits = info_nce_logits(eeg, audio, extra, temperature=torch.exp(self.log_temperature))
        if self.training:
            self._enqueue(audio)
        return {
            "loss": loss,
            "logits": logits,
            "eeg_embedding": eeg,
            "audio_embedding": audio,
            "queue_filled": self.queue_filled.float(),
        }


class TokenMetrics(nn.Module):
    """Module wrapper around token usage metrics."""

    def __init__(self, codebook_size: int):
        super().__init__()
        self.codebook_size = int(codebook_size)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        return token_usage_metrics(tokens, self.codebook_size)

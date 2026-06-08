from __future__ import annotations

import torch
import torch.nn as nn

from .model import EEGEncoder
from .utils import mean_pool_masked


class EEGSpeechAlignmentModel(nn.Module):
    def __init__(
        self,
        n_channels_eeg: int,
        hidden_dim: int,
        speech_embedding_dim: int,
        prosody_dim: int,
        num_labels: int,
        latent_dim: int = 192,
        use_label_head: bool = True,
        use_subject_demo_head: bool = False,
        num_subjects: int | None = None,
        subject_embedding_dim: int = 64,
    ):
        super().__init__()
        self.encoder = EEGEncoder(in_channels=n_channels_eeg, hidden_dim=hidden_dim)
        self.projector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )
        self.speech_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, speech_embedding_dim),
        )
        self.prosody_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, prosody_dim),
        )
        self.label_head = None
        if use_label_head:
            self.label_head = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, num_labels),
            )
        self.subject_embedding = None
        self.subject_demo_head = None
        if use_subject_demo_head:
            if num_subjects is None:
                raise ValueError("num_subjects is required when use_subject_demo_head=True")
            self.subject_embedding = nn.Embedding(num_subjects, subject_embedding_dim)
            self.subject_demo_head = nn.Sequential(
                nn.LayerNorm(latent_dim + subject_embedding_dim),
                nn.Linear(latent_dim + subject_embedding_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, speech_embedding_dim),
            )

    def forward(
        self,
        eeg: torch.Tensor,
        subject_indices: torch.Tensor | None = None,
        valid_steps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        sequence_latent = self.encoder(eeg)
        pooled_latent = mean_pool_masked(sequence_latent, valid_steps=valid_steps)
        neural_speech_latent = self.projector(pooled_latent)
        speech_embedding = self.speech_head(neural_speech_latent)
        prosody = self.prosody_head(neural_speech_latent)
        outputs = {
            "sequence_latent": sequence_latent,
            "pooled_latent": pooled_latent,
            "neural_speech_latent": neural_speech_latent,
            "speech_embedding": speech_embedding,
            "prosody": prosody,
        }
        if self.label_head is not None:
            outputs["label_logits"] = self.label_head(neural_speech_latent)
        if self.subject_embedding is not None and self.subject_demo_head is not None and subject_indices is not None:
            subject_latent = self.subject_embedding(subject_indices.long())
            outputs["subject_conditioned_embedding"] = self.subject_demo_head(
                torch.cat([neural_speech_latent, subject_latent], dim=-1)
            )
        return outputs

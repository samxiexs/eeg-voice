from __future__ import annotations

import torch
import torch.nn as nn

from .model import EEGEncoder, PromptClassifier, VectorQuantizerEMA, WaveformDecoder


class SubjectConditionedEEG2WaveVQModel(nn.Module):
    def __init__(
        self,
        n_channels_eeg: int = 14,
        hidden_dim: int = 128,
        codebook_size: int = 512,
        vq_beta: float = 0.25,
        vq_decay: float = 0.99,
        output_samples: int = 24000,
        num_labels: int = 16,
        num_subjects: int = 1,
        subject_embedding_dim: int = 64,
    ):
        super().__init__()
        self.encoder = EEGEncoder(in_channels=n_channels_eeg, hidden_dim=hidden_dim)
        self.subject_embedding = nn.Embedding(num_subjects, subject_embedding_dim)
        self.subject_to_hidden = nn.Linear(subject_embedding_dim, hidden_dim)
        self.quantizer = VectorQuantizerEMA(
            num_embeddings=codebook_size,
            embedding_dim=hidden_dim,
            beta=vq_beta,
            decay=vq_decay,
        )
        self.decoder = WaveformDecoder(hidden_dim=hidden_dim, output_samples=output_samples)
        self.classifier = PromptClassifier(hidden_dim=hidden_dim, num_labels=num_labels)

    def forward(
        self,
        eeg: torch.Tensor,
        subject_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encoder(eeg)
        subject_latent = self.subject_to_hidden(self.subject_embedding(subject_indices.long())).unsqueeze(-1)
        z_e = z_e + subject_latent
        z_q, vq_loss, codes, perplexity = self.quantizer(z_e)
        recon = self.decoder(z_q)
        logits = self.classifier(z_q)
        return recon, vq_loss, codes, perplexity, logits

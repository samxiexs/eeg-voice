from __future__ import annotations

import torch
import torch.nn as nn

from .model import EEGEncoder
from .utils import adaptive_avg_pool_masked, mean_pool_masked


class EEGSpeechAlignmentModel(nn.Module):
    def __init__(
        self,
        n_channels_eeg: int,
        hidden_dim: int,
        speech_embedding_dim: int,
        prosody_dim: int,
        num_labels: int,
        latent_dim: int = 192,
        target_steps: int = 1,
        use_label_head: bool = True,
        use_subject_demo_head: bool = False,
        num_subjects: int | None = None,
        subject_embedding_dim: int = 64,
        use_codec_scale_head: bool = False,
        use_phoneme_head: bool = False,
        num_phoneme_tokens: int = 0,
        phoneme_steps: int = 0,
    ):
        super().__init__()
        self.target_steps = int(target_steps)
        self.phoneme_steps = int(phoneme_steps)
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
        self.prosody_head = None
        if int(prosody_dim) > 0:
            self.prosody_head = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, max(latent_dim // 2, 1)),
                nn.GELU(),
                nn.Linear(max(latent_dim // 2, 1), prosody_dim),
            )
        self.label_head = None
        if use_label_head:
            self.label_head = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, num_labels),
            )
        self.codec_scale_head = None
        if use_codec_scale_head:
            self.codec_scale_head = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, max(latent_dim // 2, 1)),
                nn.GELU(),
                nn.Linear(max(latent_dim // 2, 1), 1),
            )
        self.phoneme_head = None
        if use_phoneme_head:
            if int(num_phoneme_tokens) <= 0 or int(phoneme_steps) <= 0:
                raise ValueError("num_phoneme_tokens and phoneme_steps are required when use_phoneme_head=True")
            self.phoneme_head = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, int(num_phoneme_tokens)),
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
        if self.target_steps == 1:
            pooled_sequence_latent = pooled_latent.unsqueeze(-1)
        else:
            pooled_sequence_latent = adaptive_avg_pool_masked(
                sequence_latent,
                target_steps=self.target_steps,
                valid_steps=valid_steps,
            )
        batch_size, hidden_dim, target_steps = pooled_sequence_latent.shape
        step_latent = pooled_sequence_latent.transpose(1, 2).reshape(batch_size * target_steps, hidden_dim)
        neural_step_latent = self.projector(step_latent).view(batch_size, target_steps, -1)
        speech_sequence = self.speech_head(neural_step_latent.reshape(batch_size * target_steps, -1)).view(
            batch_size,
            target_steps,
            -1,
        )
        neural_speech_latent = neural_step_latent.mean(dim=1)
        speech_embedding = speech_sequence.mean(dim=1)
        outputs = {
            "sequence_latent": sequence_latent,
            "pooled_latent": pooled_latent,
            "pooled_sequence_latent": pooled_sequence_latent,
            "neural_step_latent": neural_step_latent,
            "neural_speech_latent": neural_speech_latent,
            "speech_sequence": speech_sequence,
            "speech_embedding": speech_embedding,
        }
        if self.prosody_head is not None:
            outputs["prosody"] = self.prosody_head(neural_speech_latent)
        if self.label_head is not None:
            outputs["label_logits"] = self.label_head(neural_speech_latent)
        if self.codec_scale_head is not None:
            outputs["codec_log_rms"] = self.codec_scale_head(neural_speech_latent).squeeze(-1)
        if self.phoneme_head is not None:
            phoneme_latent = neural_step_latent
            if phoneme_latent.shape[1] != self.phoneme_steps:
                phoneme_latent = nn.functional.adaptive_avg_pool1d(
                    phoneme_latent.transpose(1, 2),
                    self.phoneme_steps,
                ).transpose(1, 2)
            outputs["phoneme_logits"] = self.phoneme_head(
                phoneme_latent.reshape(batch_size * self.phoneme_steps, -1)
            ).view(batch_size, self.phoneme_steps, -1)
        if self.subject_embedding is not None and self.subject_demo_head is not None and subject_indices is not None:
            subject_latent = self.subject_embedding(subject_indices.long())
            outputs["subject_conditioned_embedding"] = self.subject_demo_head(
                torch.cat([neural_speech_latent, subject_latent], dim=-1)
            )
        return outputs

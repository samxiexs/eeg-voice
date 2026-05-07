"""Token-centric EEG voice model v0.1.

The tokenizer remains the main model path. Voice/content branches consume the
quantized latent representation only as downstream probes or alignment heads.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .heads import PhonemeSequenceHead, SegmentContrastiveHead, TokenMetrics, VoiceProfileHead, pool_tokens
from .tokenizer import BrainStyleEEGTokenizerConfig, BrainStyleEEGTokenizerV0


@dataclass
class TokenCentricEEGVoiceConfig:
    tokenizer: BrainStyleEEGTokenizerConfig
    audio_embedding_dim: int = 11
    projection_dim: int = 256
    contrastive_temperature: float = 0.07
    phoneme_classes: int = 48
    pitch_dim: int = 1
    timbre_dim: int = 8
    speaker_dim: int = 128
    style_classes: int = 5
    dropout: float = 0.1


class TokenCentricEEGVoiceModelV01(nn.Module):
    """EEG -> tokens as the trunk, with optional downstream alignment heads."""

    def __init__(self, config: TokenCentricEEGVoiceConfig):
        super().__init__()
        self.config = config
        dim = config.tokenizer.dim
        self.tokenizer = BrainStyleEEGTokenizerV0(config.tokenizer)
        self.content_projection = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, config.projection_dim),
            nn.GELU(),
            nn.Linear(config.projection_dim, config.projection_dim),
        )
        self.phoneme_head = PhonemeSequenceHead(
            dim=dim,
            num_phoneme_classes=config.phoneme_classes,
            dropout=config.dropout,
        )
        self.segment_head = SegmentContrastiveHead(
            eeg_dim=dim,
            audio_dim=config.audio_embedding_dim,
            proj_dim=config.projection_dim,
            temperature=config.contrastive_temperature,
        )
        self.voice_profile_head = VoiceProfileHead(
            dim=dim,
            pitch_dim=config.pitch_dim,
            timbre_dim=config.timbre_dim,
            speaker_dim=config.speaker_dim,
            style_classes=config.style_classes,
            dropout=config.dropout,
        )
        self.token_metrics = TokenMetrics(config.tokenizer.codebook_size)

    def forward(
        self,
        eeg: torch.Tensor,
        sensor_pos: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        sensor_type: torch.Tensor | None = None,
        audio_embedding: torch.Tensor | None = None,
        phoneme_labels: torch.Tensor | None = None,
        voice_targets: dict[str, torch.Tensor] | None = None,
        compute_loss: bool = True,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        token_out = self.tokenizer(
            eeg=eeg,
            sensor_pos=sensor_pos,
            channel_mask=channel_mask,
            sensor_type=sensor_type,
            compute_loss=compute_loss,
        )
        z_q = token_out["z_q"]
        tokens = token_out["tokens"]
        content_embedding = self.content_projection(pool_tokens(z_q))
        phoneme_out = self.phoneme_head(z_q, phoneme_labels)
        voice_out = self.voice_profile_head(z_q, voice_targets)

        out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "z": token_out["z"],
            "z_q": z_q,
            "tokens": tokens,
            "x_rec": token_out["x_rec"],
            "target": token_out["target"],
            "commitment_loss": token_out["commitment_loss"],
            "token_metrics": self.token_metrics(tokens),
            "content_embedding": content_embedding,
            "phoneme_logits": phoneme_out["logits"],
            "pitch_pred": voice_out["pitch_pred"],
            "timbre_pred": voice_out["timbre_pred"],
            "speaker_embedding": voice_out["speaker_embedding"],
            "style_logits": voice_out["style_logits"],
        }
        if compute_loss and "losses" in token_out:
            out["tokenizer_losses"] = token_out["losses"]

        branch_losses = {}
        if "loss" in phoneme_out:
            branch_losses["phoneme_loss"] = phoneme_out["loss"]
        if "loss" in voice_out:
            branch_losses["voice_profile_loss"] = voice_out["loss"]
        if audio_embedding is not None:
            retrieval_out = self.segment_head(z_q, audio_embedding)
            out["retrieval_logits"] = retrieval_out["logits"]
            out["retrieval_eeg_embedding"] = retrieval_out["eeg_embedding"]
            out["retrieval_audio_embedding"] = retrieval_out["audio_embedding"]
            branch_losses["retrieval_loss"] = retrieval_out["loss"]

        if branch_losses:
            out["branch_losses"] = branch_losses
            if compute_loss:
                total = sum(branch_losses.values())
                if "losses" in token_out:
                    total = total + token_out["losses"]["loss"]
                out["loss"] = total
        elif compute_loss and "losses" in token_out:
            out["loss"] = token_out["losses"]["loss"]
        return out

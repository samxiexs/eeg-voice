"""Direct EEG-only -> EnCodec latent model.

Unlike the factored model, this path does not accept subject ids or learned
speaker embeddings. Content and voice/style information must be carried by EEG.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.feis_factored.encoder import SpatialTemporalEEGEncoder


@dataclass
class DirectEEG2SpeechConfig:
    n_channels_eeg: int = 14
    d_model: int = 256
    cond_dim: int = 32
    num_labels: int = 16
    num_stages: int = 2
    target_steps: int = 75
    target_dim: int = 128
    num_blocks: int = 5
    kernel_size: int = 5
    channel_dropout: float = 0.2
    dropout: float = 0.2
    num_transformer_layers: int = 3
    num_heads: int = 8
    ff_mult: int = 4


class DirectEEG2Speech(nn.Module):
    """Predict decoder-compatible latent speech directly from EEG."""

    def __init__(self, cfg: DirectEEG2SpeechConfig):
        super().__init__()
        self.cfg = cfg
        self.stage_embedding = nn.Embedding(cfg.num_stages, cfg.cond_dim)
        nn.init.normal_(self.stage_embedding.weight, std=0.02)

        self.encoder = SpatialTemporalEEGEncoder(
            in_channels=cfg.n_channels_eeg,
            d_model=cfg.d_model,
            cond_dim=cfg.cond_dim,
            target_steps=cfg.target_steps,
            num_blocks=cfg.num_blocks,
            kernel_size=cfg.kernel_size,
            channel_dropout=cfg.channel_dropout,
            dropout=cfg.dropout,
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.d_model * cfg.ff_mult,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_model = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_transformer_layers)
        self.latent_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.target_dim),
        )
        self.content_classifier = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.num_labels),
        )
        self.log_rms_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(self, eeg: torch.Tensor, stage_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        cond = self.stage_embedding(stage_idx.long())
        seq = self.encoder(eeg, cond).transpose(1, 2)  # [B, T, d_model]
        seq = self.sequence_model(seq)
        pooled = seq.mean(dim=1)
        pred_latent = self.latent_head(seq)
        return {
            "pred_latent": pred_latent,
            "pred_log_rms": self.log_rms_head(pooled).squeeze(-1),
            "content_logits": self.content_classifier(pooled),
            # Metadata-only voice diagnostics use the generated latent summary.
            # No subject id or separate subject-supervised head is used.
            "voice_embed": pred_latent.mean(dim=1),
            "pooled": pooled,
            "seq": seq,
        }

    @torch.no_grad()
    def generate(self, eeg: torch.Tensor, stage_idx: torch.Tensor) -> torch.Tensor:
        return self.forward(eeg, stage_idx)["pred_latent"]

    @torch.no_grad()
    def generate_full(self, eeg: torch.Tensor, stage_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(eeg, stage_idx)
        return out["pred_latent"], out["pred_log_rms"]

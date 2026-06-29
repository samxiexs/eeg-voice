from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .encoder import SpatialTemporalEEGEncoder, TransformerEEGEncoder
from .model import _masked_time_mean


@dataclass
class RetrievalFirstConfig:
    n_channels_eeg: int = 62
    d_model: int = 256
    cond_dim: int = 64
    num_labels: int = 11
    num_stages: int = 1
    core_steps: int = 64
    core_dim: int = 80
    audio_embed_dim: int = 768
    num_blocks: int = 6
    kernel_size: int = 5
    channel_dropout: float = 0.15
    dropout: float = 0.15
    num_channel_experts: int = 1
    encoder_kind: str = "cnn"
    transformer_layers: int = 4
    transformer_heads: int = 4
    patch_stride: int = 4
    instance_norm: bool = True
    use_channel_reliability: bool = True
    semantic_token_vocab: int = 0
    semantic_token_steps: int = 50


class KaraOneRetrievalFirst(nn.Module):
    """Subject-agnostic EEG encoder for v6.1 retrieval-first training.

    The main output is an EEG embedding aligned to a waveform-derived speech SSL
    embedding. A small Mel residual head is optional and only used after
    retrieval is meaningful; generation can therefore fall back to retrieved
    train-bank active-core priors instead of template regression.
    """

    def __init__(self, cfg: RetrievalFirstConfig):
        super().__init__()
        self.cfg = cfg
        self.stage_embedding = nn.Embedding(cfg.num_stages, cfg.cond_dim)
        nn.init.normal_(self.stage_embedding.weight, std=0.02)
        if str(cfg.encoder_kind) in {"transformer", "conformer"}:
            self.encoder = TransformerEEGEncoder(
                in_channels=cfg.n_channels_eeg,
                d_model=cfg.d_model,
                cond_dim=cfg.cond_dim,
                target_steps=cfg.core_steps,
                kernel_size=cfg.kernel_size,
                channel_dropout=cfg.channel_dropout,
                dropout=cfg.dropout,
                num_channel_experts=cfg.num_channel_experts,
                instance_norm=cfg.instance_norm,
                encoder_kind=str(cfg.encoder_kind),
                transformer_layers=int(cfg.transformer_layers),
                transformer_heads=int(cfg.transformer_heads),
                patch_stride=int(cfg.patch_stride),
                use_channel_reliability=bool(cfg.use_channel_reliability),
            )
        else:
            self.encoder = SpatialTemporalEEGEncoder(
                in_channels=cfg.n_channels_eeg,
                d_model=cfg.d_model,
                cond_dim=cfg.cond_dim,
                target_steps=cfg.core_steps,
                num_blocks=cfg.num_blocks,
                kernel_size=cfg.kernel_size,
                channel_dropout=cfg.channel_dropout,
                dropout=cfg.dropout,
                num_channel_experts=cfg.num_channel_experts,
                instance_norm=cfg.instance_norm,
                use_channel_reliability=bool(cfg.use_channel_reliability),
            )
        d = int(cfg.d_model)
        self.eeg_embed_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, cfg.audio_embed_dim),
        )
        self.core_delta_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, cfg.core_dim),
        )
        self.content_classifier = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, cfg.num_labels),
        )
        self.log_rms_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.log_peak_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.semantic_token_vocab = int(cfg.semantic_token_vocab)
        self.semantic_token_steps = int(cfg.semantic_token_steps)
        if self.semantic_token_vocab > 0:
            self.semantic_token_head = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(d, self.semantic_token_vocab),
            )

    def forward(
        self,
        eeg: torch.Tensor,
        stage_idx: torch.Tensor,
        eeg_valid_len: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        in_len = int(eeg.shape[-1])
        cond = self.stage_embedding(stage_idx.long())
        encoded, aux = self.encoder(eeg, cond, eeg_valid_len)
        seq = encoded.transpose(1, 2)
        pooled = _masked_time_mean(seq, eeg_valid_len, in_len)
        out = {
            "eeg_embed": self.eeg_embed_head(pooled),
            "pred_core_delta": self.core_delta_head(seq),
            "content_logits": self.content_classifier(pooled),
            "pred_log_rms": self.log_rms_head(pooled).squeeze(-1),
            "pred_log_peak": self.log_peak_head(pooled).squeeze(-1),
            "pooled": pooled,
        }
        if self.semantic_token_vocab > 0:
            token_seq = torch.nn.functional.interpolate(
                seq.transpose(1, 2),
                size=self.semantic_token_steps,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            out["semantic_token_logits"] = self.semantic_token_head(token_seq)
        out.update(aux)
        return out

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.direct_eeg2speech.encoder import SpatialTemporalEEGEncoder


@dataclass
class FEISMelConfig:
    n_channels_eeg: int = 14
    d_model: int = 192
    cond_dim: int = 16
    num_labels: int = 16
    target_steps: int = 64
    mel_dim: int = 80
    num_blocks: int = 5
    kernel_size: int = 5
    num_heads: int = 6
    num_cross_layers: int = 2
    ff_mult: int = 4
    channel_moe: bool = True
    moe_num_experts: int = 4
    moe_top_k: int = 2
    channel_dropout: float = 0.2
    dropout: float = 0.2


class FEISEEGToMel(nn.Module):
    """EEG-only FEIS model. No stage, talker, or identity input is accepted."""

    def __init__(self, cfg: FEISMelConfig):
        super().__init__()
        self.cfg = cfg
        self.null_condition = nn.Parameter(torch.zeros(cfg.cond_dim), requires_grad=False)
        self.encoder = SpatialTemporalEEGEncoder(
            in_channels=cfg.n_channels_eeg,
            d_model=cfg.d_model,
            cond_dim=cfg.cond_dim,
            target_steps=cfg.target_steps,
            num_blocks=cfg.num_blocks,
            kernel_size=cfg.kernel_size,
            channel_dropout=cfg.channel_dropout,
            dropout=cfg.dropout,
            use_channel_moe=cfg.channel_moe,
            moe_num_experts=cfg.moe_num_experts,
            moe_top_k=cfg.moe_top_k,
        )
        self.query = nn.Parameter(torch.randn(cfg.target_steps, cfg.d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.d_model * cfg.ff_mult,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.num_cross_layers)
        self.mel_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.mel_dim),
        )
        self.content_classifier = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.num_labels),
        )
        self.contrast_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.mel_dim),
        )
        self.log_rms_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(self, eeg: torch.Tensor) -> dict[str, torch.Tensor]:
        cond = self.null_condition.unsqueeze(0).expand(eeg.shape[0], -1)
        enc, aux = self.encoder(eeg, cond)
        memory = enc.transpose(1, 2)
        pooled = memory.mean(dim=1)
        queries = self.query.unsqueeze(0).expand(eeg.shape[0], -1, -1)
        decoded = self.decoder(queries, memory)
        out = {
            "pred_mel": self.mel_head(decoded),
            "content_logits": self.content_classifier(pooled),
            "contrast_embed": self.contrast_head(pooled),
            "pred_log_rms": self.log_rms_head(pooled).squeeze(-1),
            "pooled": pooled,
            "eeg_tokens": memory,
        }
        out.update(aux)
        return out


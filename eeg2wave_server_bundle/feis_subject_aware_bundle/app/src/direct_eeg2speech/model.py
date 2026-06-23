"""Direct EEG-only -> EnCodec latent model.

The model accepts only EEG and stage indices. Any talker/style information must
be carried by the neural signal itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.direct_eeg2speech.diffusion import LatentDiffusion
from src.direct_eeg2speech.encoder import SpatialTemporalEEGEncoder


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
    use_channel_moe: bool = False
    moe_num_experts: int = 4
    moe_top_k: int = 2
    use_latent_diffusion: bool = False
    diffusion_num_steps: int = 200
    diffusion_sample_steps: int = 24
    diffusion_layers: int = 2
    diffusion_time_dim: int = 128


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
            use_channel_moe=cfg.use_channel_moe,
            moe_num_experts=cfg.moe_num_experts,
            moe_top_k=cfg.moe_top_k,
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
        self.diffusion = (
            LatentDiffusion(
                latent_dim=cfg.target_dim,
                cond_dim=cfg.d_model,
                d_model=cfg.d_model,
                num_steps=cfg.diffusion_num_steps,
                num_layers=cfg.diffusion_layers,
                num_heads=cfg.num_heads,
                ff_mult=cfg.ff_mult,
                dropout=cfg.dropout,
                time_dim=cfg.diffusion_time_dim,
            )
            if cfg.use_latent_diffusion
            else None
        )

    def forward(
        self,
        eeg: torch.Tensor,
        stage_idx: torch.Tensor,
        *,
        target_seq: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cond = self.stage_embedding(stage_idx.long())
        enc, enc_aux = self.encoder(eeg, cond)
        seq = enc.transpose(1, 2)  # [B, T, d_model]
        seq = self.sequence_model(seq)
        pooled = seq.mean(dim=1)
        pred_latent = self.latent_head(seq)
        out = {
            "pred_latent": pred_latent,
            "pred_log_rms": self.log_rms_head(pooled).squeeze(-1),
            "content_logits": self.content_classifier(pooled),
            "latent_summary": pred_latent.mean(dim=1),
            "pooled": pooled,
            "seq": seq,
        }
        out.update(enc_aux)
        if self.diffusion is not None and target_seq is not None:
            out.update(
                self.diffusion.training_losses(
                    target_seq,
                    cond_seq=seq,
                    coarse_latent=pred_latent,
                )
            )
        return out

    @torch.no_grad()
    def generate(self, eeg: torch.Tensor, stage_idx: torch.Tensor, sample_steps: int | None = None) -> torch.Tensor:
        out = self.forward(eeg, stage_idx)
        if self.diffusion is None:
            return out["pred_latent"]
        return self.diffusion.sample_ddim(
            tuple(out["pred_latent"].shape),
            cond_seq=out["seq"],
            coarse_latent=out["pred_latent"],
            sample_steps=sample_steps or self.cfg.diffusion_sample_steps,
        )

    @torch.no_grad()
    def generate_full(
        self,
        eeg: torch.Tensor,
        stage_idx: torch.Tensor,
        sample_steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(eeg, stage_idx)
        pred_latent = out["pred_latent"]
        if self.diffusion is not None:
            pred_latent = self.diffusion.sample_ddim(
                tuple(pred_latent.shape),
                cond_seq=out["seq"],
                coarse_latent=pred_latent,
                sample_steps=sample_steps or self.cfg.diffusion_sample_steps,
            )
        return pred_latent, out["pred_log_rms"]

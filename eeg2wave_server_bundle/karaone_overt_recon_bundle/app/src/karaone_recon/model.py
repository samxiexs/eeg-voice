from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .encoder import SpatialTemporalEEGEncoder


@dataclass
class KaraOneConfig:
    n_channels_eeg: int = 62
    d_model: int = 256
    cond_dim: int = 64
    num_subjects: int = 14
    num_labels: int = 11
    num_stages: int = 1
    target_steps: int = 150
    target_dim: int = 128
    content_dim: int = 128
    speaker_dim: int = 64
    num_blocks: int = 6
    kernel_size: int = 5
    channel_dropout: float = 0.15
    dropout: float = 0.15
    num_experts: int = 1


class KaraOneEEG2Codec(nn.Module):
    def __init__(self, cfg: KaraOneConfig):
        super().__init__()
        self.cfg = cfg
        self.stage_embedding = nn.Embedding(cfg.num_stages, cfg.cond_dim)
        self.subject_condition = nn.Embedding(cfg.num_subjects, cfg.cond_dim)
        nn.init.normal_(self.stage_embedding.weight, std=0.02)
        nn.init.normal_(self.subject_condition.weight, std=0.02)

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
        d = cfg.d_model
        self.content_seq_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, cfg.content_dim),
        )
        self.content_embed_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.content_dim))
        self.content_classifier = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, cfg.num_labels))
        self.subject_classifier = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, cfg.num_subjects))

        self.speaker_embedding = nn.Embedding(cfg.num_subjects, cfg.speaker_dim)
        nn.init.normal_(self.speaker_embedding.weight, std=0.02)
        self.speaker_to_proto = nn.Linear(cfg.speaker_dim, cfg.target_dim)

        expert_in = cfg.content_dim + cfg.speaker_dim
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(expert_in, d),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(d, d),
                    nn.GELU(),
                    nn.Linear(d, cfg.target_dim),
                )
                for _ in range(max(1, int(cfg.num_experts)))
            ]
        )
        self.router = nn.Sequential(
            nn.LayerNorm(d + cfg.speaker_dim),
            nn.Linear(d + cfg.speaker_dim, d),
            nn.GELU(),
            nn.Linear(d, len(self.experts)),
        )
        self.log_rms_head = nn.Sequential(
            nn.LayerNorm(cfg.content_dim + cfg.speaker_dim),
            nn.Linear(cfg.content_dim + cfg.speaker_dim, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def forward(self, eeg: torch.Tensor, subject_idx: torch.Tensor, stage_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        cond = self.stage_embedding(stage_idx.long()) + self.subject_condition(subject_idx.long())
        seq = self.encoder(eeg, cond).transpose(1, 2)
        pooled = seq.mean(dim=1)
        content_seq = self.content_seq_head(seq)
        content_embed = self.content_embed_head(pooled)
        speaker = self.speaker_embedding(subject_idx.long())
        speaker_proto_pred = self.speaker_to_proto(speaker)

        speaker_seq = speaker.unsqueeze(1).expand(-1, content_seq.shape[1], -1)
        expert_input = torch.cat([content_seq, speaker_seq], dim=-1)
        expert_outputs = torch.stack([expert(expert_input) for expert in self.experts], dim=2)
        router_logits = self.router(torch.cat([pooled, speaker], dim=-1))
        router_probs = torch.softmax(router_logits, dim=-1)
        pred_latent = (expert_outputs * router_probs[:, None, :, None]).sum(dim=2)

        pred_log_rms = self.log_rms_head(torch.cat([content_seq.mean(dim=1), speaker], dim=-1)).squeeze(-1)
        return {
            "pred_latent": pred_latent,
            "pred_log_rms": pred_log_rms,
            "content_embed": content_embed,
            "content_logits": self.content_classifier(pooled),
            "subject_logits": self.subject_classifier(pooled),
            "speaker_proto_pred": speaker_proto_pred,
            "router_logits": router_logits,
            "router_probs": router_probs,
            "pooled": pooled,
        }

    @torch.no_grad()
    def generate_full(self, eeg: torch.Tensor, subject_idx: torch.Tensor, stage_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(eeg, subject_idx, stage_idx)
        return out["pred_latent"], out["pred_log_rms"]


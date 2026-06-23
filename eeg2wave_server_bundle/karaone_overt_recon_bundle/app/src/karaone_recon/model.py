from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .encoder import SpatialTemporalEEGEncoder


def _masked_time_mean(seq: torch.Tensor, eeg_valid_len: torch.Tensor | None, in_len: int) -> torch.Tensor:
    """Mean over time that ignores the zero-padded tail of each EEG window.

    `eeg_valid_len` (samples, out of `in_len`) is computed by the dataset but was
    previously unused, so padding diluted the utterance embedding. We map the
    valid fraction onto the encoder's output frames (padding sits at the end) and
    average only over the valid frames. Falls back to a plain mean when no length
    is given.
    """
    if eeg_valid_len is None:
        return seq.mean(dim=1)
    b, t_out, _ = seq.shape
    frac = (eeg_valid_len.float() / float(max(in_len, 1))).clamp(min=1.0 / t_out, max=1.0)
    valid_frames = (frac * t_out).ceil().clamp(min=1.0)  # [B]
    idx = torch.arange(t_out, device=seq.device).unsqueeze(0)  # [1, T]
    mask = (idx < valid_frames.unsqueeze(1)).to(seq.dtype)  # [B, T]
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (seq * mask.unsqueeze(-1)).sum(dim=1) / denom


@dataclass
class KaraOneConfig:
    n_channels_eeg: int = 62
    d_model: int = 256
    cond_dim: int = 64
    num_subjects: int = 14  # retained for checkpoint/data compatibility; UNUSED by the model (subject-agnostic)
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
    num_channel_experts: int = 1


class KaraOneEEG2Codec(nn.Module):
    def __init__(self, cfg: KaraOneConfig):
        super().__init__()
        self.cfg = cfg
        # Conditioning is task-only (stage); no subject-ID lookup. The model is
        # fully subject-agnostic: everything below is derived from the EEG itself.
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
            num_channel_experts=cfg.num_channel_experts,
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

        # Global utterance/voice embedding inferred from the EEG (replaces the old
        # per-subject speaker lookup table). cfg.speaker_dim is just its width.
        self.global_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, cfg.speaker_dim),
        )

        # Cross-modal CLIP head: projects the EEG utterance embedding into the
        # audio-latent space so a contrastive loss can align EEG with speech
        # (Defossez et al. 2022). This is the alignment signal that complements
        # the frame-wise regression; the audio side stays frozen (the EnCodec
        # target summary), so no subject identity is involved.
        self.clip_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, cfg.target_dim),
        )

        # Output projection head (content + global -> latent). With num_experts=1
        # this is a plain MLP; the channel-selecting MoE now lives in the encoder
        # (ChannelMoEFrontend), which is where channel filtering/clustering belongs.
        # num_experts>1 keeps an optional soft output mixture for ablation only.
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

    def forward(
        self,
        eeg: torch.Tensor,
        subject_idx: torch.Tensor,
        stage_idx: torch.Tensor,
        eeg_valid_len: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # NOTE: subject_idx is accepted for API compatibility (eval/synth/train call
        # sites) but is intentionally UNUSED — the model is subject-agnostic and the
        # output depends only on the EEG (and the task `stage`).
        del subject_idx
        in_len = int(eeg.shape[-1])
        cond = self.stage_embedding(stage_idx.long())
        encoded, channel_aux = self.encoder(eeg, cond)
        seq = encoded.transpose(1, 2)
        pooled = _masked_time_mean(seq, eeg_valid_len, in_len)  # ignores zero-padded tail
        content_seq = self.content_seq_head(seq)
        content_embed = self.content_embed_head(pooled)
        global_embed = self.global_head(pooled)  # EEG-derived global voice/utterance context

        global_seq = global_embed.unsqueeze(1).expand(-1, content_seq.shape[1], -1)
        expert_input = torch.cat([content_seq, global_seq], dim=-1)
        expert_outputs = torch.stack([expert(expert_input) for expert in self.experts], dim=2)
        router_logits = self.router(torch.cat([pooled, global_embed], dim=-1))
        router_probs = torch.softmax(router_logits, dim=-1)
        pred_latent = (expert_outputs * router_probs[:, None, :, None]).sum(dim=2)

        pred_log_rms = self.log_rms_head(torch.cat([content_seq.mean(dim=1), global_embed], dim=-1)).squeeze(-1)
        out = {
            "pred_latent": pred_latent,
            "pred_log_rms": pred_log_rms,
            "content_embed": content_embed,
            "content_logits": self.content_classifier(pooled),
            "clip_embed": self.clip_head(pooled),  # EEG side of the EEG<->audio contrastive alignment
            "router_logits": router_logits,
            "router_probs": router_probs,
            "pooled": pooled,
        }
        out.update(channel_aux)  # channel_gate, channel_assign, channel_balance (if encoder MoE on)
        return out

    @torch.no_grad()
    def generate_full(
        self,
        eeg: torch.Tensor,
        subject_idx: torch.Tensor,
        stage_idx: torch.Tensor,
        eeg_valid_len: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(eeg, subject_idx, stage_idx, eeg_valid_len)
        return out["pred_latent"], out["pred_log_rms"]


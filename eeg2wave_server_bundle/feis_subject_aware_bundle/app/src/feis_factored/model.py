"""Factored EEG->speech model.

    EEG ──► encoder(+FiLM stage) ──► content path  (decode WHAT, speaker-independent)
    subject_id ──► speaker embedding (WHO, given, not from EEG)
    (content_seq , speaker_embed) ──► generator ──► EnCodec-latent sequence [T,D]

Heads:
  - content_seq   : per-frame content features (drive the generator)
  - content_embed : pooled, for supervised contrastive + matching content prototype
  - content_logits: 16-way content classifier (read-out metric)
  - generator     : fuses content + speaker -> target latent

The split content(EEG) / speaker(id) is deliberate: it stops the model from
"cheating" by reading subject identity off the EEG to inflate scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .encoder import SpatialTemporalEEGEncoder


class _GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer (DANN): identity forward, negated grad backward."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd: float = 1.0):
    return _GradReverse.apply(x, lambd)


@dataclass
class FactoredConfig:
    n_channels_eeg: int = 14
    d_model: int = 256
    cond_dim: int = 32          # FiLM conditioning = STAGE (hear/imagine), not subject
    num_subjects: int = 20
    num_labels: int = 16
    num_stages: int = 2
    target_steps: int = 75      # EnCodec frames
    target_dim: int = 128       # EnCodec latent dim
    content_dim: int = 128
    speaker_dim: int = 64
    num_blocks: int = 5
    kernel_size: int = 5
    channel_dropout: float = 0.2
    dropout: float = 0.2
    adv_lambda: float = 1.0     # gradient-reversal strength (content disentanglement)


class FactoredEEG2Speech(nn.Module):
    def __init__(self, cfg: FactoredConfig):
        super().__init__()
        self.cfg = cfg
        # FiLM conditioning of the EEG encoder uses STAGE only.
        self.stage_embedding = nn.Embedding(cfg.num_stages, cfg.cond_dim)
        nn.init.normal_(self.stage_embedding.weight, std=0.02)
        self.encoder = SpatialTemporalEEGEncoder(
            in_channels=cfg.n_channels_eeg, d_model=cfg.d_model, cond_dim=cfg.cond_dim,
            target_steps=cfg.target_steps, num_blocks=cfg.num_blocks,
            kernel_size=cfg.kernel_size, channel_dropout=cfg.channel_dropout, dropout=cfg.dropout,
        )
        d = cfg.d_model
        # CONTENT (decoded from EEG)
        self.content_seq_head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d, cfg.content_dim),
        )
        self.content_embed_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.content_dim))
        self.content_classifier = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, cfg.num_labels),
        )
        # SPEAKER (from known subject id) + grounding to audio voice prototype
        self.speaker_embedding = nn.Embedding(cfg.num_subjects + 1, cfg.speaker_dim)
        nn.init.normal_(self.speaker_embedding.weight, std=0.02)
        self.speaker_to_proto = nn.Linear(cfg.speaker_dim, cfg.target_dim)   # -> match audio speaker prototype
        # ADVERSARY: try to predict subject FROM content (via GRL) -> forces content
        # to be speaker-independent (voice-conversion / DANN style disentanglement).
        self.subject_adversary = nn.Sequential(
            nn.LayerNorm(cfg.content_dim), nn.Linear(cfg.content_dim, cfg.content_dim), nn.GELU(),
            nn.Linear(cfg.content_dim, cfg.num_subjects),
        )
        # GENERATOR: (content_seq [T,content_dim] + speaker [speaker_dim]) -> latent [T, target_dim]
        self.generator = nn.Sequential(
            nn.Linear(cfg.content_dim + cfg.speaker_dim, d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, cfg.target_dim),
        )
        # ENERGY: predict decoded-wav target log-RMS from (content pooled + speaker).
        # Lets synthesis restore loudness with a MODEL-predicted scale (no target leak),
        # fixing the v1 "quiet, mean-collapsed" 17%-RMS output.
        self.log_rms_head = nn.Sequential(
            nn.LayerNorm(cfg.content_dim + cfg.speaker_dim),
            nn.Linear(cfg.content_dim + cfg.speaker_dim, d), nn.GELU(),
            nn.Linear(d, 1),
        )

    @property
    def unknown_subject_index(self) -> int:
        return self.cfg.num_subjects

    def forward(self, eeg: torch.Tensor, subject_idx: torch.Tensor | None,
                stage_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        b = eeg.shape[0]
        cond = self.stage_embedding(stage_idx.long())                  # [B, cond_dim]
        seq = self.encoder(eeg, cond).transpose(1, 2)                  # [B, T, d_model]
        pooled = seq.mean(dim=1)                                       # [B, d_model]

        content_seq = self.content_seq_head(seq)                      # [B, T, content_dim]
        content_embed = self.content_embed_head(pooled)              # [B, content_dim]
        content_logits = self.content_classifier(pooled)            # [B, num_labels]

        if subject_idx is None:
            sid = torch.full((b,), self.unknown_subject_index, device=eeg.device, dtype=torch.long)
        else:
            sid = subject_idx.long().clamp(min=0, max=self.cfg.num_subjects)
        speaker = self.speaker_embedding(sid)                        # [B, speaker_dim]
        speaker_proto_pred = self.speaker_to_proto(speaker)          # [B, target_dim]

        T = content_seq.shape[1]
        spk_seq = speaker.unsqueeze(1).expand(-1, T, -1)             # [B, T, speaker_dim]
        gen_in = torch.cat([content_seq, spk_seq], dim=-1)
        pred_latent = self.generator(gen_in)                        # [B, T, target_dim]

        # energy / loudness prediction (content pooled + speaker)
        content_pooled = content_seq.mean(dim=1)                    # [B, content_dim]
        pred_log_rms = self.log_rms_head(
            torch.cat([content_pooled, speaker], dim=-1)).squeeze(-1)   # [B]

        # adversary on content (gradient reversal -> remove subject info from content)
        subject_adv_logits = self.subject_adversary(grad_reverse(content_embed, self.cfg.adv_lambda))

        return {
            "pred_latent": pred_latent,
            "pred_log_rms": pred_log_rms,
            "content_seq": content_seq,
            "content_embed": content_embed,
            "content_logits": content_logits,
            "speaker": speaker,
            "speaker_proto_pred": speaker_proto_pred,
            "subject_adv_logits": subject_adv_logits,
            "pooled": pooled,
        }

    @torch.no_grad()
    def generate(self, eeg, subject_idx, stage_idx):
        return self.forward(eeg, subject_idx, stage_idx)["pred_latent"]

    @torch.no_grad()
    def generate_full(self, eeg, subject_idx, stage_idx):
        """Return both predicted latent and predicted log-RMS (for scaled synthesis)."""
        out = self.forward(eeg, subject_idx, stage_idx)
        return out["pred_latent"], out["pred_log_rms"]

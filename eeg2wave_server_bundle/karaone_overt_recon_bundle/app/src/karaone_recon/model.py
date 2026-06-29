from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import SpatialTemporalEEGEncoder, TransformerEEGEncoder
from .losses import grad_reverse


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
    encoder_kind: str = "cnn"           # cnn | transformer | conformer
    transformer_layers: int = 4
    transformer_heads: int = 4
    patch_stride: int = 4
    use_channel_reliability: bool = False
    decoder_scale_dim: int = 1
    # WS1 cross-subject domain adaptation
    instance_norm: bool = False        # per-trial RevIN normalization in the encoder (no subject id)
    use_domain_adv: bool = False       # subject-adversarial DANN head (train-only; inference subject-agnostic)
    # WS3 HuBERT auxiliary content head (0/absent => disabled)
    hubert_dim: int = 0
    hubert_steps: int = 50
    # v3 alignment-aware semantic-token heads (0/absent => disabled).
    semantic_token_vocab: int = 0
    semantic_token_steps: int = 50
    # v4.2 optional speech-core shift diagnostic head.
    shift_bins: int = 0
    # v5 active speech-core duration head (0/absent => disabled).
    duration_bins: int = 0


class KaraOneEEG2Codec(nn.Module):
    def __init__(self, cfg: KaraOneConfig):
        super().__init__()
        self.cfg = cfg
        # Conditioning is task-only (stage); no subject-ID lookup. The model is
        # fully subject-agnostic: everything below is derived from the EEG itself.
        self.stage_embedding = nn.Embedding(cfg.num_stages, cfg.cond_dim)
        nn.init.normal_(self.stage_embedding.weight, std=0.02)

        if str(cfg.encoder_kind) in {"transformer", "conformer"}:
            self.encoder = TransformerEEGEncoder(
                in_channels=cfg.n_channels_eeg,
                d_model=cfg.d_model,
                cond_dim=cfg.cond_dim,
                target_steps=cfg.target_steps,
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
                target_steps=cfg.target_steps,
                num_blocks=cfg.num_blocks,
                kernel_size=cfg.kernel_size,
                channel_dropout=cfg.channel_dropout,
                dropout=cfg.dropout,
                num_channel_experts=cfg.num_channel_experts,
                instance_norm=cfg.instance_norm,
                use_channel_reliability=bool(cfg.use_channel_reliability),
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
        self.ctc_classifier = nn.Sequential(nn.LayerNorm(cfg.content_dim), nn.Linear(cfg.content_dim, cfg.num_labels + 1))

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
        self.log_peak_head = nn.Sequential(
            nn.LayerNorm(cfg.content_dim + cfg.speaker_dim),
            nn.Linear(cfg.content_dim + cfg.speaker_dim, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )
        self.frame_energy_head = nn.Sequential(
            nn.LayerNorm(cfg.content_dim + cfg.speaker_dim),
            nn.Linear(cfg.content_dim + cfg.speaker_dim, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )
        self.active_head = nn.Sequential(
            nn.LayerNorm(cfg.content_dim + cfg.speaker_dim),
            nn.Linear(cfg.content_dim + cfg.speaker_dim, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )
        self.lag_head = nn.Sequential(
            nn.LayerNorm(d + cfg.speaker_dim),
            nn.Linear(d + cfg.speaker_dim, d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d, 2),
        )
        self.decoder_scale_head = nn.Sequential(
            nn.LayerNorm(cfg.speaker_dim),
            nn.Linear(cfg.speaker_dim, d),
            nn.GELU(),
            nn.Linear(d, max(1, int(cfg.decoder_scale_dim))),
        )

        # WS1: subject-adversarial DANN head. Trains to classify the subject from the
        # pooled EEG embedding through a gradient-reversal layer, so the ENCODER is
        # pushed to drop subject-specific information (cross-subject generalization).
        # Train-only: at inference the head is unused and `subject_idx` never feeds
        # the generative path, so the model stays subject-agnostic.
        self.use_domain_adv = bool(cfg.use_domain_adv)
        if self.use_domain_adv:
            self.subject_classifier = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(d, max(2, cfg.num_subjects)),
            )

        # WS3: HuBERT auxiliary content head. Maps the per-frame content sequence to a
        # HuBERT-feature sequence (interpolated to `hubert_steps`), giving a
        # content-bearing semantic target alongside the rendered mel/latent target.
        self.hubert_dim = int(cfg.hubert_dim)
        self.hubert_steps = int(cfg.hubert_steps)
        if self.hubert_dim > 0:
            self.hubert_head = nn.Sequential(
                nn.LayerNorm(cfg.content_dim),
                nn.Linear(cfg.content_dim, d),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(d, self.hubert_dim),
            )
        self.semantic_token_vocab = int(cfg.semantic_token_vocab)
        self.semantic_token_steps = int(cfg.semantic_token_steps)
        if self.semantic_token_vocab > 0:
            self.semantic_token_head = nn.Sequential(
                nn.LayerNorm(cfg.content_dim),
                nn.Linear(cfg.content_dim, d),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(d, self.semantic_token_vocab),
            )
        self.shift_bins = int(cfg.shift_bins)
        if self.shift_bins > 0:
            self.shift_head = nn.Sequential(
                nn.LayerNorm(d + cfg.speaker_dim),
                nn.Linear(d + cfg.speaker_dim, d),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(d, self.shift_bins),
            )
        self.duration_bins = int(cfg.duration_bins)
        if self.duration_bins > 0:
            self.duration_head = nn.Sequential(
                nn.LayerNorm(d + cfg.speaker_dim),
                nn.Linear(d + cfg.speaker_dim, d),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(d, self.duration_bins),
            )

    def forward(
        self,
        eeg: torch.Tensor,
        subject_idx: torch.Tensor,
        stage_idx: torch.Tensor,
        eeg_valid_len: torch.Tensor | None = None,
        lambda_domain: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        # NOTE: subject_idx is accepted for API compatibility (eval/synth/train call
        # sites) and is UNUSED by the generative path — the reconstruction depends only
        # on the EEG (and the task `stage`). It is consumed ONLY by the optional
        # subject-adversarial DANN head (train-time), which exists to REMOVE subject
        # information, keeping inference subject-agnostic.
        del subject_idx
        in_len = int(eeg.shape[-1])
        cond = self.stage_embedding(stage_idx.long())
        encoded, channel_aux = self.encoder(eeg, cond, eeg_valid_len)
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
        pred_log_peak = self.log_peak_head(torch.cat([content_seq.mean(dim=1), global_embed], dim=-1)).squeeze(-1)
        pred_frame_log_energy = self.frame_energy_head(expert_input).squeeze(-1)
        pred_active_logits = self.active_head(expert_input).squeeze(-1)
        lag_params = self.lag_head(torch.cat([pooled, global_embed], dim=-1))
        pred_log_decoder_scale = self.decoder_scale_head(global_embed)
        out = {
            "pred_latent": pred_latent,
            "pred_log_rms": pred_log_rms,
            "pred_log_peak": pred_log_peak,
            "pred_frame_log_energy": pred_frame_log_energy,
            "pred_active_logits": pred_active_logits,
            "pred_lag_mu": lag_params[:, 0],
            "pred_lag_log_sigma": lag_params[:, 1].clamp(min=-5.0, max=3.0),
            "pred_log_decoder_scale": pred_log_decoder_scale,
            "content_embed": content_embed,
            "content_logits": self.content_classifier(pooled),
            "ctc_logits": self.ctc_classifier(content_seq),
            "clip_embed": self.clip_head(pooled),  # EEG side of the EEG<->audio contrastive alignment
            "router_logits": router_logits,
            "router_probs": router_probs,
            "pooled": pooled,
        }
        if self.use_domain_adv:
            # Gradient-reversal: classifier learns subject; encoder unlearns it.
            out["subject_logits"] = self.subject_classifier(grad_reverse(pooled, lambda_domain))
        if self.hubert_dim > 0:
            # Interpolate the content sequence to the HuBERT frame rate, then project.
            content_t = F.interpolate(
                content_seq.transpose(1, 2), size=self.hubert_steps, mode="linear", align_corners=False
            ).transpose(1, 2)
            out["pred_hubert"] = self.hubert_head(content_t)  # [B, hubert_steps, hubert_dim]
        if self.semantic_token_vocab > 0:
            token_t = F.interpolate(
                content_seq.transpose(1, 2), size=self.semantic_token_steps, mode="linear", align_corners=False
            ).transpose(1, 2)
            out["semantic_token_logits"] = self.semantic_token_head(token_t)
        if self.shift_bins > 0:
            shift_logits = self.shift_head(torch.cat([pooled, global_embed], dim=-1))
            shift_probs = torch.softmax(shift_logits, dim=-1)
            grid = torch.linspace(-1.0, 1.0, steps=self.shift_bins, device=shift_logits.device, dtype=shift_logits.dtype)
            out["pred_shift_logits"] = shift_logits
            out["pred_shift_mu"] = (shift_probs * grid.unsqueeze(0)).sum(dim=-1)
        if self.duration_bins > 0:
            duration_logits = self.duration_head(torch.cat([pooled, global_embed], dim=-1))
            duration_probs = torch.softmax(duration_logits, dim=-1)
            grid = torch.arange(1, self.duration_bins + 1, device=duration_logits.device, dtype=duration_logits.dtype)
            out["pred_duration_logits"] = duration_logits
            out["pred_duration_mu"] = (duration_probs * grid.unsqueeze(0)).sum(dim=-1)
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

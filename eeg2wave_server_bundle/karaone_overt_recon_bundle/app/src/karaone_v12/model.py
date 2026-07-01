from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.karaone_v11.model import KaraOneV11Config, KaraOneV11TokenGenerator
from src.karaone_v9.model import resize_sequence


@dataclass
class KaraOneV12Config(KaraOneV11Config):
    sample_rate: int = 16000
    duration_sec: float = 2.0
    active_mask_steps: int = 200
    max_lag_sec: float = 1.0


class TimeAnchorHead(nn.Module):
    def __init__(self, cfg: KaraOneV12Config):
        super().__init__()
        self.cfg = cfg
        self.scalar_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 3),
        )
        self.active_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, 1))
        self.boundary_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, 2))

    def forward(self, pooled: torch.Tensor, prosody_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.scalar_head(pooled)
        onset = torch.sigmoid(raw[:, 0]) * float(self.cfg.duration_sec)
        duration = 0.05 + torch.sigmoid(raw[:, 1]) * max(0.05, float(self.cfg.duration_sec) - 0.05)
        duration = torch.minimum(duration, torch.clamp(float(self.cfg.duration_sec) - onset, min=0.05))
        lag = torch.tanh(raw[:, 2]) * float(self.cfg.max_lag_sec)
        active_tokens = resize_sequence(prosody_tokens, int(self.cfg.active_mask_steps))
        active_logits = self.active_head(active_tokens).squeeze(-1)
        boundary_logits = self.boundary_head(active_tokens)
        return {
            "pred_onset_sec": onset,
            "pred_duration_sec": duration,
            "pred_center_sec": onset + 0.5 * duration,
            "pred_lag_sec": lag,
            "pred_active_mask_logits": active_logits,
            "pred_token_boundary_logits": boundary_logits,
        }


class TemporalPlacementAdapter(nn.Module):
    """Places an active sequence into a fixed temporal canvas using onset/duration."""

    def __init__(self, cfg: KaraOneV12Config):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        active_seq: torch.Tensor,
        onset_sec: torch.Tensor,
        duration_sec: torch.Tensor,
        *,
        target_steps: int | None = None,
    ) -> torch.Tensor:
        steps = int(target_steps or active_seq.shape[1])
        if active_seq.ndim != 3:
            raise ValueError(f"Expected [B,T,D], got {tuple(active_seq.shape)}")
        out = active_seq.new_zeros(active_seq.shape[0], steps, active_seq.shape[2])
        for idx in range(active_seq.shape[0]):
            start = int(round(float(onset_sec[idx].detach().cpu()) / max(float(self.cfg.duration_sec), 1e-6) * steps))
            dur = int(round(float(duration_sec[idx].detach().cpu()) / max(float(self.cfg.duration_sec), 1e-6) * steps))
            start = max(0, min(steps - 1, start))
            dur = max(1, min(steps - start, dur))
            resized = F.interpolate(active_seq[idx : idx + 1].transpose(1, 2), size=dur, mode="linear", align_corners=False).transpose(1, 2)
            out[idx : idx + 1, start : start + dur] = resized
        return out


class KaraOneV12TokenGenerator(KaraOneV11TokenGenerator):
    """v11 token generator plus time-anchor prediction and temporal placement."""

    def __init__(self, cfg: KaraOneV12Config, codec_codebook: torch.Tensor | None = None):
        super().__init__(cfg, codec_codebook=codec_codebook)
        self.cfg: KaraOneV12Config = cfg
        self.time_anchor_head = TimeAnchorHead(cfg)
        self.temporal_placement = TemporalPlacementAdapter(cfg)

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        out = super().forward(*args, **kwargs)
        out.update(self.time_anchor_head(out["pooled"], out["prosody_tokens"]))
        out["pred_codec_seq_placed"] = self.temporal_placement(
            out["pred_codec_seq"],
            out["pred_onset_sec"],
            out["pred_duration_sec"],
            target_steps=out["pred_codec_seq"].shape[1],
        )
        return out

    @torch.no_grad()
    def generate_codec_tokens(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        was_training = self.training
        self.eval()
        out = self.forward(*args, **kwargs)
        out["pred_codec_token_ids"] = out["codec_token_logits"].argmax(dim=-1)
        out["pred_codec_seq"] = self.codec_codebook[out["pred_codec_token_ids"].clamp(0, self.codec_codebook.shape[0] - 1)]
        out["pred_codec_seq_placed"] = self.temporal_placement(
            out["pred_codec_seq"],
            out["pred_onset_sec"],
            out["pred_duration_sec"],
            target_steps=out["pred_codec_seq"].shape[1],
        )
        if was_training:
            self.train()
        return out


__all__ = ["KaraOneV12Config", "KaraOneV12TokenGenerator", "TemporalPlacementAdapter", "TimeAnchorHead"]

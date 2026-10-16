"""Coordinate-aware variable-channel EEG content model.

The encoder is a focused migration of ``open_vocab_v3/cp_temporal.py`` from
the reference bundle.  Dataset identity is used only for an input affine and
post-fusion normalization; labels, audio, subjects and conditions are never
model inputs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradientReverse(torch.autograd.Function):
    """Identity forward with a configurable reversed backward signal."""

    @staticmethod
    def forward(ctx, value: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = float(strength)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.strength * gradient, None


def reverse_gradient(value: torch.Tensor, strength: float) -> torch.Tensor:
    return _GradientReverse.apply(value, float(strength))


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype).unsqueeze(-1)
    return (value * weight).sum(1) / weight.sum(1).clamp_min(1.0)


class ConformerBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float, kernel: int = 31):
        super().__init__()
        hidden = dimension * 4
        self.ff1_norm = nn.LayerNorm(dimension)
        self.ff1 = nn.Sequential(nn.Linear(dimension, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, dimension))
        self.attn_norm = nn.LayerNorm(dimension)
        self.attn = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.conv_norm = nn.LayerNorm(dimension)
        self.conv_in = nn.Conv1d(dimension, dimension * 2, 1)
        self.depthwise = nn.Conv1d(dimension, dimension, kernel, padding=kernel // 2, groups=dimension)
        self.conv_out = nn.Conv1d(dimension, dimension, 1)
        self.ff2_norm = nn.LayerNorm(dimension)
        self.ff2 = nn.Sequential(nn.Linear(dimension, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, dimension))
        self.out_norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        value = value + 0.5 * self.dropout(self.ff1(self.ff1_norm(value)))
        norm = self.attn_norm(value)
        attended = self.attn(norm, norm, norm, key_padding_mask=~mask, need_weights=False)[0]
        value = value + self.dropout(attended)
        conv = self.conv_norm(value).transpose(1, 2)
        left, gate = self.conv_in(conv).chunk(2, dim=1)
        conv = self.conv_out(F.silu(self.depthwise(left * torch.sigmoid(gate)))).transpose(1, 2)
        value = value + self.dropout(conv)
        value = value + 0.5 * self.dropout(self.ff2(self.ff2_norm(value)))
        return torch.where(mask.unsqueeze(-1), self.out_norm(value), torch.zeros_like(value))


class ConformerStack(nn.Module):
    def __init__(self, dimension: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([ConformerBlock(dimension, heads, dropout) for _ in range(layers)])

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value = layer(value, mask)
        return value


class DatasetInputAdapter(nn.Module):
    """Two scalar affine frontends; no dataset vector enters content tokens."""

    def __init__(self, datasets: int):
        super().__init__()
        self.log_gain = nn.Parameter(torch.zeros(datasets))
        self.bias = nn.Parameter(torch.zeros(datasets))

    def forward(self, eeg: torch.Tensor, dataset_id: torch.Tensor) -> torch.Tensor:
        # Keep the affine adapter expressive while preventing an accidental
        # optimizer excursion from overflowing exp() and corrupting EEG tokens.
        gain = self.log_gain[dataset_id].clamp(-5.0, 5.0).exp().view(-1, 1, 1)
        bias = self.bias[dataset_id].view(-1, 1, 1)
        return eeg * gain + bias


@dataclass
class JointState:
    local: torch.Tensor
    global_embedding: torch.Tensor
    mfcc: torch.Tensor
    phoneme_logits: torch.Tensor
    token_mask: torch.Tensor
    baseline_mfcc: torch.Tensor | None = None
    residual_mfcc: torch.Tensor | None = None
    residual_z: torch.Tensor | None = None
    predicted_duration: torch.Tensor | None = None
    duration_z: torch.Tensor | None = None
    activity_logits: torch.Tensor | None = None
    subject_logits: torch.Tensor | None = None


@dataclass
class _RawState:
    local: torch.Tensor
    global_feature: torch.Tensor
    mfcc: torch.Tensor
    phoneme_logits: torch.Tensor
    token_mask: torch.Tensor
    duration: torch.Tensor
    activity_logits: torch.Tensor
    subject_logits: torch.Tensor | None = None


class JointEEGContentModel(nn.Module):
    def __init__(self, *, dimension: int = 192, heads: int = 6, layers: int = 4,
                 local_layers: int = 2, dropout: float = 0.1, token_steps: int = 96,
                 target_frames: int = 161, mfcc_dimension: int = 39,
                 hubert_dimension: int = 768, phoneme_classes: int = 64,
                 datasets: int = 2, zero_centered: bool = False,
                 teacher_dimension: int = 0, subject_adversary_classes: int = 0,
                 subject_adversary_strength: float = 0.0,
                 duration_standardized: bool = False):
        super().__init__()
        self.token_steps = int(token_steps)
        self.target_frames = int(target_frames)
        self.zero_centered = bool(zero_centered)
        self.teacher_dimension = int(teacher_dimension)
        self.duration_standardized = bool(duration_standardized)
        self.subject_adversary_strength = float(subject_adversary_strength)
        if self.teacher_dimension < 0:
            raise ValueError("teacher_dimension must be non-negative")
        if self.duration_standardized and not self.zero_centered:
            raise ValueError("duration_standardized requires zero_centered=True")
        branch = dimension // 3
        widths = (branch, branch, dimension - 2 * branch)
        self.input_adapter = DatasetInputAdapter(datasets)
        self.temporal = nn.ModuleList([
            nn.Sequential(nn.Conv1d(1, width, kernel, padding=kernel // 2), nn.GELU(),
                          nn.Conv1d(width, width, 5, padding=2), nn.GELU())
            for width, kernel in zip(widths, (9, 33, 65))
        ])
        self.coordinate = nn.Sequential(nn.Linear(3, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        self.fusion = nn.Sequential(nn.Linear(dimension * 2, dimension), nn.GELU())
        self.dataset_norms = nn.ModuleList([nn.LayerNorm(dimension) for _ in range(datasets)])
        self.channel_score = nn.Linear(dimension, 1)
        self.position = nn.Parameter(torch.zeros(1, token_steps, dimension))
        nn.init.normal_(self.position, std=0.02)
        self.stem = ConformerStack(dimension, heads, layers, dropout)
        self.local_head = ConformerStack(dimension, heads, local_layers, dropout)
        global_dimension = self.teacher_dimension or dimension
        self.global_head = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, global_dimension))
        self.mfcc_head = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, mfcc_dimension))
        self.phoneme_head = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, phoneme_classes))
        self.duration_head = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 1))
        self.activity_head = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 1))
        # Local HuBERT alignment always lives in the encoder-width space.  The
        # optional v3 global teacher is a separate frozen PCA space.
        self.audio_projection = nn.Sequential(nn.LayerNorm(hubert_dimension), nn.Linear(hubert_dimension, dimension))
        self.subject_head = (nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, int(subject_adversary_classes)))
                             if int(subject_adversary_classes) > 0 else None)
        self.clip_logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
        # Templates are derived from the training fold and are deliberately
        # carried by checkpoint provenance rather than baked into code/config.
        self.register_buffer("target_mfcc_template", torch.zeros(mfcc_dimension, target_frames), persistent=False)
        self.register_buffer("target_mfcc_scale", torch.ones(mfcc_dimension, target_frames), persistent=False)
        self.register_buffer("hubert_template", torch.zeros(hubert_dimension), persistent=False)
        self.register_buffer("duration_log_mean", torch.zeros(()), persistent=False)
        self.register_buffer("duration_log_scale", torch.ones(()), persistent=False)
        self.register_buffer("duration_min_frames", torch.ones(()), persistent=False)
        self.register_buffer("duration_max_frames", torch.full((), 1000.0), persistent=False)
        teacher_width = max(1, self.teacher_dimension)
        self.register_buffer("teacher_mean", torch.zeros(hubert_dimension), persistent=False)
        self.register_buffer("teacher_components", torch.zeros(hubert_dimension, teacher_width), persistent=False)
        self.register_buffer("teacher_scale", torch.ones(teacher_width), persistent=False)
        self.register_buffer("teacher_bank", torch.empty(0, teacher_width), persistent=False)
        self.teacher_labels: tuple[str, ...] = ()

    def set_target_templates(self, mfcc_mean: torch.Tensor, mfcc_scale: torch.Tensor | None = None,
                             hubert_mean: torch.Tensor | None = None) -> None:
        if mfcc_mean.shape != self.target_mfcc_template.shape:
            raise ValueError(f"MFCC template shape {tuple(mfcc_mean.shape)} is invalid")
        self.target_mfcc_template.copy_(mfcc_mean.detach().to(self.target_mfcc_template))
        if mfcc_scale is not None:
            if mfcc_scale.shape != self.target_mfcc_scale.shape:
                raise ValueError("MFCC scale shape mismatch")
            self.target_mfcc_scale.copy_(mfcc_scale.detach().to(self.target_mfcc_scale).clamp_min(1e-4))
        if hubert_mean is not None:
            if hubert_mean.shape != self.hubert_template.shape:
                raise ValueError("HuBERT template shape mismatch")
            self.hubert_template.copy_(hubert_mean.detach().to(self.hubert_template))

    def set_duration_statistics(self, log_mean: torch.Tensor, log_scale: torch.Tensor,
                                minimum_frames: torch.Tensor, maximum_frames: torch.Tensor) -> None:
        """Set train-fold-only duration statistics for the standardized v3 head."""
        if not self.duration_standardized:
            raise RuntimeError("duration statistics were supplied to a legacy duration head")
        values = (log_mean, log_scale, minimum_frames, maximum_frames)
        if not all(torch.isfinite(value).all() for value in values):
            raise ValueError("duration statistics must be finite")
        if float(log_scale) <= 0 or float(minimum_frames) < 1 or float(maximum_frames) < float(minimum_frames):
            raise ValueError("invalid train-fold duration statistics")
        self.duration_log_mean.copy_(log_mean.detach().to(self.duration_log_mean))
        self.duration_log_scale.copy_(log_scale.detach().to(self.duration_log_scale).clamp_min(1e-4))
        self.duration_min_frames.copy_(minimum_frames.detach().to(self.duration_min_frames))
        self.duration_max_frames.copy_(maximum_frames.detach().to(self.duration_max_frames))

    def set_speech_teacher(self, mean: torch.Tensor, components: torch.Tensor, scale: torch.Tensor,
                           bank: torch.Tensor | None = None, labels: list[str] | tuple[str, ...] = ()) -> None:
        """Install the frozen train-fold HuBERT whitening/PCA teacher.

        ``labels`` are metadata for loss routing only; they never become model
        inputs.  Components and the content bank are checkpointed through the
        training payload, rather than inferred from validation/test data.
        """
        if self.teacher_dimension <= 0:
            raise RuntimeError("speech teacher requires model.teacher_dimension > 0")
        expected = (self.hubert_template.numel(), self.teacher_dimension)
        if mean.shape != (expected[0],) or components.shape != expected or scale.shape != (expected[1],):
            raise ValueError("speech teacher shape mismatch")
        if not torch.isfinite(mean).all() or not torch.isfinite(components).all() or not torch.isfinite(scale).all():
            raise ValueError("speech teacher values must be finite")
        self.teacher_mean.copy_(mean.detach().to(self.teacher_mean))
        self.teacher_components[:, :self.teacher_dimension].copy_(components.detach().to(self.teacher_components))
        self.teacher_scale[:self.teacher_dimension].copy_(scale.detach().to(self.teacher_scale).clamp_min(1e-4))
        if bank is not None:
            if bank.ndim != 2 or bank.shape[1] != self.teacher_dimension or len(labels) != bank.shape[0]:
                raise ValueError("speech teacher bank/label mismatch")
            if len(set(labels)) != len(labels):
                raise ValueError("speech teacher bank labels must be unique")
            self.teacher_bank = bank.detach().to(self.teacher_bank)
            self.teacher_labels = tuple(str(value) for value in labels)

    def teacher_target(self, hubert_global: torch.Tensor) -> torch.Tensor:
        if self.teacher_dimension <= 0:
            raise RuntimeError("teacher_target requires a v3 teacher-enabled model")
        centered = hubert_global - self.teacher_mean.view(1, -1)
        return (centered @ self.teacher_components[:, :self.teacher_dimension] /
                self.teacher_scale[:self.teacher_dimension].view(1, -1))

    def _duration_from_z(self, duration_z: torch.Tensor) -> torch.Tensor:
        value = (self.duration_log_mean + self.duration_log_scale * duration_z).exp()
        return value.clamp(self.duration_min_frames, self.duration_max_frames)

    def _dataset_norm(self, value: torch.Tensor, dataset_id: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(value)
        for identifier, norm in enumerate(self.dataset_norms):
            selected = dataset_id == identifier
            if selected.any():
                output[selected] = norm(value[selected])
        return output

    def project_audio(self, hubert_local: torch.Tensor) -> torch.Tensor:
        return self.audio_projection(hubert_local)

    def centered_audio(self, hubert_local: torch.Tensor) -> torch.Tensor:
        audio = self.project_audio(hubert_local)
        if not self.zero_centered:
            return audio
        return audio - self.project_audio(self.hubert_template.view(1, 1, -1)).squeeze(0)

    def _forward_raw(self, eeg: torch.Tensor, channel_xyz: torch.Tensor, channel_mask: torch.Tensor,
                     time_mask: torch.Tensor, dataset_id: torch.Tensor) -> _RawState:
        """Backbone execution before optional zero-baseline subtraction."""
        batch, channels, samples = eeg.shape
        adapted = self.input_adapter(eeg, dataset_id)
        adapted = adapted * time_mask[:, None, :].to(adapted.dtype)
        adapted = adapted * channel_mask[:, :, None].to(adapted.dtype)
        flat = adapted.reshape(batch * channels, 1, samples)
        temporal = torch.cat([branch(flat) for branch in self.temporal], dim=1)
        temporal = F.interpolate(temporal, size=self.token_steps, mode="linear", align_corners=False)
        temporal = temporal.transpose(1, 2).reshape(batch, channels, self.token_steps, -1)
        coordinates = self.coordinate(channel_xyz).unsqueeze(2).expand(-1, -1, self.token_steps, -1)
        fused = self.fusion(torch.cat((temporal, coordinates), dim=-1))
        fused_shape = fused.shape
        fused = self._dataset_norm(
            fused.reshape(batch * channels, self.token_steps, -1),
            dataset_id[:, None].expand(-1, channels).reshape(-1),
        ).reshape(fused_shape)
        score = self.channel_score(fused).squeeze(-1).masked_fill(~channel_mask.unsqueeze(-1), -1e4)
        value = (fused * torch.softmax(score, dim=1).unsqueeze(-1)).sum(1)
        token_mask = F.interpolate(time_mask.float().unsqueeze(1), size=self.token_steps, mode="nearest").squeeze(1).bool()
        stem = self.stem(value + self.position, token_mask)
        local = self.local_head(stem, token_mask)
        pooled = masked_mean(stem, token_mask)
        acoustic_tokens = F.interpolate(local.transpose(1, 2), size=self.target_frames, mode="linear", align_corners=False).transpose(1, 2)
        raw_duration = self.duration_head(pooled).squeeze(-1)
        duration = raw_duration if self.duration_standardized else F.softplus(raw_duration) + 1.0
        subject_logits = (self.subject_head(reverse_gradient(pooled, self.subject_adversary_strength))
                          if self.subject_head is not None else None)
        return _RawState(local, self.global_head(pooled), self.mfcc_head(acoustic_tokens).transpose(1, 2),
                         self.phoneme_head(pooled), token_mask, duration,
                         self.activity_head(acoustic_tokens).squeeze(-1), subject_logits)

    def forward(self, eeg: torch.Tensor, channel_xyz: torch.Tensor, channel_mask: torch.Tensor,
                time_mask: torch.Tensor, dataset_id: torch.Tensor) -> JointState:
        if eeg.ndim != 3 or channel_xyz.shape != (*eeg.shape[:2], 3):
            raise ValueError("eeg must be [B,C,T] and channel_xyz [B,C,3]")
        if channel_mask.shape != eeg.shape[:2] or time_mask.shape != (eeg.shape[0], eeg.shape[2]):
            raise ValueError("channel/time mask shape mismatch")
        if not channel_mask.any(1).all() or not time_mask.any(1).all():
            raise ValueError("each example needs at least one valid channel and time sample")
        # ``zero EEG`` is a scientific control, not a stochastic forward pass.
        # In training mode two dropout draws would otherwise make h(0)-h(0)
        # nonzero and quietly violate the zero-centered residual contract.
        if self.zero_centered and not bool(eeg.detach().abs().any()):
            token_mask = F.interpolate(time_mask.float().unsqueeze(1), size=self.token_steps, mode="nearest").squeeze(1).bool()
            baseline_mfcc = self.target_mfcc_template.unsqueeze(0).expand(eeg.shape[0], -1, -1)
            global_dimension = self.teacher_dimension or self.position.shape[-1]
            zero_duration = (self._duration_from_z(torch.zeros(eeg.shape[0], device=eeg.device, dtype=eeg.dtype))
                             if self.duration_standardized else torch.ones(eeg.shape[0], device=eeg.device, dtype=eeg.dtype))
            return JointState(
                local=torch.zeros(eeg.shape[0], self.token_steps, self.position.shape[-1], device=eeg.device, dtype=eeg.dtype),
                global_embedding=torch.zeros(eeg.shape[0], global_dimension, device=eeg.device, dtype=eeg.dtype),
                mfcc=baseline_mfcc,
                phoneme_logits=torch.zeros(eeg.shape[0], self.phoneme_head[-1].out_features, device=eeg.device, dtype=eeg.dtype),
                token_mask=token_mask, baseline_mfcc=baseline_mfcc,
                residual_mfcc=torch.zeros_like(baseline_mfcc), residual_z=torch.zeros_like(baseline_mfcc),
                predicted_duration=zero_duration,
                duration_z=torch.zeros(eeg.shape[0], device=eeg.device, dtype=eeg.dtype),
                activity_logits=torch.zeros(eeg.shape[0], self.target_frames, device=eeg.device, dtype=eeg.dtype),
                subject_logits=(torch.zeros(eeg.shape[0], self.subject_head[-1].out_features, device=eeg.device, dtype=eeg.dtype)
                                if self.subject_head is not None else None),
            )
        raw = self._forward_raw(eeg, channel_xyz, channel_mask, time_mask, dataset_id)
        if not self.zero_centered:
            duration = self._duration_from_z(raw.duration) if self.duration_standardized else raw.duration
            return JointState(raw.local, F.normalize(raw.global_feature, dim=-1), raw.mfcc,
                              raw.phoneme_logits, raw.token_mask,
                              torch.zeros_like(raw.mfcc), raw.mfcc, raw.mfcc,
                              duration, raw.duration if self.duration_standardized else None,
                              raw.activity_logits, raw.subject_logits)
        baseline = self._forward_raw(torch.zeros_like(eeg), channel_xyz, channel_mask, time_mask, dataset_id)
        residual_z = raw.mfcc - baseline.mfcc
        residual_mfcc = residual_z * self.target_mfcc_scale.unsqueeze(0)
        mfcc = self.target_mfcc_template.unsqueeze(0) + residual_mfcc
        local = raw.local - baseline.local
        global_embedding = F.normalize(raw.global_feature - baseline.global_feature, dim=-1, eps=1e-6)
        duration_z = raw.duration - baseline.duration if self.duration_standardized else None
        duration = self._duration_from_z(duration_z) if duration_z is not None else raw.duration
        return JointState(local, global_embedding, mfcc, raw.phoneme_logits, raw.token_mask,
                          self.target_mfcc_template.unsqueeze(0).expand_as(mfcc), residual_mfcc, residual_z,
                          duration, duration_z, raw.activity_logits, raw.subject_logits)


@dataclass
class RendererState:
    log_mel: torch.Tensor
    rms: torch.Tensor
    activity_logits: torch.Tensor


class AudioMFCCRenderer(nn.Module):
    """Audio-only MFCC -> acoustic renderer used as a separately gated oracle."""

    def __init__(self, hidden_dimension: int = 128, layers: int = 4, dropout: float = 0.1):
        super().__init__()
        blocks: list[nn.Module] = [nn.Conv1d(39, hidden_dimension, 5, padding=2), nn.GELU()]
        for _ in range(max(1, int(layers))):
            blocks.extend([
                nn.Conv1d(hidden_dimension, hidden_dimension, 5, padding=2, groups=hidden_dimension),
                nn.Conv1d(hidden_dimension, hidden_dimension, 1), nn.GELU(), nn.Dropout(dropout),
            ])
        self.backbone = nn.Sequential(*blocks)
        self.log_mel_head = nn.Conv1d(hidden_dimension, 80, 1)
        self.rms_head = nn.Sequential(nn.Conv1d(hidden_dimension, 1, 1), nn.Softplus())
        self.activity_head = nn.Conv1d(hidden_dimension, 1, 1)

    def forward(self, mfcc: torch.Tensor) -> RendererState:
        if mfcc.ndim != 3 or mfcc.shape[1] != 39:
            raise ValueError("renderer input must be [B,39,T]")
        state = self.backbone(mfcc)
        return RendererState(self.log_mel_head(state), self.rms_head(state).squeeze(1),
                             self.activity_head(state).squeeze(1))


class DurationConditionedNativeRenderer(nn.Module):
    """Relative MFCC plus duration to native SpeechT5 mel.

    Content remains on the 161-frame relative grid; only this audio-only
    renderer expands it to a native-duration mel sequence.  It is deliberately
    separate from the EEG model so waveform quality cannot hide weak EEG use.
    """

    def __init__(self, hidden_dimension: int = 160, layers: int = 4, dropout: float = 0.1):
        super().__init__()
        blocks: list[nn.Module] = [nn.Conv1d(40, hidden_dimension, 5, padding=2), nn.GELU()]
        for _ in range(max(1, int(layers))):
            blocks.extend([nn.Conv1d(hidden_dimension, hidden_dimension, 5, padding=2, groups=hidden_dimension),
                           nn.Conv1d(hidden_dimension, hidden_dimension, 1), nn.GELU(), nn.Dropout(dropout)])
        self.backbone = nn.Sequential(*blocks)
        self.mel_head = nn.Conv1d(hidden_dimension, 80, 1)

    def forward(self, mfcc: torch.Tensor, duration_frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if mfcc.ndim != 3 or mfcc.shape[1] != 39:
            raise ValueError("native renderer MFCC must be [B,39,T]")
        if duration_frames.ndim != 1 or duration_frames.shape[0] != mfcc.shape[0]:
            raise ValueError("duration_frames must be [B]")
        maximum = int(duration_frames.max().item()) if len(duration_frames) else 1
        duration = duration_frames.to(mfcc.dtype).clamp_min(1).log().view(-1, 1, 1).expand(-1, 1, mfcc.shape[-1])
        relative = self.mel_head(self.backbone(torch.cat((mfcc, duration), dim=1)))
        rows = []
        mask = torch.zeros(mfcc.shape[0], maximum, dtype=torch.bool, device=mfcc.device)
        for index, frames in enumerate(duration_frames.tolist()):
            frames = max(1, int(frames))
            value = F.interpolate(relative[index:index + 1], size=frames, mode="linear", align_corners=False)
            rows.append(F.pad(value, (0, maximum - frames)))
            mask[index, :frames] = True
        return torch.cat(rows, dim=0), mask

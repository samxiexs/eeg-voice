"""Recovery v3: speech-frame-masked objectives, batch CLIP, subject mixing.

Kept outside eeg2speech/ so legacy training fingerprints remain unchanged.

Why v3 exists (diagnosed 2026-09-14 on the v2 run): every cached target is
zero padded to four seconds and the padded mel frames are a constant -10, so
33% of the frames on average carried nothing but sentence duration.  Mel MAE,
the 320-way bank contrastive and retrieval were all dominated by that
duration (bank prototype PC1 correlated -0.98 with duration), the median
template beat the model, and the variance penalty manufactured trial-to-trial
noise instead of information.  v3 therefore supervises and scores only the
frames where speech was actually presented, contrasts time-resolved sequences
inside the physical batch, predicts duration with a separate head, and adds
subject mixing plus augmentation for single-trial SNR.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from eeg2speech.aligned import (AlignedState, EEG_RATE, EEG_SAMPLES, EEG_START,
                                LAGS_MS, sample_at)

MEL_HOP_S = 256 / 16000
SILENCE_MEL = -10.


@dataclass
class RecoveryState(AlignedState):
    duration_fraction: torch.Tensor | None = None


class TemporalResidual(nn.Module):
    def __init__(self, width, dilation, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 5, padding=2 * dilation,
                                   dilation=dilation, groups=width)
        self.mix = nn.Sequential(nn.Conv1d(width, width * 2, 1), nn.GELU(),
                                 nn.Dropout(dropout), nn.Conv1d(width * 2, width, 1))
        self.gain = nn.Parameter(torch.full((1, width, 1), .1))

    def forward(self, value):
        normalized = self.norm(value.transpose(1, 2)).transpose(1, 2)
        return value + self.gain * self.mix(F.gelu(self.depthwise(normalized)))


class RecoveryEEGModel(nn.Module):
    """Fixed-montage EEG -> physical-time speech sequence -> frozen decoder.

    ``subjects`` enables a per-subject signed channel mixing (identity plus a
    learned, weight-decayed delta) before the shared spatial filters.  The
    protocol is known-subject, so the subject index is a legitimate input; an
    unknown subject (``subject=None``) falls back to the shared identity path.
    """
    def __init__(self, decoder, channels=128, width=128, lag_ms=0, subjects=0, dropout=.1, positional=False):
        super().__init__()
        if lag_ms not in LAGS_MS:
            raise ValueError('unregistered lag')
        self.decoder = decoder.requires_grad_(False)
        self.lag_ms = lag_ms
        self.channels = channels
        self.subjects = int(subjects)
        if self.subjects:
            self.subject_delta = nn.Parameter(torch.zeros(self.subjects, channels, channels))
        # A learned signed filter for every channel precedes temporal processing.
        self.spatial = nn.Conv1d(channels, width, 1, bias=False)
        self.temporal = nn.Conv1d(width, width, 15, stride=4, padding=7)
        self.blocks = nn.Sequential(*[TemporalResidual(width, 2 ** i, dropout) for i in range(6)])
        self.output_norm = nn.LayerNorm(width)
        self.head = nn.Conv1d(width, decoder.normalizer.mean.numel(), 1)
        nn.init.normal_(self.head.weight, std=.01)
        nn.init.zeros_(self.head.bias)
        # Auxiliary sentence-duration head: keeps the offset cue out of the
        # speech sequence and gives inference a tail policy without oracle data.
        self.duration = nn.Linear(width, 1)
        nn.init.zeros_(self.duration.weight); nn.init.constant_(self.duration.bias, .5)
        count = (EEG_SAMPLES + 2 * 7 - 15) // 4 + 1
        self.register_buffer('token_times', EEG_START + torch.arange(count).float() * 4 / EEG_RATE)
        # Time-since-onset code.  The window is physically fixed, so this is
        # the same for every trial and carries no content or duration: it lets
        # the encoder express the onset-locked prior that a time-only template
        # already has, instead of inferring the clock from the noisy ERP.
        self.positional = bool(positional)
        if self.positional:
            self.position = nn.Parameter(torch.zeros(1, width, count))

    def mix_subject(self, x, subject):
        if subject is None or not self.subjects:
            return x
        if subject.shape != (len(x),) or bool((subject < 0).any()) or bool((subject >= self.subjects).any()):
            raise ValueError('subject index out of range')
        mixing = torch.eye(self.channels, device=x.device, dtype=x.dtype) + self.subject_delta[subject]
        return torch.bmm(mixing, x)

    def forward(self, eeg, channel_xyz, channel_mask, time_mask, subject=None):
        if eeg.shape[1:] != (self.channels, EEG_SAMPLES) or time_mask.shape != (len(eeg), EEG_SAMPLES):
            raise ValueError('fixed montage and physical EEG window required')
        if not bool(time_mask.all()) or not bool(channel_mask.any(1).all()):
            raise ValueError('invalid masks')
        if not bool(torch.isfinite(eeg).all()):
            raise ValueError('nonfinite EEG')
        x = self.mix_subject(eeg * channel_mask[:, :, None], subject)
        x = F.gelu(self.temporal(self.spatial(x)))
        if self.positional:
            x = x + self.position
        x = self.output_norm(self.blocks(x).transpose(1, 2))          # B, T, W
        duration = torch.sigmoid(self.duration(x.mean(1))).squeeze(-1)
        z = self.head(x.transpose(1, 2)).transpose(1, 2)
        z = sample_at(z, self.token_times, self.decoder.speech_times + self.lag_ms / 1000.)
        sequence = z * self.decoder.normalizer.scale + self.decoder.normalizer.mean
        return RecoveryState(sequence, self.decoder.speech_times, self.decoder(sequence),
                             self.decoder.mel_times, F.normalize(z.mean(1), dim=-1), duration)


def speech_frame_masks(duration_frames, mel_times, speech_times):
    """Boolean (B, T_mel) and (B, T_speech) masks of presented-speech frames.

    ``duration_frames`` is the cached ``source_samples_16k // 256 + 1``; mel
    frame k is speech iff k < duration_frames.  Speech-teacher frames are
    speech iff their centre lies before the last speech mel frame plus half a
    hop.  Padding frames never enter any v3 loss or gate metric.
    """
    duration_frames = duration_frames.to(mel_times.device).long()
    if bool((duration_frames < 2).any()) or bool((duration_frames > len(mel_times)).any()):
        raise ValueError('duration frames outside the fixed acoustic window')
    mel_mask = torch.arange(len(mel_times), device=mel_times.device)[None, :] < duration_frames[:, None]
    end_s = mel_times[duration_frames - 1] + MEL_HOP_S / 2
    speech_mask = speech_times.to(mel_times.device)[None, :] < end_s[:, None]
    if not bool(speech_mask.any(1).all()):
        raise ValueError('a trial has no speech frames')
    return mel_mask, speech_mask


def duration_fraction(duration_frames, mel_frames):
    return duration_frames.float() / float(mel_frames)


def masked_mean(value, mask):
    """Mean of ``value`` over positions where ``mask`` is true (broadcast on dim 1 or -1)."""
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.) / (value.numel() / weight.numel())


def recovery_loss(state, teacher, mel, normalizer, speech_mask, mel_mask, duration_target, indices, *,
                  mel_weight=1., temperature=.1, contrastive_weight=1., sequence_weight=1.,
                  delta_weight=.2, duration_weight=.5):
    if len(indices) < 2 or len(indices.unique()) != len(indices):
        raise ValueError('each physical batch must contain at least two distinct contents')
    if speech_mask.shape != teacher.shape[:2] or mel_mask.shape != (len(mel), mel.shape[-1]):
        raise ValueError('mask/target shape mismatch')
    target = normalizer(teacher).detach()
    z = normalizer(state.aligned_sequence)
    frame = speech_mask[:, :, None]
    sequence = masked_mean((z - target).square(), frame)
    both = (speech_mask[:, 1:] & speech_mask[:, :-1])[:, :, None]
    delta = masked_mean((z.diff(dim=1) - target.diff(dim=1)).abs(), both)
    # Time-resolved CLIP inside the physical batch: every trial's predicted
    # sequence must match its own teacher better than the other contents',
    # judged only on frames where both sentences were still being spoken.
    similarity = torch.einsum('itd,jtd->ijt', F.normalize(z, dim=-1), F.normalize(target, dim=-1))
    pair = (speech_mask[:, None, :] & speech_mask[None, :, :]).to(similarity.dtype)
    logits = (similarity * pair).sum(-1) / pair.sum(-1).clamp_min(1.) / temperature
    labels = torch.arange(len(z), device=z.device)
    contrastive = .5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    mel_frame = mel_mask[:, None, :]
    mel_l1 = masked_mean((state.native_mel - mel).abs(), mel_frame)
    mel_both = (mel_mask[:, 1:] & mel_mask[:, :-1])[:, None, :]
    mel_delta = masked_mean((state.native_mel.diff(dim=-1) - mel.diff(dim=-1)).abs(), mel_both)
    mel_loss = mel_l1 + .2 * mel_delta
    duration = F.l1_loss(state.duration_fraction, duration_target.to(state.duration_fraction))
    loss = (contrastive_weight * contrastive + sequence_weight * sequence + delta_weight * delta
            + mel_weight * mel_loss + duration_weight * duration)
    with torch.no_grad():
        accuracy = float((logits.argmax(1) == labels).float().mean())
        first = int(speech_mask.sum(1).min())          # frames every batch trial contains
        ratio = float(z.detach()[:, :first].var(0, unbiased=False).mean() /
                      target[:, :first].var(0, unbiased=False).mean().clamp_min(1e-8))
    metrics = {name: float(value.detach()) for name, value in
               [('contrastive', contrastive), ('sequence_mse', sequence), ('delta', delta),
                ('mel', mel_loss), ('duration', duration), ('loss', loss)]}
    metrics.update(batch_accuracy=accuracy, sequence_variance_ratio=ratio)
    return loss, metrics


def augment_eeg(eeg, channel_mask, *, channel_drop=.1, spans=2, span_max=64, noise=.1, gain=.2):
    """Training-only single-trial augmentation in normalized (MAD) units.

    Random channel dropout, up to ``spans`` zeroed spans of at most
    ``span_max`` samples (250 ms at 256 Hz), a per-trial gain jitter and
    white noise.  Invalid channels stay zero; shapes and masks are unchanged.
    """
    if channel_drop <= 0 and spans <= 0 and noise <= 0 and gain <= 0:
        return eeg
    batch, channels, samples = eeg.shape
    keep = (torch.rand(batch, channels, device=eeg.device) >= channel_drop) & channel_mask
    # Never drop every valid channel of a trial.
    keep = torch.where(keep.any(1, keepdim=True), keep, channel_mask)
    out = eeg * keep[:, :, None]
    if spans > 0 and span_max > 0:
        positions = torch.arange(samples, device=eeg.device)[None, None, :]
        starts = torch.randint(0, samples, (batch, spans, 1), device=eeg.device)
        lengths = torch.randint(0, span_max + 1, (batch, spans, 1), device=eeg.device)
        hide = ((positions >= starts) & (positions < starts + lengths)).any(1)   # B, T
        out = out * (~hide)[:, None, :]
    if gain > 0:
        out = out * (1 + (torch.rand(batch, 1, 1, device=eeg.device) * 2 - 1) * gain)
    if noise > 0:
        out = out + noise * torch.randn_like(out) * channel_mask[:, :, None]
    return out


def same_content_partners(frame, probability, rng):
    """Training-only augmentation plan: for each trial, optionally a partner trial of the same sentence.

    Returns ``{index: partner_index}`` for the trials chosen with
    ``probability``.  A partner is another observation of the same content
    (preferably the same participant in another task, else another
    participant); its EEG is averaged with the trial's own EEG to raise the
    single-trial SNR of the positive pair.  Test-time inputs stay single-trial.
    """
    if probability <= 0:
        return {}
    groups = frame.groupby('content_group').indices
    subjects = frame.subject.to_numpy()
    plan = {}
    for i in range(len(frame)):
        if rng.random() >= probability:
            continue
        others = [j for j in groups[frame.content_group.iat[i]] if j != i]
        if not others:
            continue
        same = [j for j in others if subjects[j] == subjects[i]]
        pool = same or others
        plan[i] = int(pool[rng.integers(len(pool))])
    return plan


def average_eeg(eeg, partner, channel_mask, partner_mask):
    """Mean of two trials over channels valid in both; channels valid in one keep that trial's signal."""
    both = (channel_mask & partner_mask)[:, :, None]
    only_self = (channel_mask & ~partner_mask)[:, :, None]
    only_partner = (~channel_mask & partner_mask)[:, :, None]
    return torch.where(both, .5 * (eeg + partner), torch.where(only_self, eeg, torch.where(only_partner, partner, torch.zeros_like(eeg))))


def apply_tail(mel, duration_frames):
    """Silence every mel frame at or beyond ``duration_frames`` (per trial)."""
    frames = torch.arange(mel.shape[-1], device=mel.device)[None, None, :]
    return torch.where(frames < duration_frames.to(mel.device)[:, None, None], mel, torch.full_like(mel, SILENCE_MEL))


def diverse_batches(frame, batch_size, seed):
    """One visit per trial per epoch, distinct contents per physical update.

    A final singleton is carried with a different-content trial; at most that
    partner is repeated. No trial is dropped, including tail subjects.
    """
    import numpy as np
    if batch_size < 2:
        raise ValueError('recovery requires batch_size >= 2')
    rng = np.random.default_rng(seed)
    groups = {c: rng.permutation(rows.index).tolist() for c, rows in frame.groupby('content_group')}
    if len(groups) < 2:
        raise ValueError('need at least two contents')
    remaining = {c: list(ids) for c, ids in groups.items()}
    batches = []
    while remaining:
        # Prefer large groups, random ties; prevents a large one-content tail.
        labels = list(remaining); rng.shuffle(labels)
        labels.sort(key=lambda c: len(remaining[c]), reverse=True)
        chosen = labels[:batch_size]
        ids = [remaining[c].pop() for c in chosen]
        remaining = {c: values for c, values in remaining.items() if values}
        if len(ids) == 1:
            others = [c for c in groups if c != chosen[0]]
            ids.append(int(rng.choice(groups[rng.choice(others)])))
        batches.append(ids)
    return batches

"""Recovery experiment: signed spatial mixing and content-diverse updates.

Kept outside eeg2speech/ so legacy training fingerprints remain unchanged.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from eeg2speech.aligned import (AlignedState, EEG_RATE, EEG_SAMPLES, EEG_START,
                                LAGS_MS, acoustic_loss, sample_at)


class TemporalResidual(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 5, padding=2 * dilation,
                                   dilation=dilation, groups=width)
        self.mix = nn.Sequential(nn.Conv1d(width, width * 2, 1), nn.GELU(),
                                 nn.Conv1d(width * 2, width, 1))
        self.gain = nn.Parameter(torch.full((1, width, 1), .1))

    def forward(self, value):
        normalized = self.norm(value.transpose(1, 2)).transpose(1, 2)
        return value + self.gain * self.mix(F.gelu(self.depthwise(normalized)))


class RecoveryEEGModel(nn.Module):
    """Fixed-montage EEG -> physical-time speech sequence -> frozen decoder."""
    def __init__(self, decoder, channels=128, width=128, lag_ms=0):
        super().__init__()
        if lag_ms not in LAGS_MS:
            raise ValueError('unregistered lag')
        self.decoder = decoder.requires_grad_(False)
        self.lag_ms = lag_ms
        self.channels = channels
        # A learned signed filter for every channel precedes temporal processing.
        self.spatial = nn.Conv1d(channels, width, 1, bias=False)
        self.temporal = nn.Conv1d(width, width, 15, stride=4, padding=7)
        self.blocks = nn.Sequential(*[TemporalResidual(width, 2 ** i) for i in range(6)])
        self.output_norm = nn.LayerNorm(width)
        self.head = nn.Conv1d(width, decoder.normalizer.mean.numel(), 1)
        nn.init.normal_(self.head.weight, std=.01)
        nn.init.zeros_(self.head.bias)
        count = (EEG_SAMPLES + 2 * 7 - 15) // 4 + 1
        self.register_buffer('token_times', EEG_START + torch.arange(count).float() * 4 / EEG_RATE)

    def forward(self, eeg, channel_xyz, channel_mask, time_mask):
        if eeg.shape[1:] != (self.channels, EEG_SAMPLES) or time_mask.shape != (len(eeg), EEG_SAMPLES):
            raise ValueError('fixed montage and physical EEG window required')
        if not bool(time_mask.all()) or not bool(channel_mask.any(1).all()):
            raise ValueError('invalid masks')
        if not bool(torch.isfinite(eeg).all()):
            raise ValueError('nonfinite EEG')
        x = eeg * channel_mask[:, :, None]
        x = F.gelu(self.temporal(self.spatial(x)))
        x = self.output_norm(self.blocks(x).transpose(1, 2)).transpose(1, 2)
        z = self.head(x).transpose(1, 2)
        z = sample_at(z, self.token_times, self.decoder.speech_times + self.lag_ms / 1000.)
        sequence = z * self.decoder.normalizer.scale + self.decoder.normalizer.mean
        return AlignedState(sequence, self.decoder.speech_times, self.decoder(sequence),
                            self.decoder.mel_times, F.normalize(z.mean(1), dim=-1))


def recovery_loss(state, teacher, mel, normalizer, bank, indices, *, mel_weight=1.):
    if len(indices) < 2 or len(indices.unique()) != len(indices):
        raise ValueError('each physical batch must contain at least two distinct contents')
    target = normalizer(teacher).detach()
    z = normalizer(state.aligned_sequence)
    sequence = F.mse_loss(z, target)
    delta = F.l1_loss(z.diff(dim=1), target.diff(dim=1))
    # Unit normalization at nearly-zero amplitude creates huge angular gradients
    # while the acoustic decoder sees almost no change. Bound that denominator.
    global_z = z.mean(1)
    logits = (global_z / global_z.norm(dim=-1, keepdim=True).clamp_min(1.)) @ F.normalize(bank.detach(), dim=-1).T / .07
    contrastive = F.cross_entropy(logits, indices)
    # Variance is computed ACROSS different EEG examples, not across time.
    # Matching teacher-dependent variation avoids a fixed arbitrary noise floor.
    predicted_std = (z.var(0, unbiased=False) + 1e-4).sqrt()
    target_std = (target.var(0, unbiased=False) + 1e-4).sqrt()
    variance = F.relu(.5 * target_std - predicted_std).mean()
    mel_loss = acoustic_loss(state.native_mel, mel)
    loss = contrastive + .5 * sequence + .1 * delta + .5 * variance + mel_weight * mel_loss
    metrics = {name: float(value.detach()) for name, value in
               [('contrastive', contrastive), ('sequence_mse', sequence), ('delta', delta),
                ('variance_penalty', variance), ('mel', mel_loss), ('loss', loss)]}
    metrics['sequence_variance_ratio'] = float(z.detach().var(0, unbiased=False).mean() /
                                               target.var(0, unbiased=False).mean().clamp_min(1e-8))
    return loss, metrics


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

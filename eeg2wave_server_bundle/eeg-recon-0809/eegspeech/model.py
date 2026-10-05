"""Montage-agnostic EEG encoder with a phase branch and a band-power branch.

Listening EEG carries stimulus-locked (phase) responses; imagined speech has no
external clock, so its information sits in non-phase-locked band power.  The
encoder keeps both: a convolutional branch on the filtered signal and a bank of
learnable band-pass filters whose log power is pooled over 250 ms.  Both feed a
dilated residual trunk at 32 Hz.  Frames serve the listening objectives (match
the speech features in time); the attention-pooled vector serves item decoding.

No participant index enters the model.  Optional personalisation uses a person
vector computed from that person's own unlabelled EEG (``PersonEncoder``), applied
through a FiLM whose strength is gated by the content features (the joint
representation gate of Zhang et al. 2026), zero at initialisation.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

RATE = 128                     # Hz, default model input rate (listening data); imagery runs at 256 Hz
FRAME_RATE = 32                # Hz, rate of the encoder frames whatever the input rate
BANDS = ((4, 8), (8, 13), (13, 30), (30, 45))
HIGH_BANDS = ((55, 80), (80, 120))   # used when the input rate allows; skips 50/60 Hz mains
THETA_MAX = 2.2                # rad from the vertex covered by the flattened position code


def flatten_positions(xyz):
    """Azimuthal-equidistant projection around the vertex into [0, 1]^2 (same map for every dataset)."""
    unit = xyz / xyz.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    theta = torch.arccos(unit[..., 2].clamp(-1, 1))
    phi = torch.atan2(unit[..., 1], unit[..., 0])
    return torch.stack([.5 + theta * torch.cos(phi) / (2 * THETA_MAX), .5 + theta * torch.sin(phi) / (2 * THETA_MAX)], -1)


class SpatialAttention(nn.Module):
    """Virtual channels as position-dependent softmax mixtures of the recorded electrodes (Défossez et al. 2023)."""

    def __init__(self, virtual=64, harmonics=10, init_std=.5):
        super().__init__()
        k, l = torch.meshgrid(torch.arange(harmonics), torch.arange(harmonics), indexing='ij')
        self.register_buffer('frequencies', torch.stack([k.flatten(), l.flatten()], -1).float())
        self.logits = nn.Linear(2 * harmonics * harmonics, virtual)
        # Average-referenced EEG sums to ~0 over electrodes, so a near-uniform softmax (PyTorch's default
        # init) makes every virtual channel ~0 and the encoder output input-independent.  Larger logits give
        # each virtual channel a regional mixture from the start.
        if init_std is not None:
            nn.init.normal_(self.logits.weight, std=init_std)
            nn.init.zeros_(self.logits.bias)

    def forward(self, eeg, xyz, valid):
        valid = valid & torch.isfinite(xyz).all(-1)                        # an electrode without a position is unusable
        phase = 2 * math.pi * flatten_positions(torch.nan_to_num(xyz)) @ self.frequencies.T
        logits = self.logits(torch.cat([torch.cos(phase), torch.sin(phase)], -1))           # B, C, V
        logits = logits.masked_fill(~valid[:, :, None], float('-inf'))
        return torch.einsum('bcv,bct->bvt', torch.softmax(logits, dim=1), eeg)


class Residual(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.norm = nn.GroupNorm(1, width)
        self.depthwise = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation, groups=width)
        self.pointwise = nn.Conv1d(width, 2 * width, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(F.glu(self.pointwise(F.gelu(self.depthwise(self.norm(x)))), dim=1))


def bandpass_taps(low, high, taps, rate):
    """Hamming-windowed sinc band-pass FIR (initialisation of the learnable power filters)."""
    n = np.arange(taps) - (taps - 1) / 2
    h = (2 * high / rate) * np.sinc(2 * high * n / rate) - (2 * low / rate) * np.sinc(2 * low * n / rate)
    return (h * np.hamming(taps)).astype(np.float32)


class PowerBank(nn.Module):
    """Per-channel learnable band-pass filters -> log power pooled over 250 ms, at the frame rate."""

    def __init__(self, channels, rate=RATE):
        super().__init__()
        bands = BANDS + tuple(b for b in HIGH_BANDS if b[1] < .5 * rate)
        self.channels, self.n_bands, self.stride = channels, len(bands), rate // FRAME_RATE
        self.pool, taps = rate // 4, rate // 2 + 1
        self.filters = nn.Conv1d(channels, channels * len(bands), taps, padding=taps // 2, groups=channels, bias=False)
        init = torch.from_numpy(np.stack([bandpass_taps(lo, hi, taps, rate) for lo, hi in bands]))
        self.filters.weight.data.copy_(init.repeat(channels, 1)[:, None, :])

    def forward(self, x):
        power = F.avg_pool1d(self.filters(x).square(), self.pool, self.stride,
                             padding=self.pool // 2 - self.stride // 2)
        return torch.log(power + 1e-4)


class Encoder(nn.Module):
    """EEG (any montage) at ``rate`` Hz -> frames (B, T', W) at 32 Hz, pooled (B, W) and a unit embedding (B, dim)."""

    def __init__(self, virtual=64, width=128, dim=128, dilations=(1, 2, 4, 8, 16, 1), dropout=.1,
                 branches='both', person_dim=0, spatial_init=.5, normalize_virtual=True, rate=RATE):
        super().__init__()
        if branches not in ('both', 'phase', 'power'):
            raise ValueError('branches: both | phase | power')
        if rate % FRAME_RATE:
            raise ValueError(f'rate must be a multiple of {FRAME_RATE} Hz')
        self.branches, self.person_dim, self.normalize_virtual = branches, int(person_dim), normalize_virtual
        self.rate, self.stride = int(rate), int(rate) // FRAME_RATE
        kernel = self.rate // 16 + 1
        self.spatial = SpatialAttention(virtual, init_std=spatial_init)
        self.phase = nn.Sequential(nn.Conv1d(virtual, width, kernel, padding=kernel // 2), nn.GELU(),
                                   nn.Conv1d(width, width, 2 * self.stride, stride=self.stride, padding=self.stride // 2))
        self.power = PowerBank(virtual, self.rate)
        self.power_mix = nn.Conv1d(virtual * self.power.n_bands, width, 1)
        self.merge = nn.Conv1d(2 * width, width, 1)
        self.blocks = nn.Sequential(*[Residual(width, d, dropout) for d in dilations])
        self.norm = nn.LayerNorm(width)
        self.attend = nn.Linear(width, 1)
        self.project = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, dim))
        if self.person_dim:
            self.gate = nn.Linear(width, width)
            self.film = nn.Linear(self.person_dim, 2 * width)
            nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def virtual(self, eeg, xyz, valid):
        """Per-trial RMS normalisation over valid channels, spatial attention, and (optionally) RMS
        normalisation of the virtual channels, so their scale does not depend on how selective the attention is."""
        x = eeg * valid[:, :, None]
        power = x.square().sum((1, 2)) / (valid.sum(1) * x.shape[-1]).clamp_min(1)
        v = self.spatial(x / (power.sqrt() + 1e-6)[:, None, None], xyz, valid)
        if self.normalize_virtual:
            v = v / (v.square().mean((1, 2), keepdim=True).sqrt() + 1e-6)
        return v

    def forward(self, eeg, xyz, valid, lengths=None, person=None):
        v = self.virtual(eeg, xyz, valid)
        a = self.phase(v)
        p = self.power_mix(self.power(v))
        frames = min(a.shape[-1], p.shape[-1])
        a, p = a[..., :frames], p[..., :frames]
        if self.branches == 'phase':
            p = torch.zeros_like(p)
        elif self.branches == 'power':
            a = torch.zeros_like(a)
        h = self.merge(torch.cat([a, p], 1))
        if person is not None and self.person_dim:
            gate = torch.sigmoid(self.gate(h.mean(-1)))[:, :, None]
            scale, shift = self.film(person).chunk(2, -1)
            h = h + gate * (scale[:, :, None] * h + shift[:, :, None])
        h = self.norm(self.blocks(h).transpose(1, 2))                                     # B, T', W
        if lengths is None:
            mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        else:
            mask = torch.arange(h.shape[1], device=h.device)[None] < (lengths[:, None] + self.stride - 1) // self.stride
        weights = self.attend(h).squeeze(-1).masked_fill(~mask, float('-inf')).softmax(-1)
        pooled = (weights[..., None] * h).sum(1)
        return h, mask, pooled, F.normalize(self.project(pooled), dim=-1)


class SpeechEncoder(nn.Module):
    """Stimulus features (B, F, frames at 64 Hz) -> (B, frames / 2, W) at 32 Hz."""

    def __init__(self, features=18, width=128, dilations=(1, 2, 4), dropout=.1):
        super().__init__()
        self.stem = nn.Conv1d(features, width, 5, padding=2)
        self.blocks = nn.Sequential(*[Residual(width, d, dropout) for d in dilations])
        self.norm = nn.LayerNorm(width)

    def forward(self, features):
        x = self.stem(F.avg_pool1d(features, 2))
        return self.norm(self.blocks(x).transpose(1, 2))


class PersonEncoder(nn.Module):
    """A person vector from a set of that person's unlabelled windows - never from a participant index.

    Each window's virtual channels give log-variance and band log-power; the window
    vectors of one set are averaged and projected.  Trained with ``losses.person_contrastive``
    (two disjoint sets of one person agree, different people differ).
    """

    def __init__(self, encoder: Encoder, dim):
        super().__init__()
        self.encoder = encoder
        virtual = encoder.power.channels
        self.window = nn.Sequential(nn.Linear(virtual * (1 + encoder.power.n_bands), 128), nn.GELU(), nn.Linear(128, 128))
        self.out = nn.Sequential(nn.GELU(), nn.Linear(128, dim))

    def forward(self, eeg, xyz, valid, counts):
        v = self.encoder.virtual(eeg, xyz, valid)
        features = torch.cat([torch.log(v.var(-1) + 1e-4), self.encoder.power(v).mean(-1)], -1)
        per_window = self.window(features)
        return self.out(torch.stack([part.mean(0) for part in torch.split(per_window, list(counts))]))


class Model(nn.Module):
    """Encoder + speech encoder (listening objectives) + one item head per labelled dataset."""

    ENCODER_DEFAULTS = dict(spatial_init=.5, normalize_virtual=True, rate=RATE)

    def __init__(self, heads: dict, *, features=18, person_dim=0, **encoder_args):
        super().__init__()
        encoder_args = {**self.ENCODER_DEFAULTS, **encoder_args}          # recorded in full in the checkpoint
        self.encoder = Encoder(person_dim=person_dim, **encoder_args)
        width = self.encoder.norm.normalized_shape[0]
        self.speech = SpeechEncoder(features, width)
        self.heads = nn.ModuleDict({name: nn.Linear(width, n) for name, n in heads.items()})
        self.person = PersonEncoder(self.encoder, person_dim) if person_dim else None
        self.config = dict(heads=dict(heads), features=features, person_dim=person_dim, **encoder_args)

    def save(self, path, **extra):
        torch.save(dict(config=self.config, state=self.state_dict(), **extra), path)

    @classmethod
    def load(cls, path, map_location='cpu', heads=None):
        """Rebuild from a checkpoint; ``heads`` replaces the item heads (encoder weights still load)."""
        payload = torch.load(path, map_location=map_location, weights_only=False)
        config = dict(payload['config'])
        config.setdefault('spatial_init', None)            # checkpoints from before these options existed
        config.setdefault('normalize_virtual', False)
        if heads is not None:
            config['heads'] = dict(heads)
        model = cls(**config)
        state = {k: v for k, v in payload['state'].items()
                 if not k.startswith('heads.') or (k.split('.')[1] in model.heads
                                                   and v.shape == model.state_dict()[k].shape)}
        missing, _ = model.load_state_dict(state, strict=False)
        new = ('heads.', 'person.', 'encoder.gate.', 'encoder.film.')     # modules a later stage may add
        missing = [k for k in missing if not k.startswith(new)]
        if missing:
            raise ValueError(f'{path}: checkpoint lacks {missing[:4]}')
        return model, payload

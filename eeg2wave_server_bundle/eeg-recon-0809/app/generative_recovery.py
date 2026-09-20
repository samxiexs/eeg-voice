#!/usr/bin/env python3
"""Generative (diffusion) mel decoder on top of the frozen recovery-v3 encoder.

Why this exists.  The v3 route regresses the HuBERT sequence and the native mel
under L1/MSE.  Single-trial EEG carries little sentence information, so the
loss-optimal output is the *conditional mean*: an onset-locked average of the
320 training sentences (the "syllable blob" of audio_comparison.png,
``prediction_variance_ratio`` ~ 0.09).  No real spectrogram looks like an
average of spectrograms, which is why the exported WAVs do not sound like
speech.  The legacy MFCC route sounded like speech only because its
audio-trained renderer projected *any* input onto the speech manifold; its EEG
contribution was nil.

This module keeps the frozen v3 encoder (the CLIP-aligned EEG embedding) and
replaces the deterministic decoder with a conditional diffusion model
p(mel | EEG features).  Every sample lies on the speech manifold (formants,
harmonics, pauses); the EEG enters through classifier-free guidance, so the
amount of EEG influence is explicit and is tested against the same
counterfactuals as before (zero EEG, duration-matched wrong trial, time-block
shuffle), with an audio-conditioned ceiling (teacher HuBERT) and a pooled
(all presentations of the sentence) condition.

Stages
  cache    preload EEG/mel/teacher for train+validation into .npy files
  train    conditional v-prediction diffusion over the 4 s native mel window
  export   sample validation trials under every control; vocode with the pinned HiFi-GAN
  compare  score the export with app/audio_comparison.py measures and draw the figure

Nothing here touches the test role.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('ALIGNED_TARGET_CACHE_NAME', 'targets_adapted.h5')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src')); sys.path.insert(0, str(ROOT / 'scripts'))
import aligned_recovery as recovery
from aligned_recovery_model import RecoveryEEGModel, augment_eeg, diverse_batches, duration_fraction, recovery_loss, speech_frame_masks
from aligned_recovery_eval import matched_wrong_trial_indices, train_templates
from envelope_decoder import BANDS, EnvelopeDecoder, envelope_targets, masked_correlation, partial_correlation
from eeg2speech.aligned import EEG_SAMPLES, EEG_RATE, EEG_START, sample_at
from eeg2speech.losses import counterfactual_eeg

legacy = recovery.legacy
CONTRACT = 'generative_recovery_v1'
MEL_FRAMES = 251
MEL_BINS = 80
MEL_FLOOR = -7.          # native SpeechT5 log-mel is padded at -10; speech never goes below ~-6
SILENCE_MEL = -10.
DEFAULT_ENCODER = ROOT / 'outputs/aligned_recovery_v3/full_seed322_positional/best_passed.pt'
DEFAULT_ENVELOPE = ROOT / 'outputs/envelope_decoder/trunk/best.pt'
DEFAULT_OUTPUT = ROOT / 'outputs/generative_recovery'
CONDITIONS = ('correct', 'zero', 'wrong_trial', 'time_block_shuffle', 'pooled', 'pooled_wrong', 'teacher_oracle')


# ----------------------------------------------------------------------------- cache
def cache_role(cfg, role: str, folder: Path, subjects: dict, audio_index: dict) -> None:
    """EEG (float16, normalised), masks and per-trial metadata for one role, plus the shared audio table."""
    data = legacy.dataset_for(cfg, role)
    n = len(data)
    eeg = np.lib.format.open_memmap(folder / f'{role}_eeg.npy', mode='w+', dtype=np.float16, shape=(n, 128, EEG_SAMPLES))
    channel_mask = np.zeros((n, 128), dtype=bool)
    rows = []
    started = time.monotonic()
    for i in range(n):
        item = data[i]
        eeg[i] = item['eeg'].numpy().astype(np.float16)
        channel_mask[i] = item['channel_mask'].numpy()
        rows.append(dict(trial_id=item['trial_id'], subject=item['subject'], content=item['content'],
                         subject_index=subjects[item['subject']], audio_key=data.frame.audio_key.iloc[i],
                         audio_index=audio_index[data.frame.audio_key.iloc[i]],
                         duration_frames=int(item['oracle_duration_frames'])))
        if i % 500 == 0:
            print(json.dumps(dict(role=role, cached=i + 1, total=n, seconds=round(time.monotonic() - started, 1))), flush=True)
    eeg.flush(); del eeg
    np.save(folder / f'{role}_channel_mask.npy', channel_mask)
    wrong = matched_wrong_trial_indices(data.frame)
    for i, row in enumerate(rows):
        row['wrong_index'] = int(wrong[i])
    (folder / f'{role}_rows.json').write_text(json.dumps(rows))


def cache(args, cfg):
    folder = Path(args.output) / 'cache'; folder.mkdir(parents=True, exist_ok=True)
    marker = folder / 'complete.json'
    if marker.exists():
        print('cache already complete'); return
    import h5py
    train = legacy.dataset_for(cfg, 'train')
    encoder_payload = recovery.load_checkpoint(Path(args.encoder))
    subjects = {s: i for i, s in enumerate(encoder_payload['signature']['subjects'])}
    _, manifest, target_cache, _ = legacy.artifact_paths(cfg)
    with h5py.File(target_cache, 'r') as h5:
        keys = sorted(h5['targets'])
        mel = np.stack([h5['targets'][k]['mel'][:] for k in keys]).astype(np.float32)
        teacher = np.stack([h5['targets'][k]['teacher'][:] for k in keys]).astype(np.float16)
        wave = np.stack([h5['targets'][k]['wave'][:] for k in keys]).astype(np.float32)
        frames = np.array([int(h5['targets'][k].attrs['source_samples_16k']) // 256 + 1 for k in keys])
        speech_times = h5['speech_times'][:]; mel_times = h5['mel_times'][:]
    np.save(folder / 'audio_mel.npy', mel); np.save(folder / 'audio_teacher.npy', teacher)
    np.save(folder / 'audio_wave.npy', wave); np.save(folder / 'audio_frames.npy', frames)
    np.save(folder / 'speech_times.npy', speech_times); np.save(folder / 'mel_times.npy', mel_times)
    (folder / 'audio_keys.json').write_text(json.dumps(keys))
    audio_index = {k: i for i, k in enumerate(keys)}
    for role in ('train', 'validation'):
        cache_role(cfg, role, folder, subjects, audio_index)
    legacy.atomic_json(marker, dict(contract=CONTRACT, manifest_sha256=legacy.sha256(manifest),
                                    cache_sha256=legacy.sha256(target_cache), encoder=legacy.sha256(Path(args.encoder)),
                                    subjects=sorted(subjects), roles=['train', 'validation']))
    print('cache complete')


class Role:
    """In-memory view of one cached role."""
    def __init__(self, folder: Path, role: str):
        self.eeg = np.load(folder / f'{role}_eeg.npy', mmap_mode='r')
        self.channel_mask = np.load(folder / f'{role}_channel_mask.npy')
        self.rows = json.loads((folder / f'{role}_rows.json').read_text())
        self.subject_index = np.array([r['subject_index'] for r in self.rows])
        self.audio_index = np.array([r['audio_index'] for r in self.rows])
        self.duration_frames = np.array([r['duration_frames'] for r in self.rows])
        self.wrong_index = np.array([r['wrong_index'] for r in self.rows])
        self.content = [r['content'] for r in self.rows]
        self.audio_key = [r['audio_key'] for r in self.rows]
        groups = {}
        for i, c in enumerate(self.content):
            groups.setdefault(c, []).append(i)
        self.groups = groups
        self.fold = np.zeros(len(self.rows), dtype=np.int64)       # conditioner fold per trial; set by assign_folds

    def assign_folds(self, fold_of_content: dict):
        """Every trial of a sentence shares a fold, so pooled inputs never mix fold models."""
        self.fold = np.array([fold_of_content[c] for c in self.content], dtype=np.int64)

    def __len__(self):
        return len(self.rows)


class Audio:
    def __init__(self, folder: Path):
        self.mel = torch.from_numpy(np.load(folder / 'audio_mel.npy'))
        self.teacher = np.load(folder / 'audio_teacher.npy', mmap_mode='r')
        self.frames = np.load(folder / 'audio_frames.npy')
        self.speech_times = torch.from_numpy(np.load(folder / 'speech_times.npy'))
        self.mel_times = torch.from_numpy(np.load(folder / 'mel_times.npy'))
        self.keys = json.loads((folder / 'audio_keys.json').read_text())

    def wave(self, folder: Path):
        return np.load(folder / 'audio_wave.npy', mmap_mode='r')


# ----------------------------------------------------------------------------- frozen conditioning
class FrozenConditioner(nn.Module):
    """Frozen recovery-v3 encoder(s) + frozen envelope decoder(s) -> conditioning at mel frame times.

    ``compact`` (default): the encoder head output (its normalised HuBERT-space
    prediction) projected on the top PCs of the frozen teacher space, the
    predicted duration, and the envelope decoder's prediction.  The head is the
    part of the encoder that was trained to generalise (contrastive + mel
    losses); the raw 128-d trunk (``trunk`` mode) carries a per-trial
    fingerprint that a decoder memorises.

    Several folds may be given (cross-fitting): trial ``i`` is conditioned by
    the models of ``fold[i]``, which must never have seen its sentence.
    """
    def __init__(self, encoder_paths, envelope_paths, mel_times: torch.Tensor, train_dataset,
                 *, mode: str = 'compact', components: int = 16, teacher_train: np.ndarray | None = None):
        super().__init__()
        if mode not in ('compact', 'trunk'):
            raise ValueError(mode)
        if not encoder_paths or (envelope_paths and len(envelope_paths) != len(encoder_paths)):
            raise ValueError('one encoder per fold, and one envelope decoder per fold when envelopes are used')
        self.mode, self.components = mode, int(components)
        self.encoders, self.encoder_sha256, subjects = nn.ModuleList(), [], None
        for path in encoder_paths:
            payload = recovery.load_checkpoint(Path(path)) if str(path).endswith('best_passed.pt') else torch.load(Path(path), map_location='cpu', weights_only=False)
            encoder = RecoveryEEGModel(legacy.decoder_from(payload), **payload['signature']['spec'])
            encoder.load_state_dict(payload['model']); encoder.eval().requires_grad_(False)
            if subjects is not None and list(payload['signature']['subjects']) != subjects:
                raise ValueError('fold encoders disagree on the subject table')
            subjects = list(payload['signature']['subjects'])
            self.encoders.append(encoder); self.encoder_sha256.append(legacy.sha256(Path(path)))
        self.subjects = subjects
        self.register_buffer('mel_times', mel_times.clone())
        self.envelopes, self.envelope_sha256 = None, []
        if envelope_paths:
            self.envelopes = nn.ModuleList()
            for path in envelope_paths:
                env = torch.load(Path(path), map_location='cpu', weights_only=False)
                spec = env['args']
                model = EnvelopeDecoder(channels=128, width=spec['width'], subjects=len(env['subjects']),
                                        bands=len(BANDS) if spec['band_split'] else 1, per_subject_spatial=spec['per_subject_spatial'])
                model.load_state_dict(env['model']); model.eval().requires_grad_(False)
                if list(env['subjects']) != subjects:
                    raise ValueError('encoder and envelope decoder disagree on the subject table')
                self.envelopes.append(model); self.envelope_sha256.append(legacy.sha256(Path(path)))
            count = (EEG_SAMPLES + 2 * 7 - 15) // 4 + 1
            token_times = EEG_START + np.arange(count) * 4 / EEG_RATE
            targets = envelope_targets(train_dataset, token_times)
            stack = np.stack([v for v, _ in targets.values()]); masks = np.stack([m for _, m in targets.values()])
            prior = np.where(masks.sum(0) > 0, np.nansum(stack * masks, 0) / np.maximum(masks.sum(0), 1), stack.mean(0))
            self.register_buffer('envelope_prior', torch.from_numpy(prior.astype(np.float32)))
        if self.mode == 'compact':
            # Directions of the frozen teacher space along which the head output is read: train-fold audio only.
            if teacher_train is None:
                raise ValueError('compact conditioning needs the train-fold teacher features for the PCA basis')
            flat = self.encoders[0].decoder.normalizer(torch.from_numpy(np.asarray(teacher_train, dtype=np.float32))).reshape(-1, 768)
            mu = flat.mean(0)
            _, _, basis = torch.pca_lowrank(flat - mu, q=self.components, center=False)
            self.register_buffer('pca_mean', mu); self.register_buffer('pca_basis', basis[:, :self.components].contiguous())
            self.dimension = self.components + 1 + (1 if self.envelopes is not None else 0)
        else:
            self.dimension = self.encoders[0].spatial.out_channels + (1 if self.envelopes is not None else 0)
        # Per-channel standardisation of the conditioning (fitted with fit_statistics on train trials).
        self.register_buffer('channel_mean', torch.zeros(self.dimension)); self.register_buffer('channel_scale', torch.ones(self.dimension))

    @property
    def folds(self):
        return len(self.encoders)

    @torch.no_grad()
    def fit_statistics(self, role, device, count: int = 512):
        ids = np.arange(min(count, len(role)))
        values = []
        for offset in range(0, len(ids), 64):
            chunk = ids[offset:offset + 64]
            eeg, mask, subject, fold = batch_tensors(role, chunk, device)
            values.append(self.raw(eeg, mask, subject, fold).transpose(1, 2).reshape(-1, self.dimension).cpu())
        values = torch.cat(values)
        self.channel_mean.copy_(values.mean(0).to(self.channel_mean.device)); self.channel_scale.copy_(values.std(0).clamp_min(1e-4).to(self.channel_scale.device))

    def statistics(self):
        return dict(channel_mean=self.channel_mean.cpu().tolist(), channel_scale=self.channel_scale.cpu().tolist(),
                    mode=self.mode, components=self.components, folds=self.folds)

    def load_statistics(self, state):
        if state['mode'] != self.mode or state['components'] != self.components or state.get('folds', 1) != self.folds:
            raise ValueError('conditioning statistics were fitted for a different conditioning setup')
        self.channel_mean.copy_(torch.tensor(state['channel_mean'])); self.channel_scale.copy_(torch.tensor(state['channel_scale']))

    def mix(self, eeg, channel_mask, subject, fold):
        """Per-trial subject mixing of both frozen models (per fold), applied before any averaging."""
        x = eeg * channel_mask[:, :, None]
        a = x.clone(); b = x.clone()
        for f in range(self.folds):
            rows = (fold == f).nonzero().flatten()
            if not len(rows):
                continue
            a[rows] = self.encoders[f].mix_subject(x[rows], subject[rows] if subject is not None else None)
            env = self.envelopes[f] if self.envelopes is not None else None
            if env is not None and subject is not None and env.subjects and not env.per_subject_spatial:
                b[rows] = torch.bmm(torch.eye(x.shape[1], device=x.device, dtype=x.dtype) + env.subject_delta[subject[rows]], x[rows])
        return a, b

    @torch.no_grad()
    def forward(self, eeg, channel_mask, subject, fold, *, premixed=None, noise: float = 0., channel_dropout: float = 0.):
        """Standardised (B, D, MEL_FRAMES) conditioning; optional training-time noise / channel dropout."""
        c = (self.raw(eeg, channel_mask, subject, fold, premixed=premixed) - self.channel_mean[None, :, None]) / self.channel_scale[None, :, None]
        if noise > 0:
            c = c + noise * torch.randn_like(c)
        if channel_dropout > 0:
            c = c * (torch.rand(c.shape[0], c.shape[1], 1, device=c.device) >= channel_dropout)
        return c

    @torch.no_grad()
    def raw(self, eeg, channel_mask, subject, fold, *, premixed=None):
        """Unstandardised (B, D, MEL_FRAMES) conditioning at mel frame times."""
        a, b = self.mix(eeg, channel_mask, subject, fold) if premixed is None else premixed
        out = torch.zeros(len(a), self.dimension, len(self.mel_times), device=a.device)
        for f in range(self.folds):
            rows = (fold == f).nonzero().flatten()
            if not len(rows):
                continue
            enc = self.encoders[f]
            x = F.gelu(enc.temporal(enc.spatial(a[rows])))
            if enc.positional:
                x = x + enc.position
            x = enc.output_norm(enc.blocks(x).transpose(1, 2))                       # B, T, W
            if self.mode == 'compact':
                z = enc.head(x.transpose(1, 2)).transpose(1, 2)                      # normalised HuBERT-space prediction
                duration = torch.sigmoid(enc.duration(x.mean(1))).squeeze(-1)         # B
                parts = [(z - self.pca_mean) @ self.pca_basis, duration[:, None, None].expand(-1, x.shape[1], 1)]
            else:
                parts = [x]
            if self.envelopes is not None:
                env = self.envelopes[f]
                y = F.gelu(env.temporal(env.spatial(b[rows])))
                if env.positional:
                    y = y + env.position
                y = env.output_norm(env.blocks(y).transpose(1, 2)).transpose(1, 2)
                predicted = env.head(y).squeeze(1) + env.prior_gain * self.envelope_prior[None]
                parts.append(predicted[:, :, None])
            c = sample_at(torch.cat(parts, -1), enc.token_times, self.mel_times)      # B, F, D
            out[rows] = c.transpose(1, 2)
        return out

    @torch.no_grad()
    def teacher(self, teacher_sequence, speech_times):
        """Normalised HuBERT teacher (B, 199, 768) -> (B, 768, MEL_FRAMES) for the audio-conditioned ceiling."""
        z = self.encoders[0].decoder.normalizer(teacher_sequence)
        return sample_at(z, speech_times.to(z.device), self.mel_times).transpose(1, 2)


# ----------------------------------------------------------------------------- diffusion model
def sinusoidal(timestep: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    freqs = torch.exp(-math.log(10000.) * torch.arange(half, device=timestep.device).float() / half)
    angles = timestep.float()[:, None] * freqs[None]
    return torch.cat([angles.sin(), angles.cos()], 1)


class Block(nn.Module):
    """adaLN-zero residual block: dilated depthwise conv, optional self-attention, MLP; conditioning re-injected."""
    def __init__(self, hidden, time_dim, dilation, heads, attention, dropout):
        super().__init__()
        self.cond = nn.Conv1d(hidden, hidden, 1)
        self.norm1 = nn.LayerNorm(hidden); self.film1 = nn.Linear(time_dim, 3 * hidden)
        self.depthwise = nn.Conv1d(hidden, hidden, 5, padding=2 * dilation, dilation=dilation, groups=hidden)
        self.pointwise = nn.Conv1d(hidden, hidden, 1)
        self.attention = attention
        if attention:
            self.norm2 = nn.LayerNorm(hidden); self.film2 = nn.Linear(time_dim, 3 * hidden)
            self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(hidden); self.film3 = nn.Linear(time_dim, 3 * hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden))
        for film in (self.film1, self.film3) + ((self.film2,) if attention else ()):
            nn.init.zeros_(film.weight); nn.init.zeros_(film.bias)

    @staticmethod
    def modulate(norm, film, h, t):
        scale, shift, gate = film(t)[:, None, :].chunk(3, -1)
        return norm(h) * (1 + scale) + shift, gate

    def forward(self, h, c, t):
        h = h + self.cond(c.transpose(1, 2)).transpose(1, 2)
        u, gate = self.modulate(self.norm1, self.film1, h, t)
        u = self.pointwise(F.gelu(self.depthwise(u.transpose(1, 2)))).transpose(1, 2)
        h = h + gate * u
        if self.attention:
            u, gate = self.modulate(self.norm2, self.film2, h, t)
            u, _ = self.attn(u, u, u, need_weights=False)
            h = h + gate * u
        u, gate = self.modulate(self.norm3, self.film3, h, t)
        return h + gate * self.mlp(u)


class MelDiffusion(nn.Module):
    """v-prediction diffusion over the normalised (80, 251) native mel, cosine schedule.

    Conditioning streams share one hidden-size channel sequence: EEG features
    (frozen conditioner, D channels), the audio teacher (768 channels, ceiling)
    or a learned null embedding (classifier-free guidance).
    """
    def __init__(self, eeg_dimension, teacher_dimension=768, hidden=256, blocks=6, heads=4, dropout=.1,
                 timesteps=1000, frames=MEL_FRAMES):
        super().__init__()
        self.hidden, self.timesteps, self.frames = hidden, timesteps, frames
        self.input = nn.Conv1d(MEL_BINS, hidden, 3, padding=1)
        self.eeg_input = nn.Conv1d(eeg_dimension, hidden, 3, padding=1)
        self.teacher_input = nn.Conv1d(teacher_dimension, hidden, 3, padding=1)
        self.null = nn.Parameter(torch.zeros(1, frames, hidden))
        self.position = nn.Parameter(torch.zeros(1, frames, hidden))
        nn.init.normal_(self.position, std=.02)
        self.time = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden))
        dilations = [1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8]
        self.blocks = nn.ModuleList([Block(hidden, hidden, dilations[i % len(dilations)], heads, i % 2 == 1, dropout)
                                     for i in range(blocks)])
        self.output_norm = nn.LayerNorm(hidden)
        self.output = nn.Conv1d(hidden, MEL_BINS, 3, padding=1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        steps = torch.arange(timesteps + 1).double() / timesteps
        s = .008
        alpha_bar = (torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2)
        alpha_bar = (alpha_bar / alpha_bar[0]).clamp(1e-5, 1.).float()
        self.register_buffer('alpha_bar', alpha_bar)      # index t in [0, T]; alpha_bar[0] = 1

    def condition(self, kind, value):
        if kind == 'null' or value is None:
            return self.null.expand(value.shape[0] if value is not None else 1, -1, -1)
        if kind == 'eeg':
            return self.eeg_input(value).transpose(1, 2)
        if kind == 'teacher':
            return self.teacher_input(value).transpose(1, 2)
        raise ValueError(kind)

    def forward(self, x, t, c):
        """x (B, 80, F) noisy normalised mel, t (B,) integer step in [1, T], c (B, F, hidden) conditioning stream."""
        h = self.input(x).transpose(1, 2) + self.position
        te = self.time(sinusoidal(t, self.hidden))
        for block in self.blocks:
            h = block(h, c, te)
        return self.output(self.output_norm(h).transpose(1, 2))

    def loss(self, x0, c):
        t = torch.randint(1, self.timesteps + 1, (len(x0),), device=x0.device)
        ab = self.alpha_bar[t][:, None, None]
        noise = torch.randn_like(x0)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        v = ab.sqrt() * noise - (1 - ab).sqrt() * x0
        return F.mse_loss(self(xt, t, c), v)

    def loss_at(self, x0, c, steps):
        """Denoising loss at fixed timesteps with a fixed noise seed: a low-variance validation quantity."""
        generator = torch.Generator(device='cpu').manual_seed(1234)
        noise = torch.randn(x0.shape, generator=generator).to(x0.device)
        total = 0.
        for step in steps:
            t = torch.full((len(x0),), int(step), device=x0.device)
            ab = self.alpha_bar[t][:, None, None]
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
            v = ab.sqrt() * noise - (1 - ab).sqrt() * x0
            total += float(F.mse_loss(self(xt, t, c), v))
        return total / len(steps)

    @torch.no_grad()
    def sample(self, c, *, null, steps=50, guidance=2., noise=None, clamp=(-2., 3.)):
        """DDIM (eta = 0) with classifier-free guidance on v.  ``null`` is the null stream of the same batch size."""
        batch = c.shape[0]
        x = torch.randn(batch, MEL_BINS, self.frames, device=c.device) if noise is None else noise.to(c.device)
        schedule = torch.linspace(self.timesteps, 0, steps + 1).round().long().tolist()
        for i in range(steps):
            t, t_next = schedule[i], schedule[i + 1]
            tt = torch.full((batch,), t, device=c.device)
            if guidance == 1.:
                v = self(x, tt, c)
            else:
                v_pair = self(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([c, null]))
                v_c, v_u = v_pair.chunk(2)
                v = v_u + guidance * (v_c - v_u)
            ab = self.alpha_bar[t]
            x0 = (ab.sqrt() * x - (1 - ab).sqrt() * v).clamp(*clamp)
            eps = (1 - ab).sqrt() * x + ab.sqrt() * v
            ab_next = self.alpha_bar[t_next]
            x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
        return x


class MelScaler:
    """Global affine normalisation of the clamped native mel; silence tail restored on the way back."""
    def __init__(self, mean: float, scale: float):
        self.mean, self.scale = float(mean), float(scale)

    @classmethod
    def fit(cls, mel: torch.Tensor):
        value = mel.clamp_min(MEL_FLOOR)
        return cls(float(value.mean()), float(value.std()))

    def encode(self, mel):
        return (mel.clamp_min(MEL_FLOOR) - self.mean) / self.scale

    def decode(self, x):
        mel = x * self.scale + self.mean
        return mel

    def state(self):
        return dict(mean=self.mean, scale=self.scale)


def silence_tail(mel: torch.Tensor, threshold: float = -6.):
    """Frames the generator left at the floor become vocoder silence; the tail is the sample's own decision."""
    quiet = mel.mean(1, keepdim=True) < threshold                     # B, 1, F
    return torch.where(quiet, torch.full_like(mel, SILENCE_MEL), mel)


# ----------------------------------------------------------------------------- cross-fitting
def fold_encoder(fold, ids, held, v3, audio, train_role, device, updates, output: Path, seed: int):
    """The v3 recipe (fresh init, acoustic weights, positional code, subject layer, augmentation) on one content half.

    Data come from the cache instead of h5py, which is what makes two extra
    encoders affordable.  No validation selection: the final weights are kept.
    """
    import pandas as pd
    legacy.seed_all(seed)
    spec, weights = v3['signature']['spec'], v3['signature']['weights']
    model = RecoveryEEGModel(legacy.decoder_from(v3), **spec).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=.05)
    frame = pd.DataFrame(dict(content_group=[train_role.content[i] for i in ids]))
    contents = {c: k for k, c in enumerate(sorted(set(frame.content_group)))}
    speech_times, mel_times = audio.speech_times.to(device), audio.mel_times.to(device)
    xyz = torch.zeros(1, 128, 3, device=device)
    step = epoch = 0; started = time.monotonic(); collected = []
    while step < updates:
        for batch_ids in diverse_batches(frame, 32, seed + epoch):
            if step >= updates:
                break
            rows = ids[batch_ids]
            eeg, mask, subject, _ = batch_tensors(train_role, rows, device)
            eeg = augment_eeg(eeg, mask)
            teacher = torch.from_numpy(np.asarray(audio.teacher[train_role.audio_index[rows]], dtype=np.float32)).to(device)
            mel = audio.mel[train_role.audio_index[rows]].to(device)
            frames = torch.from_numpy(train_role.duration_frames[rows]).to(device)
            mel_mask, speech_mask = speech_frame_masks(frames, mel_times, speech_times)
            for group in optimizer.param_groups:
                group['lr'] = recovery.learning_rate(step, 3e-4, updates, 200)
            model.train(); model.decoder.eval(); optimizer.zero_grad(set_to_none=True)
            state = model(eeg, xyz.expand(len(rows), -1, -1), mask, torch.ones(len(rows), EEG_SAMPLES, dtype=torch.bool, device=device), subject)
            indices = torch.tensor([contents[c] for c in frame.content_group.iloc[batch_ids]], device=device)
            loss, parts = recovery_loss(state, teacher, mel, model.decoder.normalizer, speech_mask, mel_mask,
                                        duration_fraction(frames, MEL_FRAMES), indices, mel_weight=min(1., (step + 1) / 100), **weights)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite fold-encoder loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step(); step += 1
            collected.append(parts)
            if step % 100 == 0:
                print(json.dumps(dict(fold=fold, update=step, seconds=round(time.monotonic() - started, 1),
                                      **{k: round(float(np.mean([v[k] for v in collected])), 4) for k in ('loss', 'contrastive', 'mel', 'batch_accuracy')})), flush=True)
                collected = []
        epoch += 1
    check = held_out_check(model, held, audio, train_role, device)
    print(json.dumps(dict(fold=fold, held_out=check)), flush=True)
    legacy.atomic_save(output, dict(contract='generative_recovery_crossfit_encoder', fold=fold,
                                    signature=dict(spec=spec, subjects=v3['signature']['subjects'], weights=weights, updates=updates, seed=seed,
                                                   v3_encoder=legacy.sha256(Path(DEFAULT_ENCODER))),
                                    model=model.state_dict(), decoder_spec=v3['decoder_spec'], decoder=model.decoder.state_dict(),
                                    held_out=check))


@torch.no_grad()
def held_out_check(model, held, audio, train_role, device, count: int = 256):
    """Speech-frame mel MAE and envelope r on the sentences this fold never saw, real vs zero EEG."""
    model.eval()
    ids = held[:count]
    out = {'correct': dict(mae=[], env=[]), 'zero': dict(mae=[], env=[])}
    for offset in range(0, len(ids), 32):
        rows = ids[offset:offset + 32]
        eeg, mask, subject, _ = batch_tensors(train_role, rows, device)
        truth = audio.mel[train_role.audio_index[rows]].to(device)
        frames = torch.from_numpy(train_role.duration_frames[rows]).to(device)
        for name in out:
            x = eeg if name == 'correct' else torch.zeros_like(eeg)
            state = model(x, torch.zeros(len(rows), 128, 3, device=device), mask, torch.ones(len(rows), EEG_SAMPLES, dtype=torch.bool, device=device), subject)
            for i in range(len(rows)):
                m = torch.arange(MEL_FRAMES, device=device) < frames[i]
                out[name]['mae'].append(float((state.native_mel[i][:, m] - truth[i][:, m]).abs().mean()))
                a = state.native_mel[i][:, m].mean(0); b = truth[i][:, m].mean(0); a = a - a.mean(); b = b - b.mean()
                out[name]['env'].append(float(a @ b / (a.norm() * b.norm() + 1e-8)))
    return {name: dict(native_mel_mae=float(np.mean(v['mae'])), envelope_corr=float(np.mean(v['env']))) for name, v in out.items()}


def fold_envelope(fold, ids, audio, train_role, train_dataset, subjects, device, updates, output: Path, seed: int, trunk_path: Path):
    """The envelope-decoder recipe (Broderick trunk init, correlation + partial-correlation loss) on one content half."""
    legacy.seed_all(seed)
    count = (EEG_SAMPLES + 2 * 7 - 15) // 4 + 1
    token_times = EEG_START + np.arange(count) * 4 / EEG_RATE
    targets = envelope_targets(train_dataset, token_times)
    stack = np.stack([v for v, _ in targets.values()]); masks = np.stack([m for _, m in targets.values()])
    prior = torch.from_numpy(np.where(masks.sum(0) > 0, np.nansum(stack * masks, 0) / np.maximum(masks.sum(0), 1), stack.mean(0)).astype(np.float32)).to(device)
    model = EnvelopeDecoder(channels=128, width=128, subjects=len(subjects), bands=1, per_subject_spatial=False).to(device)
    trunk = torch.load(trunk_path, map_location='cpu', weights_only=False)['trunk']
    model.load_state_dict(trunk, strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.05)
    rng = np.random.default_rng(seed); started = time.monotonic(); collected = []
    for step in range(1, updates + 1):
        lr = 3e-4 * min(1., step / 200) * (.1 + .9 * .5 * (1 + math.cos(math.pi * min(1., step / updates))))
        for group in optimizer.param_groups:
            group['lr'] = lr
        rows = rng.choice(ids, size=32, replace=False)
        eeg, mask, subject, _ = batch_tensors(train_role, rows, device)
        eeg = augment_eeg(eeg * mask[:, :, None], mask)
        keys = [train_role.audio_key[i] for i in rows]
        target = torch.stack([torch.from_numpy(targets[k][0]) for k in keys]).to(device)
        tmask = torch.stack([torch.from_numpy(targets[k][1]) for k in keys]).to(device)
        prior_batch = prior[None].expand(len(rows), -1)
        model.train(); optimizer.zero_grad(set_to_none=True)
        prediction = model(eeg, subject, prior_batch)
        loss = (1 - masked_correlation(prediction, target, tmask)).mean() + .3 * (1 - partial_correlation(prediction, target, prior_batch, tmask)).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
        collected.append(float(loss))
        if step % 100 == 0:
            print(json.dumps(dict(fold=fold, envelope_update=step, loss=round(float(np.mean(collected)), 4), seconds=round(time.monotonic() - started, 1))), flush=True); collected = []
    torch.save(dict(contract='envelope_decoder_v1', model=model.state_dict(), update=updates, subjects=list(subjects),
                    args=dict(width=128, band_split=False, per_subject_spatial=False, initialize_trunk=str(trunk_path), fold=fold, seed=seed)), output)


def crossfit(args, cfg):
    device = legacy.device(args.device)
    folder = Path(args.output) / 'cache'
    if not (folder / 'complete.json').exists():
        raise SystemExit('run `cache` first')
    audio = Audio(folder); train_role = Role(folder, 'train'); train_dataset = legacy.dataset_for(cfg, 'train')
    root = Path(args.output) / 'crossfit'; root.mkdir(parents=True, exist_ok=True)
    halves = content_halves(set(train_role.content), args.crossfit_seed)
    v3 = recovery.load_checkpoint(Path(args.encoder))
    for f, fit_contents in enumerate(halves):
        keep = set(fit_contents)
        ids = np.array([i for i, c in enumerate(train_role.content) if c in keep])
        held = np.array([i for i, c in enumerate(train_role.content) if c not in keep])
        print(json.dumps(dict(fold=f, fit_contents=len(keep), fit_trials=len(ids), held_trials=len(held))), flush=True)
        target = root / f'encoder_fold{f}.pt'
        if not target.exists():
            fold_encoder(f, ids, held, v3, audio, train_role, device, args.crossfit_updates, target, args.crossfit_seed + f)
        target = root / f'envelope_fold{f}.pt'
        if args.envelope and not target.exists():
            fold_envelope(f, ids, audio, train_role, train_dataset, v3['signature']['subjects'], device,
                          args.crossfit_envelope_updates, target, args.crossfit_seed + f, Path(args.broderick_trunk))
    legacy.atomic_json(root / 'complete.json', dict(contract=CONTRACT, halves=[len(h) for h in halves], seed=args.crossfit_seed,
                                                    updates=args.crossfit_updates, envelope_updates=args.crossfit_envelope_updates))
    print('crossfit complete')


# ----------------------------------------------------------------------------- training
def batch_tensors(role: Role, ids, device):
    eeg = torch.from_numpy(np.asarray(role.eeg[ids], dtype=np.float32)).to(device)
    mask = torch.from_numpy(role.channel_mask[ids]).to(device)
    subject = torch.from_numpy(role.subject_index[ids]).to(device)
    fold = torch.from_numpy(role.fold[ids]).to(device)
    return eeg, mask, subject, fold


def pooled_eeg(conditioner: FrozenConditioner, role: Role, ids, device, rng=None, k=0):
    """Average the subject-mixed EEG of ``k`` (0 = all) trials sharing each trial's sentence (own trial included)."""
    outs_a, outs_b = [], []
    for i in ids:
        members = role.groups[role.content[i]]
        if k and k < len(members):
            others = [j for j in members if j != i]
            chosen = [i] + list(rng.choice(others, size=k - 1, replace=False))
        else:
            chosen = members
        eeg, mask, subject, fold = batch_tensors(role, chosen, device)
        a, b = conditioner.mix(eeg, mask, subject, fold)
        outs_a.append(a.mean(0)); outs_b.append(b.mean(0))
    return torch.stack(outs_a), torch.stack(outs_b)


class EMA:
    def __init__(self, model, decay=.999):
        self.shadow = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.lerp_(p.detach(), 1 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)


def content_halves(contents, seed: int):
    """Deterministic split of the train sentences into two halves (for cross-fitting)."""
    ordered = sorted(contents, key=lambda c: hashlib.sha256(f'{seed}:{c}'.encode()).hexdigest())
    return ordered[:len(ordered) // 2], ordered[len(ordered) // 2:]


def prepare(args, cfg, device):
    folder = Path(args.output) / 'cache'
    if not (folder / 'complete.json').exists():
        raise SystemExit('run `cache` first')
    audio = Audio(folder)
    train_dataset = legacy.dataset_for(cfg, 'train')
    key_index = {k: i for i, k in enumerate(audio.keys)}
    train_keys = [key_index[k] for k in sorted(set(train_dataset.frame.audio_key))]
    train_role, val_role = Role(folder, 'train'), Role(folder, 'validation')
    if args.crossfit:
        # Fold f's models were fitted on half f; trial i is conditioned by the fold that never saw its sentence.
        root = Path(args.output) / 'crossfit'
        encoders = [root / f'encoder_fold{f}.pt' for f in range(2)]
        envelopes = [root / f'envelope_fold{f}.pt' for f in range(2)] if args.envelope else None
        halves = content_halves(set(train_role.content), args.crossfit_seed)
        fold_of = {c: 1 for c in halves[0]} | {c: 0 for c in halves[1]}
        train_role.assign_folds(fold_of)
        val_role.assign_folds({c: int(hashlib.sha256(f'{args.crossfit_seed}:v:{c}'.encode()).hexdigest(), 16) % 2 for c in set(val_role.content)})
    else:
        encoders = [Path(args.encoder)]; envelopes = [Path(args.envelope)] if args.envelope else None
    conditioner = FrozenConditioner(encoders, envelopes, audio.mel_times, train_dataset, mode=args.conditioning,
                                    components=args.components,
                                    teacher_train=audio.teacher[train_keys] if args.conditioning == 'compact' else None).to(device)
    conditioner.fit_statistics(train_role, device)
    return folder, audio, conditioner, train_dataset, train_role, val_role


def sampled_metrics(model, conditioner, scaler, audio, role: Role, ids, device, *, steps, guidance, templates):
    """Quick sampled-mel check on a few validation trials: MAE/envelope r on speech frames for correct vs zero EEG."""
    eeg, mask, subject, fold = batch_tensors(role, ids, device)
    generator = torch.Generator(device='cpu').manual_seed(777)
    noise = torch.randn(len(ids), MEL_BINS, MEL_FRAMES, generator=generator).to(device)
    truth = audio.mel[role.audio_index[ids]].to(device)
    frames = torch.from_numpy(role.duration_frames[ids]).to(device)
    mel_mask = torch.arange(MEL_FRAMES, device=device)[None] < frames[:, None]
    null = model.condition('null', torch.zeros(len(ids), 1, device=device))
    wrong_eeg, wrong_mask, _, _ = batch_tensors(role, role.wrong_index[ids], device)
    out, per_trial = {}, {}
    for name in ('correct', 'zero', 'wrong_trial'):
        if name == 'correct':
            c = conditioner(eeg, mask, subject, fold)
        elif name == 'zero':
            c = conditioner(torch.zeros_like(eeg), mask, subject, fold)
        else:
            c = conditioner(wrong_eeg, wrong_mask, subject, fold)
        mel = scaler.decode(model.sample(model.condition('eeg', c), null=null, steps=steps, guidance=guidance, noise=noise))
        maes, corrs = [], []
        for i in range(len(ids)):
            m = mel_mask[i]
            maes.append(float((mel[i][:, m] - truth[i][:, m]).abs().mean()))
            a = mel[i][:, m].mean(0); b = truth[i][:, m].mean(0)
            a = a - a.mean(); b = b - b.mean()
            corrs.append(float(a @ b / (a.norm() * b.norm() + 1e-8)))
        per_trial[name] = np.array(corrs)
        out[name] = dict(mel_mae=float(np.mean(maes)), envelope_corr=float(np.mean(corrs)))
    # Paired against the in-distribution control (same noise seed): the honest sample-level EEG effect.
    out['envelope_gain_over_wrong'] = float(np.mean(per_trial['correct'] - per_trial['wrong_trial']))
    out['fraction_beats_wrong'] = float(np.mean(per_trial['correct'] > per_trial['wrong_trial']))
    template = templates[1].to(device)
    out['template'] = dict(mel_mae=float(np.mean([float((template[:, mel_mask[i]] - truth[i][:, mel_mask[i]]).abs().mean()) for i in range(len(ids))])))
    return out


def train(args, cfg):
    legacy.seed_all(args.seed); device = legacy.device(args.device)
    folder, audio, conditioner, train_dataset, train_role, val_role = prepare(args, cfg, device)
    train_keys = sorted(set(train_dataset.frame.audio_key))
    key_index = {k: i for i, k in enumerate(audio.keys)}
    scaler = MelScaler.fit(audio.mel[[key_index[k] for k in train_keys]])
    templates = train_templates(train_dataset)
    model = MelDiffusion(conditioner.dimension, hidden=args.hidden, blocks=args.blocks, dropout=args.dropout).to(device)
    ema = EMA(model, args.ema)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(.9, .99))
    output = Path(args.output) / args.run; output.mkdir(parents=True, exist_ok=True)
    signature = dict(contract=CONTRACT, encoder=conditioner.encoder_sha256, envelope=conditioner.envelope_sha256,
                     crossfit=bool(args.crossfit), crossfit_seed=args.crossfit_seed, seed=args.seed, hidden=args.hidden,
                     blocks=args.blocks, dropout=args.dropout, lr=args.lr, updates=args.updates, batch=args.batch_size,
                     p_teacher=args.p_teacher, p_null=args.p_null, p_pool=args.p_pool, pool_max=args.pool_max, augment=args.augment,
                     condition_noise=args.condition_noise, condition_dropout=args.condition_dropout,
                     conditioning=args.conditioning, components=args.components,
                     scaler=scaler.state(), conditioning_dimension=conditioner.dimension)
    print(json.dumps(dict(parameters=sum(p.numel() for p in model.parameters()), **signature)), flush=True)
    rng = np.random.default_rng(args.seed)
    speech_times = audio.speech_times.to(device)
    val_ids = np.array(sorted(rng.choice(len(val_role), size=min(128, len(val_role)), replace=False)))
    val_eeg, val_mask, val_subject, val_fold = batch_tensors(val_role, val_ids, device)
    val_x0 = scaler.encode(audio.mel[val_role.audio_index[val_ids]].to(device))
    with torch.no_grad():
        val_c = conditioner(val_eeg, val_mask, val_subject, val_fold)
    history, best, started, collected = [], float('inf'), time.monotonic(), []
    best_advantage = -float('inf')
    step = 0
    state_path = output / 'training_state.pt'
    if state_path.exists():
        saved = torch.load(state_path, map_location='cpu', weights_only=False)
        if saved['signature'] != signature:
            raise ValueError('resume arguments changed; use a new --run')
        model.load_state_dict(saved['model']); ema.shadow.load_state_dict(saved['ema']); optimizer.load_state_dict(saved['optimizer'])
        step, history, best = saved['step'], saved['history'], saved['best']
        best_advantage = saved.get('best_advantage', -float('inf'))
        rng = np.random.default_rng(args.seed + step)
        print(f'resuming at update {step}', flush=True)

    def save(path, extra=None):
        legacy.atomic_save(path, dict(signature=signature, model=model.state_dict(), ema=ema.shadow.state_dict(),
                                      optimizer=optimizer.state_dict(), step=step, history=history, best=best,
                                      best_advantage=best_advantage, conditioner=conditioner.statistics(),
                                      model_spec=dict(eeg_dimension=conditioner.dimension, hidden=args.hidden, blocks=args.blocks),
                                      extra=extra))

    while step < args.updates:
        ids = rng.choice(len(train_role), size=args.batch_size, replace=False)
        eeg, mask, subject, fold = batch_tensors(train_role, ids, device)
        x0 = scaler.encode(audio.mel[train_role.audio_index[ids]].to(device))
        modes = rng.choice(['eeg', 'teacher', 'null'], size=len(ids), p=[1 - args.p_teacher - args.p_null, args.p_teacher, args.p_null])
        with torch.no_grad():
            if args.augment:
                eeg = augment_eeg(eeg, mask)
            premixed = conditioner.mix(eeg, mask, subject, fold)
            pool = rng.random(len(ids)) < args.p_pool
            if pool.any():
                pooled_ids = ids[pool]
                a, b = pooled_eeg(conditioner, train_role, pooled_ids, device, rng, k=int(rng.integers(2, args.pool_max + 1)))
                pa, pb = premixed
                pa = pa.clone(); pb = pb.clone(); pa[torch.from_numpy(pool).to(device)] = a; pb[torch.from_numpy(pool).to(device)] = b
                premixed = (pa, pb)
            raw_eeg = conditioner(eeg, mask, subject, fold, premixed=premixed, noise=args.condition_noise, channel_dropout=args.condition_dropout)
            teacher = torch.from_numpy(np.asarray(audio.teacher[train_role.audio_index[ids]], dtype=np.float32)).to(device)
            raw_teacher = conditioner.teacher(teacher, speech_times)
        # The projection layers are part of the model: keep them in the graph.
        c_eeg = model.condition('eeg', raw_eeg)
        c_teacher = model.condition('teacher', raw_teacher)
        c_null = model.condition('null', torch.zeros(len(ids), 1, device=device))
        c = torch.where(torch.from_numpy(modes == 'eeg').to(device)[:, None, None], c_eeg,
                        torch.where(torch.from_numpy(modes == 'teacher').to(device)[:, None, None], c_teacher, c_null))
        lr = args.lr * min(1., (step + 1) / args.warmup) * (.05 + .95 * .5 * (1 + math.cos(math.pi * min(1., step / args.updates))))
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = model.loss(x0, c)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite diffusion loss')
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        optimizer.step(); ema.update(model); step += 1
        collected.append(float(loss))
        if step % 50 == 0:
            print(json.dumps(dict(update=step, loss=round(float(np.mean(collected)), 4), grad=round(grad, 3), lr=round(lr, 6),
                                  seconds=round(time.monotonic() - started, 1))), flush=True); collected = []
        if step % args.eval_every == 0 or step == args.updates:
            shadow = ema.shadow.eval()
            with torch.no_grad():
                fixed = (100, 300, 500, 700, 900)
                val_null = shadow.condition('null', torch.zeros(len(val_ids), 1, device=device))
                report = dict(update=step, val_loss_eeg=shadow.loss_at(val_x0, shadow.condition('eeg', val_c), fixed),
                              val_loss_null=shadow.loss_at(val_x0, val_null, fixed))
                report['val_eeg_advantage'] = report['val_loss_null'] - report['val_loss_eeg']
                report['sampled'] = sampled_metrics(shadow, conditioner, scaler, audio, val_role, val_ids[:64], device,
                                                    steps=args.quick_steps, guidance=args.quick_guidance, templates=templates)
            history.append(report)
            print(json.dumps(report), flush=True)
            legacy.atomic_json(output / 'metrics.json', dict(signature=signature, history=history))
            if report['val_loss_eeg'] < best:
                best = report['val_loss_eeg']; save(output / 'best.pt', report)
            if step >= args.min_updates and report['val_eeg_advantage'] > best_advantage:
                best_advantage = report['val_eeg_advantage']; save(output / 'best_advantage.pt', report)
            save(state_path)
    save(output / 'last.pt', history[-1] if history else None)
    print(json.dumps(dict(finished=step, best_val_loss_eeg=best, output=str(output))), flush=True)


# ----------------------------------------------------------------------------- export
def load_model(path: Path, device):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = MelDiffusion(**payload['model_spec']).to(device)
    model.load_state_dict(payload['ema']); model.eval()
    scaler = MelScaler(**payload['signature']['scaler'])
    return model, scaler, payload


def export(args, cfg):
    import soundfile as sf
    legacy.seed_all(args.seed); device = legacy.device(args.device)
    folder, audio, conditioner, train_dataset, _, role = prepare(args, cfg, device)
    model, scaler, payload = load_model(Path(args.checkpoint), device)
    trained_with = payload['signature']['encoder']
    if ([trained_with] if isinstance(trained_with, str) else list(trained_with)) != conditioner.encoder_sha256:
        raise ValueError('checkpoint was trained with different encoder(s)')
    conditioner.load_statistics(payload['conditioner'])
    v3 = recovery.load_checkpoint(Path(args.encoder))
    regression = RecoveryEEGModel(legacy.decoder_from(v3), **v3['signature']['spec']).to(device)
    regression.load_state_dict(v3['model']); regression.eval()
    v3_subjects = {s: i for i, s in enumerate(v3['signature']['subjects'])}
    vocoder = legacy.SpeechT5HiFiGan(Path(args.hifigan), device=device)
    references, source = legacy.official_reference_transcripts(legacy.dataset_for(cfg, 'validation').frame)
    waves = audio.wave(folder)
    speech_times = audio.speech_times.to(device)
    output = Path(args.export_output); (output / 'waveforms').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    ids = np.arange(len(role)) if not args.limit else np.array(sorted(rng.choice(len(role), size=args.limit, replace=False)))
    conditions = [c for c in CONDITIONS if c not in args.skip]
    index = []
    legacy.atomic_json(output / 'export_manifest.json',
                       dict(contract=CONTRACT, checkpoint_sha256=legacy.sha256(Path(args.checkpoint)), role='validation',
                            encoder_sha256=conditioner.encoder_sha256, envelope_sha256=conditioner.envelope_sha256, crossfit=bool(args.crossfit),
                            vocoder_sha256=legacy.tree_hash(Path(args.hifigan)), steps=args.steps, guidance=args.guidance,
                            conditions=conditions, tail_policy='sample_decides', references_source=source,
                            note='diffusion samples share one noise seed per trial across conditions; nothing uses the oracle duration'))
    started = time.monotonic()
    for offset in range(0, len(ids), args.batch_size):
        chunk = ids[offset:offset + args.batch_size]
        eeg, mask, subject, fold = batch_tensors(role, chunk, device)
        wrong_eeg, wrong_mask, _, _ = batch_tensors(role, role.wrong_index[chunk], device)
        generator = torch.Generator(device='cpu')
        noise = torch.stack([torch.randn(MEL_BINS, MEL_FRAMES, generator=generator.manual_seed(int(hashlib.sha256(role.rows[i]['trial_id'].encode()).hexdigest()[:8], 16)))
                             for i in chunk]).to(device)
        null = model.condition('null', torch.zeros(len(chunk), 1, device=device))
        mels = {}
        with torch.no_grad():
            # The deterministic v3 route on the same trials, for the side-by-side.
            v3_subject = torch.tensor([v3_subjects[role.rows[i]['subject']] for i in chunk], device=device)
            state = regression(eeg, torch.zeros(len(chunk), 128, 3, device=device), mask, torch.ones(len(chunk), EEG_SAMPLES, dtype=torch.bool, device=device), v3_subject)
            cut = (state.duration_fraction * MEL_FRAMES).round().long().clamp(1, MEL_FRAMES)
            frames = torch.arange(MEL_FRAMES, device=device)[None, None]
            mels['regression'] = torch.where(frames < cut[:, None, None], state.native_mel, torch.full_like(state.native_mel, SILENCE_MEL))
            for name in conditions:
                if name == 'correct':
                    c = conditioner(eeg, mask, subject, fold)
                elif name == 'zero':
                    c = conditioner(torch.zeros_like(eeg), mask, subject, fold)
                elif name == 'wrong_trial':
                    c = conditioner(wrong_eeg, wrong_mask, subject, fold)
                elif name == 'time_block_shuffle':
                    c = conditioner(counterfactual_eeg(eeg, 'time_block_shuffle', channel_mask=mask), mask, subject, fold)
                elif name == 'pooled':
                    c = conditioner(eeg, mask, subject, fold, premixed=pooled_eeg(conditioner, role, chunk, device))
                elif name == 'pooled_wrong':
                    # Control for the pooled condition: every presentation of a DIFFERENT (duration-matched)
                    # sentence, averaged the same way.  If pooling only cleaned up the conditioning, this scores as high.
                    c = conditioner(eeg, mask, subject, fold, premixed=pooled_eeg(conditioner, role, role.wrong_index[chunk], device))
                elif name == 'teacher_oracle':
                    teacher = torch.from_numpy(np.asarray(audio.teacher[role.audio_index[chunk]], dtype=np.float32)).to(device)
                    c = conditioner.teacher(teacher, speech_times)
                stream = model.condition('teacher' if name == 'teacher_oracle' else 'eeg', c)
                mels[name] = silence_tail(scaler.decode(model.sample(stream, null=null, steps=args.steps, guidance=args.guidance, noise=noise)))
            mels['native_mel_oracle'] = audio.mel[role.audio_index[chunk]].to(device)
            rendered = {name: vocoder.synthesize(mel).cpu().numpy() for name, mel in mels.items()}
        for j, i in enumerate(chunk):
            row = role.rows[i]
            key = hashlib.sha256(row['trial_id'].encode()).hexdigest()[:20]
            trial_folder = output / 'waveforms' / key; trial_folder.mkdir(exist_ok=True)
            sf.write(trial_folder / 'original.wav', np.asarray(waves[row['audio_index']], dtype=np.float32), 16000, subtype='FLOAT')
            for name, wave in rendered.items():
                w = wave[j][:64000]; w = np.pad(w, (0, max(0, 64000 - len(w))))
                sf.write(trial_folder / f'{name}.wav', w, 16000, subtype='FLOAT')
            np.savez_compressed(trial_folder / 'mels.npz', **{name: mel[j].cpu().numpy().astype(np.float16) for name, mel in mels.items()})
            index.append(dict(trial_id=row['trial_id'], folder=key, subject=row['subject'], content=row['content'],
                              wrong_trial_id=role.rows[int(role.wrong_index[i])]['trial_id'],
                              oracle_duration_frames=row['duration_frames'], **references.get(row['trial_id'], {})))
        print(json.dumps(dict(exported=offset + len(chunk), total=len(ids), seconds=round(time.monotonic() - started, 1))), flush=True)
    legacy.atomic_json(output / 'index.json', dict(samples=index))


# ----------------------------------------------------------------------------- compare
def sharpness(mel: np.ndarray, frames: int) -> dict:
    """How speech-like is a spectrogram, independent of which sentence it is?

    spectral_contrast: mean over speech frames of (90th - 10th percentile) across bins;
    temporal_modulation: std over speech frames of the frame mean (envelope depth);
    frame_flux: mean absolute frame-to-frame change.  Real speech is high on all
    three; a conditional-mean blur is low on all three.
    """
    speech = mel[:, :frames].astype(np.float64)
    contrast = float(np.mean(np.percentile(speech, 90, axis=0) - np.percentile(speech, 10, axis=0)))
    modulation = float(np.std(speech.mean(0)))
    flux = float(np.mean(np.abs(np.diff(speech, axis=1))))
    return dict(spectral_contrast=contrast, temporal_modulation=modulation, frame_flux=flux)


def compare(args, cfg):
    import audio_comparison as ac
    import pandas as pd
    export_folder = Path(args.export_output)
    index = json.loads((export_folder / 'index.json').read_text())['samples']
    folders = {e['folder']: export_folder / 'waveforms' / e['folder'] for e in index}
    contents = {e['folder']: e['content'] for e in index}
    names = ['native_mel_oracle', 'teacher_oracle', 'regression', 'correct', 'zero', 'wrong_trial', 'time_block_shuffle', 'pooled', 'pooled_wrong']
    rng = np.random.default_rng(args.seed)
    rows = {n: [] for n in names}; sharp = {n: [] for n in names + ['original']}
    for k, entry in enumerate(index, start=1):
        scores = ac.score_trial(folders[entry['folder']], ac.speech_length(entry), names)
        for n, v in scores.items():
            rows[n].append(v)
        mels = np.load(folders[entry['folder']] / 'mels.npz')
        frames = int(entry['oracle_duration_frames'])
        for n in names:
            if n in mels:
                sharp[n].append(sharpness(mels[n].astype(np.float32), frames))
        sharp['original'].append(sharpness(mels['native_mel_oracle'].astype(np.float32), frames))
        if k % 50 == 0:
            print(json.dumps(dict(scored=k, total=len(index))), flush=True)
    measures = ['stoi', 'pesq', 'mcd', 'envelope_corr', 'modulation_corr']
    summary = {n: {m: float(np.nanmean([r[m] for r in v if m in r])) for m in measures} for n, v in rows.items() if v}
    for n, v in rows.items():
        if v:
            summary[n]['stoi_ci95'] = ac.bootstrap_interval([r['stoi'] for r in v], rng)
            summary[n]['envelope_ci95'] = ac.bootstrap_interval([r['envelope_corr'] for r in v], rng)
    sharp_summary = {n: {k: float(np.mean([r[k] for r in v])) for k in v[0]} for n, v in sharp.items() if v}
    afc_keys = [e['folder'] for e in index[:args.afc_trials]]
    afc = ac.two_alternative({k: folders[k] for k in afc_keys}, [e for e in index if e['folder'] in afc_keys], names, contents, rng)
    # Paired: does real EEG beat its own zero-EEG sample (same noise) on the same trial?
    paired = {}
    for n in ('zero', 'wrong_trial', 'time_block_shuffle', 'pooled_wrong'):
        base = 'pooled' if n == 'pooled_wrong' else 'correct'
        if rows[n] and rows[base]:
            d = np.array([a['stoi'] - b['stoi'] for a, b in zip(rows[base], rows[n])])
            e = np.array([a['envelope_corr'] - b['envelope_corr'] for a, b in zip(rows[base], rows[n])])
            paired[n] = dict(baseline=base, stoi_gain=float(np.nanmean(d)), stoi_gain_ci95=ac.bootstrap_interval(d[np.isfinite(d)], rng),
                             envelope_gain=float(np.nanmean(e)), envelope_gain_ci95=ac.bootstrap_interval(e[np.isfinite(e)], rng),
                             fraction_trials_real_better_stoi=float(np.nanmean(d > 0)))
    result = dict(contract=CONTRACT, export=str(export_folder), trials=len(index), per_condition=summary,
                  speech_likeness=sharp_summary, two_alternative_forced_choice=afc, paired_against_controls=paired)
    (export_folder / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"\n{'condition':20s} {'STOI':>6s} {'PESQ':>6s} {'MCD':>7s} {'env r':>6s} {'mod r':>6s} | contrast  modul   flux | 2AFC stoi  mcd  env")
    for n in ['original'] + names:
        s = summary.get(n, {}); sh = sharp_summary.get(n, {}); a = afc.get(n, {})
        print(f"{n:20s} {s.get('stoi', float('nan')):6.3f} {s.get('pesq', float('nan')):6.3f} {s.get('mcd', float('nan')):7.2f} "
              f"{s.get('envelope_corr', float('nan')):6.3f} {s.get('modulation_corr', float('nan')):6.3f} | "
              f"{sh.get('spectral_contrast', float('nan')):7.3f} {sh.get('temporal_modulation', float('nan')):6.3f} {sh.get('frame_flux', float('nan')):6.3f} |"
              f" {a.get('stoi', {}).get('accuracy', float('nan')):5.2f} {a.get('mcd', {}).get('accuracy', float('nan')):5.2f} {a.get('envelope_corr', {}).get('accuracy', float('nan')):5.2f}")
    for n, p in paired.items():
        print(f"  paired {p['baseline']}-EEG minus {n:14s}: STOI {p['stoi_gain']:+.4f} {p['stoi_gain_ci95']}  envelope r {p['envelope_gain']:+.4f} {p['envelope_gain_ci95']}  real better in {p['fraction_trials_real_better_stoi']:.0%} of trials")
    figure(export_folder, index, args.figure_trials)


def figure(export_folder: Path, index, count: int):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = ['native_mel_oracle', 'teacher_oracle', 'regression', 'correct', 'zero', 'wrong_trial', 'pooled', 'pooled_wrong']
    labels = {'native_mel_oracle': 'presented sentence', 'teacher_oracle': 'audio→teacher→diffusion (ceiling)',
              'regression': 'v3 regression, real EEG', 'correct': 'diffusion, real EEG', 'zero': 'diffusion, zero EEG',
              'wrong_trial': 'diffusion, wrong-trial EEG', 'pooled': 'diffusion, pooled EEG (all presentations)',
              'pooled_wrong': 'diffusion, pooled EEG of a different sentence'}
    chosen = index[:count]
    fig, axes = plt.subplots(len(rows), len(chosen), figsize=(4.2 * len(chosen), 1.6 * len(rows)), squeeze=False)
    for j, entry in enumerate(chosen):
        mels = np.load(export_folder / 'waveforms' / entry['folder'] / 'mels.npz')
        frames = int(entry['oracle_duration_frames'])
        for i, name in enumerate(rows):
            ax = axes[i, j]
            if name not in mels:
                ax.set_visible(False); continue
            if name in mels:
                ax.imshow(mels[name].astype(np.float32)[:, :frames + 20], origin='lower', aspect='auto', vmin=-7, vmax=1, cmap='magma')
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(labels[name], fontsize=7, rotation=0, ha='right', va='center')
        axes[0, j].set_title(entry.get('reference_transcript', entry['trial_id'])[:48], fontsize=8)
    fig.suptitle('DS004940 validation: native mel per condition (diffusion decoder vs v3 regression)', fontsize=10)
    fig.tight_layout()
    fig.savefig(export_folder / 'comparison.png', dpi=130)
    print(f'figure written to {export_folder / "comparison.png"}')


# ----------------------------------------------------------------------------- feature figures
FIGURE_ORDER = [('original', 'original (presented)'), ('native_mel_oracle', 'presented mel → vocoder'),
                ('teacher_oracle', 'audio → teacher → diffusion (ceiling)'), ('regression', 'v3 regression, real EEG'),
                ('correct', 'diffusion, real EEG'), ('zero', 'diffusion, zero EEG'), ('wrong_trial', 'diffusion, wrong-trial EEG'),
                ('time_block_shuffle', 'diffusion, time-block-shuffled EEG'), ('pooled', 'diffusion, pooled EEG (all presentations)'),
                ('pooled_wrong', 'diffusion, pooled EEG of a different sentence')]
FIGURE_COLOURS = {'original': 'black', 'native_mel_oracle': 'dimgray', 'teacher_oracle': 'tab:purple', 'regression': 'tab:orange',
                  'correct': 'tab:red', 'zero': 'tab:blue', 'wrong_trial': 'tab:gray', 'time_block_shuffle': 'tab:brown',
                  'pooled': 'tab:green', 'pooled_wrong': 'tab:olive'}
LISTENING_NAMES = {'0_original': 'original', '1_ceiling_audio_teacher_to_diffusion': 'teacher_oracle', '2_v3_regression_blob': 'regression',
                   '3_diffusion_real_EEG': 'correct', '4_diffusion_zero_EEG': 'zero', '5_diffusion_wrong_trial_EEG': 'wrong_trial',
                   '6_diffusion_pooled_EEG_all_presentations': 'pooled', '7_diffusion_pooled_EEG_of_a_DIFFERENT_sentence': 'pooled_wrong'}


def yin_f0(wave: np.ndarray, rate: int = 16000, frame: int = 1024, hop: int = 160, fmin: float = 60., fmax: float = 400.,
           threshold: float = .15, silence_db: float = -40.) -> tuple[np.ndarray, np.ndarray]:
    """Frame-wise fundamental frequency by YIN (de Cheveigné & Kawahara 2002); NaN where unvoiced.

    Difference function over a window of ``frame - tau_max`` samples, cumulative mean normalised, first dip
    below ``threshold`` with parabolic interpolation.  Frames quieter than ``silence_db`` relative to the
    loudest frame are unvoiced.  Good enough for contour comparison; not a reference pitch tracker.
    """
    wave = np.asarray(wave, dtype=np.float64)
    tau_min, tau_max = int(rate / fmax), int(rate / fmin)
    window = frame - tau_max
    count = max(0, 1 + (len(wave) - frame) // hop)
    f0 = np.full(count, np.nan); times = (np.arange(count) * hop + frame / 2) / rate
    rms = np.array([np.sqrt((wave[i * hop:i * hop + frame] ** 2).mean() + 1e-12) for i in range(count)])
    floor = rms.max() * 10 ** (silence_db / 20) if count else 0.
    for i in range(count):
        if rms[i] <= floor:
            continue
        x = wave[i * hop:i * hop + frame]
        head = x[:window]
        energy_head = float(head @ head)
        cumulative = np.concatenate([[0.], np.cumsum(x ** 2)])
        acf = np.correlate(x, head, mode='valid')                       # tau = 0 .. tau_max
        shifted = np.array([cumulative[t + window] - cumulative[t] for t in range(tau_max + 1)])
        d = energy_head + shifted - 2 * acf
        d[0] = 1.
        cmnd = np.ones_like(d)
        running = np.cumsum(d[1:])
        cmnd[1:] = d[1:] * np.arange(1, tau_max + 1) / np.maximum(running, 1e-12)
        candidates = np.flatnonzero(cmnd[tau_min:tau_max] < threshold)
        if len(candidates):
            tau = tau_min + candidates[0]
            while tau + 1 < tau_max and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
        else:
            tau = tau_min + int(np.argmin(cmnd[tau_min:tau_max]))
            if cmnd[tau] > .35:
                continue
        if 0 < tau < tau_max:
            a, b, c = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
            denominator = a - 2 * b + c
            tau = tau + (.5 * (a - c) / denominator if abs(denominator) > 1e-12 else 0.)
        f0[i] = rate / tau
    return times, f0


def mfcc_frames(wave: np.ndarray, coefficients: int = 13):
    import audio_comparison as ac
    from scipy.fft import dct
    spectrum = ac.log_mel(wave, bands=40)
    return dct(spectrum, type=2, axis=0, norm='ortho')[:coefficients]


def feature_figures(wavs: dict, output_stacked: Path, output_overlay: Path, title: str, duration_s: float,
                    reference: str = 'original', rate: int = 16000) -> None:
    """Stacked per-condition panels (mel + F0, MFCC, energy) and an overlay figure (envelope, F0, MFCC distance)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import soundfile as sf
    import audio_comparison as ac
    span = min(4., duration_s + .35)
    samples = int(span * rate)
    order = [name for name, _ in FIGURE_ORDER if name in wavs]
    labels = dict(FIGURE_ORDER)
    data = {}
    for name in order:
        wave = sf.read(wavs[name], dtype='float32')[0][:samples]
        wave = np.pad(wave, (0, max(0, samples - len(wave))))
        mel = ac.log_mel(wave, bands=64)
        env = ac.envelope(wave)
        times, f0 = yin_f0(wave, rate)
        data[name] = dict(wave=wave, mel=mel, mfcc=mfcc_frames(wave), env=env, f0_times=times, f0=f0)
    ref = data.get(reference)
    speech = int(duration_s * rate)
    metrics = {}
    for name in order:
        if ref is None or name == reference:
            continue
        a, b = ref['wave'][:speech], data[name]['wave'][:speech]
        metrics[name] = dict(stoi=ac.stoi(a, b), env=ac.correlation(ac.envelope(a), ac.envelope(b)))
    hop_s = 160 / rate
    # ---- stacked
    rows = len(order)
    fig, axes = plt.subplots(rows, 3, figsize=(16, 1.75 * rows + .8), squeeze=False, gridspec_kw=dict(width_ratios=[3, 2, 2]))
    vmin = min(float(np.percentile(d['mel'], 2)) for d in data.values()); vmax = max(float(np.percentile(d['mel'], 99.5)) for d in data.values())
    mfcc_scale = max(float(np.abs(d['mfcc'][1:]).max()) for d in data.values()) or 1.
    for r, name in enumerate(order):
        d = data[name]; colour = FIGURE_COLOURS.get(name, 'k')
        ax = axes[r, 0]
        ax.imshow(d['mel'], origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=vmax,
                  extent=[0, d['mel'].shape[1] * hop_s, 0, 64])
        twin = ax.twinx()
        twin.plot(d['f0_times'], d['f0'], '.', color='cyan', markersize=2.5)
        twin.set_ylim(50, 420); twin.set_ylabel('F0 (Hz)', fontsize=7, color='cyan'); twin.tick_params(labelsize=6, colors='cyan')
        ax.axvline(duration_s, color='white', linestyle=':', linewidth=.8)
        text = labels.get(name, name)
        if name in metrics:
            text += f"   STOI {metrics[name]['stoi']:.2f}  env r {metrics[name]['env']:.2f}"
        ax.set_title(text, fontsize=8, loc='left', color=colour)
        ax.set_ylabel('mel band', fontsize=7); ax.tick_params(labelsize=6); ax.set_xlim(0, span)
        ax = axes[r, 1]
        ax.imshow(d['mfcc'][1:], origin='lower', aspect='auto', cmap='coolwarm', vmin=-mfcc_scale, vmax=mfcc_scale,
                  extent=[0, d['mfcc'].shape[1] * hop_s, 1, d['mfcc'].shape[0]])
        ax.axvline(duration_s, color='k', linestyle=':', linewidth=.8)
        ax.set_ylabel('MFCC c1–c12', fontsize=7); ax.tick_params(labelsize=6); ax.set_xlim(0, span)
        if r == 0:
            ax.set_title('MFCC (c0 removed)', fontsize=8)
        ax = axes[r, 2]
        t_env = np.arange(len(d['env'])) * hop_s
        if ref is not None and name != reference:
            ax.plot(np.arange(len(ref['env'])) * hop_s, ref['env'] / (ref['env'].max() + 1e-9), color='black', linewidth=.8, alpha=.5, label='original')
        ax.plot(t_env, d['env'] / (d['env'].max() + 1e-9), color=colour, linewidth=1.2, label=labels.get(name, name)[:24])
        ax.axvline(duration_s, color='k', linestyle=':', linewidth=.8)
        ax.set_ylim(0, 1.05); ax.set_xlim(0, span); ax.tick_params(labelsize=6); ax.set_ylabel('RMS energy', fontsize=7)
        if r == 0:
            ax.set_title('energy envelope (normalised; black = original)', fontsize=8)
        if r == rows - 1:
            for c in range(3):
                axes[r, c].set_xlabel('seconds', fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, .98])
    fig.savefig(output_stacked, dpi=95); plt.close(fig)
    # ---- overlay
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for name in order:
        d = data[name]; colour = FIGURE_COLOURS.get(name, 'k'); lw = 2.2 if name == reference else 1.2
        style = '-' if name in (reference, 'correct', 'pooled', 'teacher_oracle') else '--'
        label = labels.get(name, name)
        if name in metrics:
            label += f" (STOI {metrics[name]['stoi']:.2f}, env r {metrics[name]['env']:.2f})"
        axes[0].plot(np.arange(len(d['env'])) * hop_s, d['env'] / (d['env'].max() + 1e-9), style, color=colour, linewidth=lw, label=label)
        axes[1].plot(d['f0_times'], d['f0'], '.', color=colour, markersize=3 if name == reference else 2, label=label, alpha=.9 if name == reference else .7)
        if ref is not None and name != reference:
            length = min(ref['mfcc'].shape[1], d['mfcc'].shape[1])
            distance = np.sqrt(((ref['mfcc'][1:, :length] - d['mfcc'][1:, :length]) ** 2).sum(0))
            kernel = np.ones(5) / 5
            axes[2].plot(np.arange(length) * hop_s, np.convolve(distance, kernel, mode='same'), style, color=colour, linewidth=lw, label=label)
    for ax in axes:
        ax.axvline(duration_s, color='k', linestyle=':', linewidth=.8); ax.tick_params(labelsize=8); ax.set_xlim(0, span)
    axes[0].set_ylabel('RMS energy (normalised)'); axes[0].set_title('energy envelopes, all conditions overlaid', fontsize=9, loc='left')
    voiced = np.concatenate([d['f0'][~np.isnan(d['f0'])] for d in data.values()] or [np.array([150., 250.])])
    low, high = (np.percentile(voiced, 2), np.percentile(voiced, 98)) if len(voiced) else (100., 300.)
    axes[1].set_ylabel('F0 (Hz)'); axes[1].set_ylim(max(50., low - 30), min(420., high + 30))
    axes[1].set_title('pitch contours (YIN; unvoiced frames omitted; y-range fitted to the voiced frames)', fontsize=9, loc='left')
    axes[2].set_ylabel('MFCC distance to original'); axes[2].set_title('per-frame MFCC (c1–c12) Euclidean distance to the presented sentence, 50 ms smoothed', fontsize=9, loc='left')
    axes[2].set_xlabel('seconds (dotted line = end of the presented sentence)')
    handles, names_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, names_, fontsize=7.5, loc='lower center', ncol=3, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, .07, 1, .97])
    fig.savefig(output_overlay, dpi=100); plt.close(fig)


def figures(args, cfg):
    """Per-trial feature figures for an export folder (``--export-output``) or the curated listening folder (``--listening``)."""
    made = 0
    if args.listening:
        root = Path(args.listening)
        for folder in sorted(p for p in root.glob('*/*') if p.is_dir()):
            wavs = {LISTENING_NAMES[p.stem]: p for p in sorted(folder.glob('*.wav')) if p.stem in LISTENING_NAMES}
            if 'original' not in wavs:
                continue
            import soundfile as sf
            original = sf.read(wavs['original'], dtype='float32')[0]
            # presented-speech length: last sample above -50 dB of the presented stimulus
            loud = np.flatnonzero(np.abs(original) > np.abs(original).max() * 10 ** (-50 / 20))
            duration = (loud[-1] + 1) / 16000 if len(loud) else len(original) / 16000
            title = folder.name.split('_', 1)[1].replace('_', ' ') + f'   [{folder.parent.name}]'
            feature_figures(wavs, folder / 'features_stacked.png', folder / 'features_overlay.png', title, duration)
            made += 1
            print(json.dumps(dict(figures=str(folder))), flush=True)
    if args.export_output:
        export_folder = Path(args.export_output)
        index = json.loads((export_folder / 'index.json').read_text())['samples']
        chosen = index if not args.limit else index[:args.limit]
        for entry in chosen:
            folder = export_folder / 'waveforms' / entry['folder']
            wavs = {name: folder / f'{name}.wav' for name, _ in FIGURE_ORDER if (folder / f'{name}.wav').exists()}
            duration = int(entry['oracle_duration_frames']) * 256 / 16000
            title = f"{entry.get('reference_transcript', entry['trial_id'])}   [{entry['subject']}, {entry['trial_id']}]"
            feature_figures(wavs, folder / 'features_stacked.png', folder / 'features_overlay.png', title, duration)
            made += 1
            if made % 10 == 0:
                print(json.dumps(dict(figures=made, total=len(chosen))), flush=True)
    print(json.dumps(dict(figures_written=made)))


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['cache', 'crossfit', 'train', 'export', 'compare', 'figures'])
    parser.add_argument('--config', default=str(ROOT / 'configs/aligned_speech_local_v1.yaml'))
    parser.add_argument('--encoder', default=str(DEFAULT_ENCODER))
    parser.add_argument('--envelope', default=str(DEFAULT_ENVELOPE), help='"" disables the envelope channel')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--run', default='base')
    parser.add_argument('--device', default='auto'); parser.add_argument('--seed', type=int, default=31)
    # train
    parser.add_argument('--hidden', type=int, default=256); parser.add_argument('--blocks', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=.1); parser.add_argument('--ema', type=float, default=.999)
    parser.add_argument('--lr', type=float, default=2e-4); parser.add_argument('--weight-decay', type=float, default=.01)
    parser.add_argument('--warmup', type=int, default=500); parser.add_argument('--updates', type=int, default=12000)
    parser.add_argument('--batch-size', type=int, default=32); parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--p-teacher', type=float, default=.3); parser.add_argument('--p-null', type=float, default=.15)
    parser.add_argument('--p-pool', type=float, default=.4, help='probability an EEG-conditioned example uses averaged same-sentence EEG')
    parser.add_argument('--pool-max', type=int, default=8, help='largest number of same-sentence trials averaged during training')
    parser.add_argument('--conditioning', choices=['compact', 'trunk'], default='compact',
                        help='compact: PCA of the encoder head output + duration + envelope; trunk: raw 128-d trunk features')
    parser.add_argument('--components', type=int, default=16, help='PCA components of the head output (compact mode)')
    parser.add_argument('--condition-noise', type=float, default=.3, help='training-time Gaussian noise on standardised conditioning')
    parser.add_argument('--condition-dropout', type=float, default=.1, help='training-time channel dropout on the conditioning')
    parser.add_argument('--min-updates', type=int, default=1500, help='earliest update eligible for best_advantage.pt')
    parser.add_argument('--crossfit', action='store_true', help='condition on cross-fitted (held-out) encoder features; needs the crossfit stage')
    parser.add_argument('--crossfit-seed', type=int, default=322)
    parser.add_argument('--crossfit-updates', type=int, default=2000, help='encoder updates per fold')
    parser.add_argument('--crossfit-envelope-updates', type=int, default=750)
    parser.add_argument('--broderick-trunk', default=str(ROOT / 'outputs/broderick2018/trunk/best.pt'))
    parser.add_argument('--no-augment', dest='augment', action='store_false')
    parser.add_argument('--quick-steps', type=int, default=25); parser.add_argument('--quick-guidance', type=float, default=2.)
    # export / compare
    parser.add_argument('--checkpoint'); parser.add_argument('--export-output')
    parser.add_argument('--hifigan', default=str(ROOT / 'outputs/aligned_speech_local_v1/hifigan/best'))
    parser.add_argument('--steps', type=int, default=50); parser.add_argument('--guidance', type=float, default=2.)
    parser.add_argument('--limit', type=int, default=0); parser.add_argument('--skip', nargs='*', default=[])
    parser.add_argument('--afc-trials', type=int, default=120); parser.add_argument('--figure-trials', type=int, default=4)
    parser.add_argument('--listening', help='figures: curated listening folder (outputs/generative_recovery/listening)')
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = legacy.config(args.config)
    if args.stage == 'cache':
        cache(args, cfg)
    elif args.stage == 'crossfit':
        crossfit(args, cfg)
    elif args.stage == 'train':
        train(args, cfg)
    elif args.stage == 'export':
        if not args.checkpoint or not args.export_output:
            parser.error('export needs --checkpoint and --export-output')
        export(args, cfg)
    elif args.stage == 'figures':
        if not args.export_output and not args.listening:
            parser.error('figures needs --export-output and/or --listening')
        figures(args, cfg)
    else:
        if not args.export_output:
            parser.error('compare needs --export-output')
        compare(args, cfg)


if __name__ == '__main__':
    main()

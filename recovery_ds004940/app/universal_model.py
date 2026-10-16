"""Subject-free, montage-agnostic EEG encoder for speech and music reconstruction.

Kept outside ``eeg2speech/`` (like ``aligned_recovery_model.py``) so that
earlier checkpoints' runtime fingerprints are unchanged.

What differs from recovery v3 (``RecoveryEEGModel``):

* **No subject index anywhere.**  v3 learned one 128 x 128 mixing matrix per
  participant, so it could only serve participants seen in training.  This
  model receives EEG, electrode positions and a validity mask, nothing else;
  the same weights must serve participants it has never seen (the protocol
  a generative decoder needs).  Per-trial RMS normalisation replaces
  participant-specific scaling.
* **Any electrode layout.**  A spatial-attention layer (Défossez et al.
  2023) maps the recorded electrodes onto a fixed set of virtual channels,
  with weights that are a learned function of each electrode's scalp
  position; missing electrodes are masked out of the softmax.  128-channel
  BioSemi (DS004940), 64-channel BioSemi (Di Liberto 2020) and EGI caps all
  enter the same trunk, and the output is invariant to channel order.
* **Several reconstruction chains on one trunk.**  The temporal trunk
  (strided conv + dilated residual blocks, identical to v3 and to the
  Broderick-pretrained trunk) feeds per-domain heads.  Each head predicts a
  teacher sequence that a frozen per-domain AcousticDecoder turns into a
  mel spectrogram, exactly the v3 alignment chain: speech = HuBERT layer 9
  -> 80-band 16 kHz mel (SpeechT5 HiFi-GAN); music = MERT (or EnCodec)
  -> 100-band 24 kHz mel (BigVGAN-v2).  A shared low-level head predicts the
  broadband log-energy envelope and an onset-strength curve, the auditory
  features EEG tracks in both domains.
* The v3 time-since-onset code and duration head only apply to
  onset-locked trials (DS004940 sentences); music windows are cut from
  continuous listening and carry no onset.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from numpy.polynomial import legendre
import torch
from torch import nn
import torch.nn.functional as F

from aligned_recovery_model import RecoveryState, TemporalResidual
from eeg2speech.aligned import AcousticDecoder, EEG_RATE, EEG_SAMPLES, EEG_START, LAGS_MS, sample_at

ACOUSTIC_RATE = 64                     # Hz, low-level target frame rate (one EEG token per frame)
WINDOW_S = 4.                          # audio window that every teacher / mel / acoustic grid covers
ACOUSTIC_FRAMES = int(WINDOW_S * ACOUSTIC_RATE)
THETA_MAX = 2.2                        # rad from the vertex covered by the flattened position code
TRUNK_KEYS = ('temporal', 'blocks', 'output_norm')


@dataclass
class UniversalState(RecoveryState):
    acoustic: torch.Tensor | None = None          # (B, 2, ACOUSTIC_FRAMES): envelope, onset strength


class ConfigurableAcousticDecoder(AcousticDecoder):
    """The v3 teacher -> mel decoder with a configurable number of mel bands (BigVGAN-v2 uses 100)."""
    def __init__(self, speech_times, mel_times, speech_dimension=768, hidden=256, layers=6, mel_bins=80):
        super().__init__(speech_times, mel_times, speech_dimension=speech_dimension, hidden=hidden, layers=layers)
        self.mel_bins = int(mel_bins)
        if self.mel_bins != 80:
            self.output = nn.Conv1d(hidden, self.mel_bins, 1)


def decoder_from_spec(spec, state=None):
    decoder = ConfigurableAcousticDecoder(torch.tensor(spec['speech_times']), torch.tensor(spec['mel_times']),
                                          speech_dimension=spec['speech_dimension'], hidden=spec['hidden'],
                                          layers=spec['layers'], mel_bins=spec.get('mel_bins', 80))
    if state is not None:
        decoder.load_state_dict(state)
    return decoder


def flatten_positions(xyz):
    """Azimuthal-equidistant projection around the vertex into [0, 1]^2.

    A fixed map (not a per-dataset min-max), so one electrode position has
    one code in every dataset.
    """
    unit = xyz / xyz.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    theta = torch.arccos(unit[..., 2].clamp(-1, 1))
    phi = torch.atan2(unit[..., 1], unit[..., 0])
    u = .5 + theta * torch.cos(phi) / (2 * THETA_MAX)
    v = .5 + theta * torch.sin(phi) / (2 * THETA_MAX)
    return torch.stack([u, v], -1).clamp(0, 1)


class SpatialAttention(nn.Module):
    """Virtual channels as position-dependent softmax mixtures of the recorded electrodes."""
    def __init__(self, virtual=128, harmonics=12):
        super().__init__()
        self.harmonics = int(harmonics)
        k, l = torch.meshgrid(torch.arange(harmonics), torch.arange(harmonics), indexing='ij')
        self.register_buffer('frequencies', torch.stack([k.flatten(), l.flatten()], -1).float())
        self.logits = nn.Linear(2 * harmonics * harmonics, virtual)

    def forward(self, eeg, channel_xyz, channel_mask):
        phase = 2 * math.pi * flatten_positions(channel_xyz) @ self.frequencies.T          # B, C, H*H
        logits = self.logits(torch.cat([torch.cos(phase), torch.sin(phase)], -1))           # B, C, V
        logits = logits.masked_fill(~channel_mask[:, :, None], float('-inf'))
        weights = torch.softmax(logits, dim=1)
        return torch.einsum('bcv,bct->bvt', weights, eeg)


class UniversalEEGModel(nn.Module):
    """EEG (any montage, any participant) -> per-domain teacher sequence -> frozen decoder -> mel."""
    def __init__(self, decoders: dict, *, virtual=128, width=128, dropout=.1, positional=True, lag_ms=0,
                 harmonics=12, input_norm='trial_rms'):
        super().__init__()
        if lag_ms not in LAGS_MS:
            raise ValueError('unregistered lag')
        if input_norm not in ('trial_rms', 'none'):
            raise ValueError('input_norm must be trial_rms or none')
        if 'speech' not in decoders:
            raise ValueError('a speech decoder is required (DS004940 evaluation uses it)')
        self.lag_ms, self.input_norm = lag_ms, input_norm
        self.decoders = nn.ModuleDict(decoders).requires_grad_(False)
        self.spatial_attention = SpatialAttention(virtual, harmonics)
        self.spatial = nn.Conv1d(virtual, width, 1, bias=False)
        self.temporal = nn.Conv1d(width, width, 15, stride=4, padding=7)
        self.blocks = nn.Sequential(*[TemporalResidual(width, 2 ** i, dropout) for i in range(6)])
        self.output_norm = nn.LayerNorm(width)
        self.heads = nn.ModuleDict({name: nn.Conv1d(width, decoder.normalizer.mean.numel(), 1)
                                    for name, decoder in decoders.items()})
        for head in self.heads.values():
            nn.init.normal_(head.weight, std=.01); nn.init.zeros_(head.bias)
        self.acoustic = nn.Conv1d(width, 2, 1)
        self.duration = nn.Linear(width, 1)
        nn.init.zeros_(self.duration.weight); nn.init.constant_(self.duration.bias, .5)
        count = (EEG_SAMPLES + 2 * 7 - 15) // 4 + 1
        self.register_buffer('token_times', EEG_START + torch.arange(count).float() * 4 / EEG_RATE)
        self.register_buffer('acoustic_times', torch.arange(ACOUSTIC_FRAMES).float() / ACOUSTIC_RATE)
        self.positional = bool(positional)
        if self.positional:
            self.position = nn.Parameter(torch.zeros(1, width, count))

    @property
    def decoder(self):
        """The speech decoder, so ``aligned_recovery_eval.evaluate_recovery`` runs unchanged on DS004940."""
        return self.decoders['speech']

    def train(self, mode=True):
        super().train(mode)
        self.decoders.eval()                          # frozen, always in inference mode
        return self

    def encode(self, eeg, channel_xyz, channel_mask, time_mask, locked):
        if eeg.ndim != 3 or eeg.shape[-1] != EEG_SAMPLES or time_mask.shape != (len(eeg), EEG_SAMPLES):
            raise ValueError('physical 4.6 s EEG window required')
        if channel_mask.shape != eeg.shape[:2] or not bool(time_mask.all()) or not bool(channel_mask.any(1).all()):
            raise ValueError('invalid masks')
        if not bool(torch.isfinite(eeg).all()):
            raise ValueError('nonfinite EEG')
        if channel_xyz.ndim == 2:
            channel_xyz = channel_xyz.expand(len(eeg), -1, -1)
        if channel_xyz.shape != (*eeg.shape[:2], 3):
            raise ValueError('channel_xyz must be (C, 3) or (B, C, 3)')
        x = eeg * channel_mask[:, :, None]
        if self.input_norm == 'trial_rms':
            power = x.square().sum((1, 2)) / (channel_mask.sum(1) * x.shape[-1]).clamp_min(1)
            x = x / (power.sqrt() + 1e-6)[:, None, None]
        x = self.spatial_attention(x, channel_xyz.to(x.dtype), channel_mask)
        x = F.gelu(self.temporal(self.spatial(x)))
        if self.positional:
            x = x + self.position * locked.to(x.dtype)[:, None, None]
        return self.output_norm(self.blocks(x).transpose(1, 2))           # B, T, W

    def forward(self, eeg, channel_xyz, channel_mask, time_mask, subject=None, *, domain='speech', locked=None):
        if subject is not None:
            raise ValueError('subject-free model: participant indices are not an input')
        if domain not in self.heads:
            raise ValueError(f'unknown domain {domain!r}')
        if locked is None:
            locked = torch.full((len(eeg),), domain == 'speech', dtype=torch.bool, device=eeg.device)
        hidden = self.encode(eeg, channel_xyz, channel_mask, time_mask, locked)
        lag = self.lag_ms / 1000.
        decoder = self.decoders[domain]
        duration = torch.sigmoid(self.duration(hidden.mean(1))).squeeze(-1)
        z = self.heads[domain](hidden.transpose(1, 2)).transpose(1, 2)
        z = sample_at(z, self.token_times, decoder.speech_times + lag)
        sequence = z * decoder.normalizer.scale + decoder.normalizer.mean
        acoustic = self.acoustic(hidden.transpose(1, 2)).transpose(1, 2)
        acoustic = sample_at(acoustic, self.token_times, self.acoustic_times + lag).transpose(1, 2)
        return UniversalState(sequence, decoder.speech_times, decoder(sequence), decoder.mel_times,
                              F.normalize(z.mean(1), dim=-1), duration, acoustic)


def load_trunk(model, path):
    """Initialise the temporal trunk from a Broderick trunk, a v3 recovery or a universal checkpoint.

    Only ``temporal``, ``blocks`` and ``output_norm`` transfer: a v3 or
    Broderick ``spatial`` layer mixes physical 128-channel BioSemi
    electrodes, not this model's virtual channels.
    """
    payload = torch.load(path, map_location='cpu', weights_only=False)
    state = payload.get('trunk') or payload.get('model')
    if state is None:
        raise ValueError(f'{path}: neither a trunk nor a model state')
    chosen = {k: v for k, v in state.items() if k.split('.')[0] in TRUNK_KEYS}
    own = model.state_dict()
    mismatched = [k for k, v in chosen.items() if k not in own or own[k].shape != v.shape]
    if not chosen or mismatched:
        raise ValueError(f'{path}: trunk does not match ({mismatched[:3]})')
    model.load_state_dict(chosen, strict=False)
    return sorted(chosen)


# --- targets and losses ------------------------------------------------------------

def acoustic_targets(wave, sample_rate, *, frames=ACOUSTIC_FRAMES, bands=8, fmin=50., fmax=8000.):
    """(B, 2, frames) broadband log-energy envelope and onset strength at 64 Hz, frame k at k / 64 s.

    One definition for every domain and sample rate (energy above 8 kHz is
    ignored, so 16 kHz speech and 24 kHz music are comparable).  Energies are
    divided by their per-clip mean before the log, so the curves do not
    depend on playback level.
    """
    hop = sample_rate // ACOUSTIC_RATE
    if hop * ACOUSTIC_RATE != sample_rate:
        raise ValueError('sample rate must be a multiple of 64 Hz')
    wave = wave.float()
    if wave.ndim == 1:
        wave = wave[None]
    win = 2 * hop
    n_fft = 1 << (win - 1).bit_length()
    window = torch.hann_window(win, device=wave.device)
    power = torch.stft(wave, n_fft, hop, win, window=window, center=True, return_complex=True).abs().square()
    freqs = torch.fft.rfftfreq(n_fft, 1 / sample_rate).to(wave.device)
    top = min(fmax, sample_rate / 2)
    broadband = power[:, (freqs >= fmin) & (freqs <= top)].sum(1)
    envelope = torch.log(broadband / broadband.mean(1, keepdim=True).clamp_min(1e-12) + 1e-3)
    edges = torch.logspace(math.log10(fmin), math.log10(top), bands + 1)
    onset = torch.zeros_like(envelope)
    for low, high in zip(edges[:-1].tolist(), edges[1:].tolist()):
        band = power[:, (freqs >= low) & (freqs < high)].sum(1)
        level = torch.log(band / band.mean(1, keepdim=True).clamp_min(1e-12) + 1e-3)
        onset = onset + F.relu(torch.diff(level, dim=1, prepend=level[:, :1]))
    out = torch.stack([envelope, onset / bands], 1)
    if out.shape[-1] >= frames:
        return out[..., :frames]
    return F.pad(out, (0, frames - out.shape[-1]), mode='replicate')


def correlation_loss(prediction, target, mask=None):
    """Mean (1 - Pearson r) over samples and curves on masked frames; returns (loss, r of shape (B, K))."""
    if mask is None:
        mask = torch.ones(prediction.shape[0], prediction.shape[-1], dtype=torch.bool, device=prediction.device)
    weight = mask[:, None, :].to(prediction.dtype)
    count = weight.sum(-1).clamp_min(1)
    p = prediction - (prediction * weight).sum(-1, keepdim=True) / count[..., None]
    t = target.to(prediction) - (target.to(prediction) * weight).sum(-1, keepdim=True) / count[..., None]
    r = (p * t * weight).sum(-1) / ((p.square() * weight).sum(-1).sqrt() * (t.square() * weight).sum(-1).sqrt() + 1e-6)
    return (1 - r).mean(), r


def masked_mean(value, mask):
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.) / (value.numel() / weight.numel())


def aligned_loss(state, teacher, mel, normalizer, frame_mask, mel_mask, negatives=None, *, temperature=.1,
                 contrastive_weight=1., sequence_weight=1., delta_weight=.2, mel_weight=1.):
    """The v3 recovery objective without the sentence-duration term and with explicit valid negatives.

    ``negatives`` is a (B, B) boolean matrix of pairs allowed as negatives
    in the time-resolved in-batch CLIP; music repeats itself (choruses,
    sequences, the same piece heard by every participant), so windows of the
    same passage must not be pushed apart.
    """
    target = normalizer(teacher).detach()
    z = normalizer(state.aligned_sequence)
    frame = frame_mask[:, :, None]
    sequence = masked_mean((z - target).square(), frame)
    both = (frame_mask[:, 1:] & frame_mask[:, :-1])[:, :, None]
    delta = masked_mean((z.diff(dim=1) - target.diff(dim=1)).abs(), both)
    similarity = torch.einsum('itd,jtd->ijt', F.normalize(z, dim=-1), F.normalize(target, dim=-1))
    pair = (frame_mask[:, None, :] & frame_mask[None, :, :]).to(similarity.dtype)
    logits = (similarity * pair).sum(-1) / pair.sum(-1).clamp_min(1.) / temperature
    labels = torch.arange(len(z), device=z.device)
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    allowed = torch.ones_like(eye) if negatives is None else (negatives.to(z.device) | eye)
    logits = logits.masked_fill(~allowed, float('-inf'))
    contrastive = .5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    mel_l1 = masked_mean((state.native_mel - mel).abs(), mel_mask[:, None, :])
    mel_both = (mel_mask[:, 1:] & mel_mask[:, :-1])[:, None, :]
    mel_delta = masked_mean((state.native_mel.diff(dim=-1) - mel.diff(dim=-1)).abs(), mel_both)
    mel_loss = mel_l1 + .2 * mel_delta
    loss = contrastive_weight * contrastive + sequence_weight * sequence + delta_weight * delta + mel_weight * mel_loss
    with torch.no_grad():
        accuracy = float((logits.argmax(1) == labels).float().mean())
        negatives_per_row = float((allowed.sum(1) - 1).float().mean())
    return loss, {'contrastive': float(contrastive.detach()), 'sequence_mse': float(sequence.detach()),
                  'delta': float(delta.detach()), 'mel': float(mel_loss.detach()), 'loss': float(loss.detach()),
                  'batch_accuracy': accuracy, 'negatives_per_row': negatives_per_row}


def music_negatives(pieces, starts, normalized_teacher, *, window_s=WINDOW_S, similarity=.8):
    """Valid negatives for music windows: not overlapping in the same piece, and not near-identical in content."""
    starts = torch.as_tensor(starts, dtype=torch.float32, device=normalized_teacher.device)
    codes = {p: i for i, p in enumerate(dict.fromkeys(pieces))}
    piece = torch.as_tensor([codes[p] for p in pieces], device=normalized_teacher.device)
    same = (piece[:, None] == piece[None, :]) & ((starts[:, None] - starts[None, :]).abs() < window_s)
    unit = F.normalize(normalized_teacher, dim=-1)
    content = torch.einsum('itd,jtd->ij', unit, unit) / unit.shape[1]
    return ~same & (content < similarity)


def consistency_loss(first, second, normalizer, frame_mask=None):
    """1 - frame-wise cosine between two predictions of the same trial (two disjoint electrode halves)."""
    a = F.normalize(normalizer(first.aligned_sequence), dim=-1); b = F.normalize(normalizer(second.aligned_sequence), dim=-1)
    cosine = (a * b).sum(-1)
    if frame_mask is None:
        return (1 - cosine).mean()
    return masked_mean(1 - cosine, frame_mask)

# =========================================================================================
# Electrode-space (spatial) augmentation
#
# Electrode-space (spatial) augmentation for scalp EEG.
#
# The scalp field is spatially smooth, so 128 electrodes carry few independent
# spatial components: on the DS004940 training fold 95 % of the variance lies
# in a median of 16 components per participant (range 6-32; measured
# 2026-09-23).  Channel subsets and small geometric perturbations therefore
# give new, physically plausible views of the same trial while the audio
# target, which is locked to the stimulus clock, stays exactly valid.
#
# Every transform works on ``(B, C, T)`` EEG in normalised units with a
# ``(B, C)`` validity mask and electrode positions ``(C, 3)`` or ``(B, C, 3)``
# on a head sphere (MNE head frame: x right, y nose, z up; metres).  Invalid
# channels are zero on input and stay zero on output.  All transforms of one
# sample are linear, so they are composed into one ``C x C`` matrix per sample
# and applied with a single batched matrix product.
#
# Nothing here uses a subject identity except :class:`SpatialAugmenter`'s
# optional covariance recolouring, which needs per-group covariances estimated
# from training trials; that is a data-side training augmentation only, the
# model never receives a subject index.
# =========================================================================================
# --- spherical splines (Perrin et al. 1989; same formula and defaults as MNE) ---------

def spline_g(cosang, stiffness=4, terms=50):
    """Legendre series g(x) = sum_n (2n+1) / (n^m (n+1)^m 4 pi) P_n(x)."""
    factors = [(2 * n + 1) / (n ** stiffness * (n + 1) ** stiffness * 4 * math.pi) for n in range(1, terms + 1)]
    return legendre.legval(cosang, [0.] + factors)


def spherical_spline(source, target, *, alpha=1e-5, stiffness=4, terms=50):
    """``(len(target), len(source))`` matrix that interpolates a scalp field sampled at ``source`` onto ``target``.

    Positions are projected onto the unit sphere around the origin (the
    standard montages are centred there).  Equivalent to MNE's
    ``_make_interpolation_matrix`` (regularised, with the constant term).
    """
    source = np.asarray(source, np.float64); target = np.asarray(target, np.float64)
    source = source / np.linalg.norm(source, axis=1, keepdims=True)
    target = target / np.linalg.norm(target, axis=1, keepdims=True)
    count = len(source)
    g_source = spline_g(np.clip(source @ source.T, -1, 1), stiffness, terms)
    g_target = spline_g(np.clip(target @ source.T, -1, 1), stiffness, terms)
    g_source.flat[::count + 1] += alpha
    system = np.zeros((count + 1, count + 1))
    system[:count, :count] = g_source; system[:count, count] = 1.; system[count, :count] = 1.
    rhs = np.hstack([g_target, np.ones((len(target), 1))])
    # C is symmetric, so rhs @ inv(C) = solve(C, rhs.T).T
    return np.linalg.solve(system, rhs.T).T[:, :count]


# --- geometry ------------------------------------------------------------------------

def rotation(yaw, pitch, roll):
    """Rotation about z (vertical), x (left-right axis: front/back tilt) and y (nose axis: sideways tilt), radians."""
    cz, sz, cx, sx, cy, sy = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return rz @ rx @ ry


def perturbed_positions(xyz, rng, *, rotate_deg=8., tilt_deg=5., shift_m=.01, jitter_m=.004):
    """A mis-placed cap: global rotation/tilt, a translation and per-electrode jitter, re-projected onto the sphere."""
    xyz = np.asarray(xyz, np.float64)
    radius = np.linalg.norm(xyz, axis=1, keepdims=True)
    angles = np.radians([rng.uniform(-rotate_deg, rotate_deg), rng.uniform(-tilt_deg, tilt_deg), rng.uniform(-tilt_deg, tilt_deg)])
    moved = xyz @ rotation(*angles).T
    direction = rng.normal(size=3); direction /= np.linalg.norm(direction) + 1e-12
    moved = moved + direction * rng.uniform(0, shift_m) + rng.normal(scale=jitter_m, size=xyz.shape)
    return moved / np.linalg.norm(moved, axis=1, keepdims=True) * radius


def mirrored_positions(xyz):
    """Left-right mirror image (x -> -x)."""
    out = np.array(xyz, np.float64, copy=True); out[:, 0] *= -1
    return out


def chord_distances(xyz):
    xyz = np.asarray(xyz, np.float64)
    return np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)


def smoothing_matrix(xyz, sigma):
    """Row-normalised Gaussian volume-conduction kernel over the given electrodes."""
    weights = np.exp(-chord_distances(xyz) ** 2 / (2 * sigma ** 2))
    return weights / weights.sum(1, keepdims=True)


def laplacian_matrix(xyz, neighbours=4):
    """Local (Hjorth) Laplacian: each electrode minus the mean of its nearest neighbours."""
    count = len(xyz)
    neighbours = min(neighbours, count - 1)
    out = np.eye(count)
    if neighbours < 1:
        return out
    order = np.argsort(chord_distances(xyz), axis=1)[:, 1:neighbours + 1]
    for i, near in enumerate(order):
        out[i, near] -= 1. / neighbours
    return out


def matrix_power_psd(eigenvalues, eigenvectors, power):
    return (eigenvectors * eigenvalues[None, :] ** power) @ eigenvectors.T


# --- covariance recolouring ----------------------------------------------------------

def estimate_covariance(trials, masks, *, shrinkage=.1):
    """Pairwise-valid spatial covariance of zero-mean trials ``(N, C, T)`` with masks ``(N, C)``, shrunk towards scaled identity."""
    trials = np.asarray(trials, np.float64); masks = np.asarray(masks, bool)
    channels = trials.shape[1]
    total = np.zeros((channels, channels)); counts = np.zeros((channels, channels))
    for x, m in zip(trials, masks):
        x = (x - x.mean(1, keepdims=True)) * m[:, None]
        total += x @ x.T
        counts += np.outer(m, m) * x.shape[1]
    covariance = np.where(counts > 0, total / np.maximum(counts, 1), 0.)
    observed = np.diag(counts) > 0
    scale = float(np.mean(np.diag(covariance)[observed])) if observed.any() else 1.
    covariance[~observed, :] = 0.; covariance[:, ~observed] = 0.
    covariance[np.diag_indices(channels)] = np.where(observed, np.diag(covariance), scale)
    covariance = (1 - shrinkage) * covariance + shrinkage * scale * np.eye(channels)
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    return (vectors * np.clip(values, scale * 1e-6, None)) @ vectors.T


def recolour_matrix(source_cov, target_cov, strength):
    """``C_target^(s/2) C_source^(-s/2)``: whitens with the source participant's spatial statistics, colours with the target's."""
    if strength <= 0:
        return np.eye(len(source_cov))
    s_values, s_vectors = np.linalg.eigh(source_cov); t_values, t_vectors = np.linalg.eigh(target_cov)
    floor_s = s_values.max() * 1e-6; floor_t = t_values.max() * 1e-6
    return (matrix_power_psd(np.clip(t_values, floor_t, None), t_vectors, strength / 2)
            @ matrix_power_psd(np.clip(s_values, floor_s, None), s_vectors, -strength / 2))


# --- the augmenter -------------------------------------------------------------------

@dataclass
class SpatialConfig:
    """Probabilities (per sample) and ranges of each spatial transform; all zero = identity."""
    geometry: float = 0.            # cap rotation / tilt / shift / electrode jitter
    rotate_deg: float = 8.
    tilt_deg: float = 5.
    shift_m: float = .01
    jitter_m: float = .004
    mirror: float = 0.              # left-right mirror (hemispheric asymmetry makes this risky; ablation only)
    subset: float = 0.              # montage dropout
    keep_low: float = .25
    keep_high: float = 1.
    regional: float = .5            # share of subset draws that remove one contiguous scalp region instead
    region_m: float = .05
    subset_mode: str = 'mask'       # 'mask': drop channels (montage-agnostic model); 'interpolate': re-estimate them (fixed-montage model)
    reference: float = 0.           # random re-reference: average / single electrode / local Laplacian
    conduction: float = 0.          # volume-conduction change: Gaussian smoothing or sharpening
    smooth_low_m: float = .005
    smooth_high_m: float = .02
    sharpen: float = .5
    recolour: float = 0.            # cross-participant spatial covariance recolouring
    recolour_low: float = .25
    recolour_high: float = 1.
    preserve_rms: bool = True
    # Spline regularisation.  MNE's 1e-5 (right for bad-channel repair) smooths strongly: re-interpolating
    # 64 BioSemi electrodes onto themselves keeps only 0.39 of each channel (0.23 at 128).  At 1e-9 the
    # self-map is the identity to 0.98-1.0 with unit white-noise gain, so a cap perturbation moves the field
    # without also blurring it (blurring is the separate ``conduction`` transform).
    spline_alpha: float = 1e-9

    def enabled(self):
        return any(p > 0 for p in (self.geometry, self.mirror, self.subset, self.reference, self.conduction, self.recolour))

    def validate(self):
        for name in ('geometry', 'mirror', 'subset', 'regional', 'reference', 'conduction', 'recolour'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'spatial {name} probability outside [0, 1]')
        if not 0 < self.keep_low <= self.keep_high <= 1:
            raise ValueError('spatial keep fractions must satisfy 0 < low <= high <= 1')
        if self.subset_mode not in ('mask', 'interpolate'):
            raise ValueError('subset_mode must be mask or interpolate')
        if not 0 < self.smooth_low_m <= self.smooth_high_m or self.sharpen < 0:
            raise ValueError('invalid conduction ranges')
        if not 0 <= self.recolour_low <= self.recolour_high <= 1:
            raise ValueError('recolour strengths must lie in [0, 1]')
        if not 0 < self.spline_alpha < 1e-3:
            raise ValueError('spline_alpha must lie in (0, 1e-3)')
        return self

    def as_dict(self):
        return asdict(self)


class SpatialAugmenter:
    """Per-sample composition of the spatial transforms of :class:`SpatialConfig`.

    ``covariances`` maps a group key (``'<dataset>:<participant>'``) to its
    ``(C, C)`` spatial covariance; recolouring draws the target group among
    other groups of the same dataset (same montage).  Output channels that
    were invalid stay zero and masked; a channel used as the reference, or
    dropped in ``'mask'`` mode, becomes invalid.
    """
    def __init__(self, config: SpatialConfig, covariances: dict | None = None):
        self.config = config.validate()
        self.covariances = dict(covariances or {})

    def _recolour_target(self, group, rng):
        if group not in self.covariances:
            return None
        dataset = group.split(':')[0]
        others = sorted(g for g in self.covariances if g != group and g.split(':')[0] == dataset)
        return others[rng.integers(len(others))] if others else None

    def sample_matrix(self, xyz, mask, group, rng):
        """``(C, C)`` matrix and the output mask for one sample (numpy, float64)."""
        cfg = self.config
        channels = len(mask)
        valid = np.flatnonzero(mask)
        pos = np.asarray(xyz, np.float64)[valid]
        n = len(valid)
        operator = np.eye(n)
        out_mask = mask.copy()
        changed = False
        if cfg.recolour and rng.random() < cfg.recolour:
            target = self._recolour_target(group, rng)
            if target is not None:
                source_cov = self.covariances[group][np.ix_(valid, valid)]
                target_cov = self.covariances[target][np.ix_(valid, valid)]
                operator = recolour_matrix(source_cov, target_cov, rng.uniform(cfg.recolour_low, cfg.recolour_high)) @ operator
                changed = True
        if cfg.geometry and rng.random() < cfg.geometry:
            moved = perturbed_positions(pos, rng, rotate_deg=cfg.rotate_deg, tilt_deg=cfg.tilt_deg,
                                        shift_m=cfg.shift_m, jitter_m=cfg.jitter_m)
            operator = spherical_spline(pos, moved, alpha=cfg.spline_alpha) @ operator; changed = True
        if cfg.mirror and rng.random() < cfg.mirror:
            operator = spherical_spline(pos, mirrored_positions(pos), alpha=cfg.spline_alpha) @ operator; changed = True
        if cfg.conduction and rng.random() < cfg.conduction:
            kernel = smoothing_matrix(pos, rng.uniform(cfg.smooth_low_m, cfg.smooth_high_m))
            if rng.random() < .5:
                operator = kernel @ operator
            else:
                gamma = rng.uniform(0, cfg.sharpen)
                operator = ((1 + gamma) * np.eye(n) - gamma * kernel) @ operator
            changed = True
        if cfg.reference and rng.random() < cfg.reference and n > 2:
            kind = rng.integers(3)
            if kind == 0:
                operator = (np.eye(n) - np.full((n, n), 1. / n)) @ operator
            elif kind == 1:
                ref = int(rng.integers(n))
                reference = np.eye(n); reference[:, ref] -= 1.
                operator = reference @ operator
                out_mask[valid[ref]] = False                  # the reference electrode reads zero
            else:
                operator = laplacian_matrix(pos) @ operator
            changed = True
        full = np.zeros((channels, channels))
        full[np.ix_(valid, valid)] = operator
        if cfg.subset and rng.random() < cfg.subset and n > 2:
            keep = np.zeros(channels, bool)
            candidates = np.flatnonzero(out_mask)
            if rng.random() < cfg.regional:
                centre = np.asarray(xyz, np.float64)[candidates[rng.integers(len(candidates))]]
                distance = np.linalg.norm(np.asarray(xyz, np.float64)[candidates] - centre, axis=1)
                keep[candidates[distance > cfg.region_m]] = True
            else:
                fraction = rng.uniform(cfg.keep_low, cfg.keep_high)
                count = max(2, int(round(fraction * len(candidates))))
                keep[rng.choice(candidates, size=min(count, len(candidates)), replace=False)] = True
            if keep.sum() < 2:                                 # never leave fewer than two electrodes
                keep[candidates[:2]] = True
            if cfg.subset_mode == 'mask':
                full[~keep] = 0.
                out_mask &= keep
            else:
                kept = np.flatnonzero(keep)
                targets = np.flatnonzero(out_mask)
                refill = np.zeros((channels, channels))
                refill[np.ix_(targets, kept)] = spherical_spline(np.asarray(xyz, np.float64)[kept],
                                                                 np.asarray(xyz, np.float64)[targets], alpha=cfg.spline_alpha)
                full = refill @ full
            changed = True
        full[~out_mask] = 0.
        return full, out_mask, changed

    def __call__(self, eeg, channel_mask, channel_xyz, groups=None, rng=None):
        """Augmented ``(eeg, channel_mask)``; shapes unchanged, invalid channels zero."""
        if not self.config.enabled():
            return eeg, channel_mask
        rng = rng if rng is not None else np.random.default_rng()
        masks = channel_mask.detach().cpu().numpy().astype(bool)
        xyz = channel_xyz.detach().cpu().numpy()
        batch, channels, _ = eeg.shape
        matrices = np.zeros((batch, channels, channels), np.float32)
        out_masks = np.zeros_like(masks)
        changed = np.zeros(batch, bool)
        for b in range(batch):
            positions = xyz[b] if xyz.ndim == 3 else xyz
            group = groups[b] if groups is not None else None
            matrices[b], out_masks[b], changed[b] = self.sample_matrix(positions, masks[b], group, rng)
        if not changed.any():
            return eeg, channel_mask
        operator = torch.from_numpy(matrices).to(eeg.device, eeg.dtype)
        identity = torch.eye(channels, device=eeg.device, dtype=eeg.dtype)
        use = torch.from_numpy(changed).to(eeg.device)[:, None, None]
        operator = torch.where(use, operator, identity.expand_as(operator))
        out = torch.bmm(operator, eeg)
        out_mask = torch.from_numpy(out_masks).to(channel_mask.device)
        out_mask = torch.where(use[:, :, 0], out_mask, channel_mask)
        out = out * out_mask[:, :, None]
        if self.config.preserve_rms:
            before = eeg.square().sum((1, 2)) / (channel_mask.sum(1) * eeg.shape[-1]).clamp_min(1)
            after = out.square().sum((1, 2)) / (out_mask.sum(1) * eeg.shape[-1]).clamp_min(1)
            scale = (before / after.clamp_min(1e-12)).sqrt()
            scale = torch.where(use[:, 0, 0] & (after > 0), scale, torch.ones_like(scale))
            out = out * scale[:, None, None]
        return out, out_mask


def split_views(channel_mask, rng):
    """Two disjoint halves of every sample's valid channels (random, balanced); for a two-view consistency loss."""
    masks = channel_mask.detach().cpu().numpy().astype(bool)
    first = np.zeros_like(masks); second = np.zeros_like(masks)
    for b, row in enumerate(masks):
        valid = rng.permutation(np.flatnonzero(row))
        half = len(valid) // 2
        first[b, valid[:half]] = True; second[b, valid[half:]] = True
    return torch.from_numpy(first).to(channel_mask.device), torch.from_numpy(second).to(channel_mask.device)


def group_covariances(bank_eeg, bank_mask, groups, rng, *, per_group=120, shrinkage=.1):
    """Per-group spatial covariances from a bank of training trials (``(N, C, T)`` tensors or arrays)."""
    groups = np.asarray(groups)
    out = {}
    for group in sorted(set(groups.tolist())):
        rows = np.flatnonzero(groups == group)
        rows = rng.choice(rows, size=min(per_group, len(rows)), replace=False)
        trials = np.stack([np.asarray(bank_eeg[int(i)], np.float32) for i in rows])
        masks = np.stack([np.asarray(bank_mask[int(i)], bool) for i in rows])
        out[str(group)] = estimate_covariance(trials, masks, shrinkage=shrinkage)
    return out

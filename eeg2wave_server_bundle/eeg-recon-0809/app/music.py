#!/usr/bin/env python3
"""The music route: file formats, windowed dataset, and the frozen teacher -> mel decoder.

Three parts, kept in one module because they are only ever used together:

1. **File formats** - MATLAB ``.mat`` reading for the CND files of Di Liberto
   et al. 2020 (scipy for v5, a small h5py reader for v7.3), CND accessors, a
   Standard MIDI File parser with a plain additive renderer, the BigVGAN-v2
   log-mel, and the EEG preparation helpers the two prepare scripts share.
2. **Windows** - 4.6 s EEG windows cut from continuous music listening and
   aligned to per-piece teacher / mel / envelope targets on the stimulus clock
   (:class:`PieceTargets`, :class:`MusicWindows`).
3. **Decoder** - ``python app/music.py`` trains the frozen music
   AcousticDecoder (teacher sequence -> BigVGAN 100-band mel), the music
   counterpart of the speech "adapt" decoder.  Audio only, no EEG.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from math import gcd
from pathlib import Path
import re
import sys
import time

import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src'))
from eeg2speech.aligned import EEG_RATE, EEG_SAMPLES, EEG_START, acoustic_loss
from universal_model import (ACOUSTIC_FRAMES, ACOUSTIC_RATE, WINDOW_S, ConfigurableAcousticDecoder, decoder_from_spec)


# =========================================================================================
# 1. File formats (CND, MIDI, mel, shared EEG preparation helpers)
# =========================================================================================

def load_mat(path):
    """Top-level variables of a MATLAB file as plain Python (dicts, lists, numpy arrays, scalars, str)."""
    from scipy.io import loadmat
    try:
        raw = loadmat(path, simplify_cells=True)
        return {k: _plain(v) for k, v in raw.items() if not k.startswith('__')}
    except NotImplementedError:                        # MATLAB v7.3 = HDF5
        pass
    with h5py.File(path, 'r') as h5:
        return {k: _h5_value(h5, h5[k]) for k in h5.keys() if not k.startswith('#')}


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray) and value.dtype == object:
        return _plain_cells(value)
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def _plain_cells(cells):
    if cells.ndim <= 1:
        return [_plain(v) for v in cells.reshape(-1)]
    return [[_plain(v) for v in row] for row in cells]


def _h5_value(h5, obj):
    if isinstance(obj, h5py.Group):
        fields = {k: _h5_value(h5, obj[k]) for k in obj.keys()}
        lengths = {len(v) for v in fields.values() if isinstance(v, list)}
        # A MATLAB struct array stores each field as a cell of references.
        if obj.attrs.get('MATLAB_class', b'') in (b'struct', 'struct') and fields and all(isinstance(v, list) for v in fields.values()) \
                and len(lengths) == 1 and lengths.pop() > 1:
            count = len(next(iter(fields.values())))
            return [{k: v[i] for k, v in fields.items()} for i in range(count)]
        return fields
    matlab_class = obj.attrs.get('MATLAB_class', b'')
    matlab_class = matlab_class.decode() if isinstance(matlab_class, bytes) else str(matlab_class)
    if obj.attrs.get('MATLAB_empty', 0):
        return np.array([])
    if h5py.check_dtype(ref=obj.dtype) is not None:
        cells = np.array([[_h5_value(h5, h5[ref]) for ref in row] for row in obj[()]], dtype=object) \
            if obj.ndim == 2 else np.array([_h5_value(h5, h5[ref]) for ref in obj[()]], dtype=object)
        cells = cells.T if cells.ndim == 2 else cells            # HDF5 stores MATLAB arrays transposed
        if cells.ndim == 2 and 1 in cells.shape:
            return [v for v in cells.reshape(-1)]
        return [list(row) for row in cells] if cells.ndim == 2 else list(cells)
    value = obj[()]
    if matlab_class == 'char':
        return ''.join(chr(int(c)) for c in np.asarray(value).T.reshape(-1))
    value = np.asarray(value)
    if value.ndim >= 2:
        value = value.T
    if matlab_class == 'logical':
        value = value.astype(bool)
    value = np.squeeze(value)
    return value.item() if value.ndim == 0 else value


# --- CND structures ----------------------------------------------------------------------

@dataclass
class CNDStimulus:
    names: list
    features: list          # features[f][trial] -> 1-D array at ``fs``
    fs: float
    stimulus_index: np.ndarray | None     # 1-based stimulus id per trial, when the file provides it


def _as_list(value):
    if isinstance(value, np.ndarray) and value.dtype != object and value.ndim >= 1 and value.size > 1:
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, np.ndarray) and value.dtype == object:
        return list(value.reshape(-1))
    return [value]


def cnd_stimulus(mat):
    """Stimulus features of a CND ``dataStim`` file.

    Accepts the documented layout (``stim.data`` = features x trials cell)
    and the variant described in the Di Liberto README (``stim{f}.data`` =
    one struct per feature, each with a trials cell).
    """
    stim = mat['stim']
    if isinstance(stim, list) and stim and isinstance(stim[0], dict):         # cell of per-feature structs
        features = [[np.asarray(v, np.float64).reshape(-1) for v in _as_list(s['data'])] for s in stim]
        names = [str(s.get('names', s.get('name', f'feature{i + 1}'))) for i, s in enumerate(stim)]
        fs = float(np.asarray(stim[0]['fs']).reshape(-1)[0])
        index = stim[0].get('stimIdxs')
    else:
        data = stim['data']
        if isinstance(data, list) and data and isinstance(data[0], list):       # features x trials
            features = [[np.asarray(v, np.float64).reshape(-1) for v in row] for row in data]
        else:                                                                     # one feature row
            features = [[np.asarray(v, np.float64).reshape(-1) for v in _as_list(data)]]
        raw_names = stim.get('names', [f'feature{i + 1}' for i in range(len(features))])
        names = [str(n) for n in _as_list(raw_names)]
        fs = float(np.asarray(stim['fs']).reshape(-1)[0])
        index = stim.get('stimIdxs')
    trials = {len(f) for f in features}
    if len(trials) != 1:
        raise ValueError(f'CND stimulus features have different trial counts: {sorted(trials)}')
    index = None if index is None or np.asarray(index).size == 0 else np.asarray(index, int).reshape(-1)
    if index is not None and len(index) != len(features[0]):
        raise ValueError('stimIdxs length differs from the number of stimulus trials')
    return CNDStimulus(names=names, features=features, fs=fs, stimulus_index=index)


@dataclass
class CNDRecording:
    trials: list            # (samples, channels) arrays
    fs: float
    labels: list
    chanlocs: list          # EEGLAB-style dicts (labels, X, Y, Z, ...)
    padding_start: int      # 0-based index of the first stimulus sample
    presentation: np.ndarray | None


def cnd_eeg(mat, *, channels=64):
    eeg = mat['eeg']
    trials = [np.asarray(t, np.float64) for t in _as_list(eeg['data'])]
    trials = [t if t.shape[0] >= t.shape[1] else t.T for t in trials]           # samples x channels
    chanlocs = eeg.get('chanlocs', [])
    chanlocs = chanlocs if isinstance(chanlocs, list) else _as_list(chanlocs)
    labels = [str(c.get('labels', '')).strip() for c in chanlocs if isinstance(c, dict)]
    if trials and trials[0].shape[1] < channels:
        raise ValueError(f'expected at least {channels} scalp channels, found {trials[0].shape[1]}')
    trials = [t[:, :channels] for t in trials]
    padding = eeg.get('paddingStartSample')
    padding = int(np.asarray(padding).reshape(-1)[0]) if padding is not None and np.asarray(padding).size else 0
    presentation = eeg.get('origTrialPosition')
    presentation = None if presentation is None or np.asarray(presentation).size == 0 else np.asarray(presentation, int).reshape(-1)
    return CNDRecording(trials=trials, fs=float(np.asarray(eeg['fs']).reshape(-1)[0]), labels=labels[:channels],
                        chanlocs=[c for c in chanlocs if isinstance(c, dict)][:channels], padding_start=padding,
                        presentation=presentation)


def group_identical(vectors, *, tolerance=1e-9):
    """1-based group id per vector: vectors that are numerically identical share an id (in order of first occurrence)."""
    ids, representatives = [], []
    for v in vectors:
        v = np.asarray(v, np.float64)
        for k, r in enumerate(representatives):
            if len(r) == len(v) and np.allclose(r, v, rtol=0, atol=tolerance * max(1., float(np.abs(r).max()))):
                ids.append(k + 1); break
        else:
            representatives.append(v); ids.append(len(representatives))
    return np.asarray(ids)


# --- shared EEG preparation helpers ------------------------------------------------------

def bad_channels(data, fs, *, threshold=3.5, exclude=(), band=(1., 45.)):
    """Flat channels and channels whose log-SD in ``band`` is a robust outlier (``exclude``: never flagged, e.g. the reference).

    Judged in the band the shards keep: a channel whose only defect is mains
    noise above it is removed by the filter, not by interpolation (on MUSIN-G a
    1 Hz high-pass criterion flagged such channels, 50 Hz power up to 10^5 x the
    5-45 Hz power, and excluded 70 of 240 trials).
    """
    from scipy.signal import butter, sosfiltfilt
    filtered = sosfiltfilt(butter(4, band, 'bandpass', fs=fs, output='sos'), data, axis=1)
    sd = filtered.std(1)
    log_sd = np.log(np.maximum(sd, 1e-20))
    candidate = np.ones(len(sd), bool); candidate[list(exclude)] = False
    median = np.median(log_sd[candidate]); spread = 1.4826 * np.median(np.abs(log_sd[candidate] - median)) + 1e-9
    return np.flatnonzero(candidate & ((sd < 1e-8) | (np.abs(log_sd - median) / spread > threshold)))


def ranked(items, salt):
    return sorted(items, key=lambda x: hashlib.sha256(f'{salt}:{x}'.encode()).hexdigest())


def channel_order_hash(names):
    return hashlib.sha256('\n'.join(names).encode()).hexdigest()



# --- MIDI --------------------------------------------------------------------------------

@dataclass
class Note:
    start: float
    end: float
    pitch: int
    velocity: int


def _varlen(data, pos):
    value = 0
    while True:
        byte = data[pos]; pos += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, pos


def parse_midi(path):
    """Notes (seconds) of a Standard MIDI File, format 0 or 1, with a global tempo map."""
    data = Path(path).read_bytes()
    if data[:4] != b'MThd':
        raise ValueError(f'{path}: not a MIDI file')
    length = int.from_bytes(data[4:8], 'big')
    _, tracks, division = int.from_bytes(data[8:10], 'big'), int.from_bytes(data[10:12], 'big'), int.from_bytes(data[12:14], 'big')
    if division & 0x8000:
        raise ValueError(f'{path}: SMPTE time division is not supported')
    pos = 8 + length
    tempos = [(0, 500000)]
    events = []                                                    # (tick, kind, channel, pitch, velocity)
    for _ in range(tracks):
        while data[pos:pos + 4] != b'MTrk':                      # skip unknown chunks
            pos += 8 + int.from_bytes(data[pos + 4:pos + 8], 'big')
        end = pos + 8 + int.from_bytes(data[pos + 4:pos + 8], 'big')
        pos += 8; tick = 0; status = None
        while pos < end:
            delta, pos = _varlen(data, pos); tick += delta
            byte = data[pos]
            if byte == 0xFF:
                kind = data[pos + 1]; size, pos = _varlen(data, pos + 2)
                if kind == 0x51:
                    tempos.append((tick, int.from_bytes(data[pos:pos + 3], 'big')))
                pos += size
                continue
            if byte in (0xF0, 0xF7):
                size, pos = _varlen(data, pos + 1); pos += size
                continue
            if byte & 0x80:
                status = byte; pos += 1
            if status is None:
                raise ValueError(f'{path}: running status without a status byte')
            kind, channel = status & 0xF0, status & 0x0F
            size = 1 if kind in (0xC0, 0xD0) else 2
            args = data[pos:pos + size]; pos += size
            if kind == 0x90 and args[1] > 0:
                events.append((tick, 1, channel, args[0], args[1]))
            elif kind == 0x80 or (kind == 0x90 and args[1] == 0):
                events.append((tick, 0, channel, args[0], 0))
        pos = end
    tempos.sort(key=lambda t: t[0])
    starts, seconds, current_tick, current_s, current_tempo = [], [], 0, 0., 500000
    for t, tempo in tempos:
        current_s += (t - current_tick) * current_tempo / 1e6 / division
        current_tick, current_tempo = t, tempo
        starts.append(t); seconds.append(current_s)
    tempo_values = [t for _, t in tempos]

    def to_seconds(tick):
        k = int(np.searchsorted(starts, tick, side='right')) - 1
        return seconds[k] + (tick - starts[k]) * tempo_values[k] / 1e6 / division

    notes, active = [], {}
    for tick, on, channel, pitch, velocity in sorted(events, key=lambda e: (e[0], e[1])):
        key = (channel, pitch)
        if on:
            active.setdefault(key, []).append((to_seconds(tick), velocity))
        elif active.get(key):
            start, vel = active[key].pop(0)
            notes.append(Note(start, to_seconds(tick), int(pitch), int(vel)))
    return sorted(notes, key=lambda n: (n.start, n.pitch))


def render_additive(notes, sample_rate=24000, *, tail_s=1.):
    """A plain piano-like rendering: decaying inharmonic partials, 5 ms attack, 80 ms release.

    Timbre is only approximately piano; onsets, durations and pitch are
    exact, which is what the envelope/onset targets and the alignment check use.
    """
    total = (max((n.end for n in notes), default=0.) + tail_s)
    out = np.zeros(int(math.ceil(total * sample_rate)) + 1, np.float64)
    for note in notes:
        f0 = 440. * 2 ** ((note.pitch - 69) / 12)
        length = note.end - note.start + .08
        t = np.arange(int(length * sample_rate)) / sample_rate
        decay = np.exp(-t / float(np.clip(1.2 * math.sqrt(261.6 / f0), .3, 2.5)))
        attack = np.minimum(1., t / .005)
        release = np.clip((length - t) / .08, 0., 1.)
        wave = np.zeros_like(t)
        for h in range(1, 9):
            fh = h * f0 * math.sqrt(1 + 1e-4 * h * h)
            if fh >= .45 * sample_rate:
                break
            wave += h ** -1.3 * math.exp(-.15 * (h - 1)) * np.sin(2 * math.pi * fh * t)
        start = int(round(note.start * sample_rate))
        segment = wave * decay * attack * release * (note.velocity / 127.)
        out[start:start + len(segment)] += segment[:len(out) - start]
    peak = np.abs(out).max()
    return (out / peak * .9 if peak > 0 else out).astype(np.float32)


# --- BigVGAN-v2 compatible log-mel -------------------------------------------------------

def _hz_to_mel(f):
    f = np.asarray(f, np.float64)
    linear = f / (200. / 3)
    log_region = 1000. / (200. / 3) + np.log(np.maximum(f, 1e-9) / 1000.) / (np.log(6.4) / 27.)
    return np.where(f >= 1000., log_region, linear)


def _mel_to_hz(m):
    m = np.asarray(m, np.float64)
    linear = m * (200. / 3)
    log_region = 1000. * np.exp((np.log(6.4) / 27.) * (m - 1000. / (200. / 3)))
    return np.where(m >= 1000. / (200. / 3), log_region, linear)


def mel_filterbank(sample_rate, n_fft, n_mels, fmin=0., fmax=None):
    """Slaney-scale, slaney-normalised triangular filters (librosa.filters.mel defaults)."""
    fmax = sample_rate / 2 if fmax is None else fmax
    fft_freqs = np.linspace(0, sample_rate / 2, 1 + n_fft // 2)
    mel_freqs = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_freqs)
    ramps = mel_freqs[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0, np.minimum(lower, upper))
    weights *= (2. / (mel_freqs[2:n_mels + 2] - mel_freqs[:n_mels]))[:, None]
    return weights.astype(np.float32)


BIGVGAN_24K = dict(sample_rate=24000, n_fft=1024, hop=256, win=1024, n_mels=100, fmin=0., fmax=None)


def bigvgan_mel(wave, *, sample_rate=24000, n_fft=1024, hop=256, win=1024, n_mels=100, fmin=0., fmax=None):
    """(n_mels, frames) log-mel as BigVGAN-v2's ``meldataset.mel_spectrogram`` (center=False, reflect pad)."""
    y = torch.as_tensor(np.asarray(wave, np.float32))[None]
    basis = torch.from_numpy(mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax))
    pad = (n_fft - hop) // 2
    y = torch.nn.functional.pad(y[:, None], (pad, pad), mode='reflect')[:, 0]
    spec = torch.stft(y, n_fft, hop, win, window=torch.hann_window(win), center=False, return_complex=True)
    magnitude = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    return torch.log(torch.clamp(basis @ magnitude, min=1e-5))[0].numpy()


def bigvgan_mel_times(frames, *, sample_rate=24000, hop=256):
    """Centre time (s) of each ``bigvgan_mel`` frame: frame k spans samples [k*hop - pad, k*hop - pad + n_fft)."""
    return (np.arange(frames) * hop + hop / 2) / sample_rate


# =========================================================================================
# 2. Windowed dataset over continuous music listening
# =========================================================================================

CONTRACT = 'music_eeg_v1'
TARGET_CONTRACT = 'music_targets_v1'


def interpolate(values, times, query, axis):
    """Linear interpolation of ``values`` sampled at increasing ``times`` along ``axis``; ends are held."""
    times = np.asarray(times, np.float64); query = np.clip(np.asarray(query, np.float64), times[0], times[-1])
    right = np.clip(np.searchsorted(times, query), 1, len(times) - 1); left = right - 1
    weight = ((query - times[left]) / np.maximum(times[right] - times[left], 1e-12)).astype(np.float32)
    a = np.take(values, left, axis=axis); b = np.take(values, right, axis=axis)
    shape = [1] * values.ndim; shape[axis] = len(query)
    return a * (1 - weight.reshape(shape)) + b * weight.reshape(shape)


class PieceTargets:
    """All pieces' targets in memory (a few hundred MB for Di Liberto) and their window grids."""
    def __init__(self, path, pieces=None):
        self.path = Path(path)
        with h5py.File(self.path, 'r') as h5:
            if h5.attrs.get('contract') != TARGET_CONTRACT:
                raise ValueError(f'{path}: not a {TARGET_CONTRACT} file')
            self.attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in h5.attrs.items()}
            self.teacher_grid = h5['grid/teacher_times'][:].astype(np.float32)
            self.mel_grid = h5['grid/mel_times'][:].astype(np.float32)
            available = sorted(h5['pieces'].keys(), key=int)
            chosen = available if pieces is None else [str(int(p)) for p in pieces]
            missing = sorted(set(chosen) - set(available))
            if missing:
                raise ValueError(f'targets missing for pieces {missing}')
            self.pieces = {}
            for name in chosen:
                g = h5['pieces'][name]
                self.pieces[int(name)] = dict(
                    teacher=g['teacher'][:].astype(np.float32), teacher_times=g['teacher_times'][:].astype(np.float64),
                    mel=g['mel'][:].astype(np.float32), mel_times=g['mel_times'][:].astype(np.float64),
                    acoustic=g['acoustic'][:].astype(np.float32), duration=float(g.attrs['duration_s']))
        self.acoustic_grid = np.arange(ACOUSTIC_FRAMES) / ACOUSTIC_RATE

    @property
    def teacher_dimension(self):
        return next(iter(self.pieces.values()))['teacher'].shape[1]

    def window(self, piece, start):
        p = self.pieces[int(piece)]
        teacher = interpolate(p['teacher'], p['teacher_times'], start + self.teacher_grid, axis=0)
        mel = interpolate(p['mel'], p['mel_times'], start + self.mel_grid, axis=1)
        acoustic_times = np.arange(p['acoustic'].shape[1]) / ACOUSTIC_RATE
        acoustic = interpolate(p['acoustic'], acoustic_times, start + self.acoustic_grid, axis=1)
        return teacher, mel, acoustic


class MusicWindows:
    """Windows of one role (content roles x participant roles) of a prepared music dataset."""
    def __init__(self, root, manifest, targets, normalizer, *, content_roles=('train',), subject_roles=('train',)):
        self.root = Path(root)
        frame = manifest if isinstance(manifest, pd.DataFrame) else pd.read_csv(manifest, keep_default_na=False)
        frame = frame[frame.content_role.isin(content_roles) & frame.subject_role.isin(subject_roles)]
        self.frame = frame.reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f'no music trials for content {content_roles} x participants {subject_roles}')
        self.targets = targets if isinstance(targets, PieceTargets) else PieceTargets(targets, sorted(set(self.frame.piece)))
        normalizer = normalizer if isinstance(normalizer, dict) else json.loads(Path(normalizer).read_text())
        self.center = np.asarray(normalizer['center'], np.float32)[:, None]
        self.scale = np.asarray(normalizer['scale'], np.float32)[:, None]
        self.handles = {}
        first = self._file(self.frame.iloc[0])
        self.channel_xyz = first['channel_xyz'][:].astype(np.float32)
        if str(first.attrs['channel_order_hash']) != normalizer['channel_order_hash']:
            raise ValueError('music normalizer / shard channel order mismatch')
        self.limits = np.array([self._max_start(row) for row in self.frame.itertuples()])
        if (self.limits < 0).any():
            raise ValueError('a music trial is shorter than one 4 s window')
        self.by_piece = self.frame.groupby('piece').indices
        self.by_subject = self.frame.groupby('subject').indices

    def close(self):
        for h5 in self.handles.values():
            h5.close()
        self.handles = {}

    def _file(self, row):
        path = str(row.shard_path) if not isinstance(row, pd.Series) else str(row['shard_path'])
        if path not in self.handles:
            self.handles[path] = h5py.File(self.root / path, 'r')
        return self.handles[path]

    def _max_start(self, row):
        """Largest window start (s) with both the 4 s audio window and the 4.6 s EEG window inside the trial."""
        by_eeg = (int(row.n_samples) - EEG_SAMPLES - int(row.onset_sample)) / EEG_RATE - EEG_START
        by_audio = float(row.stimulus_duration_s) - WINDOW_S
        by_targets = self.targets.pieces[int(row.piece)]['duration'] - WINDOW_S
        return min(by_eeg, by_audio, by_targets)

    def __len__(self):
        return len(self.frame)

    # --- window selection ---
    def random_keys(self, count, rng, *, min_gap_s=WINDOW_S, tries=50):
        """``count`` windows; windows of the same piece never overlap in stimulus time (distinct contents)."""
        keys, taken = [], {}
        for _ in range(count):
            for _ in range(tries):
                row = int(rng.integers(len(self.frame)))
                start = float(rng.uniform(0., self.limits[row]))
                piece = int(self.frame.piece.iat[row])
                if all(abs(start - s) >= min_gap_s for s in taken.get(piece, [])):
                    keys.append((row, start)); taken.setdefault(piece, []).append(start); break
            else:
                raise RuntimeError('could not draw non-overlapping music windows; lower the batch size')
        return keys

    def grid_keys(self, stride_s=WINDOW_S, limit=None, rng=None):
        keys = [(row, float(s)) for row in range(len(self.frame)) for s in np.arange(0., self.limits[row] + 1e-9, stride_s)]
        if limit is not None and len(keys) > limit:
            rng = rng if rng is not None else np.random.default_rng(0)
            keys = [keys[i] for i in sorted(rng.choice(len(keys), size=limit, replace=False))]
        return keys

    def partners(self, key, count, rng):
        """Other presentations (other participants or repeats) of the same piece at the same stimulus time."""
        row, start = key
        others = [int(j) for j in self.by_piece[int(self.frame.piece.iat[row])] if j != row and self.limits[j] >= start]
        if not others or count < 1:
            return []
        chosen = rng.choice(others, size=min(count, len(others)), replace=False)
        return [(int(j), start) for j in chosen]

    def background(self, key, rng):
        """The same participant listening to a different piece (random time): real, participant-specific noise."""
        row, _ = key
        subject, piece = self.frame.subject.iat[row], int(self.frame.piece.iat[row])
        others = [int(j) for j in self.by_subject[subject] if int(self.frame.piece.iat[j]) != piece]
        if not others:
            return None
        j = int(others[rng.integers(len(others))])
        return j, float(rng.uniform(0., self.limits[j]))

    def shifted(self, key, shift_s=8.):
        """Control window: same trial, a non-overlapping moment of the piece; else another piece of the same participant."""
        row, start = key
        limit = self.limits[row]
        for other in (start + shift_s, start - shift_s, 0. if start > limit / 2 else limit):
            if 0 <= other <= limit and abs(other - start) >= WINDOW_S:
                return row, float(other)
        subject, piece = self.frame.subject.iat[row], int(self.frame.piece.iat[row])
        others = [int(j) for j in self.by_subject[subject] if int(self.frame.piece.iat[j]) != piece]
        if not others:
            raise ValueError(f'no wrong-window control for {subject} piece {piece}: trial shorter than two windows')
        return others[0], float(min(start, self.limits[others[0]]))

    # --- loading ---
    def eeg(self, key):
        row, start = key
        record = self.frame.iloc[row]
        h5 = self._file(record)
        first = int(record.onset_sample) + int(round((start + EEG_START) * EEG_RATE))
        trial = f'{int(record.trial):02d}'
        x = h5['trials'][trial][:, first:first + EEG_SAMPLES].astype(np.float32)
        if x.shape[1] != EEG_SAMPLES:
            raise ValueError(f'window outside trial: {record.subject} trial {trial} start {start:.3f}')
        valid = h5['trials_valid'][trial][:].astype(bool)
        return ((x - self.center) / self.scale) * valid[:, None], valid

    def load(self, keys):
        eeg, masks, teachers, mels, acoustics = [], [], [], [], []
        for key in keys:
            x, valid = self.eeg(key)
            teacher, mel, acoustic = self.targets.window(int(self.frame.piece.iat[key[0]]), key[1])
            eeg.append(x); masks.append(valid); teachers.append(teacher); mels.append(mel); acoustics.append(acoustic)
        rows = [k[0] for k in keys]
        return dict(eeg=torch.from_numpy(np.stack(eeg)), channel_mask=torch.from_numpy(np.stack(masks)),
                    channel_xyz=torch.from_numpy(np.repeat(self.channel_xyz[None], len(keys), 0)),
                    time_mask=torch.ones(len(keys), EEG_SAMPLES, dtype=torch.bool),
                    teacher=torch.from_numpy(np.stack(teachers)), mel=torch.from_numpy(np.stack(mels)),
                    acoustic=torch.from_numpy(np.stack(acoustics)),
                    piece=[int(self.frame.piece.iat[r]) for r in rows], start=[float(k[1]) for k in keys],
                    subject=[str(self.frame.subject.iat[r]) for r in rows],
                    group=[f'{self.frame.dataset.iat[r]}:{self.frame.subject.iat[r]}' for r in rows])

    def eeg_bank_sample(self, per_subject, rng):
        """Random windows per participant, for spatial covariance estimation (recolouring)."""
        eeg, masks, groups = [], [], []
        for subject, rows in self.by_subject.items():
            for _ in range(per_subject):
                row = int(rows[rng.integers(len(rows))])
                x, valid = self.eeg((row, float(rng.uniform(0., self.limits[row]))))
                eeg.append(x); masks.append(valid); groups.append(f'{self.frame.dataset.iat[row]}:{subject}')
        return np.stack(eeg), np.stack(masks), groups


# =========================================================================================
# 3. Frozen teacher -> mel decoder (entry point)
# =========================================================================================

DECODER_CONTRACT = 'music_decoder_v1'


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def piece_roles(manifest):
    frame = pd.read_csv(manifest, keep_default_na=False)
    roles = frame.groupby('piece').content_role.agg(lambda r: sorted(set(r)))
    if any(len(v) != 1 for v in roles):
        raise ValueError('a piece has more than one content role')
    return {int(p): v[0] for p, v in roles.items()}


def fit_normalizer(decoder, targets, pieces):
    values = np.concatenate([targets.pieces[p]['teacher'] for p in pieces]).astype(np.float64)
    decoder.normalizer.mean.copy_(torch.from_numpy(values.mean(0)).float())
    decoder.normalizer.scale.copy_(torch.from_numpy(values.std(0)).clamp_min(1e-4).float())


def windows(targets, keys):
    teacher, mel = zip(*[targets.window(piece, start)[:2] for piece, start in keys])
    return torch.from_numpy(np.stack(teacher)), torch.from_numpy(np.stack(mel))


def grid(targets, pieces, stride_s):
    return [(p, float(s)) for p in pieces for s in np.arange(0., targets.pieces[p]['duration'] - WINDOW_S + 1e-9, stride_s)]


@torch.no_grad()
def validate(decoder, targets, keys, device, template, batch=32):
    decoder.eval(); errors, baseline = [], []
    for i in range(0, len(keys), batch):
        teacher, mel = windows(targets, keys[i:i + batch])
        teacher, mel = teacher.to(device), mel.to(device)
        errors.append((decoder(teacher) - mel).abs().mean((1, 2)).cpu())
        baseline.append((template.to(device)[None, :, None] - mel).abs().mean((1, 2)).cpu())
    return float(torch.cat(errors).mean()), float(torch.cat(baseline).mean())


def load_music_decoder(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('contract') != DECODER_CONTRACT:
        raise ValueError(f'{path}: not a {DECODER_CONTRACT} checkpoint')
    return decoder_from_spec(payload['decoder_spec'], payload['decoder']), payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', default=str(ROOT / 'artifacts/music/diliberto2020/targets_mert.h5'))
    parser.add_argument('--manifest', default=str(ROOT / 'artifacts/music/diliberto2020/manifest.csv'))
    parser.add_argument('--output', default=str(ROOT / 'outputs/music_decoder/mert'))
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--updates', type=int, default=4000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--seed', type=int, default=322)
    parser.add_argument('--throttle', type=float, default=0., help='sleep this fraction of each update\'s wall time')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    device = torch.device('mps' if args.device == 'auto' and torch.backends.mps.is_available() else
                          ('cpu' if args.device == 'auto' else args.device))
    roles = piece_roles(args.manifest)
    train_pieces = sorted(p for p, r in roles.items() if r == 'train')
    validation_pieces = sorted(p for p, r in roles.items() if r == 'validation')
    targets = PieceTargets(args.targets, train_pieces + validation_pieces)
    spec = dict(speech_times=targets.teacher_grid.tolist(), mel_times=targets.mel_grid.tolist(),
                speech_dimension=targets.teacher_dimension, hidden=args.hidden, layers=args.layers,
                mel_bins=next(iter(targets.pieces.values()))['mel'].shape[0])
    decoder = ConfigurableAcousticDecoder(torch.tensor(spec['speech_times']), torch.tensor(spec['mel_times']),
                                          speech_dimension=spec['speech_dimension'], hidden=args.hidden,
                                          layers=args.layers, mel_bins=spec['mel_bins'])
    fit_normalizer(decoder, targets, train_pieces)
    decoder.to(device)
    template = torch.from_numpy(np.median(np.concatenate([targets.pieces[p]['mel'] for p in train_pieces], 1), 1)).float()
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=.01)
    validation_keys = grid(targets, validation_pieces, 2.)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    best, history, start = math.inf, [], time.monotonic()
    for step in range(1, args.updates + 1):
        tick = time.monotonic()
        decoder.train()
        for group in optimizer.param_groups:
            group['lr'] = args.lr * min(1., step / 200) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / args.updates)))
        keys = [(int(p), float(rng.uniform(0., targets.pieces[int(p)]['duration'] - WINDOW_S)))
                for p in rng.choice(train_pieces, size=args.batch_size)]
        teacher, mel = windows(targets, keys)
        loss = acoustic_loss(decoder(teacher.to(device)), mel.to(device))
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.); optimizer.step()
        if args.throttle > 0:
            time.sleep(args.throttle * (time.monotonic() - tick))
        if step % args.eval_every == 0 or step == args.updates:
            mae, template_mae = validate(decoder, targets, validation_keys, device, template)
            history.append(dict(update=step, train_loss=float(loss), validation_mae=mae, template_mae=template_mae))
            print(json.dumps(dict(**history[-1], seconds=round(time.monotonic() - start, 1))), flush=True)
            if mae < best:
                best = mae
                torch.save(dict(contract=DECODER_CONTRACT, decoder_spec=spec, decoder={k: v.cpu() for k, v in decoder.state_dict().items()},
                                targets=args.targets, targets_sha256=file_sha256(args.targets), teacher=targets.attrs.get('teacher'),
                                teacher_layer=int(targets.attrs.get('teacher_layer', -1)), update=step,
                                validation_mae=mae, template_mae=template_mae, train_pieces=train_pieces,
                                validation_pieces=validation_pieces), output / 'best.pt')
            (output / 'metrics.json').write_text(json.dumps(dict(contract=DECODER_CONTRACT, history=history), indent=1))
    print(f'best validation mel MAE {best:.4f} (template {history[-1]["template_mae"]:.4f}); {output / "best.pt"}')


if __name__ == '__main__':
    main()

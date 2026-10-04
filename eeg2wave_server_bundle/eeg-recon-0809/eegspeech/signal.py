"""Signal utilities: filtering, resampling, Euclidean alignment, speech features, electrode positions."""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache

import numpy as np
from scipy import signal as sps

FEATURE_RATE = 64                      # Hz, stimulus feature frame rate (also SparrKULee / Marion EEG rate)
FEATURE_ROWS = ['envelope', 'onset'] + [f'mel{i:02d}' for i in range(16)]


# --- filtering and resampling -----------------------------------------------------------------

def resample(x, source, target, axis=-1):
    """Polyphase resampling (anti-aliased) between integer-ratio rates."""
    if source == target:
        return x
    ratio = Fraction(int(round(target)), int(round(source)))
    return sps.resample_poly(x, ratio.numerator, ratio.denominator, axis=axis)


def bandpass(x, rate, low=None, high=None, order=4, axis=-1):
    """Zero-phase Butterworth band/high/low-pass; ``None`` leaves that edge open."""
    nyquist = rate / 2
    if low and high:
        sos = sps.butter(order, [low, min(high, .95 * nyquist)], 'bandpass', fs=rate, output='sos')
    elif low:
        sos = sps.butter(order, low, 'highpass', fs=rate, output='sos')
    elif high:
        sos = sps.butter(order, min(high, .95 * nyquist), 'lowpass', fs=rate, output='sos')
    else:
        return x
    padlen = min(x.shape[axis] - 1, 3 * int(rate))
    return sps.sosfiltfilt(sos, x, axis=axis, padlen=padlen)


def notch(x, rate, frequency, axis=-1):
    if frequency >= rate / 2:
        return x
    b, a = sps.iirnotch(frequency, 30, fs=rate)
    return sps.filtfilt(b, a, x, axis=axis)


# --- Euclidean alignment -----------------------------------------------------------------------

def mean_covariance(segments, valid=None):
    """Length-weighted mean spatial covariance of demeaned (C, T) segments over channels marked valid."""
    total, weight = None, 0
    for x in segments:
        x = np.asarray(x, np.float64)
        x = x - x.mean(-1, keepdims=True)
        cov = x @ x.T
        total = cov if total is None else total + cov
        weight += x.shape[-1]
    cov = total / max(weight, 1)
    if valid is not None:
        cov = cov * np.outer(valid, valid)
    return cov


def whitening(cov, shrink=.1, valid=None):
    """Symmetric (ZCA) whitening ``R^-1/2`` with trace-scaled shrinkage.

    The symmetric root keeps each output channel as close as possible to its input
    channel, so electrode positions stay meaningful after alignment (He & Wu 2019
    use the same matrix for Euclidean alignment).  Invalid channels map to zero.
    """
    c = cov.shape[0]
    keep = np.ones(c, bool) if valid is None else np.asarray(valid, bool)
    sub = cov[np.ix_(keep, keep)]
    sub = (1 - shrink) * sub + shrink * np.trace(sub) / max(len(sub), 1) * np.eye(len(sub))
    values, vectors = np.linalg.eigh(sub)
    root = vectors @ np.diag(1 / np.sqrt(np.clip(values, 1e-12, None))) @ vectors.T
    out = np.zeros((c, c))
    out[np.ix_(keep, keep)] = root * np.sqrt(np.trace(sub) / max(len(sub), 1))   # keep the original scale
    return out.astype(np.float32)


# --- speech / audio features -------------------------------------------------------------------

def mel_filterbank(n_fft, rate, bands, low, high):
    """Triangular HTK-mel filters, (bands, n_fft // 2 + 1)."""
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = hz(np.linspace(mel(low), mel(high), bands + 2))
    freqs = np.fft.rfftfreq(n_fft, 1 / rate)
    bank = np.zeros((bands, len(freqs)))
    for i in range(bands):
        left, centre, right = edges[i:i + 3]
        bank[i] = np.clip(np.minimum((freqs - left) / (centre - left), (right - freqs) / (right - centre)), 0, None)
    return bank


@lru_cache(maxsize=4)
def _bank(n_fft, rate):
    return mel_filterbank(n_fft, rate, 16, 100., 7500.)


def speech_features(wave, rate):
    """(18, frames) z-scored features at 64 Hz: log broadband envelope, onset strength, 16 log-mel bands.

    Frame k covers audio around k / 64 s.  These are the auditory features scalp
    EEG is known to track (envelope, acoustic edges, coarse spectral energy).
    """
    wave = np.asarray(wave, np.float64)
    if wave.ndim > 1:
        wave = wave.mean(-1) if wave.shape[-1] < wave.shape[0] else wave.mean(0)
    wave = resample(wave, rate, 16000)
    hop, n_fft = 16000 // FEATURE_RATE, 512
    _, _, spec = sps.stft(wave, 16000, window='hann', nperseg=2 * hop, noverlap=hop, nfft=n_fft,
                          boundary='even', padded=True)
    power = np.abs(spec) ** 2
    bank = _bank(n_fft, 16000)
    mel = np.log(bank @ power + 1e-8)
    envelope = np.log(bank.sum(0) @ power + 1e-8)
    onset = np.clip(np.diff(mel, axis=1, prepend=mel[:, :1]), 0, None).sum(0)
    frames = int(np.ceil(len(wave) / hop))
    out = np.vstack([envelope, onset, mel])[:, :frames]
    out = (out - out.mean(1, keepdims=True)) / (out.std(1, keepdims=True) + 1e-6)
    return out.astype(np.float32)


# --- electrode positions -----------------------------------------------------------------------

@lru_cache(maxsize=8)
def _montage(name):
    import mne
    positions = mne.channels.make_standard_montage(name).get_positions()['ch_pos']
    return {k.upper(): np.asarray(v, np.float32) for k, v in positions.items()}


def positions(channels, montage='standard_1005'):
    """(C, 3) electrode positions in metres (MNE head frame) and a found mask, matched by name."""
    table = _montage(montage)
    aliases = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
    out = np.zeros((len(channels), 3), np.float32)
    found = np.zeros(len(channels), bool)
    for i, name in enumerate(channels):
        key = str(name).strip().upper()
        key = aliases.get(key, key)
        if key in table:
            out[i], found[i] = table[key], True
    return out, found


BIOSEMI64 = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
             'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
             'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
             'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
BIOSEMI128 = [f'{bank}{i}' for bank in 'ABCD' for i in range(1, 33)]

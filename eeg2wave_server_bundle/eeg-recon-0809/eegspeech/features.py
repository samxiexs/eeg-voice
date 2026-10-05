"""Per-trial EEG features for personal decoders: Euclidean-aligned filter-bank log-power.

Each labelled trial is aligned with its person's (session's) whitening matrix, band-passed into
``BANDS`` (the high-frequency bands are kept: imagined-speech information is mostly above 55 Hz)
and summarised by the log-variance of every channel in every band over the whole trial.  Within a
person this simple representation beats time-resolved band power and tangent-space covariances
(contiguous 5-fold: BCI2020 36.7 % vs 30.1 % and 32.8 %, chance 20 %; Thinking Out Loud 29.6 % vs
27.5 % and 27.1 %, chance 25 %).

Features are cached per store in ``artifacts/features/<dataset>.npz`` (recomputed when the store is newer).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import signal as sps

from . import ROOT, STORE
from .store import open_store

BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45), (55, 80), (80, 120))
CACHE = ROOT / 'artifacts' / 'features'


def bands_for(rate):
    return [(lo, min(hi, .45 * rate)) for lo, hi in BANDS if lo < .45 * rate]


def log_power(x, rate, bands):
    """(bands * channels,) log-variance of each band-passed channel."""
    out = []
    for lo, hi in bands:
        sos = sps.butter(4, [lo, hi], 'bandpass', fs=rate, output='sos')
        out.append(np.log(sps.sosfiltfilt(sos, x, axis=-1).var(-1) + 1e-6))
    return np.concatenate(out).astype(np.float32)


def _subject(args):
    name, subject = args
    store = open_store(name)
    rows = store.table[(store.table.subject == subject) & (store.table.item >= 0)]
    bands = bands_for(store.rate)
    valid = store.file['subjects'][subject]['valid'][:]
    out = []
    for row in rows.itertuples():
        x = store.alignment(subject, int(row.session)) @ store.segment(row)
        x[~valid[int(row.segment)]] = 0                      # unusable channels carry no information
        out.append(log_power(x.astype(np.float64), store.rate, bands))
    return subject, rows.index.to_numpy(), np.stack(out)


def compute(name, workers=4):
    """{subject: (table rows, features (n, bands * channels))} for every labelled trial of a store."""
    store = open_store(name)
    jobs = [(name, s) for s in store.subjects]
    with ProcessPoolExecutor(workers) as pool:
        return {subject: (rows, x) for subject, rows, x in pool.map(_subject, jobs)}


def load(name, workers=4):
    """Cached features of a store; computed on first use."""
    path = CACHE / f'{name}.npz'
    if not path.exists() or path.stat().st_mtime < (STORE / f'{name}.h5').stat().st_mtime:   # store rebuilt since
        path.parent.mkdir(parents=True, exist_ok=True)
        result = compute(name, workers)
        arrays = {}
        for subject, (rows, x) in result.items():
            arrays[f'{subject}/rows'], arrays[f'{subject}/x'] = rows, x
        np.savez(path.with_suffix('.tmp.npz'), **arrays)
        path.with_suffix('.tmp.npz').replace(path)
    data = np.load(path)
    subjects = sorted({k.split('/')[0] for k in data.files})
    return {s: (data[f'{s}/rows'], data[f'{s}/x']) for s in subjects}

"""Per-trial EEG features for personal decoders, and the within-person folds.

Features: each labelled trial is aligned with its person's (session's) whitening matrix, band-passed into
``BANDS`` and summarised by the log-variance of every channel in every band, over the whole trial and
over ``WINDOWS`` consecutive windows (its time course).  Chisco (imagined sentences, held-out runs) has
its information in the time course, not in the whole-trial power: rank percentile 0.512 with 3 windows
as deviations from the whole trial vs 0.503 without (chance 0.5).

The alignment of a fold never sees that fold's held-out trials (``load(name, held=...)``): it comes from
the person's other segments, labelled or not, so a held-out trial is processed as a new trial would be.

Folds (``blocks``): contiguous parts of every person's trials in recording order, or whole cue episodes,
or whole recording runs:

- Cue episodes (``REPEATS``): BCI2020 Track 3 had each auditory cue imagined 4 times back to back, and the
  released files list the trials in an order that is not the recording order (neighbouring trials are no
  more alike than any two; in Thinking Out Loud they are), with the 4 repetitions of a cue spread over the
  training, validation and test files.  The repetitions of one cue are near-duplicates (shared muscle
  tone, electrode state): a test trial's nearest neighbour among the training trials is the same item for
  67 % of the trials on >= 55 Hz power without any learning (chance 20 %), while within the test file,
  which holds one repetition per cue, it is 20 %.  ``episodes`` reconstructs the episodes (within each
  person and item, groups of 4 by similarity).
- Runs (``RUNS``): Chisco presents its sentences in topic blocks within a run, and the store lists runs in
  name order, so folds hold out whole runs in run order.

Features are cached per store and alignment in ``artifacts/features/`` (recomputed when the store is newer).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import signal as sps

from . import ROOT, STORE
from .store import open_store

BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45), (55, 70), (70, 90), (90, 120))
WINDOWS = 6                      # consecutive windows of the time course (pooled in groups by ``time_course``)
CACHE = ROOT / 'artifacts' / 'features'
REPEATS = {'bci2020': 4}         # imaginings per cue, for datasets whose released order is not the recording order
RUNS = {'chisco'}                # datasets whose folds hold out whole recording runs (sessions), in run order
EPISODE_MIN_HZ = 30              # the repetitions of a cue are most alike above 30 Hz
MIN_ALIGN_SEGMENTS = 20          # fewer segments left in a session after the held-out ones: align on the whole person


def bands_for(rate):
    return [(lo, min(hi, .45 * rate)) for lo, hi in BANDS if lo < .45 * rate]


def log_power(x, rate, bands, windows=0):
    """(bands * channels,) log-variance of each band-passed channel over the whole trial; with ``windows``
    also (windows, bands * channels) over consecutive equal windows of the trial."""
    edges = np.linspace(0, x.shape[-1], windows + 1).round().astype(int) if windows else []
    whole, parts = [], []
    for lo, hi in bands:
        y = sps.sosfiltfilt(sps.butter(4, [lo, hi], 'bandpass', fs=rate, output='sos'), x, axis=-1)
        whole.append(np.log(y.var(-1) + 1e-6))
        parts.append([np.log(y[:, a:b].var(-1) + 1e-6) for a, b in zip(edges[:-1], edges[1:])])
    whole = np.concatenate(whole).astype(np.float32)
    if not windows:
        return whole
    return whole, np.stack([np.concatenate([band[j] for band in parts]) for j in range(windows)]).astype(np.float32)


def band_columns(bands, features, min_hz):
    """Feature columns (band-major layout) of the bands starting at or above ``min_hz``."""
    channels = features // len(bands)
    return np.concatenate([np.arange(b * channels, (b + 1) * channels)
                           for b, (lo, _) in enumerate(bands) if lo >= min_hz])


def time_course(whole, windows, k):
    """(n, k * F) log-power of ``k`` consecutive parts of the trial (the windows averaged in groups) as
    deviations from the whole trial: the shape of each channel's and band's time course."""
    n, w, f = windows.shape
    return (windows.reshape(n, k, w // k, f).mean(2) - whole[:, None]).reshape(n, k * f)


def _subject(args):
    name, subject, held, align_by, windows, modalities = args
    store = open_store(name)
    table = store.table[store.table.subject == subject]
    if modalities:
        table = table[table.modality_name.isin(modalities)]
    rows = table[table.item >= 0]
    bands = bands_for(store.rate)
    valid = store.file['subjects'][subject]['valid'][:]
    usable = table[~table.index.isin(held)]                 # segments the alignment may use
    whole, parts = [], []
    for row in rows.itertuples():
        session = int(row.session) if align_by == 'session' else None
        if session is not None and (usable.session == session).sum() < MIN_ALIGN_SEGMENTS:
            session = None
        x = store.alignment(subject, session, modalities=modalities, exclude=held) @ store.segment(row)
        x[~valid[int(row.segment)]] = 0                      # unusable channels carry no information
        if windows:
            w, p = log_power(x.astype(np.float64), store.rate, bands, windows)
        else:
            w = log_power(x.astype(np.float64), store.rate, bands)
            p = np.zeros((0, len(w)), np.float32)
        whole.append(w)
        parts.append(p)
    return subject, rows.index.to_numpy(), np.stack(whole), np.stack(parts)


def load(name, held=(), align_by='session', modalities=None, windows=WINDOWS, workers=4):
    """{subject: (table rows, whole-trial features (n, F), window features (n, windows, F))} of the labelled
    trials of a store, and the bands; cached, computed on first use.

    ``held``: table rows whose EEG must not enter any alignment matrix (a fold's held-out trials).
    ``align_by``: 'session' (a matrix per person and session) or 'subject' (one per person, for datasets
    whose 'sessions' are parts of one recording or whose held-out runs are whole).  ``modalities``: only
    these modalities, for the features and for the alignment."""
    bands = np.array(bands_for(open_store(name).rate), float)
    held = np.unique(np.asarray(list(held), int))
    modalities = sorted(modalities) if modalities else None
    key = hashlib.sha1(json.dumps([align_by, bands.tolist(), windows, MIN_ALIGN_SEGMENTS, modalities]).encode()
                       + held.tobytes()).hexdigest()[:12]
    path = CACHE / f'{name}_{key}.npz'
    if not path.exists() or path.stat().st_mtime < (STORE / f'{name}.h5').stat().st_mtime:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(bands=bands, held=held)
        store = open_store(name)
        jobs = [(name, s, sorted(set(held.tolist()) & set(store.table.index[store.table.subject == s].tolist())),
                 align_by, windows, modalities) for s in store.subjects]
        with ProcessPoolExecutor(workers) as pool:
            for subject, rows, x, parts in pool.map(_subject, jobs):
                arrays[f'{subject}/rows'], arrays[f'{subject}/x'], arrays[f'{subject}/windows'] = rows, x, parts
        np.savez(path.with_suffix('.tmp.npz'), **arrays)
        path.with_suffix('.tmp.npz').replace(path)
    data = np.load(path)
    subjects = sorted({k.split('/')[0] for k in data.files if '/' in k})
    return ({s: (data[f'{s}/rows'], data[f'{s}/x'], data[f'{s}/windows']) for s in subjects},
            [tuple(b) for b in data['bands']])


# ----------------------------------------------------------------------------- folds
def blocks(table, folds):
    """Fold (0 .. folds - 1) of every row of ``table`` in the within-person protocol.

    Default: the contiguous parts of every (person, modality) in recording order, so back-to-back
    repetitions stay on one side.  With an ``episode`` column (``labelled`` adds it for ``REPEATS``):
    whole cue episodes, dealt to the folds per (person, modality, item), so the repetitions of one cue
    never sit on both sides and every fold keeps the class balance.  With a ``block`` column (the
    recording run, ``RUNS``): whole runs, contiguous in run order.
    """
    out = pd.Series(-1, index=table.index)
    if 'block' in table:
        for _, part in table.groupby(['subject', 'modality_name']):
            for j, chunk in enumerate(np.array_split(np.sort(part['block'].unique()), folds)):
                out[part.index[part['block'].isin(chunk)]] = j
    elif 'episode' in table:
        for _, part in table.groupby(['subject', 'modality_name', 'item']):
            episodes = part['episode'].drop_duplicates().to_numpy()            # order of first appearance
            for j, chunk in enumerate(np.array_split(episodes, folds)):
                out[part.index[part['episode'].isin(chunk)]] = j
    else:
        for _, part in table.groupby(['subject', 'modality_name']):
            for j, chunk in enumerate(np.array_split(part.index.to_numpy(), folds)):
                out[chunk] = j
    return out.to_numpy()


def within_fold(table, folds, fold):
    """Table rows held out by fold ``fold`` of the within-person protocol (see ``blocks``)."""
    return table.index.to_numpy()[blocks(table, folds) == fold].astype(int)


def labelled(store, modalities=None):
    """Labelled trials of a store (table rows), of some modalities; with an ``episode`` column where the
    paradigm repeated cues back to back and the released order is not the recording order (``REPEATS``),
    a ``block`` column (the run) where folds hold out whole runs (``RUNS``)."""
    rows = store.table[store.table.item >= 0]
    if modalities is not None:
        rows = rows[rows.modality_name.isin(modalities)]
    if store.name in REPEATS:
        rows = rows.assign(episode=episodes(store.name).reindex(rows.index).to_numpy())
    if store.name in RUNS:
        rows = rows.assign(block=rows.session.to_numpy())
    return rows


# ----------------------------------------------------------------------------- cue episodes
def capped_groups(similarity, size):
    """Average-linkage agglomerative clustering of (n, n) similarities into groups of at most ``size``."""
    n = len(similarity)
    total = np.asarray(similarity, np.float64).copy()          # summed similarity between clusters
    count = np.ones(n)
    alive = np.ones(n, bool)
    label = np.arange(n)
    while True:
        allowed = alive[:, None] & alive[None] & (count[:, None] + count[None] <= size)
        np.fill_diagonal(allowed, False)
        if not allowed.any():
            break
        a, b = np.unravel_index(np.argmax(np.where(allowed, total / np.outer(count, count), -np.inf)), (n, n))
        total[a] += total[b]
        total[:, a] += total[:, b]
        count[a] += count[b]
        alive[b] = False
        label[label == b] = a
    return [np.flatnonzero(label == g) for g in np.unique(label)]


def _episode_subject(args):
    name, subject = args
    store = open_store(name)
    rows = store.table[(store.table.subject == subject) & (store.table.item >= 0)]
    bands = [b for b in bands_for(store.rate) if b[0] >= EPISODE_MIN_HZ]
    valid = store.file['subjects'][subject]['valid'][:]
    x = []
    for row in rows.itertuples():
        eeg = store.alignment(subject) @ store.segment(row)     # unlabelled; only defines the folds
        eeg[~valid[int(row.segment)]] = 0
        x.append(log_power(eeg.astype(np.float64), store.rate, bands))
    x = np.stack(x)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    x = x - x.mean(1, keepdims=True)
    x /= np.linalg.norm(x, axis=1, keepdims=True).clip(1e-9)
    groups = np.zeros(len(rows), int)
    count = 0
    for _, part in rows.reset_index(drop=True).groupby(['modality_name', 'item']):
        index = part.index.to_numpy()
        for g in capped_groups(x[index] @ x[index].T, REPEATS[name]):
            groups[index[g]] = count
            count += 1
    return subject, rows.index.to_numpy(), groups


def episodes(name, workers=4):
    """Cue episode (int, unique within the dataset) of every labelled trial of a ``REPEATS`` dataset, by
    table row: within each person, modality and item, the trials grouped by similarity of their aligned
    >= 30 Hz log-power into groups of ``REPEATS[name]``.  Labels only decide which trials may share an
    episode (the repetitions of one cue share it); nothing here is used by any decoder."""
    path = CACHE / f'{name}_episodes.npz'
    if not path.exists() or path.stat().st_mtime < (STORE / f'{name}.h5').stat().st_mtime:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows, ids, offset = [], [], 0
        with ProcessPoolExecutor(workers) as pool:
            for _, r, g in pool.map(_episode_subject, [(name, s) for s in open_store(name).subjects]):
                rows.append(r)
                ids.append(g + offset)
                offset += g.max() + 1
        np.savez(path.with_suffix('.tmp.npz'), rows=np.concatenate(rows), episode=np.concatenate(ids))
        path.with_suffix('.tmp.npz').replace(path)
    data = np.load(path)
    return pd.Series(data['episode'], index=data['rows'])


# ----------------------------------------------------------------------------- drift
def local_baseline(x, run, held, prior_mean, window, prior=5.):
    """Causal running baseline (n, F) of a person's trial features in recording order.

    Row i's baseline is the mean of the up to ``window`` rows before it in its run (same session and
    modality), shrunk towards ``prior_mean[i]`` (its modality's mean over the training trials) with
    ``prior`` pseudo-trials.  It uses no labels and nothing recorded after the trial, as an online
    system would (adaptive bias, Vidaurre et al. 2011).  Held-out rows may lean on the held-out rows
    before them (the stream a user produces); training rows only on training rows.
    """
    out = np.empty_like(x, dtype=np.float64)
    held = np.asarray(held, bool)
    for r in np.unique(run):
        index = np.flatnonzero(run == r)
        for p, i in enumerate(index):
            before = index[:p] if held[i] else index[:p][~held[index[:p]]]
            before = before[len(before) - min(window, len(before)):]
            out[i] = (x[before].sum(0) + prior * prior_mean[i]) / (len(before) + prior)
    return out.astype(np.float32)

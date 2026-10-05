"""Samplers that turn stores into model batches.

``TrialSource``: labelled trials (heard / spoken / mouthed / imagined items) -> fixed-length crops.
``ListenSource``: stimulus-locked listening EEG -> windows with the matched stimulus features,
K mismatched candidates and, when another person heard the same stimulus, that person's
window at the same stimulus time.

Every window is optionally Euclidean-aligned with its subject's own (session) whitening
matrix, resampled to the model rate and zeroed on invalid channels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import zlib

import numpy as np
import torch

from .model import RATE
from .signal import FEATURE_RATE, bandpass, resample
from .store import Store


@dataclass
class Batch:
    eeg: torch.Tensor                    # (B, C, T) float32 at RATE
    xyz: torch.Tensor                    # (B, C, 3)
    valid: torch.Tensor                  # (B, C) bool
    lengths: torch.Tensor                # (B,) valid samples
    extra: dict = field(default_factory=dict)

    def to(self, device):
        move = lambda v: v.to(device) if torch.is_tensor(v) else v
        return Batch(move(self.eeg), move(self.xyz), move(self.valid), move(self.lengths),
                     {k: move(v) for k, v in self.extra.items()})


def collate(windows, extra_keys=()):
    """Pad channels and time to the batch maximum."""
    channels = max(w['eeg'].shape[0] for w in windows)
    samples = max(w['eeg'].shape[1] for w in windows)
    eeg = np.zeros((len(windows), channels, samples), np.float32)
    xyz = np.zeros((len(windows), channels, 3), np.float32)
    valid = np.zeros((len(windows), channels), bool)
    for i, w in enumerate(windows):
        c, t = w['eeg'].shape
        eeg[i, :c, :t], xyz[i, :c], valid[i, :c] = w['eeg'], w['xyz'], w['valid']
    extra = {}
    for key in extra_keys:
        values = [w[key] for w in windows]
        extra[key] = torch.as_tensor(np.stack(values)) if not isinstance(values[0], str) else values
    return Batch(torch.from_numpy(eeg), torch.from_numpy(xyz), torch.from_numpy(valid),
                 torch.tensor([w.get('length', w['eeg'].shape[1]) for w in windows]), extra)


def subject_folds(subjects, folds=5, seed=0):
    """Deterministic subject -> fold map (stable under adding or removing other subjects)."""
    order = sorted(subjects, key=lambda s: zlib.crc32(f'{seed}:{s}'.encode()))
    return {s: i % folds for i, s in enumerate(order)}


def within_fold(table, folds, fold):
    """Table rows held out by fold ``fold`` of the within-person protocol: the fold-th contiguous part
    of every (person, modality), in recording order, so back-to-back repetitions stay on one side."""
    return np.array([i for _, part in table.groupby(['subject', 'modality_name'])
                     for i in np.array_split(part.index.to_numpy(), folds)[fold]], int)


class Source:
    """``rate``: the model's input rate; ``max_hz``: optional low-pass (e.g. 45 Hz to exclude high-frequency EMG)."""

    def __init__(self, store: Store, subjects, *, align=True, rng=None, rate=RATE, max_hz=None):
        self.store, self.align, self.rate, self.max_hz = store, align, int(rate), max_hz
        self.subjects = sorted(set(subjects) & set(store.subjects))
        self.rng = rng or np.random.default_rng(0)
        self._xyz = {s: store.xyz(s) for s in self.subjects}
        self._valid = {s: store.file['subjects'][s]['valid'][:] for s in self.subjects}

    def window(self, row, start, length):
        """Model-rate EEG of ``length`` store samples from ``start`` within the segment ``row``."""
        x = self.store.eeg(row.subject, int(row.start) + int(start), int(length))
        valid = self._valid[row.subject][int(row.segment)]
        if self.align:
            x = self.store.alignment(row.subject, int(row.session)) @ x
        x = resample(x, self.store.rate, self.rate)
        if self.max_hz and self.max_hz < .5 * self.rate:
            x = bandpass(x, self.rate, high=self.max_hz)
        x = x.astype(np.float32)
        x[~valid] = 0
        return dict(eeg=x, xyz=self._xyz[row.subject], valid=valid)


class TrialSource(Source):
    """Labelled trials of one store, cropped to ``window_s``; ``modalities`` restricts which trials enter."""

    def __init__(self, store: Store, subjects, *, window_s, modalities=None, align=True, rng=None, cache=True,
                 rate=RATE, max_hz=None, rows=None):
        super().__init__(store, subjects, align=align, rng=rng, rate=rate, max_hz=max_hz)
        table = store.table
        keep = table.subject.isin(self.subjects) & (table.item >= 0)
        if modalities:
            keep &= table.modality_name.isin(modalities)
        if rows is not None:                       # explicit table positions (e.g. one block of each person)
            keep &= table.index.isin(rows)
        self.rows = table[keep].reset_index(drop=True)
        self.size = int(round(window_s * store.rate))
        self.items = store.items
        if cache:
            store.load(self.subjects)

    def __len__(self):
        return len(self.rows)

    def crop(self, i, start=None):
        row = self.rows.iloc[i]
        size = min(self.size, int(row.length))
        if start is None:
            start = int(self.rng.integers(0, int(row.length) - size + 1))
        w = self.window(row, start, size)
        w['length'] = w['eeg'].shape[1]
        if size < self.size:                                   # pad short trials to the common length
            pad = int(round(self.size * self.rate / self.store.rate)) - w['eeg'].shape[1]
            w['eeg'] = np.pad(w['eeg'], ((0, 0), (0, max(pad, 0))))
        w.update(item=int(row['item']), modality=int(row['modality']), index=i, subject=row['subject'])
        return w

    def crops(self, i, count):
        """Evenly spaced deterministic crops of trial i (evaluation)."""
        length = int(self.rows.iloc[i].length)
        starts = np.linspace(0, max(length - self.size, 0), count).round().astype(int)
        return [self.crop(i, s) for s in np.unique(starts)]

    def sample(self, n, by_item=True):
        """Item-balanced random trials (every item equally likely), one random crop each."""
        if by_item:
            groups = self.rows.groupby('item').indices
            keys = list(groups)
            picks = [int(self.rng.choice(groups[keys[k]])) for k in self.rng.integers(0, len(keys), n)]
        else:
            picks = self.rng.integers(0, len(self.rows), n).tolist()
        return [self.crop(i) for i in picks]

    def windows_of(self, subject, count, exclude=()):
        """``count`` random crops of one subject's trials, labels unused (person calibration sets)."""
        pool = np.setdiff1d(np.flatnonzero(self.rows.subject.to_numpy() == subject), np.asarray(list(exclude), int))
        picks = self.rng.choice(pool, size=count, replace=len(pool) < count)
        return [self.crop(int(i)) for i in picks]


class ListenSource(Source):
    """Stimulus-locked listening windows for match-mismatch and cross-person agreement."""

    def __init__(self, store: Store, subjects, *, window_s=5., mismatches=4, min_overlap=.5, align=True, rng=None,
                 rate=RATE, max_hz=None):
        super().__init__(store, subjects, align=align, rng=rng, rate=rate, max_hz=max_hz)
        table = store.table
        rows = table[table.subject.isin(self.subjects) & (table.stimulus >= 0)].reset_index(drop=True)
        self.window_s, self.mismatches = float(window_s), int(mismatches)
        self.frames = int(round(window_s * FEATURE_RATE))
        durations = {k: store.stimulus_frames(k) / FEATURE_RATE for k in rows.stimulus.unique()}
        rows['duration'] = rows.stimulus.map(durations)
        # admissible stimulus times t of a window start: inside the EEG segment, overlapping the stimulus enough
        rows['t_low'] = np.maximum(rows.offset, -(1 - min_overlap) * window_s)
        rows['t_high'] = np.minimum(rows.offset + rows.length / store.rate - window_s,
                                    rows.duration - min_overlap * window_s)
        self.rows = rows[rows.t_high >= rows.t_low].reset_index(drop=True)
        span = (self.rows.t_high - self.rows.t_low).to_numpy() + 1.
        self.weights = span / span.sum()
        self.by_stimulus = self.rows.groupby('stimulus').indices
        self.stimulus_ids = np.array(sorted(self.by_stimulus))
        self.durations = durations

    def __len__(self):
        return len(self.rows)

    def _eeg(self, row, t):
        start = int(round((t - row.offset) * self.store.rate))
        start = min(max(start, 0), int(row.length) - int(round(self.window_s * self.store.rate)))
        return self.window(row, start, int(round(self.window_s * self.store.rate)))

    def _mismatch(self, stimulus, t):
        """Stimulus features of a segment that was not heard at this time: same stimulus elsewhere if long
        enough (no overlap), otherwise another stimulus at the same relative time."""
        duration = self.durations[stimulus]
        if duration >= 3 * self.window_s:
            for _ in range(20):
                other = float(self.rng.uniform(0, duration - self.window_s))
                if abs(other - t) >= self.window_s:
                    return self.store.features(stimulus, other, self.frames)
        choices = self.stimulus_ids[self.stimulus_ids != stimulus]
        other = int(self.rng.choice(choices))
        when = float(np.clip(t, -.25 * self.window_s, max(self.durations[other] - .5 * self.window_s, 0)))
        return self.store.features(other, when, self.frames)

    def sample(self, n, partner_fraction=.5, with_features=True):
        """n anchor windows (+ partners). Returns (windows, pairs) with pairs = [(anchor index, partner index)]."""
        anchors, partners, pairs = [], [], []
        for i in self.rng.choice(len(self.rows), size=n, p=self.weights):
            row = self.rows.iloc[int(i)]
            t = float(self.rng.uniform(row.t_low, row.t_high))
            w = self._eeg(row, t)
            w.update(subject=row.subject, stimulus=int(row.stimulus), time=t)
            if with_features:
                candidates = [self.store.features(int(row.stimulus), t, self.frames)]
                candidates += [self._mismatch(int(row.stimulus), t) for _ in range(self.mismatches)]
                w['features'] = np.stack(candidates)
            anchors.append(w)
            if self.rng.random() < partner_fraction:
                pool = self.rows.iloc[self.by_stimulus[int(row.stimulus)]]
                pool = pool[(pool.subject != row.subject) & (pool.t_low <= t) & (pool.t_high >= t)]
                if len(pool):
                    other = pool.iloc[int(self.rng.integers(len(pool)))]
                    p = self._eeg(other, t)
                    p.update(subject=other.subject, stimulus=int(row.stimulus), time=t)
                    pairs.append((len(anchors) - 1, len(partners)))
                    partners.append(p)
        offset = len(anchors)
        return anchors + partners, [(a, offset + b) for a, b in pairs]

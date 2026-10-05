"""One HDF5 file per dataset (``artifacts/store/<name>.h5``) with a single layout for every dataset.

Each subject holds one concatenated EEG array plus a segment table.  A segment is
either a labelled trial (an item imagined, spoken, mouthed or heard) or a
stimulus-locked stretch of listening EEG; both kinds share the table, so the same
loaders serve every dataset.

    /                     attrs: name, rate, band, reference, unit, language, notes,
                                 items, modalities, stimuli (JSON lists)
    /stimuli/<k>          (F, frames) float16 features at FEATURE_RATE; attrs: name, rows
    /subjects/<id>/eeg    (C, samples) float16
    /subjects/<id>/xyz    (C, 3) float32 electrode positions, metres, MNE head frame
    /subjects/<id>/valid  (S, C) bool, channel validity per segment
    /subjects/<id>/segments  structured (S,): start, length, session, modality, item, stimulus, offset
    /subjects/<id>/text   (S,) optional strings (e.g. the sentence of an imagined trial)
    /subjects/<id>        attrs: channels (JSON)

``item``/``stimulus`` are indices into the root lists (-1: none).  ``offset`` is the
stimulus time in seconds at the segment's first sample (NaN without a stimulus),
so EEG sample ``i`` of a segment hears stimulus time ``offset + i / rate``.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .signal import FEATURE_RATE, mean_covariance, whitening

SEGMENT_DTYPE = np.dtype([('start', 'i8'), ('length', 'i4'), ('session', 'i2'), ('modality', 'i2'),
                          ('item', 'i4'), ('stimulus', 'i4'), ('offset', 'f4')])


class StoreWriter:
    """Builds a store subject by subject; lists of items / modalities / stimuli grow as they appear."""

    def __init__(self, path, *, name, rate, band, reference, unit, language='', notes='', items=(), modalities=()):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.path.with_suffix('.h5.tmp')
        self.file = h5py.File(self.tmp, 'w')
        self.meta = dict(name=name, rate=float(rate), band=json.dumps(band), reference=reference, unit=unit,
                         language=language, notes=notes)
        self.items, self.modalities, self.stimuli = list(items), list(modalities), []
        self.file.create_group('stimuli')
        self.file.create_group('subjects')

    @staticmethod
    def _index(table, value):
        if value is None:
            return -1
        if value not in table:
            table.append(value)
        return table.index(value)

    def add_stimulus(self, name, features, rows):
        if name in self.stimuli:
            return self.stimuli.index(name)
        k = self._index(self.stimuli, name)
        ds = self.file['stimuli'].create_dataset(str(k), data=np.asarray(features, np.float16))
        ds.attrs['name'], ds.attrs['rows'] = name, json.dumps(list(rows))
        return k

    def add_subject(self, subject, channels, xyz, segments):
        """``segments``: dicts with eeg (C, L), valid (C,) and optional session, modality, item, stimulus, offset."""
        if not segments:
            return
        group = self.file['subjects'].create_group(subject)
        group.attrs['channels'] = json.dumps(list(channels))
        xyz = np.asarray(xyz, np.float32)
        located = np.isfinite(xyz).all(1) & (np.linalg.norm(np.nan_to_num(xyz), axis=1) > 1e-6)
        group.create_dataset('xyz', data=np.where(located[:, None], xyz, 0).astype(np.float32))
        total = sum(s['eeg'].shape[-1] for s in segments)
        eeg = group.create_dataset('eeg', (len(channels), total), np.float16, chunks=(len(channels), min(total, 2048)))
        table = np.zeros(len(segments), SEGMENT_DTYPE)
        valid = np.zeros((len(segments), len(channels)), bool)
        position = 0
        for i, s in enumerate(segments):
            x = np.asarray(s['eeg'], np.float32)
            if x.shape[0] != len(channels) or not np.isfinite(x).all():
                raise ValueError(f'{subject} segment {i}: bad shape {x.shape} or non-finite values')
            eeg[:, position:position + x.shape[1]] = np.clip(x, -6e4, 6e4).astype(np.float16)
            stimulus = s.get('stimulus')
            table[i] = (position, x.shape[1], s.get('session', 0), self._index(self.modalities, s.get('modality')),
                        self._index(self.items, s.get('item')),
                        stimulus if isinstance(stimulus, (int, np.integer)) else self._index(self.stimuli, stimulus),
                        s.get('offset', np.nan))
            valid[i] = np.asarray(s.get('valid', np.ones(len(channels), bool)), bool) & located   # no position: unusable
            position += x.shape[1]
        group.create_dataset('segments', data=table)
        group.create_dataset('valid', data=valid)
        if any('text' in s for s in segments):                     # e.g. the sentence of an imagined trial
            group.create_dataset('text', data=[s.get('text', '') for s in segments], dtype=h5py.string_dtype())

    def close(self):
        for key, value in self.meta.items():
            self.file.attrs[key] = value
        for key in ('items', 'modalities', 'stimuli'):
            self.file.attrs[key] = json.dumps(getattr(self, key))
        self.file.close()
        self.tmp.replace(self.path)


class Store:
    """Read access.  ``table`` lists every segment; EEG is read as float32 on demand (or cached with ``load``)."""

    def __init__(self, path):
        self.path = Path(path)
        self.file = h5py.File(self.path, 'r')
        attrs = self.file.attrs
        self.name, self.rate = str(attrs['name']), float(attrs['rate'])
        self.items, self.modalities, self.stimuli = (json.loads(attrs[k]) for k in ('items', 'modalities', 'stimuli'))
        self.language, self.unit = str(attrs.get('language', '')), str(attrs.get('unit', ''))
        self.subjects = sorted(self.file['subjects'])
        frames = []
        for subject in self.subjects:
            group = self.file['subjects'][subject]
            table = pd.DataFrame(group['segments'][:])
            table.insert(0, 'segment', np.arange(len(table)))
            if 'text' in group:
                table['text'] = group['text'].asstr()[:]
            table.insert(0, 'subject', subject)
            frames.append(table)
        self.table = pd.concat(frames, ignore_index=True)
        self.table['modality_name'] = [self.modalities[m] if m >= 0 else '' for m in self.table['modality']]
        self.table['item_name'] = [self.items[i] if i >= 0 else '' for i in self.table['item']]
        self._cache, self._features, self._alignment = {}, {}, {}

    def __repr__(self):
        return (f'Store({self.name}: {len(self.subjects)} subjects, {len(self.table)} segments, '
                f'{self.table["length"].sum() / self.rate / 3600:.1f} h, {self.rate:g} Hz)')

    def channels(self, subject):
        return json.loads(self.file['subjects'][subject].attrs['channels'])

    def xyz(self, subject):
        return self.file['subjects'][subject]['xyz'][:]

    def load(self, subjects=None):
        """Cache the EEG of these subjects in memory (float16), for the small trial datasets."""
        for subject in subjects or self.subjects:
            if subject not in self._cache:
                self._cache[subject] = self.file['subjects'][subject]['eeg'][:]

    def eeg(self, subject, start, length):
        if subject in self._cache:
            return self._cache[subject][:, start:start + length].astype(np.float32)
        return self.file['subjects'][subject]['eeg'][:, start:start + length].astype(np.float32)

    def segment(self, row):
        """EEG (C, length) of one table row."""
        return self.eeg(row.subject, int(row.start), int(row.length))

    def features(self, stimulus, start_s, frames):
        """(F, frames) stimulus features from stimulus time ``start_s``; zeros outside the stimulus."""
        if stimulus not in self._features:
            self._features[stimulus] = self.file['stimuli'][str(stimulus)][:].astype(np.float32)
        full = self._features[stimulus]
        out = np.zeros((full.shape[0], frames), np.float32)
        first = int(round(start_s * FEATURE_RATE))
        lo, hi = max(first, 0), min(first + frames, full.shape[1])
        if hi > lo:
            out[:, lo - first:hi - first] = full[:, lo:hi]
        return out

    def stimulus_frames(self, stimulus):
        return self.file['stimuli'][str(stimulus)].shape[1]

    def alignment(self, subject, session=None, modalities=None, shrink=.1):
        """Euclidean-alignment matrix of one subject (optionally one session / some modalities).

        Computed from that subject's own EEG without labels, so it is available for
        an unseen person from calibration recordings.
        """
        key = (subject, session, tuple(modalities) if modalities else None)
        if key not in self._alignment:
            rows = self.table[self.table.subject == subject]
            if session is not None:
                rows = rows[rows.session == session]
            if modalities:
                rows = rows[rows.modality_name.isin(modalities)]
            masks = self.file['subjects'][subject]['valid'][:][rows.segment.to_numpy()]
            valid = masks.mean(0) > .5 if len(rows) else None          # channels usable in most segments
            cov = mean_covariance((self.segment(r) for r in rows.itertuples()), valid)
            self._alignment[key] = whitening(cov, shrink, valid)
        return self._alignment[key]


def open_store(name, root=None):
    from . import STORE
    return Store(Path(root or STORE) / f'{name}.h5')

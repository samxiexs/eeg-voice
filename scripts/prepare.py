"""Convert a downloaded dataset into the common store (``artifacts/store/<name>.h5``).

    python scripts/prepare.py <dataset> [--force]

    python scripts/prepare.py chisco --relabel        # stores built before 2026-10-06: mark the reading epochs

Common conventions: microvolts, average reference over valid channels, the full band (0.5-120 Hz at
256 Hz; Chisco 1-120 Hz), mains and harmonics notched.  Trials keep only the task window (the action /
imagery interval); rest recordings are kept as unlabelled ``rest`` segments for alignment.

Sources: ``data/raw/<dataset>`` from ``scripts/download.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegspeech import ROOT, STORE                                                    # noqa: E402
from eegspeech.signal import BIOSEMI128, bandpass, notch, positions, resample          # noqa: E402
from eegspeech.store import StoreWriter                                             # noqa: E402

RAW = ROOT / 'data' / 'raw'
FULL_RATE = 256                 # imagery stores keep the full band: their decodable information is mostly > 45 Hz
FULL_BAND = (.5, 120.)


def reref(x, valid):
    """Average reference over valid channels; invalid channels set to zero."""
    x = x - x[valid].mean(0, keepdims=True)
    x[~valid] = 0
    return x


def full(x, rate, valid, *, mains, scale=1.):
    """Scale to microvolts, notch the mains and its harmonics, band-pass 0.5-120 Hz (below the anti-alias band
    of 256 Hz), re-reference, resample to 256 Hz.  x (C, T)."""
    x = np.asarray(x, np.float64) * scale
    for f in np.arange(mains, rate / 2, mains):
        x = notch(x, rate, f)
    x = bandpass(x, rate, low=FULL_BAND[0], high=min(FULL_BAND[1], .45 * min(rate, FULL_RATE)))
    return resample(reref(x, valid), rate, FULL_RATE).astype(np.float32)


def thinking_out_loud(out):
    """Nieto et al. 2022 (OpenNeuro ds003626): 10 people, 4 Spanish words, pronounced / inner / visualised, 128-ch BioSemi.

    The authors' derivatives (mastoid reference, 50 Hz notch, 0.5-100 Hz, 256 Hz) are kept at
    full band.  Epochs run -0.5..4 s from the cue; the action interval is 1.0-3.5 s (facial EMG
    of the pronounced condition rises at ~1 s).  Inner / visualised trials the authors flag for
    mouth EMG are kept: covert articulation is information too (the flags stay in report.pkl).
    """
    import mne
    mne.set_log_level('ERROR')
    words = ['arriba', 'abajo', 'derecha', 'izquierda']
    conditions = {0: 'overt', 1: 'imagine', 2: 'visual'}
    writer = StoreWriter(out, name='thinking_out_loud', rate=FULL_RATE, band=[.5, 100], reference='average', unit='uV',
                         language='es', items=words, modalities=['overt', 'imagine', 'visual', 'rest'],
                         notes='authors\' derivatives kept at 0.5-100 Hz; trial window 1.0-3.5 s after the cue; '
                               'rest = 15 s baseline per session')
    xyz, found = positions(BIOSEMI128, 'biosemi128')
    root = RAW / 'ds003626' / 'derivatives'
    for subject_dir in sorted(root.glob('sub-*')):
        segments = []
        for session_dir in sorted(subject_dir.glob('ses-*')):
            base = session_dir / f'{subject_dir.name}_{session_dir.name}_'
            epochs = mne.read_epochs(f'{base}eeg-epo.fif').pick(BIOSEMI128)
            events = np.load(f'{base}events.dat', allow_pickle=True)
            data, rate = epochs.get_data(), epochs.info['sfreq']
            first, last = int(round((1.0 - epochs.tmin) * FULL_RATE)), int(round((3.5 - epochs.tmin) * FULL_RATE))
            session = int(session_dir.name.split('-')[1])
            for i, (_, word, condition, _) in enumerate(events.astype(int)):
                x = full(data[i], rate, found, mains=50, scale=1e6)          # filter the whole epoch, then crop
                segments.append(dict(eeg=x[:, first:last], valid=found, session=session,
                                     modality=conditions[condition], item=words[word]))
            baseline = mne.read_epochs(f'{base}baseline-epo.fif').pick(BIOSEMI128).get_data()[0]
            segments.append(dict(eeg=full(baseline, rate, found, mains=50, scale=1e6), valid=found, session=session,
                                 modality='rest'))
        writer.add_subject(subject_dir.name, BIOSEMI128, xyz, segments)
        print(subject_dir.name, len(segments), flush=True)
    writer.close()


def bci2020(out):
    """2020 International BCI Competition, Track 3 (OSF pq7vb): 15 people imagine 5 phrases, 64-ch BrainAmp.

    Each auditory cue was followed by four cross / imagine (2 s) cycles; the released
    epochs run -0.5..2.6 s around imagery onset and the 0-2 s imagery window is kept.
    Sessions: 0 training, 1 validation, 2 test (labels from the published answer sheet).
    Full band (0.5-120 Hz at 256 Hz): most of the decodable information is above 55 Hz.
    """
    import pandas as pd
    import scipy.io as sio
    names = ['hello', 'help me', 'stop', 'thank you', 'yes']
    writer = StoreWriter(out, name='bci2020', rate=FULL_RATE, band=list(FULL_BAND), reference='average', unit='uV',
                         language='en', items=names, modalities=['imagine'],
                         notes='imagery window 0-2 s; session 0 train / 1 validation / 2 test; 60 Hz notched')
    root = RAW / 'bci2020'
    sheet = pd.read_excel(root / 'Test set' / 'Track3_Answer Sheet_Test.xlsx', header=None)
    answers = {}
    for row in range(3):
        for col in range(sheet.shape[1]):
            value = sheet.iat[row, col]
            if isinstance(value, str) and value.startswith('Data_Sample'):
                answers[value.strip()] = sheet.iloc[row + 2:, col + 1].dropna().astype(int).to_numpy() - 1

    def read(path, key, stem):
        try:
            m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)[key]
            return m.x.transpose(2, 1, 0), np.asarray(m.y).argmax(0), np.asarray(m.t), [str(c) for c in m.clab], float(m.fs)
        except NotImplementedError:                                            # the test files are MATLAB v7.3
            with h5py.File(path, 'r') as f:
                g = f[key]
                clab = [''.join(map(chr, f[ref][:].ravel())) for ref in g['clab'][:, 0]]
                return g['x'][:], answers[stem], g['t'][:].ravel(), clab, float(g['fs'][0, 0])

    for k in range(1, 16):
        stem = f'Data_Sample{k:02d}'
        segments, channels = [], None
        for session, (folder, key) in enumerate([('Training set', 'epo_train'), ('Validation set', 'epo_validation'),
                                                 ('Test set', 'epo_test')]):
            x, y, t, clab, fs = read(root / folder / f'{stem}.mat', key, stem)
            if len(y) != len(x):
                raise ValueError(f'{stem} {folder}: {len(x)} trials, {len(y)} labels')
            channels = channels or clab
            xyz, found = positions(clab, 'standard_1005')
            onset = int(round(np.argmax(t >= 0) * FULL_RATE / fs))          # imagery onset, in samples at FULL_RATE
            for i in range(len(x)):
                segments.append(dict(eeg=full(x[i], fs, found, mains=60)[:, onset:onset + 2 * FULL_RATE], valid=found,
                                     session=session, modality='imagine', item=names[int(y[i])]))
        writer.add_subject(f'sub-{k:02d}', channels, positions(channels, 'standard_1005')[0], segments)
        print(f'sub-{k:02d}', len(segments), flush=True)
    writer.close()


CHISCO_SUBJECTS = ['01', '02', '03', '04', '05']


def chisco(out, keep_raw=False, subjects=None):
    """Chisco (Zhang et al. 2024, OpenNeuro ds005170): 5 people imagine ~6,600 everyday Chinese sentences, 122-ch.

    Each trial shows a sentence for 5 s (reading) and then has it imagined for 3.3 s; the authors'
    preprocessed fif holds both phases as separate epoch files of every run (500 Hz).  The 3.3 s
    epochs are modality 'imagine', the 5.0 s epochs 'read' (stores built before 2026-10-06 called
    both 'imagine'; ``chisco_relabel`` fixes them in place).  The epochs carry the sentence as metadata; items are the 39 semantic categories of
    ``json/textmaps.json`` (sentences without a category stay unlabelled), the sentence
    is kept as segment text.  Subjects are downloaded, converted to a part file and their
    raw files removed one at a time (61 GB of fif in total, unless ``--keep-raw``); a rerun
    skips finished parts, ``--subjects`` limits a run to some of them (one per job keeps each
    job short), and the parts are merged once all five exist.
    """
    import shutil
    import mne
    sys.path.insert(0, str(ROOT / 'scripts'))
    import download
    mne.set_log_level('ERROR')
    root = RAW / 'ds005170'
    download.run(download.openneuro_jobs('ds005170', ['json/*', 'README', 'participants.tsv']))
    textmaps = json.load(open(root / 'json' / 'textmaps.json', encoding='utf-8'))
    classes = json.load(open(root / 'json' / 'classnumber.json', encoding='utf-8'))
    names = [classes[str(k)] for k in range(len(classes))]
    meta = dict(name='chisco', rate=FULL_RATE, band=[1., FULL_BAND[1]], reference='average', unit='uV', language='zh',
                items=names, modalities=['imagine', 'read'],
                notes='authors\' epochs (1 Hz high-pass, 500 Hz) kept at full band; imagine = imagery phase 5.0-8.3 s '
                      'of each trial, read = reading phase 0-5 s; items = semantic category; session = run')
    parts = out.parent / 'chisco.parts'
    for subject in subjects or CHISCO_SUBJECTS:
        part = parts / f'sub-{subject}.h5'
        if part.exists():
            continue
        folder = root / 'derivatives' / 'preprocessed_fif' / f'sub-{subject}' / 'eeg'
        download.run(download.openneuro_jobs('ds005170', [f'derivatives/preprocessed_fif/sub-{subject}/eeg/*.fif']))
        segments, channels = [], None
        for path in sorted(folder.glob('*_eeg.fif')):
            run = int(re.search(r'run-(\d+)', path.name).group(1))
            epochs = mne.read_epochs(path)
            eeg = [c for c, kind in zip(epochs.ch_names, epochs.get_channel_types()) if kind == 'eeg'
                   and c not in ('VEO', 'HEO', 'Trigger')]
            if channels is None:
                channels = eeg
                xyz, found = positions(channels, 'standard_1005')
            if eeg != channels:
                raise ValueError(f'{path.name}: channel set differs within the subject')
            data = epochs.get_data(picks=channels)
            modality = phase(data.shape[-1] / epochs.info['sfreq'], path.name)
            words = epochs.metadata['Word'].astype(str).str.strip().tolist()
            for x, text in zip(data, words):
                category = textmaps.get(text, -1)
                segments.append(dict(eeg=full(x, epochs.info['sfreq'], found, mains=50, scale=1e6), valid=found,
                                     session=run, modality=modality, item=names[category] if category >= 0 else None,
                                     text=text))
        writer = StoreWriter(part, **meta)
        writer.add_subject(f'sub-{subject}', channels, xyz, segments)
        writer.close()
        print(f'sub-{subject}', len(segments), flush=True)
        if not keep_raw:
            shutil.rmtree(folder)                         # re-downloadable; keeps the disk footprint to one subject
    missing = [s for s in CHISCO_SUBJECTS if not (parts / f'sub-{s}.h5').exists()]
    if missing:
        print(f'parts still missing for sub-{", sub-".join(missing)}; not merged yet')
        return False
    merge(sorted(parts.glob('sub-*.h5')), out)
    shutil.rmtree(parts)
    return True


def phase(seconds, name=''):
    """Chisco trial phase of an epoch by its length: imagery 3.3 s, reading 5.0 s."""
    if abs(seconds - 3.3) < .2:
        return 'imagine'
    if abs(seconds - 5.0) < .2:
        return 'read'
    raise ValueError(f'{name}: {seconds:.2f} s epochs are neither the imagery (3.3 s) nor the reading (5.0 s) phase')


def chisco_relabel(path):
    """Mark the reading epochs of a Chisco store built before 2026-10-06 as modality 'read' (in place).

    Only the segment tables and the root attributes change; the original tables are saved first to
    ``<store>_segments_before_relabel.npz``."""
    with h5py.File(path, 'r+') as f:
        modalities = json.loads(f.attrs['modalities'])
        if 'read' in modalities:
            print(f'{path}: already relabelled')
            return
        rate = float(f.attrs['rate'])
        tables = {s: f['subjects'][s]['segments'][:] for s in f['subjects']}
        np.savez(path.with_name(f'{path.stem}_segments_before_relabel.npz'), **tables)
        read = len(modalities)
        for subject, table in tables.items():
            kind = [phase(n / rate, subject) for n in table['length']]
            table['modality'] = np.where(np.array(kind) == 'read', read, table['modality'])
            f['subjects'][subject]['segments'][...] = table
            print(subject, f'{kind.count("imagine")} imagine, {kind.count("read")} read')
        f.attrs['modalities'] = json.dumps(modalities + ['read'])
        f.attrs['notes'] = str(f.attrs['notes']).replace('imagery window 5.0-8.3 s of each trial',
                                                         'imagine = imagery phase 5.0-8.3 s of each trial, '
                                                         'read = reading phase 0-5 s')


def merge(parts, out):
    """One store from per-subject part stores with identical item / modality lists."""
    tmp = out.with_suffix('.h5.tmp')
    with h5py.File(tmp, 'w') as target:
        target.create_group('subjects')
        for i, path in enumerate(parts):
            with h5py.File(path, 'r') as source:
                if i == 0:
                    for key, value in source.attrs.items():
                        target.attrs[key] = value
                    source.copy('stimuli', target)
                for key in ('items', 'modalities'):
                    if source.attrs[key] != target.attrs[key]:
                        raise ValueError(f'{path}: {key} differ from the first part')
                for subject in source['subjects']:
                    source.copy(f'subjects/{subject}', target['subjects'])
    tmp.replace(out)


SOURCES = dict(thinking_out_loud=thinking_out_loud, bci2020=bci2020, chisco=chisco)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', choices=sorted(SOURCES))
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--keep-raw', action='store_true', help='chisco: keep the downloaded raw files')
    parser.add_argument('--subjects', nargs='*', help='chisco: only these subjects, e.g. --subjects 03')
    parser.add_argument('--relabel', action='store_true', help='chisco: fix the reading epochs of an existing store')
    args = parser.parse_args()
    target = STORE / f'{args.dataset}.h5'
    if args.relabel:
        chisco_relabel(target)
        sys.exit(0)
    if target.exists() and not args.force:
        raise SystemExit(f'{target} exists (use --force to rebuild)')
    if args.dataset == 'chisco':
        if chisco(target, keep_raw=args.keep_raw, subjects=args.subjects):
            print('wrote', target)
    else:
        SOURCES[args.dataset](target)
        print('wrote', target)

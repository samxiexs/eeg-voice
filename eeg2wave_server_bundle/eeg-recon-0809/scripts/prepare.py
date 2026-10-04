"""Convert every dataset into the common store (``artifacts/store/<name>.h5``).

    python scripts/prepare.py <dataset> [--force]

Common conventions: microvolts (SparrKULee keeps its normalised units), average
reference over valid channels, 0.5-45 Hz where the source allows (Marion and
SparrKULee are shipped at 64 Hz, so they keep <= 32 Hz), 128 Hz except the two
64 Hz sources.  Trials keep only the task window (the action / imagery interval);
rest recordings are kept as unlabelled ``rest`` segments for alignment and
person calibration.

Sources: ``data/raw/<dataset>`` from ``scripts/download.py``; DS004940, Broderick,
Marion and KaraOne are read from the harmonised shards built earlier
(``artifacts/legacy``), whose raw releases are no longer on disk.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import pickle
import re
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegspeech import ROOT, STORE                                                    # noqa: E402
from eegspeech.signal import (BIOSEMI64, BIOSEMI128, FEATURE_ROWS, bandpass, positions,    # noqa: E402
                              resample, speech_features)
from eegspeech.store import StoreWriter                                             # noqa: E402

RAW = ROOT / 'data' / 'raw'
LEGACY = ROOT / 'artifacts' / 'legacy'
RATE = 128
LOWPASS = 45.


def reref(x, valid):
    """Average reference over valid channels; invalid channels set to zero."""
    x = x - x[valid].mean(0, keepdims=True)
    x[~valid] = 0
    return x


def clean(x, rate, valid, *, lowpass=LOWPASS, target=RATE, scale=1.):
    """Scale to microvolts, low-pass, re-reference, resample.  x (C, T) at ``rate``."""
    x = np.asarray(x, np.float64) * scale
    if lowpass and lowpass < rate / 2:
        x = bandpass(x, rate, high=lowpass)
    return resample(reref(x, valid), rate, target).astype(np.float32)


def audio_features(path):
    import soundfile as sf
    wave, rate = sf.read(path, always_2d=False)
    return speech_features(wave, rate)


# --- listening -------------------------------------------------------------------------------

def ds004940(out):
    """N400 sentences (Toffolo et al.), 128-ch BioSemi; EEG epochs from 0.25 s before each spoken sentence."""
    files = sorted(glob.glob(str(LEGACY / 'ds004940' / '**' / '*.h5'), recursive=True))
    by_subject = {}
    for path in files:
        subject = re.search(r'(sub-\d+)', path).group(1)
        by_subject.setdefault(subject, []).append(path)
    writer = StoreWriter(out, name='ds004940', rate=RATE, band=[.5, LOWPASS], reference='average', unit='uV',
                         language='en', modalities=['listen'],
                         notes='OpenNeuro ds004940 v1.0.1, N400Active (session 0) and N400Passive (session 1); '
                               'segment offset -0.25 s = EEG starts 0.25 s before sentence onset')
    stimuli = ROOT / 'data' / 'ds004940' / 'stimuli'
    for subject, paths in sorted(by_subject.items()):
        segments, xyz, seen = [], None, set()
        for path in sorted(paths):
            task = re.search(r'task-(\w+)\.h5', path).group(1)
            if task in seen:
                continue
            seen.add(task)
            with h5py.File(path, 'r') as f:
                xyz = f['channel_xyz'][:]
                contents = [c.decode() if isinstance(c, bytes) else c for c in f['provenance/linguistic_content_id'][:]]
                masks = f['channel_valid_mask'][:]
                for i in range(f['eeg'].shape[0]):
                    content = contents[i].split(':')[-1]
                    if writer.stimuli.count(content) == 0:
                        wav = stimuli / f'{content}.wav'
                        if not wav.exists():
                            continue
                        writer.add_stimulus(content, audio_features(wav), FEATURE_ROWS)
                    valid = masks[i].astype(bool)
                    segments.append(dict(eeg=clean(f['eeg'][i], 256, valid, scale=1e6), valid=valid,
                                         session=0 if task == 'N400Active' else 1, modality='listen',
                                         stimulus=content, offset=-.25))
        writer.add_subject(subject, BIOSEMI128, xyz, segments)
        print(subject, len(segments), flush=True)
    writer.close()


def broderick(out):
    """Broderick et al. 2018 natural speech: 19 people x 20 audiobook runs, 128-ch BioSemi, already 0.5-45 Hz, 128 Hz."""
    import pandas as pd
    manifest = pd.read_csv(LEGACY / 'broderick2018' / 'manifest.csv')
    writer = StoreWriter(out, name='broderick2018', rate=RATE, band=[.5, LOWPASS], reference='average', unit='uV',
                         language='en', modalities=['listen'],
                         notes='Dryad doi:10.5061/dryad.070jc; audio data/ds004408/stimuli; offset -0.5 s')
    audio = ROOT / 'data' / 'ds004408' / 'stimuli'
    for piece in range(1, 21):
        writer.add_stimulus(f'audio{piece:02d}', audio_features(audio / f'audio{piece:02d}.wav'), FEATURE_ROWS)
    for path in sorted(glob.glob(str(LEGACY / 'broderick2018' / 'shards' / '*.h5'))):
        subject = Path(path).stem
        kept = set(manifest[manifest.subject == subject].trial)          # 12 runs of Subject9 were excluded (bad electrodes)
        segments = []
        with h5py.File(path, 'r') as f:
            xyz = f['channel_xyz'][:]
            for key in sorted(f['trials']):
                if int(key) not in kept:
                    continue
                ds = f['trials'][key]
                valid = f['trials_valid'][key][:].astype(bool)
                x = ds[:].astype(np.float32)
                x[~valid] = 0
                segments.append(dict(eeg=x, valid=valid, modality='listen', stimulus=f'audio{int(ds.attrs["piece"]):02d}',
                                     offset=-int(ds.attrs['onset_sample']) / RATE))
        writer.add_subject(subject, BIOSEMI128, xyz, segments)
        print(subject, len(segments), flush=True)
    writer.close()


# --- paired listening / imagery -----------------------------------------------------------------

def marion(out):
    """Marion et al. 2021: 21 musicians listen to and imagine 4 Bach chorale melodies (11 repeats each), metronome-paced."""
    writer = StoreWriter(out, name='marion2021', rate=64, band=[.1, 30], reference='average', unit='as shipped',
                         language='music', modalities=['listen', 'imagine'],
                         notes='authors\' preprocessed release (0.1-30 Hz, 64 Hz); stimulus rows = note onsets, '
                               'melodic expectation; trials start at melody onset')
    xyz, _ = positions(BIOSEMI64, 'biosemi64')
    with h5py.File(LEGACY / 'marion2021' / 'imagery.h5', 'r') as f:
        eeg, stim = f['eeg'], f['stim']
        patterns = {}
        for s in range(eeg.shape[0]):
            segments = []
            for c, condition in enumerate(['listen', 'imagine']):
                for t in range(eeg.shape[2]):
                    expectation = stim[s, c, t].astype(np.float32)
                    key = tuple(np.flatnonzero(expectation))
                    melody = patterns.setdefault(key, len(patterns))
                    name = f'melody{melody}'
                    if name not in writer.stimuli:
                        writer.add_stimulus(name, np.stack([(expectation != 0).astype(np.float32), expectation]),
                                            ['onset', 'expectation'])
                    x = eeg[s, c, t].T.astype(np.float32)
                    valid = np.ones(64, bool)
                    segments.append(dict(eeg=reref(x - x.mean(1, keepdims=True), valid), valid=valid, modality=condition,
                                         item=name, stimulus=name, offset=0.))
            writer.add_subject(f'sub-{s + 1:02d}', BIOSEMI64, xyz, segments)
        if len(patterns) != 4:
            raise SystemExit(f'expected 4 melody onset patterns, found {len(patterns)}')
    writer.close()


def karaone(out):
    """KaraOne (Zhao & Rudzicz 2015): 7 phonemic prompts + 4 words; cue (seen and heard), imagined, spoken."""
    writer = StoreWriter(out, name='karaone', rate=RATE, band=[.5, LOWPASS], reference='average', unit='uV',
                         language='en', modalities=['cue', 'imagine', 'overt', 'rest'],
                         notes='62-ch Neuroscan; phases from the authors\' epoch files; cue = prompt shown and played')
    phases = dict(stimulus='cue', thinking='imagine', speaking='overt', clearing='rest')
    for path in sorted(glob.glob(str(LEGACY / 'karaone' / 'shards' / '*.h5'))):
        with h5py.File(path, 'r') as f:
            channels = json.loads(f.attrs['channel_names'])
            xyz = f['channel_xyz'][:]
            raw = f['eeg']
            trials = {k: f['trials'][k][:] for k in f['trials']}
            valid = np.ones(len(channels), bool)
            segments = []
            for i in range(len(trials['trial'])):
                prompt = trials['prompt'][i]
                prompt = prompt.decode() if isinstance(prompt, bytes) else str(prompt)
                for phase, modality in phases.items():
                    ok = trials[f'{phase}_valid'][i]
                    ok = ok.decode() == 'True' if isinstance(ok, bytes) else bool(ok)
                    start, end = int(trials[f'{phase}_start'][i]), int(trials[f'{phase}_end'][i])
                    if not ok or end - start < 128:
                        continue
                    x = raw[:, start:end]
                    segments.append(dict(eeg=clean(x, 256, valid, lowpass=None, scale=1e6), valid=valid,
                                         modality=modality, item=None if modality == 'rest' else prompt))
        writer.add_subject(Path(path).stem, [c.title() for c in channels], xyz, segments)
        print(Path(path).stem, len(segments), flush=True)
    writer.close()


def thinking_out_loud(out):
    """Nieto et al. 2022 (OpenNeuro ds003626): 10 people, 4 Spanish words, pronounced / inner / visualised, 128-ch BioSemi.

    Epochs run -0.5..4 s from the cue; the action interval is 1.0-3.5 s (facial EMG of the
    pronounced condition rises at ~1 s).  Inner / visualised trials the authors flag for EMG
    are dropped.
    """
    import mne
    mne.set_log_level('ERROR')
    words = ['arriba', 'abajo', 'derecha', 'izquierda']
    conditions = {0: 'overt', 1: 'imagine', 2: 'visual'}
    writer = StoreWriter(out, name='thinking_out_loud', rate=RATE, band=[.5, LOWPASS], reference='average', unit='uV',
                         language='es', items=words, modalities=['overt', 'imagine', 'visual', 'rest'],
                         notes='trial window 1.0-3.5 s after the cue; rest = 15 s baseline per session')
    xyz, found = positions(BIOSEMI128, 'biosemi128')
    root = RAW / 'ds003626' / 'derivatives'
    for subject_dir in sorted(root.glob('sub-*')):
        segments = []
        for session_dir in sorted(subject_dir.glob('ses-*')):
            base = session_dir / f'{subject_dir.name}_{session_dir.name}_'
            epochs = mne.read_epochs(f'{base}eeg-epo.fif').pick(BIOSEMI128)
            events = np.load(f'{base}events.dat', allow_pickle=True)
            report = pickle.load(open(f'{base}report.pkl', 'rb'))
            emg = set(int(i) for i in np.atleast_1d(report.get('EMG_trials', [])))
            data = epochs.get_data()
            rate = epochs.info['sfreq']
            first, last = int(round((1.0 - epochs.tmin) * rate)), int(round((3.5 - epochs.tmin) * rate))
            session = int(session_dir.name.split('-')[1])
            for i, (_, word, condition, _) in enumerate(events.astype(int)):
                if i in emg and condition != 0:
                    continue
                x = clean(data[i], rate, found, scale=1e6)                     # filter the whole epoch, then crop
                a, b = int(first * RATE / rate), int(last * RATE / rate)
                segments.append(dict(eeg=x[:, a:b], valid=found, session=session, modality=conditions[condition],
                                     item=words[word]))
            baseline = mne.read_epochs(f'{base}baseline-epo.fif').pick(BIOSEMI128).get_data()[0]
            segments.append(dict(eeg=clean(baseline, rate, found, scale=1e6), valid=found, session=session,
                                 modality='rest'))
        writer.add_subject(subject_dir.name, BIOSEMI128, xyz, segments)
        print(subject_dir.name, len(segments), flush=True)
    writer.close()


def cpseed(out):
    """3M-CPSEED (Ma et al. 2025, OpenNeuro ds006465): 20 people, 10 Mandarin Pinyin syllables, overt / mouthed / imagined.

    The authors' preprocessed epochs (4-45 Hz, ICA-cleaned, 500 Hz) hold each 6 s phase
    as three 2 s epochs stored block-wise (epoch j, j + 50, j + 100 are contiguous in
    time; checked from sample continuity), so the three are rejoined into one 6 s trial.
    Labels are the ``trial_type`` codes 1-10 of the raw events (one event per trial);
    their syllable names are not published with the data, so items stay as codes.
    Converted: the 32-ch Enobio people (sub-01..15) except sub-02 (its session files are duplicates holding
    200 epochs for 50 events) and sub-10 (preprocessed data shipped as continuous EDF only).  Not converted:
    sub-16..20 (128-ch Neuracle), whose preprocessed files hold 32 unnamed channels: their correlation structure
    matches neither the Enobio order nor the 32 names in 128-ch order, so electrode positions are unknown.
    """
    import pandas as pd
    import scipy.io as sio
    modes = dict(speak='overt', intend='mouthed', imagine='imagine')
    writer = StoreWriter(out, name='cpseed', rate=RATE, band=[4, LOWPASS], reference='average', unit='uV', language='zh',
                         items=[f'p{k:02d}' for k in range(1, 11)], modalities=['overt', 'mouthed', 'imagine'],
                         notes='items p01..p10 = raw trial_type codes (Pinyin a, i, u, u-umlaut, m, f, j, l, k, ch in '
                               'some order; the mapping is not published)')
    root = RAW / 'ds006465'
    dropped = {'ECG', 'HEOR', 'HEOL', 'VEOU', 'VEOL'}
    for subject_dir in sorted((root / 'derivatives' / 'preproc').glob('sub-*')):
        subject = subject_dir.name
        if int(subject.split('-')[1]) >= 16:
            print(f'{subject}: 128-ch recording shipped as 32 unidentified channels, skipped')
            continue
        segments, channels, digests = [], None, set()
        for session_dir in sorted(subject_dir.glob('ses-*')):
            session = int(session_dir.name.split('-')[1])
            raw_dir = root / subject / session_dir.name / 'eeg'
            events_file = next(raw_dir.glob('*_events.tsv'), None)
            channels_file = (next(raw_dir.glob('*_channels.tsv'), None)                # sub-03 ses-2 lacks one;
                             or next((root / subject).glob('ses-*/eeg/*_channels.tsv'), None))   # same cap all sessions
            if events_file is None or channels_file is None:
                print(f'  {subject} {session_dir.name}: no events / channels file, skipped')
                continue
            events = pd.read_csv(events_file, sep='\t')
            table = pd.read_csv(channels_file, sep='\t')
            names = [n for n, t in zip(table['name'], table['type']) if str(t).upper() == 'EEG' and n not in dropped]
            codes = events['trial_type'].astype(int).to_numpy()
            for mode, modality in modes.items():
                path = next(session_dir.glob(f'*_{mode}.mat'), None)
                if path is None:
                    continue
                data = sio.loadmat(path)['data'].astype(np.float64)             # C, 1000, 3 * trials
                digest = hash(data[:, :50, :5].tobytes())
                if digest in digests:                                           # a session file shipped twice
                    print(f'  {path.name}: duplicate of an earlier session, skipped')
                    continue
                digests.add(digest)
                trials = data.shape[2] // 3
                if trials != len(codes) or data.shape[0] != len(names):
                    print(f'  {path.name}: {data.shape} vs {len(codes)} events / {len(names)} channels, skipped')
                    continue
                channels = channels or names
                if names != channels:
                    raise ValueError(f'{subject}: channel set changes between sessions')
                xyz, found = positions(names, 'standard_1005')
                for j in range(trials):
                    x = np.concatenate([data[:, :, j], data[:, :, j + trials], data[:, :, j + 2 * trials]], 1)
                    segments.append(dict(eeg=clean(x, 500, found, lowpass=None), valid=found, session=session,
                                         modality=modality, item=f'p{codes[j]:02d}'))
        if segments:
            writer.add_subject(subject, channels, positions(channels, 'standard_1005')[0], segments)
        print(subject, len(segments), flush=True)
    writer.close()


def bci2020(out):
    """2020 International BCI Competition, Track 3 (OSF pq7vb): 15 people imagine 5 phrases, 64-ch BrainAmp.

    Each auditory cue was followed by four cross / imagine (2 s) cycles; the released
    epochs run -0.5..2.6 s around imagery onset and the 0-2 s imagery window is kept.
    Sessions: 0 training, 1 validation, 2 test (labels from the published answer sheet).
    """
    import pandas as pd
    import scipy.io as sio
    names = ['hello', 'help me', 'stop', 'thank you', 'yes']
    writer = StoreWriter(out, name='bci2020', rate=RATE, band=[.5, LOWPASS], reference='average', unit='uV', language='en',
                         items=names, modalities=['imagine'],
                         notes='imagery window 0-2 s; session 0 train / 1 validation / 2 test')
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
            onset = int(round(np.argmax(t >= 0) * RATE / fs))               # imagery onset, in samples at RATE
            for i in range(len(x)):
                segments.append(dict(eeg=clean(x[i], fs, found)[:, onset:onset + 2 * RATE], valid=found,
                                     session=session, modality='imagine', item=names[int(y[i])]))
        writer.add_subject(f'sub-{k:02d}', channels, positions(channels, 'standard_1005')[0], segments)
        print(f'sub-{k:02d}', len(segments), flush=True)
    writer.close()


def sparrkulee(out):
    """SparrKULee (Accou et al. 2024, KU Leuven RDR doi:10.48804/K3VSND): 85 people, 64-ch BioSemi, 168 h of
    audiobooks and podcasts.  Uses the authors' preprocessed EEG (64 Hz, aligned to stimulus onset, so offset 0)
    and computes the stimulus features from the shipped 48 kHz audio.  The three recordings the README lists as
    unalignable are skipped.
    """
    import gzip
    import io
    root = RAW / 'sparrkulee'
    unaligned = {('sub-006', 'shortstories01', '06'), ('sub-017', 'shortstories01', '03'), ('sub-048', 'varyingStories05', '04')}
    writer = StoreWriter(out, name='sparrkulee', rate=64, band=[.5, 32], reference='as shipped', unit='as shipped',
                         language='nl', modalities=['listen'],
                         notes='authors\' preprocessed EEG (64 Hz), sample 0 = stimulus onset; Dutch audiobooks / podcasts')
    xyz, _ = positions(BIOSEMI64, 'biosemi64')
    pattern = re.compile(r'(sub-\d+)_ses-(\w+?)_task-\w+_run-(\d+)_desc-preproc-audio-(.+)_eeg\.npy')
    files = sorted((root / 'derivatives' / 'preprocessed_eeg').glob('sub-*/*/*_eeg.npy'))
    by_subject = {}
    for path in files:
        match = pattern.match(path.name)
        if match is None or match.groups()[:3] in unaligned:
            continue
        by_subject.setdefault(match.group(1), []).append((path, match.group(2), match.group(4)))
    sessions = {}
    for subject, recordings in sorted(by_subject.items()):
        segments = []
        for path, session, stimulus in recordings:
            if stimulus not in writer.stimuli:
                audio = root / 'stimuli' / 'eeg' / f'{stimulus}.npz.gz'
                if not audio.exists():
                    print(f'  {stimulus}: no audio, skipped')
                    continue
                with gzip.open(audio) as handle:
                    shipped = np.load(io.BytesIO(handle.read()))
                    writer.add_stimulus(stimulus, speech_features(shipped['audio'], int(shipped['fs'])), FEATURE_ROWS)
            x = np.load(path).astype(np.float32)
            x = x.T if x.shape[0] != 64 else x                                  # shipped as (time, channels)
            valid = np.ones(64, bool)
            segments.append(dict(eeg=x, valid=valid, session=sessions.setdefault(session, len(sessions)),
                                 modality='listen', stimulus=stimulus, offset=0.))
        writer.add_subject(subject, BIOSEMI64, xyz, segments)
        print(subject, len(segments), flush=True)
    writer.meta['notes'] += '; sessions: ' + json.dumps(sessions)
    writer.close()


def chisco(out, keep_raw=False):
    """Chisco (Zhang et al. 2024, OpenNeuro ds005170): 5 people imagine ~6,600 everyday Chinese sentences, 122-ch.

    The authors' preprocessed epochs (5.0-8.3 s of each trial: the imagery phase, 500 Hz)
    carry the sentence as metadata; items are the 39 semantic categories of
    ``json/textmaps.json`` (sentences without a category stay unlabelled), the sentence
    is kept as segment text.  Subjects are downloaded, converted to a part file and their
    raw files removed one at a time (61 GB of fif in total, unless ``--keep-raw``); a rerun
    skips finished parts, and the parts are merged at the end.
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
    meta = dict(name='chisco', rate=RATE, band=[.5, LOWPASS], reference='average', unit='uV', language='zh',
                items=names, modalities=['imagine'],
                notes='imagery window 5.0-8.3 s of each trial; items = semantic category; session = run')
    parts = out.parent / 'chisco.parts'
    for subject in ['01', '02', '03', '04', '05']:
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
            words = epochs.metadata['Word'].astype(str).str.strip().tolist()
            for x, text in zip(data, words):
                category = textmaps.get(text, -1)
                segments.append(dict(eeg=clean(x, epochs.info['sfreq'], found, scale=1e6), valid=found, session=run,
                                     modality='imagine', item=names[category] if category >= 0 else None, text=text))
        writer = StoreWriter(part, **meta)
        writer.add_subject(f'sub-{subject}', channels, xyz, segments)
        writer.close()
        print(f'sub-{subject}', len(segments), flush=True)
        if not keep_raw:
            shutil.rmtree(folder)                         # re-downloadable; keeps the disk footprint to one subject
    merge(sorted(parts.glob('sub-*.h5')), out)
    shutil.rmtree(parts)


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


SOURCES = dict(ds004940=ds004940, broderick2018=broderick, marion2021=marion, karaone=karaone,
               thinking_out_loud=thinking_out_loud, cpseed=cpseed, bci2020=bci2020, sparrkulee=sparrkulee, chisco=chisco)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', choices=sorted(SOURCES))
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--keep-raw', action='store_true', help='chisco: keep the downloaded fif files')
    args = parser.parse_args()
    target = STORE / f'{args.dataset}.h5'
    if target.exists() and not args.force:
        raise SystemExit(f'{target} exists (use --force to rebuild)')
    if args.dataset == 'chisco':
        chisco(target, keep_raw=args.keep_raw)
    else:
        SOURCES[args.dataset](target)
    print('wrote', target)

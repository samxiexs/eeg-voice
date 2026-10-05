"""Convert a downloaded dataset into the common store (``artifacts/store/<name>.h5``).

    python scripts/prepare.py <dataset> [--force]
    python scripts/prepare.py karaone_voices          # the KaraOne participants' own speech recordings

Common conventions: microvolts (SparrKULee keeps its normalised units) and average reference over
valid channels.  Imagined-speech datasets keep the full band (0.5-120 Hz at 256 Hz; CPSEED ships
4-45 Hz); SparrKULee is shipped at 64 Hz.  Trials keep only the task window (the action / imagery
interval); rest recordings are kept as unlabelled ``rest`` segments for alignment.

Sources: ``data/raw/<dataset>`` from ``scripts/download.py``.  The ds004940, broderick2018 and
marion2021 stores were converted from the previous project's caches, which are no longer on disk.
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
from eegspeech.signal import (BIOSEMI64, BIOSEMI128, FEATURE_ROWS, bandpass, notch, positions,    # noqa: E402
                              resample, speech_features)
from eegspeech.store import StoreWriter                                             # noqa: E402

RAW = ROOT / 'data' / 'raw'
RATE = 128                      # band-limited sources (CPSEED ships 4-45 Hz)
LOWPASS = 45.
FULL_RATE = 256                 # imagery stores keep the full band: their decodable information is mostly > 45 Hz
FULL_BAND = (.5, 120.)


def reref(x, valid):
    """Average reference over valid channels; invalid channels set to zero."""
    x = x - x[valid].mean(0, keepdims=True)
    x[~valid] = 0
    return x


def clean(x, rate, valid, *, lowpass=LOWPASS, target=RATE, scale=1., highpass=None, mains=None):
    """Scale to microvolts, notch the mains and its harmonics, band-limit, re-reference, resample.  x (C, T)."""
    x = np.asarray(x, np.float64) * scale
    for f in (np.arange(mains, rate / 2, mains) if mains else []):
        x = notch(x, rate, f)
    lowpass = min(lowpass or rate, .45 * min(rate, target))           # never above the target's anti-alias band
    x = bandpass(x, rate, low=highpass, high=lowpass)
    return resample(reref(x, valid), rate, target).astype(np.float32)


def full(x, rate, valid, *, mains, scale=1.):
    """The imagery-store version of ``clean``: 0.5-120 Hz at 256 Hz, mains and harmonics notched."""
    return clean(x, rate, valid, lowpass=FULL_BAND[1], highpass=FULL_BAND[0], target=FULL_RATE, scale=scale, mains=mains)


KARAONE_STAGES = dict(stimulus='cue', thinking='imagine', speaking='overt', clearing='rest')
KARAONE_PROMPTS = ['/iy/', '/uw/', '/piy/', '/tiy/', '/diy/', '/m/', '/n/', 'pat', 'pot', 'knew', 'gnaw']


def karaone_stages(path):
    """One row per trial with [start, end) samples (1 kHz) of each stage, from the authors' epoch_inds.mat.

    ``speaking_inds`` nominally alternates cue / speaking but some files are shifted, so each
    stage is located relative to the reliable clearing / thinking boundaries; -1 marks a stage
    that cannot be located.
    """
    import scipy.io as sio
    mat = sio.loadmat(path)
    pairs = lambda key: [tuple(int(v) for v in np.asarray(c).ravel()[:2]) for c in mat[key].ravel()]
    clearing, thinking = pairs('clearing_inds'), pairs('thinking_inds')
    mixed = sorted({m for m in pairs('speaking_inds') if m[1] > m[0]})
    rows = []
    for i, (c, t) in enumerate(zip(clearing, thinking)):
        following = clearing[i + 1][0] if i + 1 < len(clearing) else float('inf')
        ok = c[0] < c[1] <= t[0] < t[1]
        cue = [m for m in mixed if c[1] <= m[0] and m[1] <= t[0]]
        speak = [m for m in mixed if t[1] <= m[0] and m[1] <= following]
        rows.append(dict(clearing=c if ok else None, thinking=t if ok else None,
                         stimulus=cue[0] if ok and len(cue) == 1 else None,
                         speaking=speak[0] if ok and len(speak) == 1 else None))
    return rows


def karaone(out, keep_raw=False):
    """KaraOne (Zhao & Rudzicz 2015): 14 people, 7 phonemic prompts + 4 words; cue (prompt shown and played),
    imagined and spoken, 62-ch Neuroscan at 1 kHz.

    Built from the raw .cnt at full band (0.5-120 Hz, 60 Hz and harmonics notched, 256 Hz),
    one participant at a time: download, extract only the EEG and the stage indices, convert
    to a part file, delete the archive (``--keep-raw`` keeps it); parts are merged at the end.
    """
    import shutil
    import tarfile
    import mne
    sys.path.insert(0, str(ROOT / 'scripts'))
    import download
    mne.set_log_level('ERROR')
    meta = dict(name='karaone', rate=FULL_RATE, band=list(FULL_BAND), reference='average', unit='uV', language='en',
                items=KARAONE_PROMPTS, modalities=['cue', 'imagine', 'overt', 'rest'],
                notes='raw .cnt at full band; stages from epoch_inds.mat; cue = prompt shown and played')
    root, parts = RAW / 'karaone', out.parent / 'karaone.parts'
    for person in download.KARAONE_PEOPLE:
        part = parts / f'{person}.h5'
        if part.exists():
            continue
        download.run(download.karaone(argparse.Namespace(subject=person)), workers=1)
        with tarfile.open(root / f'{person}.tar.bz2', 'r:bz2') as archive:
            wanted = [m for m in archive if m.isfile() and (m.name.endswith('.cnt') or m.name.endswith('epoch_inds.mat')
                                                            or m.name.endswith('kinect_data/labels.txt'))]
            archive.extractall(root, members=wanted, filter='data')
            top = root / wanted[0].name.split('/')[0]                  # archives nest as p/spoclab/.../<person>
        folder = next(p for p in top.rglob(person) if p.is_dir())
        cnt = next(folder.rglob('*.cnt'))
        labels = [l.strip() for l in next(folder.rglob('labels.txt')).read_text().splitlines() if l.strip()]
        stages = karaone_stages(next(folder.rglob('epoch_inds.mat')))
        raw = mne.io.read_raw_cnt(cnt, preload=True)
        aux = {'M1', 'M2', 'VEO', 'HEO', 'EKG', 'EMG', 'TRIGGER'}
        names = [c for c in raw.ch_names if c.upper() not in aux]
        channels = [c.title() for c in names]
        xyz, found = positions(channels, 'standard_1005')
        source = raw.info['sfreq']
        eeg = full(raw.get_data(picks=names), source, found, mains=60, scale=1e6)       # whole recording at 256 Hz
        to_rate = lambda sample: int(round(sample * FULL_RATE / source))
        segments = []
        for trial, (label, row) in enumerate(zip(labels, stages)):
            for stage, modality in KARAONE_STAGES.items():
                span = row[stage]
                if span is None or span[1] - span[0] < .5 * source:
                    continue
                a, b = to_rate(span[0]), to_rate(span[1])
                segments.append(dict(eeg=eeg[:, a:b], valid=found, modality=modality,
                                     item=None if modality == 'rest' else label))
        writer = StoreWriter(part, **meta)
        writer.add_subject(person, channels, xyz, segments)
        writer.close()
        print(person, len(segments), flush=True)
        shutil.rmtree(top)
        if not keep_raw:
            (root / f'{person}.tar.bz2').unlink()
    merge(sorted(parts.glob('*.h5')), out)
    shutil.rmtree(parts)
    return True


def karaone_voices(out, keep_raw=False):
    """KaraOne participants' own speech: the Kinect recording of every spoken trial (16 kHz, one wav per
    trial, ``labels.txt`` in trial order) -> ``artifacts/audio/karaone/persons/<person>/``, the voice each
    person's reconstruction speaks in.  Archives are downloaded one at a time and deleted afterwards."""
    import tarfile
    sys.path.insert(0, str(ROOT / 'scripts'))
    import download
    for person in download.KARAONE_PEOPLE:
        folder = out / person
        if (folder / 'labels.txt').exists():                     # written last: the person is complete
            continue
        download.run(download.karaone(argparse.Namespace(subject=person)), workers=1)
        archive_path = RAW / 'karaone' / f'{person}.tar.bz2'
        folder.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, 'r:bz2') as archive:
            members = [m for m in archive if m.isfile() and '/kinect_data/' in m.name
                       and (m.name.endswith('.wav') or m.name.endswith('/labels.txt'))]
            for m in sorted(members, key=lambda m: m.name.endswith('labels.txt')):
                (folder / Path(m.name).name).write_bytes(archive.extractfile(m).read())
        print(person, len(members) - 1, 'recordings', flush=True)
        if not keep_raw:
            archive_path.unlink()
    return True


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


def cpseed(out):
    """3M-CPSEED (Ma et al. 2025, OpenNeuro ds006465): 10 Mandarin Pinyin syllables, overt / mouthed / imagined.

    The authors' preprocessed epochs (4-45 Hz, ICA-cleaned, 500 Hz) hold each 6 s phase
    as three 2 s epochs stored block-wise (epoch j, j + 50, j + 100 are contiguous in
    time; checked from sample continuity), so the three are rejoined into one 6 s trial.
    Labels are the ``trial_type`` codes 1-10 of the raw events (one event per trial);
    their syllable names are not published with the data, so items stay as codes.

    Channel order: the .mat files keep the raw EDF order, not the channels.tsv order
    (checked against inter-electrode distance: r = 0.75 vs 0.09).  sub-01..15 (32-ch Enobio)
    use their EDF order; sub-16..20 (128-ch Neuracle) were reduced to the same 32 electrodes,
    kept in the Neuracle EDF order (r = 0.67 / 0.53 vs 0.35 / 0.24).  The raw EDF cannot
    replace these epochs: it carries no trigger channel and the events do not line up with them.
    Not converted: sub-02 (duplicated session files, 200 epochs for 50 events), sub-10 (no epochs).
    """
    import scipy.io as sio
    import pandas as pd
    modes = dict(speak='overt', intend='mouthed', imagine='imagine')
    writer = StoreWriter(out, name='cpseed', rate=RATE, band=[4, LOWPASS], reference='average', unit='uV', language='zh',
                         items=[f'p{k:02d}' for k in range(1, 11)], modalities=['overt', 'mouthed', 'imagine'],
                         notes='authors\' 4-45 Hz ICA-cleaned epochs (no high band available); items p01..p10 = raw '
                               'trial_type codes (Pinyin a, i, u, u-umlaut, m, f, j, l, k, ch in some order)')
    root = RAW / 'ds006465'
    edf = json.load(open(root / 'edf_channels.json'))
    enobio = [c for c in edf['sub-01'] if c != 'Status']
    for subject_dir in sorted((root / 'derivatives' / 'preproc').glob('sub-*')):
        subject = subject_dir.name
        order = [c for c in edf[subject] if c != 'Status']
        channels = order if len(order) == 32 else [c for c in order if c in enobio]
        if len(channels) != 32:
            print(f'{subject}: cannot name 32 channels, skipped')
            continue
        xyz, found = positions(channels, 'standard_1005')
        segments, digests = [], set()
        for session_dir in sorted(subject_dir.glob('ses-*')):
            session = int(session_dir.name.split('-')[1])
            events_file = next((root / subject / session_dir.name / 'eeg').glob('*_events.tsv'), None)
            if events_file is None:
                print(f'  {subject} {session_dir.name}: no events, skipped')
                continue
            codes = pd.read_csv(events_file, sep='\t')['trial_type'].astype(int).to_numpy()
            for mode, modality in modes.items():
                path = next(session_dir.glob(f'*_{mode}.mat'), None)
                if path is None:
                    continue
                data = sio.loadmat(path)['data'].astype(np.float64)             # 32, 1000, 3 * trials
                digest = hash(data[:, :50, :5].tobytes())
                if digest in digests:                                           # a session file shipped twice
                    print(f'  {path.name}: duplicate of an earlier session, skipped')
                    continue
                digests.add(digest)
                trials = data.shape[2] // 3
                if trials != len(codes) or data.shape[0] != 32:
                    print(f'  {path.name}: {data.shape} vs {len(codes)} events, skipped')
                    continue
                for j in range(trials):
                    x = np.concatenate([data[:, :, j], data[:, :, j + trials], data[:, :, j + 2 * trials]], 1)
                    segments.append(dict(eeg=clean(x, 500, found, lowpass=None), valid=found, session=session,
                                         modality=modality, item=f'p{codes[j]:02d}'))
        if segments:
            writer.add_subject(subject, channels, xyz, segments)
        print(subject, len(segments), flush=True)
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


CHISCO_SUBJECTS = ['01', '02', '03', '04', '05']


def chisco(out, keep_raw=False, subjects=None):
    """Chisco (Zhang et al. 2024, OpenNeuro ds005170): 5 people imagine ~6,600 everyday Chinese sentences, 122-ch.

    The authors' preprocessed epochs (5.0-8.3 s of each trial: the imagery phase, 500 Hz)
    carry the sentence as metadata; items are the 39 semantic categories of
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
                items=names, modalities=['imagine'],
                notes='authors\' epochs (1 Hz high-pass, 500 Hz) kept at full band; imagery window 5.0-8.3 s of '
                      'each trial; items = semantic category; session = run')
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
            words = epochs.metadata['Word'].astype(str).str.strip().tolist()
            for x, text in zip(data, words):
                category = textmaps.get(text, -1)
                segments.append(dict(eeg=full(x, epochs.info['sfreq'], found, mains=50, scale=1e6), valid=found,
                                     session=run, modality='imagine', item=names[category] if category >= 0 else None,
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


SOURCES = dict(karaone=karaone, thinking_out_loud=thinking_out_loud, cpseed=cpseed, bci2020=bci2020, sparrkulee=sparrkulee,
               chisco=chisco)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', choices=sorted(SOURCES) + ['karaone_voices'])
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--keep-raw', action='store_true', help='chisco / karaone: keep the downloaded raw files')
    parser.add_argument('--subjects', nargs='*', help='chisco: only these subjects, e.g. --subjects 03')
    args = parser.parse_args()
    if args.dataset == 'karaone_voices':
        karaone_voices(ROOT / 'artifacts' / 'audio' / 'karaone' / 'persons', keep_raw=args.keep_raw)
        sys.exit(0)
    target = STORE / f'{args.dataset}.h5'
    if target.exists() and not args.force:
        raise SystemExit(f'{target} exists (use --force to rebuild)')
    if args.dataset == 'chisco':
        if chisco(target, keep_raw=args.keep_raw, subjects=args.subjects):
            print('wrote', target)
    elif args.dataset == 'karaone':
        if karaone(target, keep_raw=args.keep_raw):
            print('wrote', target)
    else:
        SOURCES[args.dataset](target)
        print('wrote', target)

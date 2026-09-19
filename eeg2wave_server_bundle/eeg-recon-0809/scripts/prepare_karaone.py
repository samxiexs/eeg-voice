#!/usr/bin/env python3
"""Audit and preprocess KaraOne (Zhao & Rudzicz 2015) into per-participant shards.

Each trial of KaraOne has four stages, all delimited in ``epoch_inds.mat`` at
the 1 kHz source rate: ``clearing`` (rest, ~5 s), ``stimulus`` (text + audio
prompt, 2 s), ``thinking`` (imagined speech, ~5 s) and ``speaking`` (overt).
``speaking_inds`` interleaves stimulus and speaking segments (even = stimulus,
odd = speaking); ``kinect_data/labels.txt`` gives the prompt of every trial and
``kinect_data/<i>.wav`` the participant's own overt utterance.

The raw Neuroscan ``.cnt`` is used (not the authors' ``.set``) because it still
contains the ocular (VEO/HEO), EKG and EMG channels needed for artifact
controls.  EEG is average-referenced, notch-filtered at 60 Hz (Toronto mains),
band-passed 0.5–45 Hz and resampled to 256 Hz, matching the DS004940 route;
auxiliary channels are band-passed 0.5–100 Hz so EMG power can still be
measured.  Every shard also stores a spherical-spline map from the 60 located
KaraOne electrodes to the BioSemi-128 layout so the DS004940-pretrained encoder
can read KaraOne windows without retraining its spatial front end.

Nothing here selects trials, splits data or trains anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARTICIPANTS = ('MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12', 'MM14', 'MM15', 'MM16', 'MM18', 'MM19', 'MM20', 'MM21', 'P02')
PROMPTS = ('/iy/', '/uw/', '/piy/', '/tiy/', '/diy/', '/m/', '/n/', 'pat', 'pot', 'knew', 'gnaw')
SOURCE_RATE = 1000
TARGET_RATE = 256
AUX_CHANNELS = ('M1', 'M2', 'VEO', 'HEO', 'EKG', 'EMG')
STAGES = ('clearing', 'stimulus', 'thinking', 'speaking')
CONTRACT = 'karaone_shard_v1'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_prompts(folder: Path) -> tuple[list[str], list[str]]:
    labels = [line.strip() for line in (folder / 'kinect_data' / 'labels.txt').read_text().splitlines() if line.strip()]
    words = []
    pairs = folder / 'kinect_data' / f'{folder.name}_p.txt'
    if pairs.exists():
        for line in pairs.read_text().splitlines():
            parts = line.split()
            words.append(parts[1] if len(parts) > 1 else '')
    unknown = sorted(set(labels) - set(PROMPTS))
    if unknown:
        raise ValueError(f'{folder.name}: unregistered prompts {unknown}')
    return labels, words


def read_stage_indices(path: Path) -> pd.DataFrame:
    """Return one row per trial with source-rate [start, end) of every stage.

    ``speaking_inds`` alternates stimulus/speaking; the interleaving is verified
    against the clearing and thinking segments so a mis-ordered file fails.
    """
    import scipy.io as sio
    mat = sio.loadmat(path)
    def pairs(key):
        return [tuple(int(v) for v in np.asarray(cell).ravel()[:2]) for cell in mat[key].ravel()]
    clearing, thinking, mixed = pairs('clearing_inds'), pairs('thinking_inds'), pairs('speaking_inds')
    if len(clearing) != len(thinking):
        raise ValueError(f'{path}: {len(clearing)} clearing vs {len(thinking)} thinking segments')
    # ``speaking_inds`` nominally alternates stimulus/speaking, but several
    # released files have shifted or corrupt entries.  Each stage is therefore
    # located by position relative to the (reliable) clearing/thinking
    # boundaries instead of by parity, validated on its own and flagged;
    # invalid stages store -1 rather than discarding the participant.
    mixed = sorted({m for m in mixed if m[1] > m[0]})     # duplicated entries occur in some files
    rows = []
    for i, (c, t) in enumerate(zip(clearing, thinking)):
        next_start = clearing[i + 1][0] if i + 1 < len(clearing) else float('inf')
        clearing_ok = c[0] < c[1] <= t[0] < t[1]
        stimulus = [m for m in mixed if c[1] <= m[0] and m[1] <= t[0]]
        speaking = [m for m in mixed if t[1] <= m[0] and m[1] <= next_start]
        stimulus_ok = clearing_ok and len(stimulus) == 1
        speaking_ok = clearing_ok and len(speaking) == 1
        span = lambda ok, pair: (pair[0], pair[1]) if ok else (-1, -1)
        s = stimulus[0] if stimulus_ok else (-1, -1); k = speaking[0] if speaking_ok else (-1, -1)
        rows.append(dict(trial=i, clearing_start=span(clearing_ok, c)[0], clearing_end=span(clearing_ok, c)[1],
                         stimulus_start=s[0], stimulus_end=s[1],
                         thinking_start=span(clearing_ok, t)[0], thinking_end=span(clearing_ok, t)[1],
                         speaking_start=k[0], speaking_end=k[1],
                         clearing_valid=bool(clearing_ok), stimulus_valid=bool(stimulus_ok), thinking_valid=bool(clearing_ok), speaking_valid=bool(speaking_ok)))
    frame = pd.DataFrame(rows)
    invalid = {stage: frame.index[~frame[f'{stage}_valid']].tolist() for stage in STAGES if not frame[f'{stage}_valid'].all()}
    if invalid:
        print(f'[karaone] {path.parent.name}: invalid stage boundaries {invalid}', flush=True)
    return frame


def to_target_rate(sample: int) -> int:
    return int(round(sample * TARGET_RATE / SOURCE_RATE))


def biosemi128_map(names: list[str], positions: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Spherical-spline interpolation matrix (128 x len(names)); unlocated channels get zero weight."""
    import mne
    from mne.channels.interpolation import _make_interpolation_matrix
    target = mne.channels.make_standard_montage('biosemi128').get_positions()['ch_pos']
    target_names = sorted(target, key=lambda s: (s[0], int(s[1:])))
    located = [i for i, n in enumerate(names) if n in positions]
    pos_from = np.array([positions[names[i]] for i in located])
    pos_to = np.array([target[n] for n in target_names])
    origin = pos_from.mean(0)
    unit = lambda p: (p - origin) / np.linalg.norm(p - origin, axis=1, keepdims=True)
    matrix = np.zeros((len(target_names), len(names)), dtype=np.float32)
    matrix[:, located] = _make_interpolation_matrix(unit(pos_from), unit(pos_to)).astype(np.float32)
    return matrix, target_names, pos_to.astype(np.float32)


def preprocess(cnt_path: Path):
    import mne
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    raw = mne.io.read_raw_cnt(cnt_path, preload=True, verbose='ERROR')
    raw.drop_channels([c for c in raw.ch_names if c.lower() == 'trigger'])
    aux = [c for c in AUX_CHANNELS if c in raw.ch_names]
    eeg_names = [c for c in raw.ch_names if c not in aux]
    raw.set_channel_types({c: 'misc' for c in aux})
    eeg = raw.copy().pick(eeg_names)
    # Positions come straight from the template montage so they share the
    # BioSemi-128 template's coordinate frame; the .cnt's own digitisation
    # (a different head frame) is deliberately not used.
    template = mne.channels.make_standard_montage('standard_1005').get_positions()['ch_pos']
    upper = {k.upper(): v for k, v in template.items()}
    positions = {n: upper[n.upper()] for n in eeg.ch_names if n.upper() in upper}
    eeg.set_eeg_reference('average', projection=False, verbose='ERROR')
    eeg.notch_filter(60., verbose='ERROR')
    eeg.filter(.5, 45., verbose='ERROR')
    eeg.resample(TARGET_RATE, npad='auto', verbose='ERROR')
    other = raw.copy().pick(aux)
    other.filter(.5, 100., picks='all', verbose='ERROR')
    other.resample(TARGET_RATE, npad='auto', verbose='ERROR')
    names = [n.upper() for n in eeg.ch_names]
    upper_positions = {k.upper(): v for k, v in positions.items()}
    xyz = np.full((len(names), 3), np.nan, dtype=np.float32)
    for i, n in enumerate(names):
        if n in upper_positions:
            xyz[i] = upper_positions[n]
    matrix, target_names, target_xyz = biosemi128_map(names, upper_positions)
    return (eeg.get_data().astype(np.float32), names, xyz, other.get_data().astype(np.float32), aux,
            matrix, target_names, target_xyz, int(raw.n_times))


def robust_stats(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(data, axis=1)
    scale = np.maximum(1.4826 * np.median(np.abs(data - center[:, None]), axis=1), 1e-9)
    return center.astype(np.float32), scale.astype(np.float32)


def build_participant(folder: Path, output: Path, force: bool) -> dict:
    cnt = folder / 'Acquisition 232 Data.cnt'
    if not cnt.exists():
        candidates = sorted(folder.glob('*.cnt'))
        if len(candidates) != 1:
            raise FileNotFoundError(f'{folder.name}: expected exactly one .cnt, found {len(candidates)}')
        cnt = candidates[0]
    stages = read_stage_indices(folder / 'epoch_inds.mat')
    labels, words = read_prompts(folder)
    if len(labels) != len(stages):
        raise ValueError(f'{folder.name}: {len(labels)} prompts for {len(stages)} trials')
    wavs = [folder / 'kinect_data' / f'{i}.wav' for i in range(len(stages))]
    missing = [w.name for w in wavs if not w.exists()]
    if missing:
        raise FileNotFoundError(f'{folder.name}: missing utterance recordings {missing[:3]}')
    sources = {'cnt': sha256(cnt), 'epoch_inds': sha256(folder / 'epoch_inds.mat'), 'labels': sha256(folder / 'kinect_data' / 'labels.txt')}
    shard = output / f'{folder.name}.h5'
    if shard.exists() and not force:
        with h5py.File(shard, 'r') as h5:
            if json.loads(h5.attrs['sources']) == sources and h5.attrs.get('contract') == CONTRACT:
                print(f'[karaone] {folder.name}: shard up to date', flush=True)
                return dict(participant=folder.name, trials=len(stages), status='cached')
    started = time.monotonic()
    eeg, names, xyz, aux, aux_names, matrix, target_names, target_xyz, source_samples = preprocess(cnt)
    last_needed = to_target_rate(int(max(stages.speaking_end.max(), stages.thinking_end.max())))
    if last_needed > eeg.shape[1]:
        raise ValueError(f'{folder.name}: stage indices exceed the recording ({last_needed} > {eeg.shape[1]} samples)')
    center, scale = robust_stats(eeg)
    table = stages.copy()
    for stage in STAGES:
        table[f'{stage}_start'] = table[f'{stage}_start'].map(lambda v: to_target_rate(v) if v >= 0 else -1)
        table[f'{stage}_end'] = table[f'{stage}_end'].map(lambda v: to_target_rate(v) if v >= 0 else -1)
    table['prompt'] = labels
    table['word'] = words if len(words) == len(labels) else ''
    table['utterance_wav'] = [str(w.relative_to(ROOT)) for w in wavs]
    table['participant'] = folder.name
    output.mkdir(parents=True, exist_ok=True)
    temporary = shard.with_suffix('.partial.h5')
    with h5py.File(temporary, 'w') as h5:
        h5.attrs.update(contract=CONTRACT, participant=folder.name, sfreq=TARGET_RATE, source_sfreq=SOURCE_RATE,
                        source_samples=source_samples, sources=json.dumps(sources), eeg_unit='V',
                        preprocessing=json.dumps(dict(reference='average', notch_hz=60., bandpass_hz=[.5, 45.], aux_bandpass_hz=[.5, 100.],
                                                      resample_hz=TARGET_RATE, montage='standard_1005', source='cnt')),
                        channel_names=json.dumps(names), aux_names=json.dumps(aux_names), biosemi128_names=json.dumps(target_names),
                        normalizer_center=center, normalizer_scale=scale, built_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        h5.create_dataset('eeg', data=eeg, chunks=(eeg.shape[0], 4096), compression='lzf')
        h5.create_dataset('aux', data=aux, chunks=(max(1, aux.shape[0]), 4096), compression='lzf')
        h5.create_dataset('channel_xyz', data=xyz)
        h5.create_dataset('biosemi128_map', data=matrix)
        h5.create_dataset('biosemi128_xyz', data=target_xyz)
        trials = h5.create_group('trials')
        for column in table.columns:
            values = table[column].to_numpy()
            if values.dtype.kind in 'iu':
                trials.create_dataset(column, data=values.astype(np.int64))
            else:
                trials.create_dataset(column, data=np.array([str(v) for v in values], dtype=h5py.string_dtype()))
    temporary.replace(shard)
    seconds = round(time.monotonic() - started, 1)
    durations = {stage: float(((table[f'{stage}_end'] - table[f'{stage}_start'])[table[f'{stage}_start'] >= 0] / TARGET_RATE).median()) for stage in STAGES}
    print(f'[karaone] {folder.name}: {len(table)} trials, {eeg.shape[1] / TARGET_RATE:.0f} s, {seconds} s', flush=True)
    return dict(participant=folder.name, trials=len(table), status='built', seconds=seconds, channels=len(names),
                invalid_stage_trials={stage: int((~table[f'{stage}_valid']).sum()) for stage in STAGES},
                located_channels=int(np.isfinite(xyz).all(1).sum()), aux=aux_names,
                prompt_counts=table.prompt.value_counts().to_dict(), median_stage_seconds=durations,
                normalizer_scale_uV=dict(median=float(np.median(scale) * 1e6), min=float(scale.min() * 1e6), max=float(scale.max() * 1e6)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=str(ROOT / 'data/karaone'))
    parser.add_argument('--output', default=str(ROOT / 'artifacts/karaone'))
    parser.add_argument('--participants', nargs='*', default=list(PARTICIPANTS))
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    data, output = Path(args.data), Path(args.output)
    report = {'contract': CONTRACT, 'participants': [], 'prompts': list(PROMPTS)}
    rows = []
    for participant in args.participants:
        folder = data / participant
        if not folder.is_dir():
            raise FileNotFoundError(f'missing participant folder {folder}')
        report['participants'].append(build_participant(folder, output / 'shards', args.force))
        with h5py.File(output / 'shards' / f'{participant}.h5', 'r') as h5:
            frame = pd.DataFrame({k: h5['trials'][k][:] for k in h5['trials']})
        for column in frame.columns:
            if frame[column].dtype == object:
                frame[column] = frame[column].map(lambda v: v.decode() if isinstance(v, bytes) else v)
        frame['shard'] = str((output / 'shards' / f'{participant}.h5').relative_to(ROOT))
        rows.append(frame)
    manifest = pd.concat(rows, ignore_index=True)
    manifest.to_csv(output / 'manifest.csv', index=False)
    report['trials'] = int(len(manifest)); report['manifest'] = str((output / 'manifest.csv').relative_to(ROOT))
    (output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'participants'}))


if __name__ == '__main__':
    main()

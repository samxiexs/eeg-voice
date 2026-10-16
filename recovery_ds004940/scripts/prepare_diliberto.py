#!/usr/bin/env python3
"""Di Liberto et al. 2020 (monophonic Bach, 64-channel BioSemi) -> harmonised music EEG shards.

Input: the extracted CND folder of ``diliBach_4dryad_CND.zip`` (``dataStim.mat``
and ``dataSub1.mat`` ... ``dataSub20.mat``; see scripts/download_music.sh).

Harmonisation matches the DS004940 shards (scripts/prepare_training_data.py):
bad electrodes interpolated with spherical splines on the biosemi64 montage,
average reference, 0.5-45 Hz band-pass, 256 Hz, volts.  Each of the 30
trials (10 pieces x 3 presentations) is kept whole, including the recording's
padding, with ``onset_sample`` marking the first stimulus sample.

Splits, both fixed by hashes so they never depend on results:
* content (piece): 1 test piece, 1 validation piece, 8 training pieces;
* participants: ``--heldout-per-group`` non-musicians (sub 1-10) and as many
  pianists (sub 11-20) are held out entirely, for the unseen-participant
  evaluation that a generative decoder needs.

Outputs (default artifacts/music/diliberto2020/): shards/sub-XX.h5,
manifest.csv, normalizer.json (per-channel median/MAD over training
participants x training pieces), stimulus.h5 (the CND envelope and note-onset
vectors per piece at 64 Hz, used to align the rendered audio), qc.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src'))
from music import bad_channels, channel_order_hash, cnd_eeg, cnd_stimulus, group_identical, load_mat, ranked

CONTRACT = 'music_eeg_v1'
DATASET = 'diliberto2020'
TARGET_RATE = 256


def channel_positions(recording):
    """(names, xyz) on the MNE biosemi64 sphere; CND labels may be 10-20 names or BioSemi A1..B32."""
    import mne
    montage = mne.channels.make_standard_montage('biosemi64')
    positions = montage.get_positions()['ch_pos']
    standard = list(positions)
    labels = recording.labels
    if len(labels) == 64 and all(l in positions for l in labels):
        names = labels
    elif len(labels) == 64 and all(re.fullmatch(r'[AB]\d{1,2}', l) for l in labels):
        order = [f'A{i}' for i in range(1, 33)] + [f'B{i}' for i in range(1, 33)]
        names = [standard[order.index(l)] for l in labels]
    elif not labels:
        names = standard                               # CND without chanlocs: BioSemi A1..B32 order
    else:
        # EEGLAB coordinates (X nose, Y left, Z up) -> MNE head frame (x right, y nose, z up).
        xyz = np.array([[-float(c['Y']), float(c['X']), float(c['Z'])] for c in recording.chanlocs])
        xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True) * .095
        return labels, xyz
    return names, np.array([positions[n] for n in names])


def harmonise(trial, fs, names, xyz):
    """(channels, samples) float32 volts at 256 Hz and the list of interpolated channel indices."""
    import mne
    data = trial.T.astype(np.float64)                         # channels x samples
    if np.median(np.abs(data - np.median(data, 1, keepdims=True))) > 1e-3:
        data = data * 1e-6                                    # CND data stored in microvolts
    bads = bad_channels(data, fs)
    info = mne.create_info(list(names), fs, 'eeg')
    raw = mne.io.RawArray(data, info, verbose='ERROR')
    raw.set_montage(mne.channels.make_dig_montage(dict(zip(names, xyz)), coord_frame='head'), verbose='ERROR')
    raw.info['bads'] = [names[i] for i in bads]
    if len(bads):
        raw.interpolate_bads(reset_bads=True, verbose='ERROR')
    raw.set_eeg_reference('average', projection=False, verbose='ERROR')
    raw.filter(.5, 45., verbose='ERROR')
    raw.resample(TARGET_RATE, npad='auto', verbose='ERROR')
    return raw.get_data().astype(np.float32), [int(i) for i in bads]


def roles(pieces, subjects, heldout_per_group):
    order = ranked(pieces, 'diliberto2020-piece')
    content = {p: 'train' for p in pieces}
    content[order[0]] = 'test'; content[order[1]] = 'validation'
    subject_role = {s: 'train' for s in subjects}
    for group in ([s for s in subjects if s <= 10], [s for s in subjects if s > 10]):
        for s in ranked(group, 'diliberto2020-subject')[:heldout_per_group]:
            subject_role[s] = 'heldout'
    return content, subject_role


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cnd', default=str(ROOT / 'data/diliberto2020/CND'), help='folder with dataStim.mat and dataSub*.mat')
    parser.add_argument('--output', default=str(ROOT / 'artifacts/music/diliberto2020'))
    parser.add_argument('--heldout-per-group', type=int, default=2)
    parser.add_argument('--max-bad-fraction', type=float, default=.15)
    parser.add_argument('--subjects', default='all', help='comma-separated subject numbers or "all"')
    args = parser.parse_args()
    cnd = Path(args.cnd); output = Path(args.output)
    stim_files = sorted(cnd.rglob('dataStim.mat'))
    if not stim_files:
        raise SystemExit(f'no dataStim.mat under {cnd}; run scripts/download_music.sh data first')
    folder = stim_files[0].parent
    subject_files = {int(re.search(r'dataSub(\d+)\.mat$', p.name).group(1)): p for p in folder.glob('dataSub*.mat')}
    subjects = sorted(subject_files) if args.subjects == 'all' else [int(s) for s in args.subjects.split(',')]
    if not subjects:
        raise SystemExit(f'no dataSub*.mat in {folder}')

    stimulus = cnd_stimulus(load_mat(stim_files[0]))
    envelopes = stimulus.features[0]
    piece_of_trial = stimulus.stimulus_index if stimulus.stimulus_index is not None else group_identical(envelopes)
    pieces = sorted(set(int(p) for p in piece_of_trial))
    print(f'{len(envelopes)} stimulus trials, {len(pieces)} pieces, features {stimulus.names}, fs {stimulus.fs} Hz', flush=True)
    content_role, subject_role = roles(pieces, subjects, args.heldout_per_group)

    output.mkdir(parents=True, exist_ok=True); (output / 'shards').mkdir(exist_ok=True)
    with h5py.File(output / 'stimulus.h5.partial', 'w') as h5:
        h5.attrs.update(contract='music_stimulus_v1', fs=stimulus.fs, names=json.dumps(stimulus.names), source=str(stim_files[0]))
        for piece in pieces:
            trial = int(np.flatnonzero(piece_of_trial == piece)[0])
            g = h5.create_group(f'pieces/{piece}')
            for name, feature in zip(stimulus.names, stimulus.features):
                g.create_dataset(re.sub(r'\W+', '_', name).strip('_').lower() or 'feature', data=feature[trial].astype(np.float32))
            g.attrs['duration_s'] = len(envelopes[trial]) / stimulus.fs
            g.attrs['envelope_feature'] = re.sub(r'\W+', '_', stimulus.names[0]).strip('_').lower()
    (output / 'stimulus.h5.partial').replace(output / 'stimulus.h5')

    rows, qc, names_seen = [], {}, None
    for subject in subjects:
        recording = cnd_eeg(load_mat(subject_files[subject]))
        if len(recording.trials) != len(envelopes):
            raise ValueError(f'sub {subject}: {len(recording.trials)} EEG trials vs {len(envelopes)} stimulus trials')
        names, xyz = channel_positions(recording)
        if names_seen is None:
            names_seen = list(names)
        elif list(names) != names_seen:
            raise ValueError(f'sub {subject}: channel order differs from sub {subjects[0]}')
        onset = int(round(recording.padding_start * TARGET_RATE / recording.fs))
        shard = output / 'shards' / f'sub-{subject:02d}.h5'
        repeats = {}
        with h5py.File(shard.with_suffix('.partial'), 'w') as h5:
            h5.attrs.update(contract=CONTRACT, dataset=DATASET, subject=f'sub-{subject:02d}', sfreq=TARGET_RATE, unit='V',
                            reference='average', bandpass_hz=json.dumps([.5, 45.]), montage='biosemi64',
                            channel_order_hash=channel_order_hash(names), source=str(subject_files[subject]))
            h5.create_dataset('channel_xyz', data=np.asarray(xyz, np.float32))
            h5.create_dataset('channel_names', data=np.array(names, dtype='S'))
            for k, trial in enumerate(recording.trials):
                piece = int(piece_of_trial[k])
                repeats[piece] = repeats.get(piece, 0) + 1
                eeg, bads = harmonise(trial, recording.fs, names, xyz)
                excluded = len(bads) > args.max_bad_fraction * len(names)
                qc.setdefault(f'sub-{subject:02d}', {})[k] = dict(piece=piece, bad_channels=[names[i] for i in bads], excluded=excluded)
                if excluded:
                    continue
                h5.create_dataset(f'trials/{k:02d}', data=eeg, chunks=(eeg.shape[0], min(1024, eeg.shape[1])))
                h5.create_dataset(f'trials_valid/{k:02d}', data=np.ones(eeg.shape[0], bool))
                h5[f'trials/{k:02d}'].attrs.update(piece=piece, repeat=repeats[piece], onset_sample=onset,
                                                    interpolated=json.dumps([names[i] for i in bads]))
                rows.append(dict(dataset=DATASET, subject=f'sub-{subject:02d}',
                                 group='pianist' if subject > 10 else 'nonmusician', trial=k, piece=piece,
                                 repeat=repeats[piece], presentation=int(recording.presentation[k]) if recording.presentation is not None else -1,
                                 shard_path=str(shard.relative_to(ROOT)) if shard.is_relative_to(ROOT) else str(shard),
                                 onset_sample=onset, n_samples=eeg.shape[1],
                                 stimulus_duration_s=len(envelopes[k]) / stimulus.fs, n_bad=len(bads),
                                 content_role=content_role[piece], subject_role=subject_role[subject]))
        shard.with_suffix('.partial').replace(shard)
        print(f'sub-{subject:02d}: {sum(1 for r in rows if r["subject"] == f"sub-{subject:02d}")} trials, '
              f'role {subject_role[subject]}', flush=True)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output / 'manifest.csv', index=False)
    # Robust per-channel scale from training participants x training pieces only (every 16th sample).
    fit = manifest[(manifest.content_role == 'train') & (manifest.subject_role == 'train')]
    samples = []
    for row in fit.itertuples():
        with h5py.File(ROOT / row.shard_path if not Path(row.shard_path).is_absolute() else row.shard_path, 'r') as h5:
            samples.append(h5['trials'][f'{row.trial:02d}'][:, ::16])
    values = np.concatenate(samples, axis=1)
    center = np.median(values, axis=1)
    scale = np.maximum(1.4826 * np.median(np.abs(values - center[:, None]), axis=1), 1e-9)
    (output / 'normalizer.json').write_text(json.dumps(dict(
        contract=CONTRACT, dataset=DATASET, fit_role='train', channel_order_hash=channel_order_hash(names_seen),
        center=center.tolist(), scale=scale.tolist(), fit_trials=len(fit)), indent=1))
    summary = dict(pieces={int(p): content_role[p] for p in pieces},
                   subjects={f'sub-{s:02d}': subject_role[s] for s in subjects},
                   trials=len(manifest), excluded_trials=sum(v['excluded'] for s in qc.values() for v in s.values()),
                   mean_bad_channels=float(manifest.n_bad.mean()) if len(manifest) else None, detail=qc)
    (output / 'qc.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'detail'}, indent=1))


if __name__ == '__main__':
    main()

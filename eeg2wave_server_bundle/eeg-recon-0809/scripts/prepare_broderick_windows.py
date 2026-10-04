#!/usr/bin/env python3
"""Broderick 2018 audiobook EEG -> continuous-speech shards for the subject-free encoder.

19 participants x 20 runs (~3 min) of "The Old Man and the Sea", 128-channel
BioSemi, from the CND release (``data/broderick2018/Natural Speech``: 128 Hz,
unfiltered, unreferenced, time-locked to the run's audio onset).  The audio is
the identical wav in OpenNeuro ds004408 (``stimuli/audio<r>.wav``); the CND
envelope of run r correlates 1.000 with its Hilbert envelope at lag 0, so the
EEG onset is sample 0 of every run.

The EEG window of a 4 s audio window starts 0.25 s before it (EEG_START), and
there is no EEG before the audio onset, so stimulus time 0 is defined as
``AUDIO_OFFSET_S`` = 0.25 s into the audio: the targets
(scripts/cache_broderick_targets.py) start there and ``onset_sample`` is 0.5 s
(in shard samples), which puts the EEG of stimulus time t at (0.25 + t) s exactly.

The output uses the continuous-window format of app/music.py (MusicWindows):
the audiobook run plays the role of a "piece", every participant heard every
run once.  Harmonisation matches the DS004940 shards: 50 Hz notch (Dublin),
bad electrodes (judged in 1-45 Hz) interpolated with spherical splines,
average reference, 0.5-45 Hz band-pass; the cap and channel order (A1..D32)
are DS004940's, so the electrode positions are copied from a DS004940 shard.
Storage is compact so the shards can replace the raw release: the native
128 Hz rate (all content is below 45 Hz) in float16 microvolts, about 2.2 GB
for all 19 participants; MusicWindows upsamples each window to 256 Hz
(manifest column ``sfreq``).

Splits: runs 19-20 are validation, runs 1-18 training, and there is no
Broderick test role (the protocol test set stays DS004940's).  Runs 19-20 are
the ones the Broderick envelope trunk (app/broderick_pretrain.py, used to
initialise the encoder) never trained on, so they stay unseen here too.
``--heldout`` participants (hash-fixed) are removed from training entirely.

Outputs (default artifacts/speech_continuous/broderick2018/): shards/SubjectN.h5,
manifest.csv, normalizer.json, qc.json.  Targets: scripts/cache_broderick_targets.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src'))
from music import bad_channels, channel_order_hash, ranked

CONTRACT = 'music_eeg_v1'              # the MusicWindows shard layout
DATASET = 'broderick2018'
SOURCE_RATE, AUDIO_RATE = 128, 16000
STORED_RATE = SOURCE_RATE                # kept native; MusicWindows upsamples windows to 256 Hz
RUNS = tuple(range(1, 21))
VALIDATION_RUNS = (19, 20)             # held out by the Broderick trunk pretraining as well
PARTICIPANTS = tuple(range(1, 20))
BIOSEMI128 = [f'{bank}{i}' for bank in 'ABCD' for i in range(1, 33)]
AUDIO_OFFSET_S = .25                    # = -EEG_START: stimulus time 0 is this far into the audio
ONSET_SAMPLE = int(round(2 * AUDIO_OFFSET_S * STORED_RATE))
DS004940_SHARD = ROOT / 'artifacts/training_data/aligned_v1/shards/aligned_v1/ds004940/sub-001/task-N400Active.h5'


def positions():
    with h5py.File(DS004940_SHARD, 'r') as h5:
        order = json.loads(h5.attrs['channel_order'])
        if order != BIOSEMI128:
            raise ValueError('DS004940 channel order is not A1..D32')
        return h5['channel_xyz'][:].astype(np.float64)


def harmonise(eeg, xyz):
    """(T, 128) CND samples -> (128, T) float16 microvolts at the native 128 Hz, interpolated channel indices."""
    import mne
    data = eeg.T.astype(np.float64)
    if np.median(np.abs(data - np.median(data, 1, keepdims=True))) > 1e-3:
        data = data * 1e-6                                    # CND data stored in microvolts
    raw = mne.io.RawArray(data, mne.create_info(BIOSEMI128, SOURCE_RATE, 'eeg'), verbose='ERROR')
    raw.notch_filter(50., verbose='ERROR')
    bads = bad_channels(raw.get_data(), SOURCE_RATE)
    raw.set_montage(mne.channels.make_dig_montage(dict(zip(BIOSEMI128, xyz)), coord_frame='head'), verbose='ERROR')
    raw.info['bads'] = [BIOSEMI128[i] for i in bads]
    if len(bads):
        raw.interpolate_bads(reset_bads=True, verbose='ERROR')
    raw.set_eeg_reference('average', projection=False, verbose='ERROR')
    raw.filter(.5, 45., verbose='ERROR')
    return (raw.get_data() * 1e6).astype(np.float16), [int(i) for i in bads]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cnd', default=str(ROOT / 'data/broderick2018/Natural Speech'))
    parser.add_argument('--audio', default=str(ROOT / 'data/ds004408/stimuli'))
    parser.add_argument('--output', default=str(ROOT / 'artifacts/speech_continuous/broderick2018'))
    parser.add_argument('--heldout', type=int, default=3)
    parser.add_argument('--max-bad-fraction', type=float, default=.15)
    parser.add_argument('--subjects', default='all')
    args = parser.parse_args()
    import scipy.io as sio
    import soundfile as sf
    cnd, output = Path(args.cnd), Path(args.output).resolve()
    subjects = list(PARTICIPANTS) if args.subjects == 'all' else [int(s) for s in args.subjects.split(',')]
    content_role = {r: 'validation' if r in VALIDATION_RUNS else 'train' for r in RUNS}
    heldout = set(ranked(list(PARTICIPANTS), 'broderick2018-subject')[:args.heldout])
    durations = {r: sf.info(str(Path(args.audio) / f'audio{r:02d}.wav')).duration for r in RUNS}
    xyz = positions()
    output.mkdir(parents=True, exist_ok=True); (output / 'shards').mkdir(exist_ok=True)
    rows, qc = [], {}
    for subject in subjects:
        shard = output / 'shards' / f'Subject{subject}.h5'
        record = qc.setdefault(f'Subject{subject}', {})
        if not shard.exists():
            with h5py.File(shard.with_suffix('.partial'), 'w') as h5:
                h5.attrs.update(contract=CONTRACT, dataset=DATASET, subject=f'Subject{subject}', sfreq=STORED_RATE, unit='uV',
                                reference='average', bandpass_hz=json.dumps([.5, 45.]), montage='biosemi128',
                                channel_order_hash=channel_order_hash(BIOSEMI128), source=str(cnd))
                h5.create_dataset('channel_xyz', data=xyz.astype(np.float32))
                h5.create_dataset('channel_names', data=np.array(BIOSEMI128, dtype='S'))
                for r in RUNS:
                    mat = sio.loadmat(cnd / 'EEG' / f'Subject{subject}' / f'Subject{subject}_Run{r}.mat')
                    if int(mat['fs'].ravel()[0]) != SOURCE_RATE or mat['eegData'].shape[1] != 128:
                        raise ValueError(f'Subject{subject} run {r}: unexpected format')
                    eeg, bads = harmonise(mat['eegData'], xyz)
                    k = r - 1
                    h5.create_dataset(f'trials/{k:02d}', data=eeg, chunks=(eeg.shape[0], min(1024, eeg.shape[1])))
                    h5.create_dataset(f'trials_valid/{k:02d}', data=np.ones(eeg.shape[0], bool))
                    h5[f'trials/{k:02d}'].attrs.update(piece=r, repeat=1, onset_sample=ONSET_SAMPLE, interpolated=json.dumps(bads))
            shard.with_suffix('.partial').replace(shard)
        with h5py.File(shard, 'r') as h5:
            for r in RUNS:
                k = r - 1
                bads = json.loads(h5[f'trials/{k:02d}'].attrs['interpolated'])
                n = h5[f'trials/{k:02d}'].shape[1]
                record[r] = dict(interpolated=len(bads), eeg_s=round(n / STORED_RATE, 2), audio_s=round(durations[r], 2))
                if len(bads) > args.max_bad_fraction * 128:
                    record[r]['excluded'] = 'too many bad electrodes'
                    continue
                rows.append(dict(dataset=DATASET, subject=f'Subject{subject}', group='all', trial=k, piece=r, repeat=1,
                                 presentation=-1, shard_path=str(shard.relative_to(ROOT)), onset_sample=ONSET_SAMPLE, n_samples=n,
                                 sfreq=STORED_RATE,
                                 stimulus_duration_s=durations[r] - AUDIO_OFFSET_S, n_bad=len(bads), content_role=content_role[r],
                                 subject_role='heldout' if subject in heldout else 'train'))
        print(f'Subject{subject}: {sum(1 for v in record.values() if "excluded" not in v)}/20 runs, '
              f'role {"heldout" if subject in heldout else "train"}, mean interpolated '
              f'{np.mean([v["interpolated"] for v in record.values()]):.1f}', flush=True)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output / 'manifest.csv', index=False)
    fit = manifest[(manifest.content_role == 'train') & (manifest.subject_role == 'train')]
    samples = []
    for row in fit.itertuples():
        with h5py.File(ROOT / row.shard_path, 'r') as h5:
            samples.append(h5['trials'][f'{row.trial:02d}'][:, ::32].astype(np.float32))
    values = np.concatenate(samples, axis=1)
    center = np.median(values, axis=1)
    scale = np.maximum(1.4826 * np.median(np.abs(values - center[:, None]), axis=1), 1e-9)
    (output / 'normalizer.json').write_text(json.dumps(dict(
        contract=CONTRACT, dataset=DATASET, fit_role='train', channel_order_hash=channel_order_hash(BIOSEMI128),
        center=center.tolist(), scale=scale.tolist(), fit_trials=len(fit)), indent=1))
    summary = dict(runs={r: content_role[r] for r in RUNS}, heldout=sorted(f'Subject{s}' for s in heldout),
                   prepared=len(manifest), excluded=[(s, r, v['excluded']) for s, v2 in qc.items() for r, v in v2.items() if 'excluded' in v],
                   mean_interpolated=float(manifest.n_bad.mean()), detail=qc)
    (output / 'qc.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'detail'}, indent=1))


if __name__ == '__main__':
    main()

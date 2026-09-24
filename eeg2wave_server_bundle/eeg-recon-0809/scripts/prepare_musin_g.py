#!/usr/bin/env python3
"""MUSIN-G (OpenNeuro ds003774; 20 participants, 12 songs, 128-channel EGI) -> harmonised music EEG shards.

Input: the per-song BIDS files ``sub-XXX/ses-NN/eeg/*_run-N_eeg.set`` and the
songs ``Code/ESongs/N.esh.wav`` (scripts/download_music.sh musin-g).

Facts this script relies on (verified 2026-09-23 on sub-001 against the
continuous recording in ``sourcedata/``):

* session / run N is song N; the file is an exact cut of the continuous
  recording from the fixation marker ``fxcl`` to the song end ``fxnd``, so the
  song starts **10.00 s** into every file (the ``stm+`` marker).  Each file's
  length minus 10 s is checked against its song's wav duration
  (``--duration-tolerance-s``); a mismatch excludes the trial.
* the annotations / events.tsv shipped inside the per-song files are copied
  unshifted from the start of the continuous recording and are wrong for
  every song except by accident; they are never read.
* channel E129 is the recording reference (Cz, identically zero).  It is
  kept as zeros through the average reference and never treated as a bad
  channel.  Electrode positions are the standard GSN-HydroCel-129 montage
  (identical to the files' own), projected onto the same 0.095 m sphere as
  the BioSemi montages.
* the songs are 8 kHz stereo (4 kHz bandwidth) as presented; the audio
  written here is the mono mix, at 8 kHz (``audio/native``) and resampled to
  24 kHz (``audio/``, what scripts/cache_music_targets.py reads; nothing above
  4 kHz exists in it).

Harmonisation matches the DS004940 and Di Liberto shards: electrodes further
than ``--max-offcap-deg`` from every DS004940 BioSemi-128 electrode (the
face, neck and eye ring of the EGI net: 20 of 129 at 15 deg) are dropped,
a 50 Hz notch, bad electrodes (judged in 1-45 Hz) interpolated with
spherical splines, then average reference, 0.5-45 Hz band-pass, 256 Hz, volts.

Splits (fixed by hashes): 1 test song, 1 validation song, 10 training songs;
``--heldout`` participants removed entirely (the unseen-participant cohort).

Outputs (default artifacts/music/musin_g/): shards/sub-XXX.h5, manifest.csv,
normalizer.json, audio/piece_NN.wav + audio/alignment.json, qc.json.
"""
from __future__ import annotations

import argparse
import json
from math import gcd
from pathlib import Path
import re
import sys
import warnings

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src'))
from music import bad_channels, channel_order_hash, ranked

CONTRACT = 'music_eeg_v1'
DATASET = 'musin_g'
TARGET_RATE = 256
AUDIO_RATE = 24000
REFERENCE = 'E129'                      # recording reference (Cz), identically zero
PARTICIPANTS = tuple(range(1, 21))      # the dataset's 20 participants (README); the split never depends on what is on disk
SPHERE_RADIUS = .095
LINE_HZ = 50.                           # eeg.json PowerLineFrequency (recorded in India)


def montage_positions(names):
    """(C, 3) GSN-HydroCel-129 positions (E129 = Cz) projected onto the 0.095 m sphere."""
    import mne
    positions = mne.channels.make_standard_montage('GSN-HydroCel-129').get_positions()['ch_pos']
    xyz = np.array([positions['Cz' if n == REFERENCE else n] for n in names], np.float64)
    return xyz / np.linalg.norm(xyz, axis=1, keepdims=True) * SPHERE_RADIUS


def coverage_distance(xyz):
    """Angle (deg) from each electrode to the nearest DS004940 BioSemi-128 electrode."""
    import mne
    biosemi = np.array(list(mne.channels.make_standard_montage('biosemi128').get_positions()['ch_pos'].values()))
    biosemi = biosemi / np.linalg.norm(biosemi, axis=1, keepdims=True)
    unit = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(unit @ biosemi.T, -1, 1))).min(1)


def harmonise(raw, keep, xyz):
    """(C, N) float32 volts at 256 Hz over ``keep`` channels, and the interpolated channel names."""
    import mne
    raw = raw.copy().pick(keep)
    raw.set_annotations(None)                                  # the shipped annotations are misaligned
    raw.set_montage(mne.channels.make_dig_montage(dict(zip(keep, xyz)), coord_frame='head'), verbose='ERROR')
    # Mains first: MNE's 45 Hz low-pass has an 11 Hz transition band and leaves 50 Hz only ~6 dB down,
    # and several EGI channels carry 50 Hz at 10^3-10^5 x their in-band power.
    raw.notch_filter(LINE_HZ, verbose='ERROR')
    data = raw.get_data()
    reference = [keep.index(REFERENCE)] if REFERENCE in keep else []
    bads = [keep[i] for i in bad_channels(data, raw.info['sfreq'], exclude=reference)]
    raw.info['bads'] = bads
    if bads:
        raw.interpolate_bads(reset_bads=True, verbose='ERROR')
    raw.set_eeg_reference('average', projection=False, verbose='ERROR')
    raw.filter(.5, 45., verbose='ERROR')
    raw.resample(TARGET_RATE, npad='auto', verbose='ERROR')
    return raw.get_data().astype(np.float32), bads


def read_song(path):
    """Mono float32 samples and rate of a presented song."""
    import soundfile as sf
    wave, rate = sf.read(path, dtype='float32', always_2d=True)
    return wave.mean(1), int(rate)


def behaviour(path):
    """{(subject, song): (enjoyment, familiarity)} from stimuli/Behavioural_data (1 = most)."""
    table = {}
    if not Path(path).is_file():
        return table
    for line in Path(path).read_text().splitlines()[1:]:
        values = line.split()
        if len(values) >= 4 and all(v.isdigit() for v in values[:4]):
            table[(int(values[0]), int(values[1]))] = (int(values[2]), int(values[3]))
    return table


def write_audio(songs, output):
    """Mono songs at native rate and at 24 kHz, plus the alignment record cache_music_targets.py reads."""
    import soundfile as sf
    from scipy.signal import resample_poly
    folder = output / 'audio'; (folder / 'native').mkdir(parents=True, exist_ok=True)
    pieces = {}
    for song, (wave, rate) in sorted(songs.items()):
        sf.write(folder / 'native' / f'piece_{song:02d}.wav', wave, rate, subtype='FLOAT')
        g = gcd(rate, AUDIO_RATE)
        high = resample_poly(wave.astype(np.float64), AUDIO_RATE // g, rate // g).astype(np.float32)
        sf.write(folder / f'piece_{song:02d}.wav', high, AUDIO_RATE, subtype='FLOAT')
        pieces[song] = dict(source=f'Code/ESongs/{song}.esh.wav', audio=str(folder / f'piece_{song:02d}.wav'),
                            native_rate=rate, bandwidth_hz=rate / 2, r=1., lag_s=0.,
                            stimulus_s=len(wave) / rate, note='distributed stimulus; onset = stm+ marker, 10 s into each file')
    (folder / 'alignment.json').write_text(json.dumps(dict(synth='none (presented audio)', sample_rate=AUDIO_RATE,
                                                           pieces=pieces), indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bids', default=str(ROOT / 'data/ds003774'))
    parser.add_argument('--output', default=str(ROOT / 'artifacts/music/musin_g'))
    parser.add_argument('--heldout', type=int, default=4, help='participants removed from training entirely')
    parser.add_argument('--max-offcap-deg', type=float, default=15., help='drop electrodes this far from every DS004940 electrode')
    parser.add_argument('--max-bad-fraction', type=float, default=.15)
    parser.add_argument('--onset-s', type=float, default=10., help='song onset inside each per-song file (fxcl -> stm+)')
    parser.add_argument('--duration-tolerance-s', type=float, default=.5)
    parser.add_argument('--subjects', default='all', help='comma-separated subject numbers or "all"')
    args = parser.parse_args()
    import mne
    bids, output = Path(args.bids), Path(args.output)
    files = sorted(bids.glob('sub-*/ses-*/eeg/*_task-MusicListening_run-*_eeg.set'))
    if not files:
        raise SystemExit(f'no per-song EEG files under {bids}; run scripts/download_music.sh musin-g first')
    by_subject = {}
    for f in files:
        match = re.search(r'sub-(\d+)_ses-(\d+)_task-MusicListening_run-(\d+)_eeg\.set$', f.name)
        subject, session, run = (int(g) for g in match.groups())
        if session != run:
            raise ValueError(f'{f.name}: session {session} != run {run}; the song numbering assumption fails')
        by_subject.setdefault(subject, {})[session] = f
    subjects = sorted(by_subject) if args.subjects == 'all' else [int(s) for s in args.subjects.split(',')]
    missing = [s for s in subjects if s not in by_subject]
    if missing:
        raise SystemExit(f'participants {missing} not downloaded yet')
    songs = {n: read_song(bids / 'Code' / 'ESongs' / f'{n}.esh.wav') for n in range(1, 13)}
    ratings = behaviour(bids / 'stimuli' / 'Behavioural_data')
    content_order = ranked(sorted(songs), 'musin_g-song')
    content_role = {s: 'train' for s in songs}
    content_role[content_order[0]] = 'test'; content_role[content_order[1]] = 'validation'
    heldout = set(ranked(PARTICIPANTS, 'musin_g-subject')[:args.heldout])

    output.mkdir(parents=True, exist_ok=True); (output / 'shards').mkdir(exist_ok=True)
    write_audio(songs, output)
    rows, qc, kept_names = [], {}, None
    for subject in subjects:
        shard = output / 'shards' / f'sub-{subject:03d}.h5'
        with h5py.File(shard.with_suffix('.partial'), 'w') as h5:
            for song, path in sorted(by_subject[subject].items()):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    raw = mne.io.read_raw_eeglab(path, preload=True, verbose='ERROR')
                names = list(raw.ch_names)
                if kept_names is None:
                    distance = coverage_distance(montage_positions(names))
                    kept_names = [n for n, d in zip(names, distance) if d <= args.max_offcap_deg]
                    dropped = [n for n, d in zip(names, distance) if d > args.max_offcap_deg]
                    xyz = montage_positions(kept_names)
                    h5_attrs = dict(kept=len(kept_names), dropped=dropped)
                    print(f'electrodes: keep {len(kept_names)}/{len(names)} within {args.max_offcap_deg:g} deg of DS004940; '
                          f'drop {dropped}', flush=True)
                if not set(kept_names) <= set(names):
                    raise ValueError(f'{path.name}: channel set differs')
                fs = raw.info['sfreq']
                after_onset = raw.n_times / fs - args.onset_s
                wave, rate = songs[song]
                mismatch = after_onset - len(wave) / rate
                record = qc.setdefault(f'sub-{subject:03d}', {})[song] = dict(file_minus_onset_s=round(after_onset, 3),
                                                                            song_s=round(len(wave) / rate, 3),
                                                                            mismatch_s=round(mismatch, 3))
                if abs(mismatch) > args.duration_tolerance_s:
                    record['excluded'] = 'duration mismatch'
                    continue
                eeg, bads = harmonise(raw, kept_names, xyz)
                record['interpolated'] = bads
                if len(bads) > args.max_bad_fraction * len(kept_names):
                    record['excluded'] = 'too many bad electrodes'
                    continue
                k = song - 1
                h5.create_dataset(f'trials/{k:02d}', data=eeg, chunks=(eeg.shape[0], min(1024, eeg.shape[1])))
                h5.create_dataset(f'trials_valid/{k:02d}', data=np.ones(eeg.shape[0], bool))
                onset = int(round(args.onset_s * TARGET_RATE))
                h5[f'trials/{k:02d}'].attrs.update(piece=song, repeat=1, onset_sample=onset, interpolated=json.dumps(bads))
                enjoyment, familiarity = ratings.get((subject, song), (-1, -1))
                rows.append(dict(dataset=DATASET, subject=f'sub-{subject:03d}', group='all', trial=k, piece=song, repeat=1,
                                 presentation=-1, shard_path=str(shard.relative_to(ROOT)) if shard.is_relative_to(ROOT) else str(shard),
                                 onset_sample=onset, n_samples=eeg.shape[1], stimulus_duration_s=len(wave) / rate,
                                 n_bad=len(bads), enjoyment=enjoyment, familiarity=familiarity,
                                 content_role=content_role[song], subject_role='heldout' if subject in heldout else 'train'))
            h5.attrs.update(contract=CONTRACT, dataset=DATASET, subject=f'sub-{subject:03d}', sfreq=TARGET_RATE, unit='V',
                            reference='average', bandpass_hz=json.dumps([.5, 45.]), montage='GSN-HydroCel-129',
                            channel_order_hash=channel_order_hash(kept_names), source=str(bids),
                            dropped_offcap=json.dumps(h5_attrs['dropped']))
            h5.create_dataset('channel_xyz', data=xyz.astype(np.float32))
            h5.create_dataset('channel_names', data=np.array(kept_names, dtype='S'))
        shard.with_suffix('.partial').replace(shard)
        done = [r for r in rows if r['subject'] == f'sub-{subject:03d}']
        print(f'sub-{subject:03d}: {len(done)}/12 songs, role {"heldout" if subject in heldout else "train"}, '
              f'max |duration mismatch| {max(abs(v["mismatch_s"]) for v in qc[f"sub-{subject:03d}"].values()):.2f} s', flush=True)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output / 'manifest.csv', index=False)
    fit = manifest[(manifest.content_role == 'train') & (manifest.subject_role == 'train')]
    if fit.empty:
        raise SystemExit('no training participant x training song prepared; the normalizer cannot be fitted')
    samples = []
    for row in fit.itertuples():
        path = Path(row.shard_path) if Path(row.shard_path).is_absolute() else ROOT / row.shard_path
        with h5py.File(path, 'r') as h5:
            samples.append(h5['trials'][f'{row.trial:02d}'][:, ::16])
    values = np.concatenate(samples, axis=1)
    center = np.median(values, axis=1)
    scale = np.maximum(1.4826 * np.median(np.abs(values - center[:, None]), axis=1), 1e-9)
    (output / 'normalizer.json').write_text(json.dumps(dict(
        contract=CONTRACT, dataset=DATASET, fit_role='train', channel_order_hash=channel_order_hash(kept_names),
        center=center.tolist(), scale=scale.tolist(), fit_trials=len(fit)), indent=1))
    summary = dict(songs={s: content_role[s] for s in sorted(songs)}, heldout=sorted(f'sub-{s:03d}' for s in heldout),
                   prepared=len(manifest), electrodes=len(kept_names), dropped_offcap=h5_attrs['dropped'],
                   excluded=[(s, song, v['excluded']) for s, v2 in qc.items() for song, v in v2.items() if 'excluded' in v],
                   mean_interpolated=float(manifest.n_bad.mean()), detail=qc)
    (output / 'qc.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'detail'}, indent=1))


if __name__ == '__main__':
    main()

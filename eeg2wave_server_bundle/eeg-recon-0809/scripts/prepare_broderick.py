#!/usr/bin/env python3
"""Preprocess Broderick et al. 2018 "Natural Speech" (Dryad doi:10.5061/dryad.070jc).

19 participants x 20 runs (~3 min each) of an audiobook with 128-channel
BioSemi EEG at 128 Hz (unfiltered, unreferenced) and the speech envelope at
128 Hz.  The release contains no audio, so this dataset serves one purpose
here: pretraining the EEG encoder trunk on envelope tracking with ~19 hours
of natural speech before it is fine-tuned on DS004940.

Per participant one HDF5 shard is written with the runs concatenated:
``eeg`` (128, T) at the native 128 Hz after average reference, 50 Hz notch
(Dublin mains) and 0.5-45 Hz band-pass; ``envelope`` (T,); a run table; and
per-channel robust normalization statistics.  Channels are assumed to be in
BioSemi A1..D32 order, the same canonical order as the DS004940 shards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RATE = 128
CONTRACT = 'broderick_shard_v1'
BIOSEMI128 = [f'{bank}{i}' for bank in 'ABCD' for i in range(1, 33)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def preprocess_run(eeg: np.ndarray) -> np.ndarray:
    """(128, T) at 128 Hz: average reference, 50 Hz notch, 0.5-45 Hz band-pass."""
    from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt
    x = eeg - eeg.mean(0, keepdims=True)
    b, a = iirnotch(50., 30., fs=SOURCE_RATE)
    x = filtfilt(b, a, x, axis=1)
    sos = butter(4, [.5, 45.], btype='bandpass', fs=SOURCE_RATE, output='sos')
    return sosfiltfilt(sos, x, axis=1).astype(np.float32)


def build_participant(root: Path, subject: int, output: Path, force: bool) -> dict:
    import scipy.io as sio
    folder = root / 'EEG' / f'Subject{subject}'
    runs = [folder / f'Subject{subject}_Run{r}.mat' for r in range(1, 21)]
    missing = [p.name for p in runs if not p.exists()]
    if missing:
        raise FileNotFoundError(f'Subject{subject}: missing {missing[:3]}')
    sources = {p.name: sha256(p) for p in runs}
    shard = output / f'Subject{subject}.h5'
    if shard.exists() and not force:
        with h5py.File(shard, 'r') as h5:
            if h5.attrs.get('contract') == CONTRACT and json.loads(h5.attrs['sources']) == sources:
                return dict(participant=f'Subject{subject}', status='cached')
    started = time.monotonic()
    pieces, envelopes, table = [], [], []
    offset = 0
    for r, path in enumerate(runs, start=1):
        mat = sio.loadmat(path)
        if int(mat['fs'].ravel()[0]) != SOURCE_RATE:
            raise ValueError(f'{path}: unexpected sampling rate')
        eeg = np.asarray(mat['eegData'], dtype=np.float64).T          # 128 x T
        if eeg.shape[0] != 128:
            raise ValueError(f'{path}: expected 128 channels, found {eeg.shape[0]}')
        env = np.asarray(sio.loadmat(root / 'Stimuli' / 'Envelopes' / f'audio{r}_128Hz.mat')['env'], dtype=np.float32).ravel()
        length = min(eeg.shape[1], len(env))            # EEG runs are a few samples longer than the envelope
        pieces.append(preprocess_run(eeg[:, :length])); envelopes.append(env[:length])
        table.append(dict(run=r, start=offset, end=offset + length, eeg_samples=int(eeg.shape[1]), envelope_samples=int(len(env))))
        offset += length
    eeg = np.concatenate(pieces, 1); envelope = np.concatenate(envelopes)
    center = np.median(eeg, axis=1).astype(np.float32)
    scale = np.maximum(1.4826 * np.median(np.abs(eeg - center[:, None]), axis=1), 1e-9).astype(np.float32)
    output.mkdir(parents=True, exist_ok=True)
    temporary = shard.with_suffix('.partial.h5')
    with h5py.File(temporary, 'w') as h5:
        h5.attrs.update(contract=CONTRACT, participant=f'Subject{subject}', sfreq=SOURCE_RATE, sources=json.dumps(sources),
                        channel_names=json.dumps(BIOSEMI128), channel_order_assumption='BioSemi ActiveTwo 128 export order A1..D32',
                        preprocessing=json.dumps(dict(reference='average', notch_hz=50., bandpass_hz=[.5, 45.], rate_hz=SOURCE_RATE)),
                        normalizer_center=center, normalizer_scale=scale, built_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        h5.create_dataset('eeg', data=eeg, chunks=(128, 2048), compression='lzf')
        h5.create_dataset('envelope', data=envelope, chunks=(8192,), compression='lzf')
        group = h5.create_group('runs')
        for key in table[0]:
            group.create_dataset(key, data=np.array([row[key] for row in table], dtype=np.int64))
    temporary.replace(shard)
    return dict(participant=f'Subject{subject}', status='built', seconds=round(time.monotonic() - started, 1),
                duration_s=round(eeg.shape[1] / SOURCE_RATE, 1), runs=len(table), scale_uV_median=float(np.median(scale)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=str(ROOT / 'data/broderick2018/Natural Speech'))
    parser.add_argument('--output', default=str(ROOT / 'artifacts/broderick2018'))
    parser.add_argument('--subjects', nargs='*', type=int, default=list(range(1, 20)))
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    report = {'contract': CONTRACT, 'participants': []}
    for subject in args.subjects:
        entry = build_participant(Path(args.data), subject, Path(args.output) / 'shards', args.force)
        report['participants'].append(entry); print(json.dumps(entry), flush=True)
    (Path(args.output) / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

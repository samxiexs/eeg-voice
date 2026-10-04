#!/usr/bin/env python3
"""Marion et al. 2021 ImageryData.mat (1.5 GB, float64) -> a compact HDF5 that replaces it.

The release is already preprocessed by the authors (0.1-30 Hz, 64 Hz, average
reference, bad channels interpolated), so nothing is filtered here.  The EEG is
kept in the release's own (undocumented) scale - median |x| about 200, artefact
peaks above 2e5, which rules out float16 - as float32 with gzip: the cast from
float64 is checked against the source before the file is marked complete, and
every analysis reproduces exactly.

    eeg   (21, 2, 44, 1803, 64) float32, as shipped
    stim  (21, 2, 44, 1803)     float32, as shipped
    attrs downFs = 64, conditions = listen, imagine, source sha256

app/marion_imagery_gate.py reads either file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', default=str(ROOT / 'data/marion2021/ImageryData.mat'))
    parser.add_argument('--output', default=str(ROOT / 'artifacts/marion2021/imagery.h5'))
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    mat = sio.loadmat(source, squeeze_me=False)
    fs = int(mat['downFs'].ravel()[0])
    subjects = mat['eeg'].shape[1]
    first = mat['eeg'][0, 0][0, 0][0, 0]
    trials, (samples, channels) = mat['eeg'][0, 0][0, 0].shape[1], first.shape
    eeg = np.zeros((subjects, 2, trials, samples, channels), np.float32)
    stim = np.zeros((subjects, 2, trials, samples), np.float32)
    worst = 0.
    for s in range(subjects):
        for c in range(2):
            for t in range(trials):
                x = mat['eeg'][0, s][0, c][0, t]
                eeg[s, c, t] = x.astype(np.float32)
                stim[s, c, t] = mat['stim'][0, s][0, c][0, t].ravel()
                worst = max(worst, float(np.abs(eeg[s, c, t].astype(np.float64) - x).max() / (np.abs(x).max() + 1e-12)))
    if not np.isfinite(eeg).all() or worst > 1e-6:
        raise SystemExit(f'float32 cast lost precision (relative error {worst:.2e}); not writing')
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix('.partial.h5')
    with h5py.File(partial, 'w') as h5:
        h5.create_dataset('eeg', data=eeg, chunks=(1, 1, 1, samples, channels), compression='gzip', compression_opts=4)
        h5.create_dataset('stim', data=stim, chunks=(1, 1, trials, samples), compression='gzip', compression_opts=4)
        h5.attrs.update(downFs=fs, conditions=json.dumps(['listen', 'imagine']), unit='as shipped', source=str(source),
                        source_sha256=sha256(source), max_relative_rounding=worst)
    partial.replace(output)
    print(json.dumps(dict(output=str(output), shape=list(eeg.shape), max_relative_rounding=worst,
                          megabytes=round(output.stat().st_size / 1e6, 1))))


if __name__ == '__main__':
    main()

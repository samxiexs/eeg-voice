#!/usr/bin/env python3
"""Discrete speech units for the content-first route (k-means over HuBERT layer 9).

The mel/HuBERT regression route asked EEG to reproduce a 768-dimensional
sequence and produced a generic speech template.  The content route asks a
smaller question: *which unit is being spoken*, as a sequence of discrete
symbols that a CTC head can emit without any frame alignment.

k-means is fitted on the **train-fold** sentences only, on presented-speech
frames only, and the resulting centroids are stored with the unit sequence of
every sentence (validation and test sentences are encoded with those frozen
centroids, never used to fit them).  Repeated symbols are kept; the collapsed
sequence is stored alongside, as CTC targets must not contain the blank-free
repeats of a frame-rate labelling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = 'speech_units_v1'


def kmeans(data: np.ndarray, clusters: int, seed: int, iterations: int = 60) -> np.ndarray:
    """k-means++ initialisation followed by Lloyd iterations (float32, no sklearn)."""
    rng = np.random.default_rng(seed)
    centres = [data[rng.integers(len(data))]]
    distance = ((data - centres[0]) ** 2).sum(1)
    for _ in range(1, clusters):
        probabilities = distance / distance.sum()
        centres.append(data[rng.choice(len(data), p=probabilities)])
        distance = np.minimum(distance, ((data - centres[-1]) ** 2).sum(1))
    centres = np.stack(centres)
    for step in range(iterations):
        assignment = assign(data, centres)
        updated = np.stack([data[assignment == k].mean(0) if (assignment == k).any() else data[rng.integers(len(data))]
                            for k in range(clusters)])
        shift = float(np.linalg.norm(updated - centres, axis=1).mean())
        centres = updated
        if step % 10 == 0 or shift < 1e-4:
            print(json.dumps(dict(iteration=step, mean_shift=round(shift, 5))), flush=True)
        if shift < 1e-4:
            break
    return centres


def assign(data: np.ndarray, centres: np.ndarray, block: int = 8192) -> np.ndarray:
    out = np.empty(len(data), dtype=np.int16)
    squared = (centres ** 2).sum(1)
    for start in range(0, len(data), block):
        chunk = data[start:start + block]
        out[start:start + block] = (squared - 2 * chunk @ centres.T).argmin(1)
    return out


def collapse(sequence: np.ndarray) -> np.ndarray:
    keep = np.ones(len(sequence), dtype=bool)
    keep[1:] = sequence[1:] != sequence[:-1]
    return sequence[keep]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(ROOT / 'artifacts/aligned_speech_local_v1/manifest.csv'))
    parser.add_argument('--cache', default=str(ROOT / 'artifacts/aligned_speech_local_v1/targets_adapted.h5'))
    parser.add_argument('--output', default=str(ROOT / 'artifacts/aligned_speech_local_v1/speech_units.h5'))
    parser.add_argument('--clusters', type=int, default=50)
    parser.add_argument('--seed', type=int, default=31)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, keep_default_na=False)
    roles = manifest.drop_duplicates('audio_key').set_index('audio_key').role.to_dict()
    with h5py.File(args.cache, 'r') as h5:
        keys = sorted(h5['targets'])
        speech_times = h5['speech_times'][:]
        teacher = {k: h5['targets'][k]['teacher'][:].astype(np.float32) for k in keys}
        frames = {k: int(h5['targets'][k].attrs['source_samples_16k']) for k in keys}
    speaking = {k: speech_times < frames[k] / 16000 for k in keys}
    train_keys = [k for k in keys if roles.get(k) == 'train']
    if not train_keys:
        raise RuntimeError('no train-fold audio in the manifest')
    fit_data = np.concatenate([teacher[k][speaking[k]] for k in train_keys])
    print(json.dumps(dict(train_sentences=len(train_keys), fit_frames=int(len(fit_data)), clusters=args.clusters)), flush=True)
    centres = kmeans(fit_data, args.clusters, args.seed)
    inertia = float(np.mean(((fit_data - centres[assign(fit_data, centres)]) ** 2).sum(1)))
    output = Path(args.output)
    with h5py.File(output, 'w') as h5:
        h5.attrs.update(contract=CONTRACT, clusters=args.clusters, seed=args.seed,
                        manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                        fit_role='train', fit_frames=len(fit_data), inertia=inertia,
                        frame_rate_hz=float(1 / (speech_times[1] - speech_times[0])))
        h5.create_dataset('centres', data=centres)
        group = h5.create_group('units')
        lengths, collapsed_lengths = [], []
        for key in keys:
            sequence = assign(teacher[key][speaking[key]], centres)
            item = group.create_group(key)
            item.create_dataset('frames', data=sequence)
            item.create_dataset('collapsed', data=collapse(sequence))
            item.attrs['role'] = roles.get(key, 'unknown')
            lengths.append(len(sequence)); collapsed_lengths.append(len(collapse(sequence)))
    report = dict(contract=CONTRACT, clusters=args.clusters, sentences=len(keys), inertia=inertia,
                  frames_per_sentence=dict(mean=float(np.mean(lengths)), min=int(np.min(lengths)), max=int(np.max(lengths))),
                  units_per_sentence=dict(mean=float(np.mean(collapsed_lengths)), min=int(np.min(collapsed_lengths)), max=int(np.max(collapsed_lengths))),
                  output=str(output.relative_to(ROOT)))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

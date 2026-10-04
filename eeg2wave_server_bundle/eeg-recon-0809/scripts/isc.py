"""Is there stimulus-locked activity to transfer?  Inter-subject and split-half correlation, per modality.

    python scripts/isc.py marion2021            # listening vs paced imagery of the same melodies
    python scripts/isc.py ds004940 --band 1 8   # positive control: listening to sentences

Perception decoders exploit activity locked to the stimulus clock.  Imagery has that
clock only when it is paced (Marion: metronome).  For every modality this reports
  * correlated components (CorrCA, Parra et al. 2018) fitted on half the people and
    scored on the other half: inter-subject correlation (ISC) of the top components,
    against a null of independent circular shifts per person;
  * split-half reliability inside each person (odd vs even repetitions), the test
    that survives people imagining at slightly different times;
  * the listening components applied to imagery (shared spatial patterns);
  * for labelled items, the same after removing each person's mean over items: what is left
    is item-specific (a metronome or cue common to all items cancels).
Responses are averaged over repetitions of the same item / stimulus first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegspeech import ROOT                               # noqa: E402
from eegspeech.signal import bandpass, whitening         # noqa: E402
from eegspeech.store import open_store                    # noqa: E402


def responses(store, modality, band, min_length):
    """{subject: {key: (odd mean, even mean)}} of band-passed, repetition-averaged responses (T, C)."""
    table = store.table[(store.table.modality_name == modality) & (store.table.length >= min_length)]
    key = 'item' if (table.item >= 0).all() else 'stimulus'
    out = {}
    for subject, part in table.groupby('subject'):
        groups = {}
        for row in part.itertuples():
            x = store.segment(row)[:, :min_length].astype(np.float64)
            x = bandpass(x - x.mean(1, keepdims=True), store.rate, *band)
            groups.setdefault(getattr(row, key), []).append(x.T)
        out[subject] = {k: (np.mean(v[0::2], 0), np.mean(v[1::2], 0) if len(v) > 1 else None) for k, v in groups.items()}
    return out


def stack(data, subjects, keys):
    """(subjects, T * keys, C) full-repetition means, concatenated over keys."""
    full = lambda s, k: data[s][k][0] if data[s][k][1] is None else (data[s][k][0] + data[s][k][1]) / 2
    return np.stack([np.concatenate([full(s, k) for k in keys]) for s in subjects])


def corrca(x, components=3, shrink=.2):
    """x (N, T, C) -> spatial filters (C, components) maximising inter-subject correlation."""
    x = x - x.mean(1, keepdims=True)
    within = sum(xi.T @ xi for xi in x)
    total = x.sum(0).T @ x.sum(0)
    between = total - within
    root = whitening(within, shrink)
    values, vectors = np.linalg.eigh(root @ between @ root)
    return root @ vectors[:, ::-1][:, :components]


def isc(x, w):
    """Mean pairwise correlation across people of each component, x (N, T, C) -> (components,)."""
    y = x @ w
    y = (y - y.mean(1, keepdims=True)) / (y.std(1, keepdims=True) + 1e-12)
    n = len(y)
    corr = np.einsum('itk,jtk->ijk', y, y) / y.shape[1]
    return (corr.sum((0, 1)) - np.trace(corr)) / (n * (n - 1))


def shifted(x, rng):
    return np.stack([np.roll(xi, rng.integers(xi.shape[0] // 10, xi.shape[0] - xi.shape[0] // 10), 0) for xi in x])


def item_specific(data):
    """Remove each person's mean response over items (whatever is common to all items, e.g. a shared metronome)."""
    out = {}
    for subject, groups in data.items():
        odd = np.mean([a for a, _ in groups.values()], 0)               # each half minus its own mean, so the two
        even = [b for _, b in groups.values() if b is not None]          # halves share no noise (unbiased split-half)
        even = np.mean(even, 0) if len(even) == len(groups) else None
        out[subject] = {k: (a - odd, None if b is None or even is None else b - even) for k, (a, b) in groups.items()}
    return out


def analyse(store, modality, band, min_length, rng, perms, max_keys=60, specific=False):
    data = responses(store, modality, band, min_length)
    if specific:
        data = item_specific(data)
    keys = sorted(set.intersection(*[set(v) for v in data.values()]))
    if len(keys) > max_keys:                               # bounded memory: a random subset of shared stimuli
        keys = sorted(rng.choice(keys, max_keys, replace=False).tolist())
    subjects = sorted(data)
    x = stack(data, subjects, keys)
    halves = [np.arange(len(subjects))[::2], np.arange(len(subjects))[1::2]]
    observed, null = [], []
    for fit, test in (halves, halves[::-1]):
        w = corrca(x[fit])
        observed.append(isc(x[test], w))
        null.append([isc(shifted(x[test], rng), w) for _ in range(perms)])
    observed, null = np.mean(observed, 0), np.mean(null, 0)
    p = ((null >= observed).sum(0) + 1) / (perms + 1)
    reliability, reliability_null = [], []
    for s in subjects:
        pairs = [(data[s][k][0], data[s][k][1]) for k in keys if data[s][k][1] is not None]
        if not pairs:
            continue
        odd, even = (np.concatenate(v) for v in zip(*pairs))
        r = lambda a, b: np.mean([np.corrcoef(a[:, c], b[:, c])[0, 1] for c in range(a.shape[1])])
        reliability.append(r(odd, even))
        reliability_null.append(np.mean([r(odd, np.roll(even, rng.integers(len(even) // 10, len(even) - len(even) // 10), 0))
                                         for _ in range(20)]))
    return dict(subjects=len(subjects), keys=len(keys), seconds=x.shape[1] / store.rate,
                isc=observed.round(4).tolist(), p=p.round(4).tolist(),
                split_half=float(np.mean(reliability)) if reliability else None,
                split_half_null=float(np.mean(reliability_null)) if reliability else None,
                split_half_above_null=int(np.sum(np.array(reliability) > np.array(reliability_null)))), x, subjects


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset')
    parser.add_argument('--band', type=float, nargs=2, default=(1., 8.))
    parser.add_argument('--seconds', type=float, default=None, help='response length (default: shortest segment)')
    parser.add_argument('--perms', type=int, default=200)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    store = open_store(args.dataset)
    rng = np.random.default_rng(0)
    report, stacks = dict(dataset=args.dataset, band=args.band), {}
    for modality in [m for m in store.modalities if m != 'rest']:
        part = store.table[store.table.modality_name == modality]
        length = int(args.seconds * store.rate) if args.seconds else int(part.length.min())
        report[modality], stacks[modality], subjects = analyse(store, modality, args.band, length, rng, args.perms)
        print(modality, json.dumps(report[modality]), flush=True)
        if (part.item >= 0).all() and part.item.nunique() > 1:
            report[f'{modality}_item_specific'], _, _ = analyse(store, modality, args.band, length, rng, args.perms,
                                                                specific=True)
            print(modality, 'item-specific', json.dumps(report[f'{modality}_item_specific']), flush=True)
    if {'listen', 'imagine'} <= set(stacks):
        listen, imagine = stacks['listen'], stacks['imagine']
        w = corrca(listen)
        observed = isc(imagine, w)
        null = np.array([isc(shifted(imagine, rng), w) for _ in range(args.perms)])
        report['listen_components_on_imagine'] = dict(isc=observed.round(4).tolist(),
                                                     p=(((null >= observed).sum(0) + 1) / (args.perms + 1)).round(4).tolist())
        print('listen components on imagine', json.dumps(report['listen_components_on_imagine']))
    out = Path(args.out or ROOT / 'outputs' / f'isc_{args.dataset}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, 'w'), indent=1)

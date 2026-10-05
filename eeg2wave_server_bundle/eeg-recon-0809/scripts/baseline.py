"""Linear floor for every item dataset: Euclidean alignment + filter-bank log-variance + logistic regression.

    python scripts/baseline.py [--datasets thinking_out_loud cpseed ...] [--out outputs/baseline.json]

Protocols (all evaluated on target-modality trials, usually imagined):
  within       5-fold cross-validation inside each person over contiguous blocks of the recording
               (person-specific; contiguous folds keep back-to-back repetitions of one cue together)
  cross        5 subject folds (same folds as scripts/train.py): train on the others' target trials
  cross_aux    as cross, but training also uses the others' spoken / mouthed / heard trials
Cross-person runs use the channels every person of the dataset has.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegspeech import ROOT                                    # noqa: E402
from eegspeech.data import subject_folds                      # noqa: E402
from eegspeech.features import bands_for, log_power            # noqa: E402
from eegspeech.metrics import summarize                        # noqa: E402
from eegspeech.store import open_store                         # noqa: E402


def features(store, rows, channels, align=True, max_hz=None):
    """(N, bands * len(channels)) log-variance of band-passed, aligned trials (bands starting below ``max_hz``)."""
    bands = [b for b in bands_for(store.rate) if b[0] < (max_hz or np.inf)]
    out = []
    for row in rows.itertuples():
        names = store.channels(row.subject)
        x = store.segment(row)
        if align:
            x = store.alignment(row.subject, int(row.session)) @ x
        out.append(log_power(x[[names.index(c) for c in channels]].astype(np.float64), store.rate, bands))
    return np.asarray(out, np.float32)


def fit_predict(x_train, y_train, x_test, classes, l2=1e-2, steps=200):
    """Multinomial logistic regression (L-BFGS) on standardised features."""
    x_train, x_test = np.asarray(x_train, np.float32), np.asarray(x_test, np.float32)
    mean, std = x_train.mean(0), x_train.std(0) + 1e-6
    xt = torch.as_tensor((x_train - mean) / std)
    yt = torch.as_tensor(np.searchsorted(classes, y_train))
    w = torch.zeros(xt.shape[1], len(classes), requires_grad=True)
    b = torch.zeros(len(classes), requires_grad=True)
    optimizer = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(xt @ w + b, yt) + l2 * w.square().sum()
        loss.backward()
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        return classes[(torch.as_tensor((x_test - mean) / std) @ w + b).argmax(1).numpy()]


def run(name, spec, folds=5, align=True, max_hz=None):
    store = open_store(name)
    table = store.table[store.table.item >= 0]
    target = table[table.modality_name == spec['target']]
    aux = table[table.modality_name.isin(spec['modalities'])]
    common = sorted(set.intersection(*[set(store.channels(s)) for s in store.subjects]))
    order = store.channels(store.subjects[0])
    common = [c for c in order if c in common] or order
    chance = 1 / target.item.nunique()
    x_target = features(store, target, common, align, max_hz)
    y_target = target.item.to_numpy()
    classes = np.unique(y_target)
    result = {'chance': chance, 'channels': len(common), 'trials': int(len(target))}

    within = {}
    for subject in store.subjects:
        idx = np.flatnonzero(target.subject.to_numpy() == subject)      # recording order
        if len(idx) < 2 * len(classes):
            continue
        correct = 0
        for test in np.array_split(idx, 5):                              # contiguous blocks: no leakage between
            train = np.setdiff1d(idx, test)                              # repetitions recorded back to back
            correct += (fit_predict(x_target[train], y_target[train], x_target[test], classes) == y_target[test]).sum()
        within[subject] = correct / len(idx)
    result['within'] = summarize(within, chance)

    assignment = subject_folds(store.subjects, folds)
    fold_of = target.subject.map(assignment).to_numpy()
    cross = {}
    for f in range(folds):
        train, test = fold_of != f, fold_of == f
        if not test.any():
            continue
        predicted = fit_predict(x_target[train], y_target[train], x_target[test], classes)
        for subject in np.unique(target.subject.to_numpy()[test]):
            mine = target.subject.to_numpy()[test] == subject
            cross[subject] = float((predicted[mine] == y_target[test][mine]).mean())
    result['cross'] = summarize(cross, chance)

    if aux.modality_name.nunique() > 1:
        x_aux = features(store, aux, common, align, max_hz)
        y_aux = aux.item.to_numpy()
        aux_fold = aux.subject.map(assignment).to_numpy()
        cross_aux = {}
        for f in range(folds):
            test = fold_of == f
            if not test.any():
                continue
            predicted = fit_predict(x_aux[aux_fold != f], y_aux[aux_fold != f], x_target[test], classes)
            for subject in np.unique(target.subject.to_numpy()[test]):
                mine = target.subject.to_numpy()[test] == subject
                cross_aux[subject] = float((predicted[mine] == y_target[test][mine]).mean())
        result['cross_aux'] = summarize(cross_aux, chance)
    result['per_subject'] = dict(within=within, cross=cross)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='*')
    parser.add_argument('--no-align', action='store_true')
    parser.add_argument('--max-hz', type=float, help='only bands below this frequency (e.g. 45: EMG control)')
    parser.add_argument('--out', default=str(ROOT / 'outputs' / 'baseline.json'))
    args = parser.parse_args()
    specs = yaml.safe_load(open(ROOT / 'configs' / 'plan.yaml'))['imagery']['sources']
    results = {}
    for name, spec in specs.items():
        if args.datasets and name not in args.datasets:
            continue
        try:
            results[name] = run(name, spec, align=not args.no_align, max_hz=args.max_hz)
        except FileNotFoundError:
            print(f'skipping {name}: no store')
            continue
        r = results[name]
        print(f"{name:18s} chance {r['chance']:.3f} | within {r['within'].get('mean', float('nan')):.3f} "
              f"| cross {r['cross'].get('mean', float('nan')):.3f}"
              + (f" | cross+aux {r['cross_aux'].get('mean', float('nan')):.3f}" if 'cross_aux' in r else ''), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, 'w'), indent=1)

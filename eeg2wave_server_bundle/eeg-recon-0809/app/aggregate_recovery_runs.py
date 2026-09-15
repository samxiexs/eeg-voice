#!/usr/bin/env python3
"""Aggregate recovery-v3 runs (seeds / variants) into one seed-level summary.

For every run directory the selected checkpoint is the evaluation with the
lowest speech-frame mel MAE among evaluations that pass the validation
controls (the same rule that writes ``best_passed.pt``); runs with no passing
evaluation are reported with their best metric and flagged.  Seed-level means
carry a t-interval so that a claim rests on replication, not on one seed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from aligned_recovery_eval import passes_validation_controls

KEYS = ('native_mel_mae', 'template_mae', 'zero_gain', 'wrong_trial_gain', 'time_block_shuffle_gain',
        'envelope_corr', 'envelope_template_corr', 'envelope_zero_gain', 'retrieval_r1', 'retrieval_mrr', 'duration_corr')
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def select(history):
    passing = [row for row in history if passes_validation_controls(row['validation'])]
    pool = passing or history
    best = min(pool, key=lambda row: row['validation']['native_mel_mae'])
    return best, bool(passing)


def summarize(runs):
    rows = []
    for path in runs:
        metrics = json.loads((Path(path) / 'metrics.json').read_text())
        best, passed = select(metrics['history'])
        rows.append(dict(run=str(path), seed=metrics['signature'].get('seed'), update=best['update'], passed=passed,
                         **{k: best['validation'].get(k) for k in KEYS}))
    summary = {}
    n = len(rows)
    for key in KEYS:
        values = [r[key] for r in rows if r[key] is not None]
        if not values:
            continue
        mean = sum(values) / len(values)
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else float('nan')
        half = T95.get(len(values) - 1, 1.96) * sd / math.sqrt(len(values)) if len(values) > 1 else float('nan')
        summary[key] = dict(mean=mean, sd=sd, ci95=[mean - half, mean + half], n=len(values))
    gates = dict(runs=n, passing_runs=sum(r['passed'] for r in rows),
                 zero_gain_positive=sum(1 for r in rows if (r['zero_gain'] or 0) > 0),
                 wrong_trial_gain_positive=sum(1 for r in rows if (r['wrong_trial_gain'] or 0) > 0),
                 time_shuffle_gain_positive=sum(1 for r in rows if (r['time_block_shuffle_gain'] or 0) > 0))
    return dict(runs=rows, seed_level=summary, gates=gates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', help='run directories containing metrics.json')
    parser.add_argument('--output')
    args = parser.parse_args()
    result = summarize(args.runs)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"{'seed':>5} {'update':>6} {'pass':>5} " + ' '.join(f'{k[:14]:>14}' for k in KEYS))
    for r in result['runs']:
        print(f"{str(r['seed']):>5} {r['update']:>6} {str(r['passed']):>5} " +
              ' '.join(f"{(r[k] if r[k] is not None else float('nan')):>14.4f}" for k in KEYS))
    print('seed-level mean [95% t-interval]:')
    for key, v in result['seed_level'].items():
        print(f"  {key:26s} {v['mean']:+.4f}  [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  n={v['n']}")
    print('gates:', json.dumps(result['gates']))


if __name__ == '__main__':
    main()

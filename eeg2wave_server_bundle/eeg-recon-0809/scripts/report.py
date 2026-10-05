"""Pool the held-out people of all folds of each variant and print one table.

    python scripts/report.py outputs/imagery            # runs named <variant>_f<k>/evaluation.json
    python scripts/report.py outputs/imagery --json outputs/imagery/summary.json

Every person is held out exactly once per variant, so pooling folds gives one accuracy per
person; the table shows mean (SEM) over people, chance, and the one-sided Wilcoxon p.
Paired comparisons between variants use the same people.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegspeech.metrics import summarize                 # noqa: E402


def collect(root):
    pooled = defaultdict(lambda: defaultdict(dict))      # variant -> (dataset, metric) -> {subject: accuracy}
    chance = {}
    for path in sorted(Path(root).glob('*/evaluation.json')):
        variant = re.sub(r'_f\d+$', '', path.parent.name)
        report = json.load(open(path))
        for dataset, result in report['results'].items():
            for metric, values in result.get('per_subject', {}).items():
                pooled[variant][(dataset, metric)].update(values)
                chance[(dataset, metric)] = result.get(metric, {}).get('chance')
    return pooled, chance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root')
    parser.add_argument('--json')
    parser.add_argument('--reference', help='variant the others are compared with (paired Wilcoxon over people)')
    args = parser.parse_args()
    pooled, chance = collect(args.root)
    keys = sorted({k for v in pooled.values() for k in v})
    summary = {}
    print(f"{'dataset':18s} {'metric':10s} " + ' '.join(f'{v:>22s}' for v in pooled))
    for key in keys:
        cells = []
        for variant, table in pooled.items():
            if key not in table:
                cells.append(f"{'-':>22s}")
                continue
            s = summarize(table[key], chance.get(key) or float('nan'))
            summary.setdefault(variant, {})['/'.join(key)] = s
            mark = '*' if s.get('p_wilcoxon', 1) < .05 else ' '
            cells.append(f"{s['mean']:.3f} ({s['sem']:.3f}){mark} n={s['subjects']:<3d}")
        print(f'{key[0]:18s} {key[1]:10s} ' + ' '.join(cells) + f'  chance {chance.get(key) or float("nan"):.3f}')
    if args.reference and args.reference in pooled:
        print(f'\npaired vs {args.reference} (Wilcoxon over people, two-sided)')
        for variant, table in pooled.items():
            if variant == args.reference:
                continue
            for key in keys:
                a, b = pooled[args.reference].get(key, {}), table.get(key, {})
                common = sorted(set(a) & set(b))
                if len(common) >= 5:
                    diff = np.array([b[s] - a[s] for s in common])
                    diff[np.abs(diff) < 1e-9] = 0
                    p = stats.wilcoxon(diff, zero_method='zsplit').pvalue if np.any(diff != 0) else 1.
                    print(f'{variant:20s} {key[0]:18s} {key[1]:10s} diff {diff.mean():+.3f}  p={p:.3f}  n={len(common)}')
    if args.json:
        json.dump(summary, open(args.json, 'w'), indent=1)

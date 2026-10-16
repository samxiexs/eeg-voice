#!/usr/bin/env python3
"""Protocol check against Défossez et al. (2023): the same alignment model per dataset under switchable protocols.

  train     --dataset ds004408|brennan2019 --split sentence|content --subjects all|train --minutes N
  evaluate  paper-style word-anchored top-1 / top-10 (writes outputs/protocol_check/<run>/protocol.md)
  plan      only build and print the split plan (seconds per split, training windows, test candidates)

The paper's protocol is --split sentence --subjects all; ours is --split content --subjects train.
Example:
  .venv-aligned-local/bin/python scripts/protocol_check.py train --run brod-sentence-all --dataset ds004408 --split sentence --subjects all --minutes 30
  .venv-aligned-local/bin/python scripts/protocol_check.py evaluate --run brod-sentence-all
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('NUMBA_DISABLE_JIT', '1'); os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['train', 'evaluate', 'plan']); p.add_argument('--run', required=True)
    p.add_argument('--dataset', choices=['ds004408', 'brennan2019', 'etard', 'kymata_english'], default='ds004408')
    p.add_argument('--split', choices=['sentence', 'content'], default='sentence'); p.add_argument('--subjects', choices=['all', 'train'], default='all')
    p.add_argument('--minutes', type=float); p.add_argument('--steps', type=int); p.add_argument('--device', default='mps'); p.add_argument('--threads', type=int, default=6)
    p.add_argument('--seed', type=int, default=11); p.add_argument('--validate-minutes', type=float, default=5); p.add_argument('--bootstrap', type=int, default=2000)
    a = p.parse_args()
    from eegspeech.neurospeech.config import Config
    from eegspeech.neurospeech import protocol_check as P
    cfg = Config(); cache_dir = ROOT/'artifacts/softalign/cache_v2'; folder = ROOT/'artifacts/protocol_check'/a.run
    if a.command == 'plan':
        cache, plan, everyone = P.setup(cfg, cache_dir, a.dataset, a.split, a.subjects)
        print(json.dumps({'seconds_per_split': dict(plan.summary), 'train_windows': int(sum(len(v) for v in plan.grid['train'].values())),
                          'test_candidates': int(sum(len(v) for v in plan.words['test'].values())), 'trained_subjects': len(cache.subjects), 'all_subjects': len(everyone)}, indent=1)); return
    if a.command == 'train':
        deadline = time.time()+a.minutes*60 if a.minutes else time.time()+365*86400
        P.train(cfg, cache_dir, folder, a.dataset, a.split, a.subjects, deadline, a.device, a.threads, seed=a.seed, max_steps=a.steps, validate_minutes=a.validate_minutes)
        print(json.dumps({'run': a.run, 'done': True})); return
    print(json.dumps(P.evaluate(cfg, cache_dir, folder, ROOT/'outputs/protocol_check'/a.run, a.run, a.device, a.bootstrap), indent=1, default=str))


if __name__ == '__main__':
    main()

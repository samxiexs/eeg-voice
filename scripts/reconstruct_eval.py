#!/usr/bin/env python3
"""Adjusted evaluation of EEG-side reconstruction runs: literature-style 1-8 Hz envelope correlations over whole test
stretches (with the audio-side ceiling and circular-shift / no-EEG nulls), plus the fine-grained, word and sentence
measures. Writes outputs/reconstruct/<run>/eval2/evaluation.md.

  .venv-aligned-local/bin/python scripts/reconstruct_eval.py --run v3/eeg2-wavlm
"""
from pathlib import Path
import argparse
import json
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('NUMBA_DISABLE_JIT', '1'); os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', required=True, nargs='+'); p.add_argument('--device', default='mps'); p.add_argument('--checkpoint', default='best.pt')
    p.add_argument('--max-records', type=int, help='records per dataset (smoke checks)'); p.add_argument('--sentences', type=int, default=5)
    p.add_argument('--bootstrap', type=int, default=2000); p.add_argument('--permutations', type=int, default=1000); p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    from eegspeech.neurospeech.config import Config
    from eegspeech.neurospeech import softalign as S, recon_eval as E
    cfg = Config(); cache = S.Cache(ROOT/'artifacts/softalign'/('cache_v2_smoke' if a.smoke else 'cache_v2'), cfg.fold_subjects('P1', 'F0', 'train'))
    for run in a.run:
        E.evaluate(cfg, cache, ROOT/'artifacts/reconstruct'/run, ROOT/'outputs/reconstruct'/run/'eval2', run, a.device, a.checkpoint,
                   a.max_records, a.sentences, a.bootstrap, a.permutations)
        print(json.dumps({'run': run, 'evaluated': True}), flush=True)


if __name__ == '__main__':
    main()

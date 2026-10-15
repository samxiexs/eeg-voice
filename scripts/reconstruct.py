#!/usr/bin/env python3
"""EEG -> speech reconstruction (generation) on top of the two-tower alignment.

  audio     stage 1, no EEG: Mel decoder from the synthesis code of real audio (--route wavlm: before the projection,
            embedding: the 128-d shared embedding after it), then a test report with whole-sentence listening
  train     stage 2: only a small bridge learns EEG -> code; loss = code frame by frame + word-level contrast
  zeroshot  the stage-1 decoder on the latency-compensated EEG embedding, no EEG training (embedding route only)
  evaluate  frames, words and stitched sentences, against wrong-time and no-EEG nulls
  status    progress of a run
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
BASE, OUT = ROOT/'artifacts/reconstruct', ROOT/'outputs/reconstruct'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['audio', 'train', 'zeroshot', 'evaluate', 'status'])
    p.add_argument('--run', required=True); p.add_argument('--align', default='quick2/main', help='alignment run under artifacts/softalign')
    p.add_argument('--audio', help='stage-1 run under artifacts/reconstruct (train, zeroshot)')
    p.add_argument('--route', choices=['embedding', 'wavlm'], default='wavlm'); p.add_argument('--noise', type=float, default=0.)
    p.add_argument('--word-weight', type=float, default=1.)
    p.add_argument('--device', default='mps'); p.add_argument('--threads', type=int, default=6); p.add_argument('--seed', type=int, default=11)
    p.add_argument('--minutes', type=float); p.add_argument('--steps', type=int); p.add_argument('--validate-minutes', type=float, default=10)
    p.add_argument('--init', help='start the bridge from this checkpoint'); p.add_argument('--lr', type=float)
    p.add_argument('--limit', type=int, default=300, help='development windows per dataset at each check')
    p.add_argument('--checkpoint', default='best.pt'); p.add_argument('--permutations', type=int, default=1000); p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--sentences', type=int, default=5, help='listening sentences per dataset'); p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); folder = BASE/a.run
    if a.command == 'status':
        path = folder/'status.json'; print(path.read_text() if path.exists() else f'{a.run}: not started'); return
    from eegspeech.neurospeech.config import Config
    from eegspeech.neurospeech import softalign as S, reconstruct as R
    cfg = Config(); cache = S.Cache(ROOT/'artifacts/softalign'/('cache_v2_smoke' if a.smoke else 'cache_v2'), cfg.fold_subjects('P1', 'F0', 'train'))
    align = ROOT/'artifacts/softalign'/a.align; audio = BASE/a.audio if a.audio else None
    deadline = time.time()+a.minutes*60 if a.minutes else time.time()+365*86400
    if a.command == 'audio':
        R.train_audio_decoder(cache, align, folder, a.route, deadline, a.device, a.threads, seed=a.seed, max_steps=a.steps,
                              validate_minutes=a.validate_minutes, noise=a.noise, peak=a.lr or 1e-3)
        R.evaluate_audio_decoder(cfg, cache, folder, OUT/a.run, a.run, a.device, a.sentences, a.threads)
        print(json.dumps({'run': a.run, 'done': True})); return
    if a.command == 'evaluate':
        R.evaluate_reconstruction(cfg, cache, folder, OUT/a.run, a.run, a.device, a.checkpoint, a.threads, a.permutations, a.bootstrap, a.sentences)
        print(json.dumps({'run': a.run, 'evaluated': True})); return
    if audio is None: raise SystemExit('give --audio: the stage-1 run to decode with')
    if a.command == 'zeroshot': print(json.dumps(R.zero_shot(cfg, cache, align, audio, folder, a.device, a.limit))); return
    R.train_generator(cfg, cache, align, audio, folder, deadline, a.device, a.threads, seed=a.seed, word_weight=a.word_weight,
                      validate_minutes=a.validate_minutes, max_steps=a.steps, peak=a.lr or 5e-4, limit=a.limit, init=a.init)
    print(json.dumps({'run': a.run, 'done': True}))


if __name__ == '__main__':
    main()

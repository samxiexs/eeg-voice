#!/usr/bin/env python3
"""Generative (flow-matching) Mel decoder on top of the v3 two-stage reconstruction.

  train     --stage audio: conditioned on true codes of real audio (no EEG)
            --stage eeg:   conditioned on codes the frozen v3 bridge predicts from EEG (--bridge), usually --init from
                           an audio-stage run; --eeg-share < 1 mixes in true codes
  evaluate  the adjusted evaluation with the flow decoder (writes outputs/flow/<run>/evaluation.md)

Example:
  .venv-aligned-local/bin/python scripts/flow_decoder.py train --run flow-audio --stage audio --audio v3/audio-wavlm --minutes 60
  .venv-aligned-local/bin/python scripts/flow_decoder.py train --run flow-eeg --stage eeg --audio v3/audio-wavlm --bridge v3/eeg2-wavlm \
      --init artifacts/flow/flow-audio/flow.pt --minutes 60
  .venv-aligned-local/bin/python scripts/flow_decoder.py evaluate --run flow-eeg --steps 8
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
    p.add_argument('command', choices=['train', 'evaluate']); p.add_argument('--run', required=True)
    p.add_argument('--stage', choices=['audio', 'eeg'], default='audio'); p.add_argument('--audio', default='v3/audio-wavlm', help='stage-1 run under artifacts/reconstruct')
    p.add_argument('--bridge', help='v3 EEG-side run under artifacts/reconstruct (stage eeg)'); p.add_argument('--init', help='flow checkpoint to start from')
    p.add_argument('--align', default='quick2/main'); p.add_argument('--eeg-share', type=float, default=1.)
    p.add_argument('--minutes', type=float); p.add_argument('--steps', type=int, default=8); p.add_argument('--guidance', type=float, default=1.)
    p.add_argument('--device', default='mps'); p.add_argument('--threads', type=int, default=6); p.add_argument('--seed', type=int, default=11)
    p.add_argument('--validate-minutes', type=float, default=10); p.add_argument('--max-steps', type=int); p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--max-records', type=int, help='smoke checks only'); p.add_argument('--sentences', type=int, default=5)
    p.add_argument('--bootstrap', type=int, default=2000); p.add_argument('--permutations', type=int, default=1000)
    a = p.parse_args()
    from eegspeech.neurospeech.config import Config
    from eegspeech.neurospeech import softalign as S, flow_decoder as D
    cfg = Config(); cache = S.Cache(ROOT/'artifacts/softalign/cache_v2', cfg.fold_subjects('P1', 'F0', 'train'))
    folder = ROOT/'artifacts/flow'/a.run; base = ROOT/'artifacts/reconstruct'
    if a.command == 'train':
        if a.stage == 'eeg' and not a.bridge: raise SystemExit('stage eeg needs --bridge (a v3 EEG-side run)')
        deadline = time.time()+a.minutes*60 if a.minutes else time.time()+365*86400
        D.train_flow(cfg, cache, ROOT/'artifacts/softalign'/a.align, base/a.audio, folder, deadline, a.device, a.threads, seed=a.seed, stage=a.stage,
                     bridge=base/a.bridge if a.bridge else None, init=a.init, eeg_share=a.eeg_share, validate_minutes=a.validate_minutes,
                     max_steps=a.max_steps, peak=a.lr, steps=a.steps)
        print(json.dumps({'run': a.run, 'done': True})); return
    D.evaluate_flow(cfg, cache, folder, ROOT/'outputs/flow'/a.run, a.run, a.device, a.steps, a.guidance, a.max_records, a.sentences, a.bootstrap, a.permutations)
    print(json.dumps({'run': a.run, 'evaluated': True}))


if __name__ == '__main__':
    main()

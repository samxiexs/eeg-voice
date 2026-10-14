#!/usr/bin/env python3
"""Two-tower soft alignment of EEG and speech on one laptop: cache, align, decoder, bridge, evaluate, overnight."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('NUMBA_DISABLE_JIT', '1'); os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['cache', 'align', 'decoder', 'bridge', 'evaluate', 'overnight', 'status'])
    p.add_argument('--run', default='overnight'); p.add_argument('--device', default='cpu'); p.add_argument('--threads', type=int, default=6)
    p.add_argument('--until', help='HH:MM wall-clock deadline of a training stage'); p.add_argument('--steps', type=int)
    p.add_argument('--smoke', action='store_true', help='one stimulus per dataset, a few steps')
    p.add_argument('--start', default='18:30'); p.add_argument('--full-until', default='23:40'); p.add_argument('--align-until', default='02:30')
    p.add_argument('--decoder-until', default='03:15'); p.add_argument('--post-until', default='04:00')
    p.add_argument('--variant', default='full', choices=['full', 'synchronous', 'no-region-prior']); p.add_argument('--no-audio', action='store_true')
    a = p.parse_args()
    from eegspeech.neurospeech.config import Config
    from eegspeech.neurospeech import softalign as S
    cfg = Config(); cache_dir = ROOT/'artifacts/softalign'/('cache_smoke' if a.smoke else 'cache')
    folder = ROOT/'artifacts/softalign'/a.run; out = ROOT/'outputs/softalign'/a.run; folder.mkdir(parents=True, exist_ok=True)
    deadline = S.wall(a.until) if a.until else time.time()+365*86400
    train = cfg.fold_subjects('P1', 'F0', 'train')
    if a.command == 'status':
        print(json.dumps(json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {'status': 'not started'}, indent=1, ensure_ascii=False)); return
    if a.command == 'cache':
        S.build_cache(cfg, cache_dir, 1 if a.smoke else None); return
    cache = S.Cache(cache_dir, train) if a.command != 'overnight' else None
    if a.command == 'align': print(json.dumps(S.train_alignment(cfg, cache, folder, deadline, a.device, a.threads, max_steps=a.steps, variant=a.variant)['mean_rank_percentile'])); return
    if a.command == 'decoder': print(S.train_decoder(cache, folder, deadline, a.device, a.threads, max_steps=a.steps)); return
    if a.command == 'bridge': print(S.train_bridge(cfg, cache, folder, deadline, a.device, a.threads, max_steps=a.steps)); return
    if a.command == 'evaluate': S.evaluate(cfg, cache, folder, out, a.device, threads=a.threads, audio=not a.no_audio); return
    # overnight: wait for the start time, cache, align, then decoder and bridge side by side, then evaluate.
    start = S.wall(a.start)
    while time.time() < start:
        S.status(folder, stage=f'waiting for {a.start}'); time.sleep(min(60, max(1, start-time.time())))
    S.status(folder, stage='cache'); S.build_cache(cfg, cache_dir, None)
    logs = ROOT/'logs/softalign'; logs.mkdir(parents=True, exist_ok=True); me = [sys.executable, str(Path(__file__).resolve())]
    # One job at a time: three processes sharing the Mac GPU ran ten times slower each.
    middle = time.strftime('%H:%M', time.localtime((S.wall(a.full_until)+S.wall(a.align_until))/2))
    runs = {'full': a.run, 'synchronous': a.run+'-synchronous', 'no-region-prior': a.run+'-no-region-prior'}
    plan = [(['align', '--run', runs['full'], '--variant', 'full', '--until', a.full_until], 'full_align'),
            (['align', '--run', runs['synchronous'], '--variant', 'synchronous', '--until', middle], 'synchronous_align'),
            (['align', '--run', runs['no-region-prior'], '--variant', 'no-region-prior', '--until', a.align_until], 'no-region-prior_align'),
            (['decoder', '--run', a.run, '--until', a.decoder_until], 'decoder'),
            (['bridge', '--run', a.run, '--until', a.post_until], 'bridge'),
            (['evaluate', '--run', runs['synchronous'], '--no-audio'], 'synchronous_evaluate'),
            (['evaluate', '--run', runs['no-region-prior'], '--no-audio'], 'no-region-prior_evaluate'),
            (['evaluate', '--run', a.run], 'full_evaluate')]
    for arguments, name in plan:
        S.status(folder, stage=name)
        code = subprocess.run(me+arguments+['--device', a.device, '--threads', '6'], stdout=open(logs/f'{a.run}_{name}.log', 'w'), stderr=subprocess.STDOUT).returncode
        S.status(folder, **{f'{name}_exit': code})
    S.status(folder, stage='done')


if __name__ == '__main__':
    main()

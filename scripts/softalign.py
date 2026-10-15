#!/usr/bin/env python3
"""Two-tower alignment of EEG and speech on one laptop.

  cache      speech-model layers, envelope and Mel of every clean stimulus, block by block (once)
  align      train one variant for --minutes (or until --until HH:MM, or for --steps)
  evaluate   retrieval of unseen speech: tie-aware ranks, permutation test, bootstrap, generation gate
  evaluate-words  top-1 / top-10 among word-anchored candidates (the rule of Défossez et al. 2023)
  decoder    Mel decoder on the audio-side embedding (test version)   | only after the run passed
  bridge     EEG -> audio-side embedding                               | the retrieval gate,
  audio      test audio of unseen speech                               | or with --force-audio
  compare    single-factor variants against the main model at the same training time
  overnight  all of the above, one job at a time; it waits only when --start is given
  status     progress of a run and its parts

Nothing here starts by itself; every command is run by hand.
"""
from pathlib import Path
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('NUMBA_DISABLE_JIT', '1'); os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
BASE, OUT, LOGS = ROOT/'artifacts/softalign', ROOT/'outputs/softalign', ROOT/'logs/softalign'


def alive(pid):
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


def deadline(a, S):
    if a.minutes: return time.time()+a.minutes*60
    if a.until: return S.wall(a.until)
    if a.steps: return time.time()+365*86400
    raise SystemExit('give --minutes, --until or --steps')


def gate_passed(run):
    path = OUT/run/'retrieval.json'
    return path.exists() and json.loads(path.read_text())['generation_gate_passed']


def show(run):
    top = BASE/run
    if not top.exists(): print(f'{run}: not started'); return
    pid = int((top/'orchestrator.pid').read_text()) if (top/'orchestrator.pid').exists() else None
    print(f"{run}: {'running' if pid and alive(pid) else 'not running'}"+(f' (pid {pid})' if pid else ''))
    for path in sorted(top.rglob('status.json')):
        v = json.loads(path.read_text()); parts = [str(v.get('stage', '?'))]
        parts += [f'{k} {v[k]:.3f}' if isinstance(v[k], float) else f'{k} {v[k]}' for k in ('step', 'validation', 'best', 'child') if v.get(k) is not None]
        failed = [k[6:] for k, x in v.items() if k.startswith('exit: ') and x]
        if failed: parts.append('FAILED: '+', '.join(failed))
        print(f"  {path.parent.relative_to(BASE)}: {', '.join(parts)} (updated {v.get('updated', '?')})")
    logs = sorted((LOGS/run).glob('*.log'), key=lambda q: q.stat().st_mtime)
    if logs:
        last = logs[-1]; print(f'  latest log {last.relative_to(ROOT)}, last written {(time.time()-last.stat().st_mtime)/60:.0f} min ago:')
        for line in last.read_text(errors='replace').splitlines()[-3:]: print('    '+line[:160])


def overnight(a, S):
    """Cache; main model with a snapshot at the variants' training time; each variant; their retrieval;
    the comparison; decoder, bridge and test audio only if the main model passed the retrieval gate."""
    top, logs = BASE/a.run, LOGS/a.run; top.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    pid_file = top/'orchestrator.pid'
    if pid_file.exists() and alive(int(pid_file.read_text())): raise SystemExit(f'{a.run} is already running (pid {pid_file.read_text()})')
    pid_file.write_text(str(os.getpid())); child = {}

    def stop(signum, frame):
        process = child.get('process')
        if process and process.poll() is None: process.terminate()
        S.status(top, stage=f"stopped by signal {signum} during {child.get('name')}", child=None); sys.exit(1)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    if a.start:
        begin = S.wall(a.start)
        while time.time() < begin: S.status(top, stage=f'waiting for {a.start}', pid=os.getpid()); time.sleep(min(60, max(1, begin-time.time())))
    me = [sys.executable, str(Path(__file__).resolve())]
    common = ['--device', a.device, '--threads', str(a.threads)]+(['--smoke'] if a.smoke else [])
    length = (lambda minutes: ['--steps', '40']) if a.smoke else (lambda minutes: ['--minutes', str(minutes)])
    stats = ['--permutations', '50', '--bootstrap', '100'] if a.smoke else []
    m = a.variant_minutes
    jobs = [('cache', ['cache'], True),
            ('main: alignment', ['align', '--run', f'{a.run}/main', '--variant', 'main', *length(a.main_minutes), '--snapshot-minutes', str(m)], True),
            ('main: retrieval', ['evaluate', '--run', f'{a.run}/main', *stats], False)]
    for v in a.variants:
        jobs += [(f'{v}: alignment', ['align', '--run', f'{a.run}/{v}', '--variant', v, *length(m), '--horizon-minutes', str(a.main_minutes)], False),
                 (f'{v}: retrieval', ['evaluate', '--run', f'{a.run}/{v}', *stats], False)]
    jobs += [(f'main at {m} min: retrieval', ['evaluate', '--run', f'{a.run}/main', '--checkpoint', f'best_{m}min.pt', *stats], False),
             ('comparison', ['compare', '--run', a.run, '--variant-minutes', str(m), '--variants', *a.variants], False)]
    S.status(top, stage='starting', pid=os.getpid(), began=time.strftime('%Y-%m-%d %H:%M'), smoke=a.smoke,
             plan=[name for name, _, _ in jobs]+['decoder, bridge, audio: only if the main model passes the retrieval gate'])

    def run(name, arguments):
        child['name'] = name; S.status(top, stage=name, stage_began=time.strftime('%H:%M'))
        with open(logs/(re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')+'.log'), 'w') as log:
            child['process'] = subprocess.Popen(me+arguments+common, stdout=log, stderr=subprocess.STDOUT)
            S.status(top, child=child['process'].pid); code = child['process'].wait()
        S.status(top, child=None, **{f'exit: {name}': code}); return code

    for name, arguments, required in jobs:
        if run(name, arguments) and required: S.status(top, stage=f'stopped: {name} failed, see {logs.relative_to(ROOT)}'); return
    if gate_passed(f'{a.run}/main') or a.force_audio:
        force = ['--force-audio'] if a.force_audio else []
        for name, arguments in (('decoder', ['decoder', '--run', f'{a.run}/main', *length(a.decoder_minutes)]),
                                ('bridge', ['bridge', '--run', f'{a.run}/main', *length(a.bridge_minutes)]),
                                ('audio', ['audio', '--run', f'{a.run}/main'])):
            if run(name, arguments+force): break
    else: S.status(top, generation='skipped: the main model did not pass the retrieval gate')
    S.status(top, stage='done', finished=time.strftime('%Y-%m-%d %H:%M'))


def main():
    from eegspeech.neurospeech import softalign as S
    others = [v for v in S.VARIANTS if v != 'main']
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['cache', 'align', 'evaluate', 'evaluate-words', 'decoder', 'bridge', 'audio', 'compare', 'overnight', 'status'])
    p.add_argument('--run', default='night', help='run name; overnight puts its parts under <run>/<variant>')
    p.add_argument('--device', default='cpu'); p.add_argument('--threads', type=int, default=6)
    p.add_argument('--smoke', action='store_true', help='small cache (one stimulus per split and dataset) and a few steps: checks the code, not the method')
    p.add_argument('--variant', default='main', choices=list(S.VARIANTS))
    p.add_argument('--minutes', type=int, help='training time of align / decoder / bridge'); p.add_argument('--until', help='HH:MM instead of --minutes')
    p.add_argument('--steps', type=int, help='stop after this many steps')
    p.add_argument('--horizon-minutes', type=int, help='length of the learning-rate schedule (default: the training time)')
    p.add_argument('--snapshot-minutes', type=int, help='also keep the best checkpoint at this training time as best_<m>min.pt')
    p.add_argument('--validate-minutes', type=int, default=20, help='minutes between validations during alignment')
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--checkpoint', default='best.pt'); p.add_argument('--permutations', type=int, default=1000); p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--force-audio', action='store_true', help='generate audio although retrieval did not pass the gate (diagnostic only)')
    p.add_argument('--start', help='overnight: wait until HH:MM'); p.add_argument('--main-minutes', type=int, default=210)
    p.add_argument('--variant-minutes', type=int, default=50); p.add_argument('--decoder-minutes', type=int, default=35); p.add_argument('--bridge-minutes', type=int, default=35)
    p.add_argument('--variants', nargs='*', default=others, choices=others)
    p.add_argument('--runs', nargs='*', help='compare: these parts of --run, the first one as the reference')
    a = p.parse_args()
    if a.command == 'status': return show(a.run)
    if a.command == 'overnight': return overnight(a, S)
    from eegspeech.neurospeech.config import Config
    cfg = Config(); cache_dir = BASE/('cache_v2_smoke' if a.smoke else 'cache_v2'); folder = BASE/a.run
    if a.command == 'cache': S.build_cache(cfg, cache_dir, a.smoke); return
    if a.command == 'compare':
        if a.runs: S.compare([OUT/a.run/r/'retrieval.json' for r in a.runs], OUT/a.run/'comparison.md'); return
        matched = [OUT/a.run/f'main@best_{a.variant_minutes}min'/'retrieval.json']+[OUT/a.run/v/'retrieval.json' for v in a.variants]
        S.compare([q for q in matched if q.exists()], OUT/a.run/'comparison.md', a.variant_minutes, OUT/a.run/'main'/'retrieval.json'); return
    cache = S.Cache(cache_dir, cfg.fold_subjects('P1', 'F0', 'train'))
    if a.command == 'align':
        score = S.train_alignment(cfg, cache, folder, deadline(a, S), a.device, a.threads, validate_minutes=a.validate_minutes, seed=a.seed, max_steps=a.steps, variant=a.variant,
                                  snapshot_minutes=a.snapshot_minutes, horizon=a.horizon_minutes*60 if a.horizon_minutes else None)
        print(json.dumps({'run': a.run, 'variant': a.variant, 'validation_mean_rank_percentile': score['mean_rank_percentile']})); return
    if a.command == 'evaluate':
        out = OUT/(a.run if a.checkpoint == 'best.pt' else f'{a.run}@{Path(a.checkpoint).stem}')
        summary = S.evaluate_retrieval(cfg, cache, folder, out, a.run, a.device, a.checkpoint, a.threads, a.permutations, a.bootstrap)
        print(json.dumps({'run': a.run, 'checkpoint': a.checkpoint, 'generation_gate_passed': summary['generation_gate_passed']})); return
    if a.command == 'evaluate-words':
        out = OUT/(a.run if a.checkpoint == 'best.pt' else f'{a.run}@{Path(a.checkpoint).stem}')
        S.evaluate_word_anchored(cfg, cache, folder, out, a.run, a.device, a.checkpoint, a.threads, a.bootstrap); print(json.dumps({'run': a.run, 'done': True})); return
    passed = gate_passed(a.run)
    if not passed and not a.force_audio:
        raise SystemExit(f'{a.run}: retrieval has not passed the generation gate (see {OUT/a.run}/retrieval.md); --force-audio runs it as a diagnostic')
    if a.command == 'decoder': print(S.train_decoder(cache, folder, deadline(a, S), a.device, a.threads, max_steps=a.steps)); return
    if a.command == 'bridge': print(S.train_bridge(cfg, cache, folder, deadline(a, S), a.device, a.threads, max_steps=a.steps)); return
    S.evaluate_audio(cfg, cache, folder, OUT/a.run, a.run, a.device, threads=a.threads, forced=not passed)


if __name__ == '__main__':
    main()

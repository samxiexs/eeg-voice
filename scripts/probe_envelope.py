#!/usr/bin/env python3
"""Sanity check of the processed store: linear reconstruction of the speech envelope from EEG.

Classical backward model per subject (ridge over 0–250 ms of EEG after each audio
sample, 1–8 Hz, 64 Hz), fitted on training content and scored on held-out content
of the same person. Published values for this kind of model are roughly r = 0.05–0.2.
A clear positive means EEG and audio are paired correctly in the processed data;
the null is the same prediction scored against circularly shifted envelopes.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import numpy as np
import soundfile as sf
from scipy import signal
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('NUMBA_DISABLE_JIT', '1')
RATE, LAGS, GUARD = 64, 17, 10.0
BAND = signal.butter(2, [1, 8], btype='band', fs=RATE, output='sos')


def envelope(path, start, seconds, length):
    """Broadband 25 ms RMS of the canonical audio at 64 Hz on the record's own time base."""
    with sf.SoundFile(path) as f:
        f.seek(round(start*24000)); wave = f.read(round(seconds*24000), dtype='float32')
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(wave, 300), 600)[::375][:length]
    return np.sqrt(np.square(frames).mean(1)+1e-12)


def record_pair(folder, record, audio_path):
    eeg = np.load(folder/record['path']/'eeg.npy').T.astype(np.float64)
    eeg = signal.resample_poly(eeg, 8, 25, axis=0)
    env = envelope(audio_path, record['audio_start'], record['duration'], len(eeg))
    n = min(len(eeg), len(env)); g = int(GUARD*RATE)
    if n <= 2*g+5*RATE: return None
    eeg = signal.sosfiltfilt(BAND, eeg[:n], axis=0)[g:n-g]; env = signal.sosfiltfilt(BAND, env[:n])[g:n-g]
    eeg[:, ~np.asarray(record['good_channels'], bool)] = 0
    return eeg.astype(np.float32), env.astype(np.float32)


def lagged(eeg):
    """Envelope at t from EEG at t .. t+250 ms (the brain responds after the sound)."""
    return np.concatenate([np.pad(eeg[k:], ((0, k), (0, 0))) for k in range(LAGS)], 1)[:len(eeg)-LAGS]


def fit(pairs, scale):
    x = [lagged(e) for e, _ in pairs]; y = [v[:len(e)-LAGS] for e, v in pairs]
    mean = np.concatenate(x).mean(0); sd = np.concatenate(x).std(0)+1e-6
    xx = sum(((a-mean)/sd).T@((a-mean)/sd) for a in x); xy = sum(((a-mean)/sd).T@(b-b.mean()) for a, b in zip(x, y))
    return {lam: (np.linalg.solve(xx+lam*np.trace(xx)/len(xx)*np.eye(len(xx)), xy), mean, sd) for lam in scale}


def predict(model, pairs):
    w, mean, sd = model
    return [(((lagged(e)-mean)/sd)@w, v[:len(e)-LAGS]) for e, v in pairs]


def corr(pairs):
    p = np.concatenate([a for a, _ in pairs]); t = np.concatenate([b for _, b in pairs])
    return float(np.corrcoef(p, t)[0, 1])


def main():
    from eegspeech.neurospeech.config import Config
    from eegspeech.utils import read_json
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', default=str(ROOT/'configs/neurospeech_v2/probe_mac.yaml'))
    parser.add_argument('--output', default=str(ROOT/'artifacts/neurospeech_probe/envelope_check.json')); a = parser.parse_args()
    cfg = Config(a.config); registry = cfg.registry(); rng = np.random.default_rng(11); scale = (1e-3, 1e-2, 1e-1, 1, 10)
    results = {}
    for dataset in ('ds004408', 'brennan2019', 'etard', 'kymata_english'):
        folder = cfg.dir('processed')/dataset; manifest = read_json(folder/'manifest.json'); splits = registry['content_splits'][dataset]
        people = sorted(s.split(':', 1)[1] for s in registry['subjects'][dataset]); results[dataset] = []
        for person in people:
            records = [r for r in manifest['records'].values() if r['subject'] == person and r['task'] == 'perceived' and r['metadata'].get('condition', 'clean') == 'clean']
            train, test = [], []
            for r in records:
                pair = record_pair(folder, r, folder/manifest['contents'][r['audio']]['path'])
                if pair is not None: (train if splits[r['content_group']] == 'train' else test).append(pair)
            if len(train) < 2 or not test: continue
            half = len(train)//2; models = fit(train[:half], scale); other = fit(train[half:], scale)
            # Ridge strength from training content only: fit on one half, score on the other, both ways.
            lam = max(scale, key=lambda s: corr(predict(models[s], train[half:]))+corr(predict(other[s], train[:half])))
            held = predict(fit(train, (lam,))[lam], test); r = corr(held); null = []
            for _ in range(200):
                shifted = [(p, np.roll(t, int(rng.integers(RATE*5, len(t)-RATE*5)))) for p, t in held if len(t) > RATE*12]
                if shifted: null.append(corr(shifted))
            results[dataset].append({'subject': person, 'r': r, 'null_95': float(np.quantile(null, .95)), 'p': float((1+np.sum(np.asarray(null) >= r))/(1+len(null))),
                                     'train_minutes': sum(len(v) for _, v in train)/RATE/60, 'test_minutes': sum(len(v) for _, v in test)/RATE/60, 'ridge': lam})
            print(json.dumps({'dataset': dataset, **results[dataset][-1]}), flush=True)
    summary = {d: {'subjects': len(v), 'median_r': float(np.median([x['r'] for x in v])), 'median_null_95': float(np.median([x['null_95'] for x in v])),
                   'subjects_above_null_95': int(sum(x['r'] > x['null_95'] for x in v))} for d, v in results.items() if v}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(json.dumps({'summary': summary, 'subjects': results}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

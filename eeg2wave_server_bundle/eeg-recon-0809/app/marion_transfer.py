#!/usr/bin/env python3
"""Does perceived-speech pre-training help a listen -> imagine decoder?  (Marion et al. 2021)

The decoder is trained on LISTENING trials only - all 21 musicians, repetitions
with rep % 4 != 3; rep % 4 == 3 is the listening validation set used for model
selection - and tested on every IMAGERY trial, which neither training nor
selection ever sees.  The listener's own listening data are in training: the
calibration a practical imagined-speech system would have.

The network is the subject-free encoder (app/universal_model.py; electrode
positions, no participant index) with a 1-channel head predicting the smoothed
note-onset train at the 64 Hz token rate from 4 s windows (no onset clock).
Arms differ only in the encoder's initial weights:
  scratch   random initialisation
  speech    outputs/universal/speech_seed322 (DS004940, mel loss)
  mfcc      outputs/universal/mfcc_broderick_seed322 (DS004940 + Broderick, MFCC-80 loss)
plus a pooled linear ridge (lagged raw EEG, same training trials) as the reference.

Scoring is the balanced design of app/marion_imagery_gate.py: a test trial's
prediction is correlated with all four melodies' onset trains, every melody is
in training, so correct-minus-wrong and 4-way identification are unbiased;
per-participant accuracy -> group mean with a sign-flip test.  Input: the HDF5
release of scripts/convert_marion.py (shipped scale; per-trial channel z-score
after a 0.5 Hz high-pass, clipped at 10 SD; upsampled to 256 Hz per window).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
import h5py
import numpy as np
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src'))
from eeg2speech.aligned import EEG_SAMPLES, EEG_START          # noqa: E402

FS, UP = 64, 4                                   # shipped rate; upsampling to the encoder's 256 Hz
LEAD = int(round(-EEG_START * FS))               # 16 samples: EEG window starts 0.25 s before the audio window
SPAN = 4 * FS                                    # 256 tokens of one 4 s audio window at 64 Hz
PAD = 8
CHECKPOINTS = {'speech': ROOT / 'outputs/universal/speech_seed322/best_passed.pt',
               'mfcc': ROOT / 'outputs/universal/mfcc_broderick_seed322/best_passed.pt'}


def onset_train(stim, sigma=3.):
    impulses = (np.abs(stim) > 0).astype(np.float64)
    half = int(4 * sigma)
    kernel = np.exp(-.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    return np.convolve(impulses, kernel, 'same').astype(np.float32)


def load(path):
    with h5py.File(path, 'r') as h5:
        eeg, stim = h5['eeg'][:], h5['stim'][:]
    sos = butter(2, .5, 'highpass', fs=FS, output='sos')
    x = np.empty(eeg.shape, np.float32)                               # (S, 2, T, 1803, 64), one participant at a time
    for s in range(eeg.shape[0]):
        y = sosfiltfilt(sos, eeg[s].astype(np.float64), axis=-2)
        y = (y - y.mean(-2, keepdims=True)) / (y.std(-2, keepdims=True) + 1e-8)
        x[s] = np.clip(y, -10, 10)
    del eeg
    melodies, keys = np.zeros(stim.shape[:3], int), {}
    reps = np.zeros(stim.shape[:3], int)
    for s in range(stim.shape[0]):
        for c in range(2):
            seen = {}
            for t in range(stim.shape[2]):
                m = keys.setdefault(tuple(np.flatnonzero(stim[s, c, t])), len(keys))
                melodies[s, c, t] = m; reps[s, c, t] = seen.get(m, 0); seen[m] = reps[s, c, t] + 1
    if len(keys) != 4:
        raise SystemExit(f'expected 4 melodies, found {len(keys)}')
    targets = {}
    for (s, c, t), m in np.ndenumerate(melodies):
        targets.setdefault(m, onset_train(stim[s, c, t]))
    onsets = np.stack([[[onset_train(stim[s, c, t]) for t in range(stim.shape[2])] for c in range(2)] for s in range(stim.shape[0])])
    return x, onsets, melodies, reps, targets


def windows(x, starts):
    """x (1803, 64) z-scored trial; starts = audio-window starts in 64 Hz samples -> (n, 64, 1178) at 256 Hz."""
    out = []
    for k in starts:
        first = k - LEAD
        seg = x[first - PAD:first + EEG_SAMPLES // UP + 1 + PAD].T
        up = resample_poly(seg, UP, 1, axis=1)
        out.append(up[:, PAD * UP:PAD * UP + EEG_SAMPLES])
    return np.stack(out).astype(np.float32)


def pearson(a, b):
    a = a - a.mean(-1, keepdim=True); b = b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-8)


class Decoder(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Conv1d(encoder.output_norm.normalized_shape[0], 1, 1)

    def forward(self, eeg, xyz):
        n = len(eeg)
        hidden = self.encoder.encode(eeg, xyz.expand(n, -1, -1), torch.ones(n, eeg.shape[1], dtype=torch.bool, device=eeg.device),
                                     torch.ones(n, EEG_SAMPLES, dtype=torch.bool, device=eeg.device),
                                     torch.zeros(n, dtype=torch.bool, device=eeg.device))
        return self.head(hidden.transpose(1, 2))[:, 0, LEAD:LEAD + SPAN]                # tokens of the 4 s audio window


def build(arm, device):
    import universal_train as ut
    from universal_model import UniversalEEGModel
    model, payload = ut.load_universal(CHECKPOINTS['mfcc'])
    if arm == 'scratch':
        torch.manual_seed(0)
        model = UniversalEEGModel(model.decoders, **payload['model_spec'])
    elif arm != 'mfcc':
        model, _ = ut.load_universal(CHECKPOINTS[arm])
    return Decoder(model).to(device)


def biosemi64(device):
    import mne
    xyz = np.array(list(mne.channels.make_standard_montage('biosemi64').get_positions()['ch_pos'].values()))
    return torch.tensor(xyz / np.linalg.norm(xyz, axis=1, keepdims=True) * .095, dtype=torch.float32, device=device)


def last_start(n_samples):
    """Largest audio-window start (64 Hz samples) whose padded EEG window fits in the trial."""
    return n_samples - (EEG_SAMPLES // UP + 1 + PAD) + LEAD


def trial_starts(n_samples):
    last = last_start(n_samples)
    starts = list(range(LEAD + PAD, last, SPAN))
    if starts[-1] < last:
        starts.append(last)                                              # the tail, overlapping the previous window
    return starts


@torch.no_grad()
def predict_trial(model, x, xyz, device):
    """Stitched 64 Hz prediction over the windows covering the trial (uncovered samples = NaN)."""
    starts = trial_starts(len(x))
    out = np.full(len(x), np.nan, np.float32)
    batch = torch.from_numpy(windows(x, starts)).to(device)
    pred = model(batch, xyz).cpu().numpy()
    for k, p in zip(starts, pred):
        out[k:k + SPAN] = p
    return out


def score(pred, melody, targets):
    ok = np.isfinite(pred)
    r = {m: float(np.corrcoef(pred[ok], y[ok])[0, 1]) for m, y in targets.items()}
    wrong = float(np.mean([v for m, v in r.items() if m != melody]))
    return r[melody] - wrong, int(max(r, key=r.get) == melody)


def summarise(per_subject, label):
    delta = np.array([np.mean([d for d, _ in v]) for v in per_subject]); acc = np.array([np.mean([h for _, h in v]) for v in per_subject])
    rng = np.random.default_rng(0); flips = rng.choice([-1, 1], size=(20000, len(acc)))
    p = float((np.abs((flips * (acc - .25)).mean(1)) >= abs((acc - .25).mean())).mean())
    return dict(label=label, delta=float(delta.mean()), delta_positive=int((delta > 0).sum()), accuracy=float(acc.mean()),
                accuracy_se=float(acc.std(ddof=1) / math.sqrt(len(acc))), p_accuracy=p, per_subject_accuracy=acc.round(4).tolist())


@torch.no_grad()
def evaluate(model, data, xyz, device, condition, rep_filter, chunk=16):
    """Per-participant (delta, hit) rows; windows of ``chunk`` trials go through the encoder in one batch."""
    x, onsets, melodies, reps, targets = data
    model.eval()
    starts = trial_starts(x.shape[3])
    trials = [(s, t) for s in range(x.shape[0]) for t in range(x.shape[2]) if rep_filter(reps[s, condition, t])]
    out = {s: [] for s in range(x.shape[0])}
    for i in range(0, len(trials), chunk):
        group = trials[i:i + chunk]
        batch = torch.from_numpy(np.concatenate([windows(x[s, condition, t], starts) for s, t in group])).to(device)
        pred = model(batch, xyz).cpu().numpy().reshape(len(group), len(starts), SPAN)
        for (s, t), p in zip(group, pred):
            stitched = np.full(x.shape[3], np.nan, np.float32)
            for k, w in zip(starts, p):
                stitched[k:k + SPAN] = w
            out[s].append(score(stitched, melodies[s, condition, t], targets))
    return [out[s] for s in range(x.shape[0])]


def train_arm(arm, data, device, args, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    x, onsets, melodies, reps, targets = data
    xyz = biosemi64(device)
    model = build(arm, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.05)
    train = [(s, t) for s in range(x.shape[0]) for t in range(x.shape[2]) if reps[s, 0, t] % 4 != 3]
    lo, hi = LEAD + PAD, last_start(x.shape[3]) + 1
    best, best_state, history = -np.inf, None, []
    started = time.monotonic()
    for step in range(1, args.updates + 1):
        tick = time.monotonic()
        lr = args.lr * min(1., step / 100) * .5 * (1 + math.cos(math.pi * step / args.updates))
        for g in optimizer.param_groups:
            g['lr'] = lr
        picks = [train[i] for i in rng.integers(len(train), size=args.batch)]
        ks = rng.integers(lo, hi, size=args.batch)
        eeg = torch.from_numpy(np.concatenate([windows(x[s, 0, t], [k]) for (s, t), k in zip(picks, ks)])).to(device)
        target = torch.from_numpy(np.stack([onsets[s, 0, t, k:k + SPAN] for (s, t), k in zip(picks, ks)])).to(device)
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = (1 - pearson(model(eeg, xyz), target)).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
        if args.throttle > 0:
            time.sleep(args.throttle * (time.monotonic() - tick))
        if step % args.eval_every == 0 or step == args.updates:
            val = summarise(evaluate(model, data, xyz, device, 0, lambda r: r % 4 == 3), 'listen_val')
            history.append(dict(step=step, loss=float(loss), listen_val_delta=val['delta'], listen_val_acc=val['accuracy'],
                                seconds=round(time.monotonic() - started)))
            print(json.dumps({'arm': arm, 'seed': seed, **history[-1]}), flush=True)
            if val['delta'] > best:
                best = val['delta']; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    selected = max(history, key=lambda h: h['listen_val_delta'])['step']
    return dict(arm=arm, seed=seed, selected_step=selected, history=history,
                listen_val=summarise(evaluate(model, data, xyz, device, 0, lambda r: r % 4 == 3), 'listen_val'),
                imagine=summarise(evaluate(model, data, xyz, device, 1, lambda r: True), 'imagine'))


def linear_reference(data):
    """Pooled ridge on lagged raw EEG (64 Hz, lags -125..+406 ms) trained on the same listening trials."""
    x, onsets, melodies, reps, targets = data
    lags = list(range(-8, 27))
    def lagged(trial):
        out = np.zeros((len(trial), len(lags) * trial.shape[1]), np.float32)
        for i, lag in enumerate(lags):
            block = out[:, i * trial.shape[1]:(i + 1) * trial.shape[1]]
            if lag >= 0:
                block[:len(trial) - lag] = trial[lag:]
            else:
                block[-lag:] = trial[:len(trial) + lag]
        return out
    xtx = 0; xty = 0
    for s in range(x.shape[0]):
        for t in range(x.shape[2]):
            if reps[s, 0, t] % 4 != 3:
                a = lagged(x[s, 0, t]).astype(np.float64); xtx = xtx + a.T @ a; xty = xty + a.T @ onsets[s, 0, t]
    values, vectors = np.linalg.eigh(xtx)
    scale = np.trace(xtx) / len(xtx)
    best = None
    for alpha in (1e-2, 1e-1, 1., 10., 100.):
        w = vectors @ ((vectors.T @ xty) / (values + alpha * scale))
        val = [[score(lagged(x[s, 0, t]) @ w, melodies[s, 0, t], targets) for t in range(x.shape[2]) if reps[s, 0, t] % 4 == 3]
               for s in range(x.shape[0])]
        summary = summarise(val, 'listen_val')
        if best is None or summary['delta'] > best[0]['delta']:
            best = (summary, w, alpha)
    summary, w, alpha = best
    imagine = [[score(lagged(x[s, 1, t]) @ w, melodies[s, 1, t], targets) for t in range(x.shape[2])] for s in range(x.shape[0])]
    return dict(arm='linear', alpha=alpha, listen_val=summary, imagine=summarise(imagine, 'imagine'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default=str(ROOT / 'artifacts/marion2021/imagery.h5'))
    parser.add_argument('--arms', default='linear,scratch,speech,mfcc')
    parser.add_argument('--seeds', default='0')
    parser.add_argument('--updates', type=int, default=1500)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--throttle', type=float, default=1.)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', default=str(ROOT / 'outputs/marion2021/transfer.json'))
    args = parser.parse_args()
    device = torch.device('mps' if args.device == 'auto' and torch.backends.mps.is_available() else
                          ('cpu' if args.device == 'auto' else args.device))
    data = load(args.data)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(output.read_text()) if output.exists() else {}
    for arm in args.arms.split(','):
        for seed in ([0] if arm == 'linear' else [int(s) for s in args.seeds.split(',')]):
            key = f'{arm}_seed{seed}'
            if key in results:
                continue
            results[key] = linear_reference(data) if arm == 'linear' else train_arm(arm, data, device, args, seed)
            output.write_text(json.dumps(results, indent=1))
            r = results[key]
            print(f"{key:16s} listen-val acc {r['listen_val']['accuracy']:.3f} | imagine acc {r['imagine']['accuracy']:.3f} "
                  f"(p={r['imagine']['p_accuracy']:.4f}) delta {r['imagine']['delta']:+.4f} ({r['imagine']['delta_positive']}/21)", flush=True)


if __name__ == '__main__':
    main()

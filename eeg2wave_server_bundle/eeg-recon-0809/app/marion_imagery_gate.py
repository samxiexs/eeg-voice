#!/usr/bin/env python3
"""Listen -> imagine gate on Marion et al. 2021 ("The music of silence", Zenodo).

21 musicians, 4 Bach chorale melodies x 11 repetitions, each once listened to and
once imagined over the same metronome (a click per 2.4 s bar).  Data ship
preprocessed: 0.1-30 Hz, 64 Hz, average reference, 1803 x 64 per trial, plus an
onset vector per trial with the onsets within 500 ms of a click removed.

A ridge backward model maps lagged EEG to the smoothed note-onset train.  The
held-out melody is never in training (leave-one-melody-out), so a positive
score means the decoder generalises to unseen melodic content.  Every test
trial is scored against the three other melodies' onset trains as well: they
share the metronome and bar structure, so correct-minus-wrong and 4-way
melody identification cancel anything locked to the click rather than to the
imagined notes.

Directions: LL, II (within condition), LI (train on listening, test on
imagery: the transfer the imagined-speech plan relies on) and IL.  Two model
families: per participant, and subject-free (train on the other 20).

``--features encoder --encoder <universal checkpoint>`` replaces the lagged
EEG by the hidden states of a trained subject-free encoder
(app/universal_train.py), read over 4 s windows with no onset clock; the
ridge read-out and every control stay the same, so the table measures how
much listen -> imagine information the encoder's representation keeps
(``encoder-acoustic``: only its envelope / onset head, 2 channels).
Electrode positions: the standard BioSemi-64 order (Fp1, AF7, ... ), which
the shipped 64-column matrices are assumed to follow.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
FS = 64
RATE = 32
LAGS = tuple(range(-4, 14))          # -125 .. +406 ms at 32 Hz (EEG read at t + lag)
BAND = (0.5, 8.0)
ONSET_SIGMA = 3.0                    # samples at 64 Hz (47 ms)
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)   # relative to mean(diag(X'X))
CONDITIONS = ('listen', 'imagine')
DIRECTIONS = (('listen', 'listen'), ('imagine', 'imagine'), ('listen', 'imagine'), ('imagine', 'listen'))


def zscore(x, axis=0):
    return (x - x.mean(axis, keepdims=True)) / (x.std(axis, keepdims=True) + 1e-8)


def onset_train(stim):
    impulses = (np.abs(stim.ravel()) > 0).astype(np.float64)
    half = int(4 * ONSET_SIGMA)
    kernel = np.exp(-.5 * (np.arange(-half, half + 1) / ONSET_SIGMA) ** 2)
    return zscore(np.convolve(impulses, kernel, 'same')[::FS // RATE]).astype(np.float32)


def lagged(eeg):
    """(frames, lags*channels) with zero padding outside the trial."""
    frames, channels = eeg.shape
    out = np.zeros((frames, len(LAGS) * channels), np.float32)
    for k, lag in enumerate(LAGS):
        block = out[:, k * channels:(k + 1) * channels]
        if lag >= 0:
            block[:frames - lag] = eeg[lag:]
        else:
            block[-lag:] = eeg[:frames + lag]
    return out


class EncoderFeatures:
    """(1803, 64) shipped EEG -> (902, D) features of a trained universal encoder at 32 Hz."""
    def __init__(self, path, device, acoustic_only=False):
        import mne
        import torch
        import universal_train as ut
        from eeg2speech.aligned import EEG_SAMPLES, EEG_START
        self.torch, self.samples, self.start = torch, EEG_SAMPLES, EEG_START
        self.model, _ = ut.load_universal(path)
        self.device = torch.device(device)
        self.model.eval().to(self.device)
        xyz = np.array(list(mne.channels.make_standard_montage('biosemi64').get_positions()['ch_pos'].values()))
        self.xyz = torch.tensor(xyz / np.linalg.norm(xyz, axis=1, keepdims=True) * .095, dtype=torch.float32, device=self.device)
        self.acoustic_only = acoustic_only
        self.high = butter(2, .5, 'highpass', fs=FS, output='sos')

    def __call__(self, raw):
        from scipy.signal import resample_poly
        torch = self.torch
        x = resample_poly(sosfiltfilt(self.high, raw.astype(np.float64), axis=0).T, 4, 1, axis=1)   # 64 ch, 256 Hz
        front = int(round(-self.start * 256))
        duration = raw.shape[0] / FS
        starts = np.arange(0., duration, 4.)
        padded = np.pad(x, ((0, 0), (front, int(starts[-1] * 256) + self.samples - front - x.shape[1] + 1)))
        windows = np.stack([padded[:, int(s * 256):int(s * 256) + self.samples] for s in starts]).astype(np.float32)
        n = len(windows)
        with torch.no_grad():
            eeg = torch.from_numpy(windows).to(self.device)
            mask = torch.ones(n, 64, dtype=torch.bool, device=self.device)
            hidden = self.model.encode(eeg, self.xyz.expand(n, -1, -1), mask, torch.ones(n, self.samples, dtype=torch.bool, device=self.device),
                                       torch.zeros(n, dtype=torch.bool, device=self.device))
            if self.acoustic_only:
                hidden = self.model.acoustic(hidden.transpose(1, 2)).transpose(1, 2)
            hidden = hidden.cpu().numpy()                                                     # windows, tokens, D
        token_times = self.model.token_times.cpu().numpy()
        grid = np.arange(0, raw.shape[0], FS // RATE) / FS                                    # the EEG path's [::FS // RATE] grid
        out = np.zeros((len(grid), hidden.shape[-1]), np.float32)
        for k, s in enumerate(starts):
            inside = (grid >= s) & (grid < s + 4.)
            for d in range(hidden.shape[-1]):
                out[inside, d] = np.interp(grid[inside] - s, token_times, hidden[k, :, d])
        return out


def load(path, featurize=None):
    mat = sio.loadmat(path, squeeze_me=False)
    assert int(mat['downFs'].ravel()[0]) == FS
    sos = butter(4, BAND, 'bandpass', fs=FS, output='sos')
    keys, subjects = {}, []
    for s in range(mat['eeg'].shape[1]):
        per_condition = {}
        for c, name in enumerate(CONDITIONS):
            eeg_cells, stim_cells = mat['eeg'][0, s][0, c], mat['stim'][0, s][0, c]
            trials = []
            for t in range(eeg_cells.shape[1]):
                stim = stim_cells[0, t].ravel()
                key = tuple(np.flatnonzero(stim))
                melody = keys.setdefault(key, len(keys))
                if featurize is None:
                    eeg = sosfiltfilt(sos, eeg_cells[0, t].astype(np.float64), axis=0)[::FS // RATE]
                else:
                    eeg = featurize(eeg_cells[0, t])
                trials.append(dict(melody=melody, eeg=zscore(eeg).astype(np.float32), y=onset_train(stim)))
            per_condition[name] = trials
        subjects.append(per_condition)
        mat['eeg'][0, s] = None                       # release the float64 copy as we go
    if len(keys) != 4:
        raise SystemExit(f'expected 4 distinct melody onset patterns, found {len(keys)}')
    targets = {}
    for per_condition in subjects:
        for trials in per_condition.values():
            for trial in trials:
                targets.setdefault(trial['melody'], trial['y'])
    return subjects, targets


def blocks(trials):
    """Per-melody sums of X'X and X'y (lagged matrices are rebuilt on demand to bound memory)."""
    xtx, xty = {}, {}
    for trial in trials:
        x = lagged(trial['eeg']).astype(np.float64)
        m = trial['melody']
        xtx[m] = xtx.get(m, 0) + x.T @ x
        xty[m] = xty.get(m, 0) + x.T @ trial['y']
    return xtx, xty


def solve_all(xtx, xty):
    """Ridge weights for every alpha in ALPHAS via one eigendecomposition."""
    values, vectors = np.linalg.eigh(xtx)
    projected = vectors.T @ xty
    scale = np.trace(xtx) / len(xtx)
    return [vectors @ (projected / (values + a * scale)) for a in ALPHAS]


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def pick_alpha(xtx, xty, trials, train_melodies):
    """Inner leave-one-melody-out on the training melodies (score = correct r)."""
    scores = np.zeros(len(ALPHAS))
    for inner in train_melodies:
        rest = [m for m in train_melodies if m != inner]
        weights = solve_all(sum(xtx[m] for m in rest), sum(xty[m] for m in rest))
        for trial in trials:
            if trial['melody'] == inner:
                x = lagged(trial['eeg'])
                scores += [corr(x @ w, trial['y']) for w in weights]
    return int(np.argmax(scores))


def score_trials(weights, trials, targets, held_out):
    rows = []
    for trial in trials:
        if trial['melody'] not in held_out:
            continue
        prediction = lagged(trial['eeg']) @ weights
        r = {m: corr(prediction, y) for m, y in targets.items()}
        correct = r[trial['melody']]
        wrong = float(np.mean([v for m, v in r.items() if m != trial['melody']]))
        rows.append(dict(melody=trial['melody'], correct=correct, wrong=wrong,
                         hit=int(max(r, key=r.get) == trial['melody']), prediction=prediction))
    return rows


def pooled_hits(rows, targets):
    hits = []
    for m in sorted({row['melody'] for row in rows}):
        mean = np.mean([row['prediction'] for row in rows if row['melody'] == m], 0)
        r = {k: corr(mean, y) for k, y in targets.items()}
        hits.append(int(max(r, key=r.get) == m))
    return hits


def summarise(rows):
    return dict(correct=float(np.mean([r['correct'] for r in rows])),
                wrong=float(np.mean([r['wrong'] for r in rows])),
                delta=float(np.mean([r['correct'] - r['wrong'] for r in rows])),
                accuracy=float(np.mean([r['hit'] for r in rows])), trials=len(rows))


def within(subjects, targets):
    """Per-participant models, leave-one-melody-out."""
    out = []
    for s, per_condition in enumerate(subjects):
        prepared = {c: blocks(per_condition[c]) for c in CONDITIONS}
        result = {}
        for train_c, test_c in DIRECTIONS:
            xtx, xty = prepared[train_c]
            rows, pooled = [], []
            for held in range(4):
                train_melodies = [m for m in range(4) if m != held]
                a = pick_alpha(xtx, xty, per_condition[train_c], train_melodies)
                w = solve_all(sum(xtx[m] for m in train_melodies), sum(xty[m] for m in train_melodies))[a]
                fold = score_trials(w, per_condition[test_c], targets, {held})
                rows += fold
                pooled += pooled_hits(fold, targets)
            result[f'{train_c[0].upper()}{test_c[0].upper()}'] = dict(summarise(rows), pooled_accuracy=float(np.mean(pooled)))
        out.append(result)
        print(f'participant {s + 1:2d}: ' + '  '.join(f"{k} d={v['delta']:+.3f} acc={v['accuracy']:.2f}" for k, v in result.items()), flush=True)
    return out


def subject_free(subjects, targets):
    """Train on the other 20 participants (no participant parameters), test on the held-out one."""
    prepared = [{c: blocks(p[c]) for c in CONDITIONS} for p in subjects]
    out = []
    for s in range(len(subjects)):
        others = [o for o in range(len(subjects)) if o != s]
        result = {}
        for train_c, test_c in DIRECTIONS:
            rows, pooled = [], []
            for held in range(4):
                train_melodies = [m for m in range(4) if m != held]
                xtx = {m: sum(prepared[o][train_c][0][m] for o in others) for m in range(4)}
                xty = {m: sum(prepared[o][train_c][1][m] for o in others) for m in range(4)}
                # Inner LOMO on the pooled training participants, scored on a 5-participant subset for speed.
                trials = [t for o in others[:5] for t in subjects[o][train_c]]
                a = pick_alpha(xtx, xty, trials, train_melodies)
                w = solve_all(sum(xtx[m] for m in train_melodies), sum(xty[m] for m in train_melodies))[a]
                fold = score_trials(w, subjects[s][test_c], targets, {held})
                rows += fold
                pooled += pooled_hits(fold, targets)
            result[f'{train_c[0].upper()}{test_c[0].upper()}'] = dict(summarise(rows), pooled_accuracy=float(np.mean(pooled)))
        out.append(result)
        print(f'held-out {s + 1:2d}: ' + '  '.join(f"{k} d={v['delta']:+.3f} acc={v['accuracy']:.2f}" for k, v in result.items()), flush=True)
    return out


def group(per_subject, seed=0):
    rng = np.random.default_rng(seed)
    table = {}
    for key in per_subject[0]:
        delta = np.array([p[key]['delta'] for p in per_subject])
        accuracy = np.array([p[key]['accuracy'] for p in per_subject])
        flips = rng.choice([-1, 1], size=(20000, len(delta)))
        p_delta = float((np.abs((flips * delta).mean(1)) >= abs(delta.mean())).mean())
        centred = accuracy - .25
        p_acc = float((np.abs((flips * centred).mean(1)) >= abs(centred.mean())).mean())
        table[key] = dict(correct=float(np.mean([p[key]['correct'] for p in per_subject])),
                          wrong=float(np.mean([p[key]['wrong'] for p in per_subject])),
                          delta=float(delta.mean()), delta_se=float(delta.std(ddof=1) / np.sqrt(len(delta))),
                          delta_positive=int((delta > 0).sum()), p_delta=p_delta,
                          accuracy=float(accuracy.mean()), p_accuracy=p_acc,
                          pooled_accuracy=float(np.mean([p[key]['pooled_accuracy'] for p in per_subject])),
                          participants=len(per_subject))
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default=str(ROOT / 'data/marion2021/ImageryData.mat'))
    parser.add_argument('--output', default=str(ROOT / 'outputs/marion2021/imagery_gate.json'))
    parser.add_argument('--skip-subject-free', action='store_true')
    parser.add_argument('--features', choices=['eeg', 'encoder', 'encoder-acoustic'], default='eeg')
    parser.add_argument('--encoder', help='universal checkpoint for --features encoder / encoder-acoustic')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.features != 'eeg' and not args.encoder:
        parser.error('--features encoder needs --encoder')

    featurize = None
    if args.features != 'eeg':
        import sys
        sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src'))
        featurize = EncoderFeatures(args.encoder, args.device, acoustic_only=args.features == 'encoder-acoustic')
    subjects, targets = load(args.data, featurize)
    print(f'{len(subjects)} participants, {len(targets)} melodies, band {BAND} Hz, lags {LAGS[0] / RATE * 1000:.0f}..{LAGS[-1] / RATE * 1000:.0f} ms')
    report = dict(band=BAND, rate=RATE, lags=LAGS, onset_sigma_64hz=ONSET_SIGMA, alphas=ALPHAS, features=args.features,
                  encoder=args.encoder)
    per = within(subjects, targets)
    report['within'] = dict(group=group(per), participants=per)
    if not args.skip_subject_free:
        free = subject_free(subjects, targets)
        report['subject_free'] = dict(group=group(free), participants=free)
    for family in ('within', 'subject_free'):
        if family not in report:
            continue
        print(f'\n{family}: r correct / wrong / delta (+-se, n>0, p) / 4-way acc (p) / pooled acc')
        for key, g in report[family]['group'].items():
            print(f"  {key}: {g['correct']:+.3f} / {g['wrong']:+.3f} / {g['delta']:+.3f} (+-{g['delta_se']:.3f}, "
                  f"{g['delta_positive']}/{g['participants']}, p={g['p_delta']:.4f}) / {g['accuracy']:.3f} "
                  f"(p={g['p_accuracy']:.4f}) / {g['pooled_accuracy']:.3f}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f'\nwrote {args.output}')


if __name__ == '__main__':
    main()

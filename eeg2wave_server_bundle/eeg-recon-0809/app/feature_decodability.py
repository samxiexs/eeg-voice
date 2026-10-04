#!/usr/bin/env python3
"""Which speech representation can scalp EEG actually encode?  (Broderick 2018 audiobook.)

19 participants x 20 runs (~3 min) of "The Old Man and the Sea", 128-ch BioSemi.
EEG comes from the harmonised Broderick shards (scripts/prepare_broderick_windows.py:
128 Hz, bad electrodes interpolated, average reference, 0.5-45 Hz; sample 0 =
run onset), or from the raw CND release (128 Hz, time-locked to run onset) if
the shards are absent;
the audio is the identical wav in OpenNeuro ds004408 (the CND envelope of run r
correlates 1.000 with the Hilbert envelope of ds004408 audio<r>.wav at lag 0).

For every dimension of several candidate targets a per-participant backward
ridge model (EEG 0.5-8 Hz, lags 0-406 ms, 32 Hz) is fit with 5 run-block folds
(alpha picked per dimension on an inner split of the training runs) and scored
on held-out runs against a circular-shift null.  Continuous speech has no
onset-locked time prior, so the null-corrected r is the EEG contribution.

Targets (all at 32 Hz): the broadband envelope; 80 log-mel bands; MFCC c0-c19
(orthonormal DCT of the same log-mel, so the full MFCC carries exactly the
log-mel information and the question is only which coefficients EEG reaches);
8 grouped log-mel bands; and the top 16 PCs of HuBERT-base layers 3/6/9/12.
The summary per representation is the null-corrected, variance-weighted
explained fraction  sum_k var_k r_k^2 / sum_k var_k  on held-out runs.
``--partial-envelope`` first regresses every target on the envelope and its
onsets, so the scores measure spectral / SSL information beyond loudness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import soundfile as sf
from scipy.fft import dct
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
CND = ROOT / 'data/broderick2018/Natural Speech'
SHARDS = ROOT / 'artifacts/speech_continuous/broderick2018/shards'
AUDIO = ROOT / 'data/ds004408/stimuli'
HUBERT = ROOT / 'models/aligned_local_base/hubert'
EEG_RATE, RATE, AUDIO_RATE = 128, 32, 16000
HOP = AUDIO_RATE // RATE
LAGS = tuple(range(14))              # 0 .. 406 ms
ALPHAS = (1e-2, 1e-1, 1., 1e1, 1e2, 1e3, 1e4)    # relative to mean(diag(X'X))
HUBERT_LAYERS = (3, 6, 9, 12)
RUNS = 20


def mel_filterbank(n_fft, bands=80, fmin=50., fmax=8000.):
    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)
    points = 700 * (10 ** (np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), bands + 2) / 2595) - 1)
    freqs = np.fft.rfftfreq(n_fft, 1 / AUDIO_RATE)
    bank = np.zeros((bands, len(freqs)))
    for b in range(bands):
        lo, mid, hi = points[b:b + 3]
        bank[b] = np.clip(np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)), 0, None)
    return bank


def audio_targets(wave, frames, hubert=None):
    n_fft = 1024
    padded = np.pad(wave, (n_fft // 2, n_fft // 2 + frames * HOP))
    window = np.hanning(n_fft)
    stft = np.stack([np.fft.rfft(padded[i * HOP:i * HOP + n_fft] * window) for i in range(frames)])
    logmel = np.log(np.abs(stft) ** 2 @ mel_filterbank(n_fft).T + 1e-6)          # (frames, 80)
    mfcc = dct(logmel, type=2, norm='ortho', axis=1)[:, :20]
    band8 = logmel.reshape(frames, 8, 10).mean(2)
    envelope = np.abs(hilbert(wave)) ** .3
    envelope = resample_poly(envelope, RATE, AUDIO_RATE)[:frames]
    envelope = np.pad(envelope, (0, frames - len(envelope)), mode='edge')
    out = dict(envelope=envelope[:, None], logmel=logmel, mfcc=mfcc, band8=band8)
    if hubert is not None:
        out.update(hubert(wave, frames))
    return out


def hubert_extractor(device):
    import torch
    from transformers import HubertModel
    model = HubertModel.from_pretrained(str(HUBERT), local_files_only=True).eval().to(device)

    def run(wave, frames):
        chunk = 20 * AUDIO_RATE
        layers = {l: [] for l in HUBERT_LAYERS}
        with torch.no_grad():
            for start in range(0, len(wave), chunk):
                piece = wave[start:start + chunk]
                piece = (piece - piece.mean()) / (piece.std() + 1e-7)
                hidden = model(torch.tensor(piece, dtype=torch.float32, device=device)[None],
                               output_hidden_states=True).hidden_states
                for l in HUBERT_LAYERS:
                    layers[l].append(hidden[l][0].float().cpu().numpy())
        grid = np.arange(frames) / RATE
        out = {}
        for l, parts in layers.items():
            feats = np.concatenate(parts)                                    # 50 Hz, 20 ms hop, centre at 10 ms + 20 ms * i
            times = .01 + .02 * np.arange(len(feats))
            out[f'hubert{l}'] = np.stack([np.interp(grid, times, feats[:, d]) for d in range(feats.shape[1])], 1)
        return out
    return run


def pca_top(per_run, k=16):
    stacked = np.concatenate(per_run)
    mean = stacked.mean(0)
    _, _, vt = np.linalg.svd(stacked - mean, full_matrices=False)
    return [(x - mean) @ vt[:k].T for x in per_run]


def partial_out_envelope(matrices, slices):
    """Residualise every non-envelope column on [envelope, positive envelope derivative, 1], fit over all runs."""
    a, b = slices['envelope']
    def basis(m):
        env = m[:, a]
        onset = np.clip(np.diff(env, prepend=env[0]), 0, None)
        return np.stack([env, onset, np.ones_like(env)], 1)
    stacked_basis = np.concatenate([basis(m) for m in matrices])
    stacked = np.concatenate(matrices)
    coef, *_ = np.linalg.lstsq(stacked_basis, stacked, rcond=None)
    coef[:, a:b] = 0                                                  # keep the envelope itself as the reference
    return [m - basis(m) @ coef for m in matrices]


def build_targets(device, use_hubert=True, partial=False):
    hubert = hubert_extractor(device) if use_hubert else None
    runs = []
    for r in range(1, RUNS + 1):
        wave, rate = sf.read(AUDIO / f'audio{r:02d}.wav')
        wave = resample_poly(wave.mean(1) if wave.ndim > 1 else wave, AUDIO_RATE, rate).astype(np.float32)
        frames = int(len(wave) / AUDIO_RATE * RATE)
        runs.append(audio_targets(wave, frames, hubert))
        print(f'audio run {r:2d}: {frames / RATE:.1f} s', flush=True)
    for key in [k for k in runs[0] if k.startswith('hubert')]:
        for run, pcs in zip(runs, pca_top([run[key] for run in runs])):
            run[key] = pcs
    names = list(runs[0])
    slices, start = {}, 0
    for name in names:
        slices[name] = (start, start + runs[0][name].shape[1])
        start = slices[name][1]
    matrices = [np.concatenate([run[n] for n in names], 1).astype(np.float64) for run in runs]
    if partial:
        matrices = partial_out_envelope(matrices, slices)
    variance = np.concatenate(matrices).var(0)
    return matrices, slices, variance


def raw_run(subject, r):
    """(T, 128) at 128 Hz from run onset, average referenced: the harmonised shard, else the raw CND file."""
    shard = SHARDS / f'Subject{subject}.h5'
    if shard.exists():
        import h5py
        with h5py.File(shard, 'r') as h5:
            return h5['trials'][f'{r - 1:02d}'][:].astype(np.float64).T
    eeg = sio.loadmat(CND / f'EEG/Subject{subject}/Subject{subject}_Run{r}.mat')['eegData'].astype(np.float64)
    return eeg - eeg.mean(1, keepdims=True)


def load_eeg(subject, frames_per_run):
    sos = butter(4, (.5, 8.), 'bandpass', fs=EEG_RATE, output='sos')
    runs = []
    for r in range(1, RUNS + 1):
        eeg = raw_run(subject, r)
        eeg = sosfiltfilt(sos, eeg, axis=0)[::EEG_RATE // RATE]
        eeg = np.pad(eeg, ((0, max(0, frames_per_run[r - 1] + LAGS[-1] - len(eeg))), (0, 0)))
        eeg = eeg[:frames_per_run[r - 1] + LAGS[-1]]
        runs.append(((eeg - eeg.mean(0)) / (eeg.std(0) + 1e-8)).astype(np.float32))
    return runs


def lagged(eeg, frames):
    return np.concatenate([eeg[lag:lag + frames] for lag in LAGS], 1).astype(np.float32)


def zscore_columns(y):
    return (y - y.mean(0)) / (y.std(0) + 1e-8)


def ridge_all(xtx, xty):
    values, vectors = np.linalg.eigh(xtx)
    projected = vectors.T @ xty
    scale = np.trace(xtx) / len(xtx)
    return [vectors @ (projected / (values + a * scale)[:, None]) for a in ALPHAS]


def column_corr(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    return (a * b).sum(0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-12)


def participant(subject, targets, rng, folds=5, shifts=20):
    frames = [len(t) for t in targets]
    eeg = load_eeg(subject, frames)
    x = [lagged(e, f) for e, f in zip(eeg, frames)]
    y = [zscore_columns(t).astype(np.float32) for t in targets]
    xtx = [(xi.T @ xi).astype(np.float64) for xi in x]
    xty = [(xi.T @ yi).astype(np.float64) for xi, yi in zip(x, y)]
    per_run_r, per_run_null = [], []
    blocks = np.array_split(np.arange(RUNS), folds)
    for test in blocks:
        train = [r for r in range(RUNS) if r not in test]
        inner_val, inner_fit = train[-4:], train[:-4]
        weights = ridge_all(sum(xtx[r] for r in inner_fit), sum(xty[r] for r in inner_fit))
        score = np.stack([np.mean([column_corr(x[r] @ w, y[r]) for r in inner_val], 0) for w in weights])
        best = score.argmax(0)                                         # alpha per target dimension
        weights = ridge_all(sum(xtx[r] for r in train), sum(xty[r] for r in train))
        chosen = np.stack([weights[a][:, d] for d, a in enumerate(best)], 1)
        for r in test:
            prediction = x[r] @ chosen
            per_run_r.append(column_corr(prediction, y[r]))
            n = len(prediction)
            per_run_null.append(np.mean([column_corr(np.roll(prediction, int(rng.integers(10 * RATE, n - 10 * RATE)), 0), y[r])
                                         for _ in range(shifts)], 0))
    return np.mean(per_run_r, 0), np.mean(per_run_null, 0)


def summarise(r, null, slices, variance):
    out = {}
    corrected = r - null
    for name, (a, b) in slices.items():
        v = variance[a:b]
        out[name] = dict(dims=b - a,
                         explained_fraction=float((v * np.clip(corrected[:, a:b], 0, None) ** 2).sum(1).mean() / v.sum()),
                         mean_r=float(r[:, a:b].mean()), mean_null=float(null[:, a:b].mean()),
                         per_dim_r=[float(v) for v in corrected[:, a:b].mean(0)],
                         best_dim=int(corrected[:, a:b].mean(0).argmax()),
                         best_r=float(corrected[:, a:b].mean(0).max()))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--subjects', default='all')
    parser.add_argument('--no-hubert', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--partial-envelope', action='store_true',
                        help='score what EEG decodes of each target beyond the broadband envelope and its onsets')
    parser.add_argument('--output', default=str(ROOT / 'outputs/broderick2018/feature_decodability.json'))
    args = parser.parse_args()
    subjects = list(range(1, 20)) if args.subjects == 'all' else [int(s) for s in args.subjects.split(',')]

    targets, slices, variance = build_targets(args.device, not args.no_hubert, args.partial_envelope)
    rng = np.random.default_rng(0)
    rs, nulls = [], []
    for s in subjects:
        r, null = participant(s, targets, rng)
        rs.append(r)
        nulls.append(null)
        a, b = slices['envelope']
        print(f'participant {s:2d}: envelope r {r[a]:.3f} (null {null[a]:+.3f})  '
              f'mfcc c0 {r[slices["mfcc"][0]]:.3f}  logmel mean {r[slices["logmel"][0]:slices["logmel"][1]].mean():.3f}', flush=True)
    summary = summarise(np.stack(rs), np.stack(nulls), slices, variance)
    print('\nrepresentation: dims / explained fraction (null-corrected, variance-weighted) / mean r / best dim r')
    for name, s in summary.items():
        print(f"  {name:9s}: {s['dims']:3d} / {s['explained_fraction']:.4f} / {s['mean_r']:.3f} (null {s['mean_null']:+.3f}) / "
              f"dim {s['best_dim']} r={s['best_r']:.3f}")
    print('  mfcc per coefficient r:', ' '.join(f'{v:.3f}' for v in summary['mfcc']['per_dim_r']))
    print('  band8 per band r:      ', ' '.join(f'{v:.3f}' for v in summary['band8']['per_dim_r']))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(dict(subjects=subjects, lags=LAGS, rate=RATE, alphas=ALPHAS,
                                                 partial_envelope=args.partial_envelope,
                                                 summary=summary), indent=2))
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""G1 gate: does single-trial EEG linearly track the presented speech envelope?

A ridge backward model maps lagged EEG (0-375 ms after each acoustic frame)
to the log-mel envelope of the presented sentence on speech frames only.  It
is fit on train-fold contents and scored on validation-fold contents, pooled
and per subject, against three references: the train-fold median envelope
profile (the time-only prior every EEG model must beat), same-subject
wrong-trial EEG, and a circular-shift null of the EEG-explained residual.  No
network is involved, so a null result here points at alignment or
preprocessing rather than at the model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('ALIGNED_TARGET_CACHE_NAME', 'targets_adapted.h5')
import numpy as np

import aligned_speech as legacy

RATE = 32                     # working rate (Hz) for EEG and envelope
LAGS = tuple(range(13))       # 0 .. 375 ms in 1/32 s steps
EEG_START = -.25


def envelope_of(mel, duration_frames, mel_times):
    """Log-mel envelope (mean over bins) on speech frames, resampled to RATE."""
    profile = mel[:, :duration_frames].mean(0)
    end = float(mel_times[duration_frames - 1])
    grid = np.arange(0, end, 1 / RATE)
    return grid, np.interp(grid, mel_times[:duration_frames], profile)


def downsample(eeg):
    channels, samples = eeg.shape
    factor = 256 // RATE
    return eeg[:, :samples // factor * factor].reshape(channels, -1, factor).mean(-1)


def lagged(low, grid, lags=LAGS):
    """(frames, channels*lags): block-averaged EEG read at t + lag for each lag."""
    factor = 256 // RATE
    first = EEG_START + (factor - 1) / 2 / 256
    columns = []
    for lag in lags:
        index = np.clip(np.round((grid + lag / RATE - first) * RATE).astype(int), 0, low.shape[1] - 1)
        columns.append(low[:, index].T)
    return np.concatenate(columns, 1)


def collect(dataset, mel_times):
    rows = []
    for i in range(len(dataset)):
        record = dataset[i]
        grid, envelope = envelope_of(record['mel'].numpy(), int(record['oracle_duration_frames']), mel_times)
        rows.append(dict(subject=record['subject'], content=record['content'], trial_id=record['trial_id'],
                         low=downsample(record['eeg'].numpy()).astype(np.float32),
                         y=envelope.astype(np.float32), grid=grid.astype(np.float32)))
    return rows


def profile_at(profile, grid):
    return np.interp(grid, np.arange(len(profile)) / RATE, profile).astype(np.float32)


def design(row, standardizer, profile, lags=None):
    mean, scale = standardizer[row['subject']]
    prior = profile_at(profile, row['grid'])
    x = (lagged(row['low'], row['grid']) - mean) / scale
    if lags is not None:
        # Column blocks are ordered by lag, so a single lag is one contiguous block.
        channels = x.shape[1] // len(LAGS)
        x = np.concatenate([x[:, LAGS.index(l) * channels:(LAGS.index(l) + 1) * channels] for l in lags], 1)
    return np.concatenate([x, prior[:, None], np.ones((len(row['y']), 1), np.float32)], 1)


def fit_standardizer(rows):
    result = {}
    for subject in sorted({r['subject'] for r in rows}):
        total = count = square = 0
        for r in rows:
            if r['subject'] == subject:
                x = lagged(r['low'], r['grid']).astype(np.float64)
                total = total + x.sum(0); square = square + (x ** 2).sum(0); count += len(x)
        mean = total / count
        result[subject] = (mean.astype(np.float32), (np.sqrt(np.maximum(square / count - mean ** 2, 0)) + 1e-6).astype(np.float32))
    return result


def median_profile(rows):
    longest = max(len(r['y']) for r in rows)
    stack = np.full((len(rows), longest), np.nan, np.float32)
    for i, r in enumerate(rows):
        stack[i, :len(r['y'])] = r['y']
    return np.nanmedian(stack, 0)


def normal_equations(rows, standardizer, profile, lags=None):
    xtx = xty = None
    for row in rows:
        x = design(row, standardizer, profile, lags).astype(np.float64)
        xtx = x.T @ x if xtx is None else xtx + x.T @ x
        xty = x.T @ row['y'] if xty is None else xty + x.T @ row['y']
    return xtx, xty


def solve(xtx, xty, alpha):
    penalty = alpha * np.eye(len(xtx)); penalty[-2:, -2:] = 0      # prior and bias unpenalized
    return np.linalg.solve(xtx + penalty, xty).astype(np.float32)


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean(); d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-12 else 0.


def score(rows, weights, standardizer, profile, wrong, rng, shifts=20):
    """Per-trial correlations for the model, the prior alone and the controls."""
    out = []
    for i, row in enumerate(rows):
        x = design(row, standardizer, profile)
        prediction = x @ weights; prior = x[:, -2]
        other = design(rows[wrong[i]], standardizer, profile)
        length = min(len(other), len(x)); swapped = x.copy()
        swapped[:length, :-2] = other[:length, :-2]
        if length < len(x):
            swapped[length:, :-2] = other[-1, :-2]
        residual = prediction - prior; truth_residual = row['y'] - prior
        null = []
        for _ in range(shifts):
            offset = int(rng.integers(RATE // 2, max(RATE // 2 + 1, len(x) - RATE // 2)))
            null.append(pearson(np.roll(residual, offset), truth_residual))
        out.append(dict(subject=row['subject'], content=row['content'], trial_id=row['trial_id'],
                        r_model=pearson(prediction, row['y']), r_prior=pearson(prior, row['y']),
                        r_wrong_trial=pearson(swapped @ weights, row['y']),
                        r_residual=pearson(residual, truth_residual), r_residual_shift_null=float(np.mean(null))))
    return out


def summarize(records):
    keys = ('r_model', 'r_prior', 'r_wrong_trial', 'r_residual', 'r_residual_shift_null')
    subjects = sorted({r['subject'] for r in records})
    per_subject = {s: {k: float(np.mean([r[k] for r in records if r['subject'] == s])) for k in keys} for s in subjects}
    gains = np.array([per_subject[s]['r_model'] - per_subject[s]['r_prior'] for s in subjects])
    residual = np.array([per_subject[s]['r_residual'] - per_subject[s]['r_residual_shift_null'] for s in subjects])
    rng = np.random.default_rng(31)
    def ci(values):
        boots = [rng.choice(values, len(values)).mean() for _ in range(2000)]
        return [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]
    return dict(trials=len(records), subjects=len(subjects),
                **{k: float(np.mean([r[k] for r in records])) for k in keys},
                per_subject=per_subject,
                gain_over_prior_subject_mean=float(gains.mean()), gain_over_prior_ci95=ci(gains),
                subjects_with_positive_gain=int((gains > 0).sum()),
                residual_over_shift_subject_mean=float(residual.mean()), residual_over_shift_ci95=ci(residual),
                subjects_with_positive_residual=int((residual > 0).sum()))


def wrong_indices(rows):
    result = []
    for i, row in enumerate(rows):
        candidates = [j for j, r in enumerate(rows) if r['subject'] == row['subject'] and r['content'] != row['content']]
        if not candidates:
            raise ValueError('no same-subject wrong trial')
        candidates.sort(key=lambda j: hashlib.sha256(f"{row['trial_id']}:{rows[j]['trial_id']}".encode()).hexdigest())
        result.append(candidates[0])
    return result


def run_check(train_rows, validation_rows, wrong, alphas=(1e1, 1e2, 1e3, 1e4, 1e5), seed=31, per_subject=True):
    rng = np.random.default_rng(seed)
    standardizer = fit_standardizer(train_rows)
    profile = median_profile(train_rows)
    # An inner content split of the train fold selects alpha; validation never does.
    contents = sorted({r['content'] for r in train_rows}); rng.shuffle(contents)
    inner = set(contents[:max(1, len(contents) // 10)])
    fit_rows = [r for r in train_rows if r['content'] not in inner]
    inner_rows = [r for r in train_rows if r['content'] in inner]
    inner_wrong = wrong_indices(inner_rows)
    xtx, xty = normal_equations(fit_rows, standardizer, profile)
    selection = {alpha: float(np.mean([r['r_model'] for r in score(inner_rows, solve(xtx, xty, alpha), standardizer,
                                                                     profile, inner_wrong, rng, shifts=1)])) for alpha in alphas}
    alpha = max(selection, key=selection.get)
    weights = solve(*normal_equations(train_rows, standardizer, profile), alpha)
    result = dict(alpha=alpha, alpha_selection_inner_r={str(k): v for k, v in selection.items()},
                  lags_ms=[int(l * 1000 / RATE) for l in LAGS], rate_hz=RATE,
                  pooled=summarize(score(validation_rows, weights, standardizer, profile, wrong, rng)))
    # Single-lag models: the lag with the best held-out r locates the neural
    # response relative to the acoustic frame (a systematic marker offset would
    # show up as a peak at the wrong lag or at the scan boundary).
    scan = {}
    for lag in LAGS:
        w = solve(*normal_equations(train_rows, standardizer, profile, (lag,)), alpha / len(LAGS))
        values = []
        for r in validation_rows:
            x = design(r, standardizer, profile, (lag,))
            values.append(pearson(x @ w - x[:, -2], r['y'] - x[:, -2]))   # residual r: EEG beyond the prior
        scan[int(lag * 1000 / RATE)] = float(np.mean(values))
    result['single_lag_residual_r'] = scan
    result['best_single_lag_ms'] = max(scan, key=scan.get)
    if per_subject:
        records = []
        for subject in sorted({r['subject'] for r in validation_rows}):
            train_s = [r for r in train_rows if r['subject'] == subject]
            valid_s = [r for r in validation_rows if r['subject'] == subject]
            if not train_s or not valid_s:
                continue
            weights_s = solve(*normal_equations(train_s, standardizer, profile), alpha)
            records.extend(score(valid_s, weights_s, standardizer, profile, wrong_indices(valid_s), rng))
        result['per_subject_models'] = summarize(records)
    return result


def report_line(name, block):
    return (f"[{name}] r_model={block['r_model']:.4f} r_prior={block['r_prior']:.4f} r_wrong_trial={block['r_wrong_trial']:.4f}"
            f" | gain over prior: subject mean {block['gain_over_prior_subject_mean']:+.4f} CI95 {np.round(block['gain_over_prior_ci95'], 4).tolist()}"
            f" ({block['subjects_with_positive_gain']}/{block['subjects']} subjects > 0)"
            f" | residual r={block['r_residual']:.4f} vs shift null {block['r_residual_shift_null']:.4f}"
            f" ({block['subjects_with_positive_residual']}/{block['subjects']} subjects > null)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(legacy.ROOT / 'configs/aligned_speech_local_v1.yaml'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--role', choices=['validation', 'test'], default='validation')
    args = parser.parse_args()
    cfg = legacy.config(args.config)
    train = legacy.dataset_for(cfg, 'train'); held = legacy.dataset_for(cfg, args.role)
    mel_times = train.mel_times.numpy()
    print('loading train fold', flush=True); train_rows = collect(train, mel_times)
    print('loading held-out fold', flush=True); held_rows = collect(held, mel_times)
    result = run_check(train_rows, held_rows, wrong_indices(held_rows))
    result.update(contract='linear_envelope_check_v1', role=args.role, train_trials=len(train_rows))
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    legacy.atomic_json(output / 'linear_envelope_check.json', result)
    print(json.dumps(dict(alpha=result['alpha'], inner=result['alpha_selection_inner_r'])), flush=True)
    print('single-lag residual r by lag (ms): ' + ' '.join(f'{k}:{v:.3f}' for k, v in result['single_lag_residual_r'].items())
          + f"  -> best {result['best_single_lag_ms']} ms", flush=True)
    for name in ('pooled', 'per_subject_models'):
        print(report_line(name, result[name]), flush=True)


if __name__ == '__main__':
    main()

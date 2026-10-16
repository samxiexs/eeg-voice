#!/usr/bin/env python3
"""KaraOne (imagined speech, the research target): honest baselines and DS004940 transfer.

Two entry points, ``python app/karaone.py baselines`` and ``python app/karaone.py transfer``
(both driven by app/run_karaone.sh).

=== 1. Honest linear baselines ===

What "honest" means here:

* **Block-aware cross-validation.**  Folds are contiguous runs of trials in
  recording order, so slow drift and session effects cannot leak a prompt's
  identity from temporally adjacent trials.  The interleaved (shuffled) split
  that most imagined-speech papers use is also reported, only to show how much
  it inflates accuracy.
* **Permutation test.**  The block-CV accuracy of the primary classifier is
  compared with the distribution obtained by permuting prompt labels within
  the participant, giving a per-participant p-value instead of a comparison
  with 1/11.
* **Controls.**  The same pipeline is run on the rest ("clearing") stage,
  where any above-chance accuracy can only be leakage; on ocular/EMG/EKG
  channels alone, which must not decode the prompt if the EEG result is to be
  called neural; and on low-band (0.5–20 Hz) versus high-band (20–45 Hz)
  EEG, because prompt-specific EMG lives mostly above 20 Hz.

Features are Riemannian: shrinkage covariance -> log-Euclidean tangent space,
classified with shrinkage LDA (primary, closed form) and L2 multinomial
logistic regression (secondary).  No deep model is involved.

=== 2. DS004940 encoder transfer ===

For every participant and stage (stimulus = heard prompt, thinking = imagined,
clearing = rest control) each trial is read as the encoder's fixed 1178-sample
window starting 250 ms before the stage onset, mapped from the 62 KaraOne
electrodes to the BioSemi-128 layout with the spherical-spline matrix stored
in the shard, and passed through the frozen encoder.  The masked time-mean of
the predicted speech sequence (the encoder's content space) is the trial
embedding.  Embeddings are then scored exactly like the linear baselines:
block-aware cross-validation, shrinkage LDA, label-permutation p-values.

Three comparisons matter:

* pretrained encoder vs the same architecture with random weights (does the
  DS004940 training add anything beyond the architecture's filtering);
* pretrained encoder vs the Riemannian baseline on the native channels;
* **cross-stage transfer**: an LDA fitted on heard-prompt embeddings is tested
  on imagined-prompt embeddings of held-out blocks (and the reverse).  This is
  the listen -> imagine evidence the imagined-speech goal ultimately needs.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('ALIGNED_TARGET_CACHE_NAME', 'targets_adapted.h5')
import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src')); sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_karaone import PROMPTS, STAGES, TARGET_RATE


# --- 1. Honest linear baselines ----------------------------------------------------------



BINARY_TASKS = {
    # From the authors' getClasses.m: 1 = positive class.
    'consonant_vs_vowel': {'/iy/': 0, '/uw/': 0, '/m/': 1, '/n/': 1, '/piy/': 1, '/tiy/': 1, '/diy/': 1, 'gnaw': 1, 'knew': 1, 'pat': 1, 'pot': 1},
    'nasal': {'/iy/': 0, '/uw/': 0, '/m/': 1, '/n/': 1, '/piy/': 0, '/tiy/': 0, '/diy/': 0, 'gnaw': 1, 'knew': 1, 'pat': 0, 'pot': 0},
    'bilabial': {'/iy/': 0, '/uw/': 0, '/m/': 1, '/n/': 0, '/piy/': 1, '/tiy/': 0, '/diy/': 0, 'gnaw': 0, 'knew': 0, 'pat': 1, 'pot': 1},
    'iy': {'/iy/': 1, '/uw/': 0, '/m/': 0, '/n/': 0, '/piy/': 1, '/tiy/': 1, '/diy/': 1, 'gnaw': 0, 'knew': 0, 'pat': 0, 'pot': 0},
    'uw': {'/iy/': 0, '/uw/': 1, '/m/': 0, '/n/': 0, '/piy/': 0, '/tiy/': 0, '/diy/': 0, 'gnaw': 0, 'knew': 0, 'pat': 0, 'pot': 0},
}


WINDOWS = {'stimulus': (0., 2.), 'thinking': (.25, 4.25), 'speaking': (0., 2.), 'clearing': (.5, 4.5)}


class Shard:
    def __init__(self, path: Path):
        self.h5 = h5py.File(path, 'r')
        self.participant = str(self.h5.attrs['participant'])
        self.names = json.loads(self.h5.attrs['channel_names']); self.aux_names = json.loads(self.h5.attrs['aux_names'])
        self.center = np.asarray(self.h5.attrs['normalizer_center'], dtype=np.float32)[:, None]
        self.scale = np.asarray(self.h5.attrs['normalizer_scale'], dtype=np.float32)[:, None]
        trials = self.h5['trials']
        self.prompts = [v.decode() if isinstance(v, bytes) else str(v) for v in trials['prompt'][:]]
        self.onsets = {stage: trials[f'{stage}_start'][:].astype(int) for stage in STAGES}
        self.ends = {stage: trials[f'{stage}_end'][:].astype(int) for stage in STAGES}

    def __len__(self):
        return len(self.prompts)

    def valid(self, stage: str) -> np.ndarray:
        """Trials whose stage boundaries are usable (a few released speaking indices are corrupt)."""
        return self.onsets[stage] >= 0

    def windows(self, stage: str, span=None, source='eeg') -> np.ndarray:
        """(valid trials, channels, samples) at 256 Hz; EEG in robust units, aux in volts."""
        start_s, end_s = span or WINDOWS[stage]
        a, b = int(round(start_s * TARGET_RATE)), int(round(end_s * TARGET_RATE))
        data = self.h5[source]
        out = np.stack([data[:, onset + a: onset + b] for onset in self.onsets[stage][self.valid(stage)]]).astype(np.float32)
        if source == 'eeg':
            out = (out - self.center[None]) / self.scale[None]
        return out


def bandpass_fft(windows: np.ndarray, low: float, high: float, rate: int = TARGET_RATE) -> np.ndarray:
    spectrum = np.fft.rfft(windows, axis=-1)
    freqs = np.fft.rfftfreq(windows.shape[-1], 1 / rate)
    spectrum[..., (freqs < low) | (freqs > high)] = 0
    return np.fft.irfft(spectrum, n=windows.shape[-1], axis=-1).astype(np.float32)


def shrinkage_covariance(window: np.ndarray, shrink: float = .1) -> np.ndarray:
    x = window - window.mean(1, keepdims=True)
    cov = x @ x.T / x.shape[1]
    return (1 - shrink) * cov + shrink * np.trace(cov) / len(cov) * np.eye(len(cov))


def logm_spd(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.log(np.maximum(values, 1e-12))) @ vectors.T


def tangent_features(windows: np.ndarray) -> np.ndarray:
    """Log-Euclidean tangent vectors (upper triangle, off-diagonals scaled by sqrt 2)."""
    n = windows.shape[1]; iu = np.triu_indices(n)
    weight = np.where(iu[0] == iu[1], 1., np.sqrt(2.))
    return np.stack([logm_spd(shrinkage_covariance(w))[iu] * weight for w in windows]).astype(np.float64)


def bandpower_features(windows: np.ndarray, bands=((4, 8), (8, 13), (13, 30), (30, 45)), rate: int = TARGET_RATE) -> np.ndarray:
    power = np.abs(np.fft.rfft(windows, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(windows.shape[-1], 1 / rate)
    return np.concatenate([np.log(power[..., (freqs >= lo) & (freqs < hi)].mean(-1) + 1e-12) for lo, hi in bands], axis=1)


def aux_features(windows: np.ndarray, names: list[str], rate: int = TARGET_RATE) -> np.ndarray:
    """Log band power of ocular (0.5-10 Hz), EKG (0.5-40 Hz) and EMG (20-100 Hz) channels."""
    power = np.abs(np.fft.rfft(windows, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(windows.shape[-1], 1 / rate)
    bands = {'VEO': (.5, 10), 'HEO': (.5, 10), 'EKG': (.5, 40), 'EMG': (20, 100), 'M1': (.5, 45), 'M2': (.5, 45)}
    columns = []
    for i, name in enumerate(names):
        lo, hi = bands.get(name, (.5, 45))
        columns.append(np.log(power[:, i, (freqs >= lo) & (freqs < hi)].mean(-1) + 1e-30))
        if name == 'EMG':   # a second EMG band catches lower-frequency jaw/lip activity
            columns.append(np.log(power[:, i, (freqs >= 10) & (freqs < 20)].mean(-1) + 1e-30))
    return np.stack(columns, 1)


class ShrinkageLDA:
    """Shrinkage LDA in the dual (sample) space: exact for n << d without forming a d x d inverse.

    Precision of the shrunk within-class covariance
    ``(1-g) S + g tau I`` (tau = trace(S)/d) is applied through the Woodbury
    identity, so a fit costs O(n^3 + n d k) instead of O(d^3).
    """
    def __init__(self, shrink: float = .2):
        self.shrink = shrink

    def fit(self, x, y):
        self.classes = np.unique(y)
        self.means = np.stack([x[y == c].mean(0) for c in self.classes])
        centered = np.concatenate([x[y == c] - m for c, m in zip(self.classes, self.means)])
        dof = max(1, len(centered) - len(self.classes))
        tau = (centered ** 2).sum() / dof / x.shape[1]
        ridge = self.shrink * tau
        a = (1 - self.shrink) / dof
        # (ridge I + a C^T C)^-1 M^T = (M^T - C^T (ridge/a I + C C^T)^-1 C M^T) / ridge
        gram = centered @ centered.T
        inner = np.linalg.solve(ridge / a * np.eye(len(gram)) + gram, centered @ self.means.T)
        self.weights = (self.means.T - centered.T @ inner) / ridge            # d x k
        self.bias = -.5 * np.einsum('kd,dk->k', self.means, self.weights) + np.log(np.array([(y == c).mean() for c in self.classes]))
        return self

    def decision(self, x):
        return x @ self.weights + self.bias

    def predict(self, x):
        return self.classes[self.decision(x).argmax(1)]


class Logistic:
    """L2 multinomial logistic regression (L-BFGS)."""
    def __init__(self, penalty: float = 1.):
        self.penalty = penalty

    def fit(self, x, y):
        from scipy.optimize import minimize
        self.classes = np.unique(y); k = len(self.classes); n, d = x.shape
        target = np.searchsorted(self.classes, y); onehot = np.eye(k)[target]
        def objective(flat):
            w = flat[:d * k].reshape(d, k); b = flat[d * k:]
            logits = x @ w + b; logits -= logits.max(1, keepdims=True)
            log_prob = logits - np.log(np.exp(logits).sum(1, keepdims=True))
            loss = -(onehot * log_prob).sum() / n + .5 * self.penalty * (w ** 2).sum() / n
            grad_logits = (np.exp(log_prob) - onehot) / n
            return loss, np.concatenate([(x.T @ grad_logits + self.penalty * w / n).ravel(), grad_logits.sum(0)])
        result = minimize(objective, np.zeros(d * k + k), jac=True, method='L-BFGS-B', options=dict(maxiter=300))
        self.w = result.x[:d * k].reshape(d, k); self.b = result.x[d * k:]
        return self

    def predict(self, x):
        return self.classes[(x @ self.w + self.b).argmax(1)]


class Standardizer:
    def fit(self, x):
        self.mean = x.mean(0); self.std = x.std(0) + 1e-8; return self

    def __call__(self, x):
        return (x - self.mean) / self.std


def block_folds(n: int, k: int) -> list[np.ndarray]:
    """Contiguous folds in recording order."""
    edges = np.linspace(0, n, k + 1).astype(int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(k)]


def interleaved_folds(n: int, k: int) -> list[np.ndarray]:
    return [np.arange(i, n, k) for i in range(k)]


def cross_validate(features: np.ndarray, labels: np.ndarray, folds, make_model) -> tuple[float, float, np.ndarray]:
    """Returns accuracy, balanced accuracy and per-trial predictions."""
    predictions = np.empty(len(labels), dtype=labels.dtype)
    for test in folds:
        train = np.setdiff1d(np.arange(len(labels)), test)
        scaler = Standardizer().fit(features[train])
        model = make_model().fit(scaler(features[train]), labels[train])
        predictions[test] = model.predict(scaler(features[test]))
    classes = np.unique(labels)
    recall = [float((predictions[labels == c] == c).mean()) for c in classes]
    return float((predictions == labels).mean()), float(np.mean(recall)), predictions


def permutation_p_value(features, labels, folds, make_model, observed, permutations, rng) -> tuple[float, list[float]]:
    null = []
    for _ in range(permutations):
        shuffled = rng.permutation(labels)
        null.append(cross_validate(features, shuffled, folds, make_model)[0])
    null = np.array(null)
    return float((np.sum(null >= observed) + 1) / (permutations + 1)), null.tolist()


def evaluate_condition(features: np.ndarray, labels: np.ndarray, *, folds: int, permutations: int, rng, seed: int) -> dict:
    n = len(labels)
    block, inter = block_folds(n, folds), interleaved_folds(n, folds)
    lda = lambda: ShrinkageLDA(.2)
    result = {'n': int(n), 'chance': float(1 / len(np.unique(labels)))}
    accuracy, balanced, _ = cross_validate(features, labels, block, lda)
    result['block_lda'] = dict(accuracy=accuracy, balanced_accuracy=balanced)
    if permutations:
        p, null = permutation_p_value(features, labels, block, lda, accuracy, permutations, rng)
        result['block_lda'].update(p_value=p, null_mean=float(np.mean(null)), null_95=float(np.quantile(null, .95)))
    accuracy, balanced, _ = cross_validate(features, labels, inter, lda)
    result['interleaved_lda'] = dict(accuracy=accuracy, balanced_accuracy=balanced)
    accuracy, balanced, _ = cross_validate(features, labels, block, lambda: Logistic(1.))
    result['block_logistic'] = dict(accuracy=accuracy, balanced_accuracy=balanced)
    return result


def evaluate_participant(shard: Shard, *, folds: int, permutations: int, seed: int, stages) -> dict:
    rng = np.random.default_rng(seed)
    all_labels = np.array(shard.prompts)
    report = {'participant': shard.participant, 'trials': len(shard), 'prompt_counts': {p: int((all_labels == p).sum()) for p in PROMPTS}, 'stages': {}}
    for stage in stages:
        started = time.monotonic()
        labels = all_labels[shard.valid(stage)]
        eeg = shard.windows(stage); aux = shard.windows(stage, source='aux')
        block = {}
        block['eeg_tangent'] = evaluate_condition(tangent_features(eeg), labels, folds=folds, permutations=permutations, rng=rng, seed=seed)
        block['eeg_bandpower'] = evaluate_condition(bandpower_features(eeg), labels, folds=folds, permutations=0, rng=rng, seed=seed)
        block['eeg_low_band_tangent'] = evaluate_condition(tangent_features(bandpass_fft(eeg, .5, 20.)), labels, folds=folds, permutations=0, rng=rng, seed=seed)
        block['eeg_high_band_tangent'] = evaluate_condition(tangent_features(bandpass_fft(eeg, 20., 45.)), labels, folds=folds, permutations=0, rng=rng, seed=seed)
        block['aux_only'] = evaluate_condition(aux_features(aux, shard.aux_names), labels, folds=folds, permutations=permutations, rng=rng, seed=seed)
        binary = {}
        tangent = tangent_features(eeg)
        for task, mapping in BINARY_TASKS.items():
            y = np.array([mapping[p] for p in labels])
            accuracy, balanced, _ = cross_validate(tangent, y, block_folds(len(y), folds), lambda: ShrinkageLDA(.2))
            binary[task] = dict(block_lda_accuracy=accuracy, block_lda_balanced_accuracy=balanced, positive_rate=float(y.mean()))
        block['binary_tasks'] = binary
        block['seconds'] = round(time.monotonic() - started, 1)
        report['stages'][stage] = block
        print(json.dumps(dict(participant=shard.participant, stage=stage, block_lda=round(block['eeg_tangent']['block_lda']['accuracy'], 3),
                              p=block['eeg_tangent']['block_lda'].get('p_value'), interleaved=round(block['eeg_tangent']['interleaved_lda']['accuracy'], 3),
                              aux_only=round(block['aux_only']['block_lda']['accuracy'], 3), high_band=round(block['eeg_high_band_tangent']['block_lda']['accuracy'], 3),
                              chance=round(block['eeg_tangent']['chance'], 3), seconds=block['seconds'])), flush=True)
    return report


def summarize(reports: list[dict], stages) -> dict:
    def ci(values):
        values = np.asarray(values, dtype=float); n = len(values)
        if n < 2:
            return [float('nan'), float('nan')]
        t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160}.get(n, 1.96)
        half = t * values.std(ddof=1) / np.sqrt(n); return [float(values.mean() - half), float(values.mean() + half)]
    summary = {}
    for stage in stages:
        rows = [r['stages'][stage] for r in reports if stage in r['stages']]
        if not rows:
            continue
        entry = {'participants': len(rows)}
        for condition in ('eeg_tangent', 'eeg_bandpower', 'eeg_low_band_tangent', 'eeg_high_band_tangent', 'aux_only'):
            block = [r[condition]['block_lda']['accuracy'] for r in rows]
            inter = [r[condition]['interleaved_lda']['accuracy'] for r in rows]
            entry[condition] = dict(block_lda_mean=float(np.mean(block)), block_lda_ci95=ci(block), interleaved_lda_mean=float(np.mean(inter)),
                                    block_logistic_mean=float(np.mean([r[condition]['block_logistic']['accuracy'] for r in rows])))
            p_values = [r[condition]['block_lda'].get('p_value') for r in rows]
            if all(p is not None for p in p_values):
                entry[condition]['participants_p_below_0.05'] = int(sum(p < .05 for p in p_values))
                entry[condition]['p_values'] = p_values
        entry['chance'] = rows[0]['eeg_tangent']['chance']
        entry['binary_tasks'] = {task: dict(block_lda_mean=float(np.mean([r['binary_tasks'][task]['block_lda_accuracy'] for r in rows])),
                                            balanced_mean=float(np.mean([r['binary_tasks'][task]['block_lda_balanced_accuracy'] for r in rows])))
                                 for task in BINARY_TASKS}
        summary[stage] = entry
    return summary


def baselines_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shards', default=str(ROOT / 'artifacts/karaone/shards'))
    parser.add_argument('--output', default=str(ROOT / 'outputs/karaone/baselines'))
    parser.add_argument('--participants', nargs='*')
    parser.add_argument('--stages', nargs='*', default=list(STAGES))
    parser.add_argument('--folds', type=int, default=6)
    parser.add_argument('--permutations', type=int, default=200)
    parser.add_argument('--seed', type=int, default=31)
    args = parser.parse_args(argv)
    shards = sorted(Path(args.shards).glob('*.h5'))
    if args.participants:
        shards = [s for s in shards if s.stem in args.participants]
    if not shards:
        raise FileNotFoundError('no KaraOne shards; run scripts/prepare_karaone.py first')
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    reports = []
    for path in shards:
        report = evaluate_participant(Shard(path), folds=args.folds, permutations=args.permutations, seed=args.seed, stages=args.stages)
        (output / f'{report["participant"]}.json').write_text(json.dumps(report, indent=2) + '\n')
        reports.append(report)
    summary = dict(contract='karaone_baselines_v1', folds=args.folds, permutations=args.permutations, windows_s=WINDOWS,
                   participants=[r['participant'] for r in reports], stages=summarize(reports, args.stages))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f"{'stage':10s} {'EEG block':>10s} {'CI95':>18s} {'p<.05':>6s} {'interleaved':>12s} {'aux-only':>9s} {'low band':>9s} {'high band':>10s} {'chance':>7s}")
    for stage, entry in summary['stages'].items():
        e = entry['eeg_tangent']
        print(f"{stage:10s} {e['block_lda_mean']:10.3f} [{e['block_lda_ci95'][0]:.3f}, {e['block_lda_ci95'][1]:.3f}] {e.get('participants_p_below_0.05', 0):>3d}/{entry['participants']:<2d} {e['interleaved_lda_mean']:12.3f} "
              f"{entry['aux_only']['block_lda_mean']:9.3f} {entry['eeg_low_band_tangent']['block_lda_mean']:9.3f} {entry['eeg_high_band_tangent']['block_lda_mean']:10.3f} {entry['chance']:7.3f}")


# --- 2. DS004940 encoder transfer --------------------------------------------------------



PRE_ONSET_S = .25


EEG_SAMPLES = 1178


CONTENT_SECONDS = {'stimulus': 2., 'thinking': 4., 'clearing': 4., 'speaking': 2.}


DEFAULT_CHECKPOINT = ROOT / 'outputs/aligned_recovery_v3/full_seed322_positional/best_passed.pt'


def load_encoder(checkpoint: Path, device, random_init: bool = False):
    import aligned_recovery as recovery
    from aligned_recovery_model import RecoveryEEGModel
    legacy = recovery.legacy
    payload = recovery.load_checkpoint(checkpoint)
    spec = dict(payload['signature']['spec']); spec['subjects'] = 0        # unknown participants: shared identity path
    model = RecoveryEEGModel(legacy.decoder_from(payload), **spec)
    if not random_init:
        state = {k: v for k, v in payload['model'].items() if not k.startswith('subject_delta')}
        model.load_state_dict(state, strict=False)
    return model.to(device).eval(), payload['signature']


def stage_windows(shard: Shard, stage: str) -> np.ndarray:
    """(trials, 128, 1178) encoder inputs in the BioSemi-128 layout."""
    mapping = shard.h5['biosemi128_map'][:]
    a = -int(round(PRE_ONSET_S * TARGET_RATE))
    eeg = shard.h5['eeg']
    out = []
    for onset in shard.onsets[stage][shard.valid(stage)]:
        window = (eeg[:, onset + a: onset + a + EEG_SAMPLES] - shard.center) / shard.scale
        out.append(mapping @ window)
    return np.stack(out).astype(np.float32)


def embed(model, windows: np.ndarray, content_seconds: float, device, batch: int = 32) -> np.ndarray:
    times = model.decoder.speech_times
    mask = (times < content_seconds).float()[None, :, None].to(device)
    xyz = torch.zeros(1, 128, 3, device=device); channel_mask = torch.ones(1, 128, dtype=torch.bool, device=device)
    time_mask = torch.ones(1, EEG_SAMPLES, dtype=torch.bool, device=device)
    out = []
    for start in range(0, len(windows), batch):
        eeg = torch.from_numpy(windows[start:start + batch]).to(device)
        n = len(eeg)
        state = model(eeg, xyz.expand(n, -1, -1), channel_mask.expand(n, -1), time_mask.expand(n, -1), None)
        z = model.decoder.normalizer(state.aligned_sequence)
        out.append(((z * mask).sum(1) / mask.sum(1)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def score(features, labels, folds, permutations, rng) -> dict:
    lda = lambda: ShrinkageLDA(.2)
    block = block_folds(len(labels), folds)
    accuracy, balanced, _ = cross_validate(features, labels, block, lda)
    result = {'n': int(len(labels)), 'chance': float(1 / len(np.unique(labels))), 'block_lda': dict(accuracy=accuracy, balanced_accuracy=balanced)}
    if permutations:
        p, null = permutation_p_value(features, labels, block, lda, accuracy, permutations, rng)
        result['block_lda'].update(p_value=p, null_mean=float(np.mean(null)), null_95=float(np.quantile(null, .95)))
    accuracy, balanced, _ = cross_validate(features, labels, interleaved_folds(len(labels), folds), lda)
    result['interleaved_lda'] = dict(accuracy=accuracy, balanced_accuracy=balanced)
    return result


def cross_stage(source_features, source_labels, target_features, target_labels, folds: int, permutations: int, rng) -> dict:
    """Fit on one stage, test on another stage of the held-out blocks (trials aligned by index)."""
    n = min(len(source_labels), len(target_labels))
    source_features, source_labels = source_features[:n], source_labels[:n]
    target_features, target_labels = target_features[:n], target_labels[:n]
    def run(labels_target):
        predictions = np.empty(n, dtype=target_labels.dtype)
        for test in block_folds(n, folds):
            train = np.setdiff1d(np.arange(n), test)
            scaler = Standardizer().fit(source_features[train])
            model = ShrinkageLDA(.2).fit(scaler(source_features[train]), source_labels[train])
            predictions[test] = model.predict(scaler(target_features[test]))
        return float((predictions == labels_target).mean())
    observed = run(target_labels)
    null = [run(rng.permutation(target_labels)) for _ in range(permutations)]
    return dict(accuracy=observed, chance=float(1 / len(np.unique(target_labels))),
                p_value=float((np.sum(np.array(null) >= observed) + 1) / (permutations + 1)) if permutations else None)


def transfer_participant(shard: Shard, encoders: dict, device, *, folds: int, permutations: int, seed: int, stages) -> dict:
    rng = np.random.default_rng(seed)
    all_labels = np.array(shard.prompts)
    report = {'participant': shard.participant, 'stages': {}, 'cross_stage': {}}
    embeddings = {name: {} for name in encoders}
    labels_by_stage = {}
    for stage in stages:
        started = time.monotonic()
        labels = all_labels[shard.valid(stage)]; labels_by_stage[stage] = labels
        windows = stage_windows(shard, stage)
        entry = {}
        for name, model in encoders.items():
            features = embed(model, windows, CONTENT_SECONDS[stage], device)
            embeddings[name][stage] = features
            entry[name] = score(features, labels, folds, permutations if name == 'pretrained' else 0, rng)
        entry['native_tangent'] = score(tangent_features(shard.windows(stage)), labels, folds, 0, rng)
        entry['seconds'] = round(time.monotonic() - started, 1)
        report['stages'][stage] = entry
        print(json.dumps(dict(participant=shard.participant, stage=stage,
                              **{name: round(entry[name]['block_lda']['accuracy'], 3) for name in list(encoders) + ['native_tangent']},
                              p_pretrained=entry['pretrained']['block_lda'].get('p_value'), chance=round(entry['pretrained']['chance'], 3))), flush=True)
    if 'stimulus' in stages and 'thinking' in stages:
        valid = shard.valid('stimulus') & shard.valid('thinking')
        keep_s = valid[shard.valid('stimulus')]; keep_t = valid[shard.valid('thinking')]
        for name in encoders:
            report['cross_stage'][name] = {
                'heard_to_imagined': cross_stage(embeddings[name]['stimulus'][keep_s], labels_by_stage['stimulus'][keep_s],
                                                 embeddings[name]['thinking'][keep_t], labels_by_stage['thinking'][keep_t], folds, permutations, rng),
                'imagined_to_heard': cross_stage(embeddings[name]['thinking'][keep_t], labels_by_stage['thinking'][keep_t],
                                                 embeddings[name]['stimulus'][keep_s], labels_by_stage['stimulus'][keep_s], folds, permutations, rng)}
        print(json.dumps(dict(participant=shard.participant, cross_stage={n: {k: round(v['accuracy'], 3) for k, v in r.items()} for n, r in report['cross_stage'].items()})), flush=True)
    return report


def transfer_summary(reports, stages, encoders) -> dict:
    def mean_ci(values):
        values = np.asarray(values, float); n = len(values)
        if n < 2:
            return dict(mean=float(values.mean()) if n else float('nan'), ci95=[float('nan')] * 2, n=n)
        t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160}.get(n, 1.96)
        half = t * values.std(ddof=1) / np.sqrt(n); return dict(mean=float(values.mean()), ci95=[float(values.mean() - half), float(values.mean() + half)], n=n)
    summary = {'stages': {}, 'cross_stage': {}}
    for stage in stages:
        rows = [r['stages'][stage] for r in reports if stage in r['stages']]
        entry = {'chance': rows[0]['pretrained']['chance']}
        for name in list(encoders) + ['native_tangent']:
            entry[name] = dict(block=mean_ci([r[name]['block_lda']['accuracy'] for r in rows]), interleaved=mean_ci([r[name]['interleaved_lda']['accuracy'] for r in rows]))
            p_values = [r[name]['block_lda'].get('p_value') for r in rows]
            if all(p is not None for p in p_values):
                entry[name]['participants_p_below_0.05'] = int(sum(p < .05 for p in p_values))
        summary['stages'][stage] = entry
    for name in encoders:
        rows = [r['cross_stage'][name] for r in reports if name in r.get('cross_stage', {})]
        if rows:
            summary['cross_stage'][name] = {direction: {**mean_ci([r[direction]['accuracy'] for r in rows]),
                                                        'participants_p_below_0.05': int(sum((r[direction]['p_value'] or 1) < .05 for r in rows)),
                                                        'chance': rows[0][direction]['chance']}
                                            for direction in ('heard_to_imagined', 'imagined_to_heard')}
    return summary


def transfer_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT))
    parser.add_argument('--shards', default=str(ROOT / 'artifacts/karaone/shards'))
    parser.add_argument('--output', default=str(ROOT / 'outputs/karaone/transfer'))
    parser.add_argument('--participants', nargs='*')
    parser.add_argument('--stages', nargs='*', default=['stimulus', 'thinking', 'clearing'])
    parser.add_argument('--folds', type=int, default=6)
    parser.add_argument('--permutations', type=int, default=200)
    parser.add_argument('--seed', type=int, default=31)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed); torch.set_num_threads(4)
    device = torch.device(args.device)
    encoders = {'pretrained': load_encoder(Path(args.checkpoint), device)[0], 'random_init': load_encoder(Path(args.checkpoint), device, random_init=True)[0]}
    shards = sorted(Path(args.shards).glob('*.h5'))
    if args.participants:
        shards = [s for s in shards if s.stem in args.participants]
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    reports = []
    for path in shards:
        report = transfer_participant(Shard(path), encoders, device, folds=args.folds, permutations=args.permutations, seed=args.seed, stages=args.stages)
        (output / f'{report["participant"]}.json').write_text(json.dumps(report, indent=2) + '\n')
        reports.append(report)
    summary = dict(contract='karaone_transfer_v1', checkpoint=str(Path(args.checkpoint).relative_to(ROOT)) if Path(args.checkpoint).is_relative_to(ROOT) else args.checkpoint,
                   folds=args.folds, permutations=args.permutations, content_seconds=CONTENT_SECONDS, pre_onset_s=PRE_ONSET_S,
                   participants=[r['participant'] for r in reports], **transfer_summary(reports, args.stages, encoders))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f"{'stage':10s} {'pretrained':>11s} {'CI95':>16s} {'p<.05':>6s} {'random_init':>12s} {'native TS':>10s} {'chance':>7s}")
    for stage, entry in summary['stages'].items():
        b = entry['pretrained']['block']
        print(f"{stage:10s} {b['mean']:11.3f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}] {entry['pretrained'].get('participants_p_below_0.05', 0):>3d}/{b['n']:<2d} "
              f"{entry['random_init']['block']['mean']:12.3f} {entry['native_tangent']['block']['mean']:10.3f} {entry['chance']:7.3f}")
    for name, entry in summary['cross_stage'].items():
        for direction, value in entry.items():
            print(f"cross-stage {name:12s} {direction:18s} {value['mean']:.3f} [{value['ci95'][0]:.3f}, {value['ci95'][1]:.3f}]  p<.05 {value['participants_p_below_0.05']}/{value['n']}  chance {value['chance']:.3f}")

def main():
    commands = {'baselines': baselines_main, 'transfer': transfer_main}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        raise SystemExit('usage: python app/karaone.py baselines|transfer [options]')
    commands[sys.argv[1]](sys.argv[2:])


if __name__ == '__main__':
    main()

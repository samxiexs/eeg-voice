#!/usr/bin/env python3
"""Does the DS004940-pretrained EEG encoder carry prompt information on KaraOne?

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
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src')); sys.path.insert(0, str(ROOT / 'scripts'))
from karaone_baselines import (Shard, ShrinkageLDA, Standardizer, block_folds, cross_validate, interleaved_folds,
                               permutation_p_value, tangent_features)
from prepare_karaone import TARGET_RATE

PRE_ONSET_S = .25
EEG_SAMPLES = 1178
# Frames of the predicted speech sequence that count as "content" per stage.
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


@torch.no_grad()
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


def evaluate_participant(shard: Shard, encoders: dict, device, *, folds: int, permutations: int, seed: int, stages) -> dict:
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


def summarize(reports, stages, encoders) -> dict:
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


def main():
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
    args = parser.parse_args()
    torch.manual_seed(args.seed); torch.set_num_threads(4)
    device = torch.device(args.device)
    encoders = {'pretrained': load_encoder(Path(args.checkpoint), device)[0], 'random_init': load_encoder(Path(args.checkpoint), device, random_init=True)[0]}
    shards = sorted(Path(args.shards).glob('*.h5'))
    if args.participants:
        shards = [s for s in shards if s.stem in args.participants]
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    reports = []
    for path in shards:
        report = evaluate_participant(Shard(path), encoders, device, folds=args.folds, permutations=args.permutations, seed=args.seed, stages=args.stages)
        (output / f'{report["participant"]}.json').write_text(json.dumps(report, indent=2) + '\n')
        reports.append(report)
    summary = dict(contract='karaone_transfer_v1', checkpoint=str(Path(args.checkpoint).relative_to(ROOT)) if Path(args.checkpoint).is_relative_to(ROOT) else args.checkpoint,
                   folds=args.folds, permutations=args.permutations, content_seconds=CONTENT_SECONDS, pre_onset_s=PRE_ONSET_S,
                   participants=[r['participant'] for r in reports], **summarize(reports, args.stages, encoders))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f"{'stage':10s} {'pretrained':>11s} {'CI95':>16s} {'p<.05':>6s} {'random_init':>12s} {'native TS':>10s} {'chance':>7s}")
    for stage, entry in summary['stages'].items():
        b = entry['pretrained']['block']
        print(f"{stage:10s} {b['mean']:11.3f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}] {entry['pretrained'].get('participants_p_below_0.05', 0):>3d}/{b['n']:<2d} "
              f"{entry['random_init']['block']['mean']:12.3f} {entry['native_tangent']['block']['mean']:10.3f} {entry['chance']:7.3f}")
    for name, entry in summary['cross_stage'].items():
        for direction, value in entry.items():
            print(f"cross-stage {name:12s} {direction:18s} {value['mean']:.3f} [{value['ci95'][0]:.3f}, {value['ci95'][1]:.3f}]  p<.05 {value['participants_p_below_0.05']}/{value['n']}  chance {value['chance']:.3f}")


if __name__ == '__main__':
    main()

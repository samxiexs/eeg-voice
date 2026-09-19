#!/usr/bin/env python3
"""Semantic-congruency (N400) probe on DS004940 Active trials.

Every Active sentence ends in a congruent (NPC) or incongruent (NPI) final
word.  This probe asks whether the final word's congruency can be read from
(a) the raw EEG after the critical word (ERP mean amplitudes, Riemannian
covariance) and (b) the trained encoder's internal tokens over the same
window — i.e. whether the representation carries linguistic information
beyond the acoustic envelope.

Protocol: fit on train-role trials, evaluate on validation-role trials
(content-disjoint, known participants); permutation null by shuffling the
training labels (200 times).  Controls: the same window *before* the
critical word (must be at chance unless context or acoustics differ), and
audio-only features of the final word (if the audio alone separates NPC from
NPI, an EEG result could be acoustic rather than semantic).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('ALIGNED_TARGET_CACHE_NAME', 'targets_adapted.h5')
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src'))
import aligned_recovery as recovery
from aligned_recovery_model import RecoveryEEGModel
from karaone_baselines import ShrinkageLDA, Standardizer, tangent_features

legacy = recovery.legacy
RATE = 256; EEG_START = -.25; WINDOW_S = .9
BINS = ((0., .3), (.3, .5), (.5, .7), (.7, .9))


def critical_word_onsets(frame: pd.DataFrame) -> np.ndarray:
    """Seconds from stimulus onset to the critical (final) word, from the BIDS events."""
    onsets = np.full(len(frame), np.nan)
    for path, rows in frame.groupby('source_event_path'):
        events = pd.read_csv(ROOT / path, sep='\t', keep_default_na=False)
        for index, row in rows.iterrows():
            event = events.iloc[int(row.source_event_row)]
            onsets[index] = float(event['onset']) - float(event['stim_onset_s_'])
    return onsets


def labels_of(frame: pd.DataFrame) -> np.ndarray:
    prefix = frame.stim_file.str[:3]
    return np.where(prefix == 'NPC', 1, np.where(prefix == 'NPI', 0, -1))


def erp_features(eeg: np.ndarray, start: int) -> np.ndarray:
    """Mean amplitude per channel in four post-onset bins; eeg is (channels, samples)."""
    return np.concatenate([eeg[:, start + int(a * RATE): start + int(b * RATE)].mean(1) for a, b in BINS])


def collect(model, dataset, subjects, device, onsets, labels, pre_word: bool):
    """Per-trial feature sets for the window after (or before) the critical word."""
    hooks = {}
    handle = model.output_norm.register_forward_hook(lambda module, inputs, output: hooks.__setitem__('tokens', output.detach()))
    token_times = model.token_times.cpu().numpy()
    rows = []
    with torch.inference_mode():
        for offset in range(0, len(dataset), 32):
            ids = [i for i in range(offset, min(len(dataset), offset + 32)) if labels[i] >= 0]
            if not ids:
                continue
            batch = legacy.move(torch.utils.data.default_collate([dataset[i] for i in ids]), device)
            subject = torch.tensor([subjects[s] for s in batch['subject']], device=device) if subjects else None
            model(batch['eeg'], batch['channel_xyz'], batch['channel_mask'], batch['time_mask'], subject)
            tokens = hooks['tokens'].cpu().numpy(); eeg = batch['eeg'].cpu().numpy(); mel = batch['mel'].cpu().numpy()
            for j, i in enumerate(ids):
                onset = onsets[i] - WINDOW_S if pre_word else onsets[i]
                if onset < 0:
                    continue
                start = int(round((onset - EEG_START) * RATE))
                if start + int(WINDOW_S * RATE) > eeg.shape[-1]:
                    continue
                window = eeg[j, :, start: start + int(WINDOW_S * RATE)]
                selected = (token_times >= onset) & (token_times < onset + WINDOW_S)
                mel_start = int(round(onset / (256 / 16000)))
                rows.append(dict(index=i, label=int(labels[i]), subject=batch['subject'][j], content=batch['content'][j],
                                 erp=erp_features(eeg[j], start), tangent=tangent_features(window[None])[0],
                                 tokens=tokens[j, selected].mean(0), audio=mel[j][:, mel_start: mel_start + int(WINDOW_S * 16000 / 256)].mean(1)))
    handle.remove()
    return rows


def content_permutation(rows, rng):
    """Permute labels at the SENTENCE level, not the trial level.

    Each sentence is heard by ~17 participants and carries one label, so a
    trial-level permutation destroys that structure and yields an
    over-optimistic null: any feature that merely identifies the sentence
    beats it.  Permuting whole sentences keeps the structure in the null.
    """
    contents = sorted({r['content'] for r in rows})
    labels = [next(r['label'] for r in rows if r['content'] == c) for c in contents]
    mapping = dict(zip(contents, rng.permutation(labels)))
    return np.array([mapping[r['content']] for r in rows])


def evaluate(train_rows, test_rows, key, permutations, rng):
    x_train = np.stack([r[key] for r in train_rows]); y_train = np.array([r['label'] for r in train_rows])
    x_test = np.stack([r[key] for r in test_rows]); y_test = np.array([r['label'] for r in test_rows])
    scaler = Standardizer().fit(x_train)
    def accuracy(y):
        model = ShrinkageLDA(.2).fit(scaler(x_train), y)
        predictions = model.predict(scaler(x_test))
        return float((predictions == y_test).mean()), predictions
    observed, predictions = accuracy(y_train)
    null = [accuracy(content_permutation(train_rows, rng))[0] for _ in range(permutations)]
    subjects = sorted({r['subject'] for r in test_rows})
    per_subject = {s: float(np.mean([p == r['label'] for p, r in zip(predictions, test_rows) if r['subject'] == s])) for s in subjects}
    values = np.array(list(per_subject.values())); n = len(values)
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120}.get(n, 1.96)
    half = t * values.std(ddof=1) / np.sqrt(n) if n > 1 else float('nan')
    balanced = float(np.mean([(predictions[y_test == c] == c).mean() for c in (0, 1)]))
    contents = sorted({r['content'] for r in test_rows})
    per_content = [float(np.mean([p == r['label'] for p, r in zip(predictions, test_rows) if r['content'] == c])) for c in contents]
    votes = [float(np.mean([p for p, r in zip(predictions, test_rows) if r['content'] == c])) for c in contents]
    truths = [next(r['label'] for r in test_rows if r['content'] == c) for c in contents]
    majority = float(np.mean([(v > .5) == bool(y) for v, y in zip(votes, truths)]))
    return dict(content_mean_accuracy=float(np.mean(per_content)), content_majority_accuracy=majority, n_contents=len(contents),
                accuracy=observed, balanced_accuracy=balanced, p_value=float((np.sum(np.array(null) >= observed) + 1) / (permutations + 1)) if permutations else None,
                null_mean=float(np.mean(null)) if null else None, subject_mean=float(values.mean()), subject_ci95=[float(values.mean() - half), float(values.mean() + half)],
                subjects_above_0_5=int((values > .5).sum()), n_subjects=n, n_test=len(test_rows), positive_rate_test=float(y_test.mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=str(ROOT / 'outputs/aligned_recovery_v3/full_seed322_positional/best_passed.pt'))
    parser.add_argument('--config', default=str(ROOT / 'configs/aligned_speech_local_v1.yaml'))
    parser.add_argument('--roles', nargs='*', default=['validation'])
    parser.add_argument('--permutations', type=int, default=200)
    parser.add_argument('--seed', type=int, default=31)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default=str(ROOT / 'outputs/aligned_recovery_v3/congruency_probe'))
    args = parser.parse_args()
    torch.set_num_threads(4); device = torch.device(args.device); rng = np.random.default_rng(args.seed)
    cfg = legacy.config(args.config)
    payload = recovery.load_checkpoint(Path(args.checkpoint))
    train = legacy.dataset_for(cfg, 'train')
    model = RecoveryEEGModel(legacy.decoder_from(payload), **payload['signature']['spec']).to(device)
    model.load_state_dict(payload['model']); model.eval()
    subjects = {s: i for i, s in enumerate(payload['signature']['subjects'])} if payload['signature']['spec'].get('subjects') else None
    features = {}
    for role in ['train', *args.roles]:
        dataset = train if role == 'train' else legacy.dataset_for(cfg, role)
        onsets = critical_word_onsets(dataset.frame); labels = labels_of(dataset.frame)
        features[role] = {window: collect(model, dataset, subjects, device, onsets, labels, pre_word=(window == 'pre_word')) for window in ('post_word', 'pre_word')}
        print(json.dumps(dict(role=role, trials={w: len(v) for w, v in features[role].items()})), flush=True)
    report = dict(contract='congruency_probe_v1', checkpoint=str(args.checkpoint), window_s=WINDOW_S, bins=BINS, permutations=args.permutations, results={})
    for role in args.roles:
        report['results'][role] = {}
        for window in ('post_word', 'pre_word'):
            report['results'][role][window] = {}
            for key, name in (('erp', 'eeg_erp_bins'), ('tangent', 'eeg_riemannian'), ('tokens', 'encoder_tokens'), ('audio', 'audio_only_mel')):
                result = evaluate(features['train'][window], features[role][window], key, args.permutations, rng)
                report['results'][role][window][name] = result
                print(f"{role:10s} {window:9s} {name:16s} acc {result['accuracy']:.3f} bal {result['balanced_accuracy']:.3f} p={result['p_value']:.3f} (sentence-null {result['null_mean']:.3f}) | subjects {result['subject_mean']:.3f} [{result['subject_ci95'][0]:.3f}, {result['subject_ci95'][1]:.3f}] {result['subjects_above_0_5']}/{result['n_subjects']}>0.5 | per-sentence {result['content_mean_accuracy']:.3f}, vote {result['content_majority_accuracy']:.3f} of {result['n_contents']}", flush=True)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / 'congruency_probe.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

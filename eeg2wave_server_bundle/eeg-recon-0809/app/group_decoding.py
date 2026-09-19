#!/usr/bin/env python3
"""How much sentence information is there when EEG trials of the same sentence are pooled?

Each DS004940 sentence was heard by ~17 participants.  This script evaluates a
trained encoder on held-out sentences while averaging k trials of the same
sentence (k = 1, 2, 4, 8, all), in two ways:

* **input averaging** — the participant-mixed EEG windows are averaged before
  the shared encoder (SNR grows with k, as in classic ERP averaging);
* **output averaging** — every trial is decoded alone and the predicted mel /
  content embeddings are averaged afterwards (what a system with repeated
  presentations would do).

Metrics are the speech-frame metrics of the recovery route: mel MAE against the
median template, envelope correlation, and retrieval among the split's unseen
sentences.  This is a group-level statement about the information present in
EEG, not a single-trial BCI result, and it uses no oracle audio at inference.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('ALIGNED_TARGET_CACHE_NAME', 'targets_adapted.h5')
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src'))
import aligned_recovery as recovery
from aligned_recovery_model import RecoveryEEGModel, speech_frame_masks
from aligned_recovery_eval import train_templates, pearson, chance_mrr

legacy = recovery.legacy


def load(checkpoint: Path, cfg, role: str, device):
    payload = recovery.load_checkpoint(checkpoint)
    train = legacy.dataset_for(cfg, 'train'); data = legacy.dataset_for(cfg, role)
    model = RecoveryEEGModel(legacy.decoder_from(payload), **payload['signature']['spec']).to(device)
    model.load_state_dict(payload['model']); model.eval()
    subjects = {s: i for i, s in enumerate(payload['signature']['subjects'])} if payload['signature']['spec'].get('subjects') else None
    return model, train, data, subjects


@torch.no_grad()
def mixed_inputs(model, batch, subjects, device):
    """Participant-specific mixing applied outside the model so inputs can be averaged afterwards."""
    eeg = batch['eeg'] * batch['channel_mask'][:, :, None]
    if subjects is None or not model.subjects:
        return eeg
    index = torch.tensor([subjects[s] for s in batch['subject']], device=device)
    return model.mix_subject(eeg, index)


@torch.no_grad()
def decode(model, eeg, device):
    n = len(eeg)
    xyz = torch.zeros(n, model.channels, 3, device=device); mask = torch.ones(n, model.channels, dtype=torch.bool, device=device)
    times = torch.ones(n, eeg.shape[-1], dtype=torch.bool, device=device)
    state = model(eeg, xyz, mask, times, None)
    return state.native_mel, model.decoder.normalizer(state.aligned_sequence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=str(ROOT / 'outputs/aligned_recovery_v3/full_seed322_positional/best_passed.pt'))
    parser.add_argument('--config', default=str(ROOT / 'configs/aligned_speech_local_v1.yaml'))
    parser.add_argument('--role', choices=['validation', 'test'], default='validation')
    parser.add_argument('--ks', nargs='*', type=int, default=[1, 2, 4, 8, 0], help='trials averaged per sentence; 0 = all')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=31)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default=str(ROOT / 'outputs/aligned_recovery_v3/group_decoding'))
    args = parser.parse_args()
    torch.set_num_threads(4); device = torch.device(args.device)
    cfg = legacy.config(args.config)
    model, train, data, subjects = load(Path(args.checkpoint), cfg, args.role, device)
    _, speech_median, _, _ = train_templates(train)
    speech_median = speech_median.to(device)
    rng = np.random.default_rng(args.seed)
    # ---- decode every trial once; keep mixed inputs, outputs and targets grouped by sentence
    groups = {}
    for offset in range(0, len(data), 16):
        ids = list(range(offset, min(len(data), offset + 16)))
        batch = legacy.move(torch.utils.data.default_collate([data[i] for i in ids]), device)
        eeg = mixed_inputs(model, batch, subjects, device)
        mel, z = decode(model, eeg, device)
        mel_mask, speech_mask = speech_frame_masks(batch['oracle_duration_frames'], model.decoder.mel_times, model.decoder.speech_times)
        teacher = model.decoder.normalizer(batch['teacher'])
        for i, content in enumerate(batch['content']):
            g = groups.setdefault(content, dict(eeg=[], mel=[], z=[], target=batch['mel'][i], teacher=teacher[i], mel_mask=mel_mask[i], speech_mask=speech_mask[i]))
            g['eeg'].append(eeg[i]); g['mel'].append(mel[i]); g['z'].append(z[i])
    contents = sorted(groups)
    prototypes = F.normalize(torch.stack([(groups[c]['teacher'] * groups[c]['speech_mask'][:, None]).sum(0) / groups[c]['speech_mask'].sum() for c in contents]), dim=-1)
    # Time-resolved candidates: per-frame unit teacher sequences and their speech masks (the training objective's similarity).
    teacher_frames = F.normalize(torch.stack([groups[c]['teacher'] for c in contents]), dim=-1)          # C, T, D
    teacher_masks = torch.stack([groups[c]['speech_mask'] for c in contents]).float()                    # C, T
    # Oracle-free variant: a fixed window inside every sentence (shortest is 1.59 s).
    fixed = (model.decoder.speech_times < 1.5).float()
    durations = torch.tensor([float(groups[c]['speech_mask'].sum()) for c in contents])
    zero_mel, _ = decode(model, torch.zeros(1, model.channels, data[0]['eeg'].shape[-1], device=device), device)
    zero_mel = zero_mel[0]

    def metrics(mel, z, g):
        m, s = g['mel_mask'], g['speech_mask']; truth = g['target']
        embedding = F.normalize((z * s[:, None]).sum(0) / s.sum(), dim=-1)
        rank = int((prototypes @ embedding).argsort(descending=True).tolist().index(contents.index(g['name'])) + 1)
        pair = teacher_masks * s.float()[None, :]
        frame_similarity = ((F.normalize(z, dim=-1)[None] * teacher_frames).sum(-1) * pair).sum(-1) / pair.sum(-1).clamp_min(1.)
        rank_frames = int(frame_similarity.argsort(descending=True).tolist().index(contents.index(g['name'])) + 1)
        fixed_similarity = ((F.normalize(z, dim=-1)[None] * teacher_frames).sum(-1) * fixed[None]).sum(-1) / fixed.sum()
        rank_fixed = int(fixed_similarity.argsort(descending=True).tolist().index(contents.index(g['name'])) + 1)
        # Duration-only shortcut: rank candidates by closeness of their duration to the target's oracle duration.
        rank_duration = int((durations - float(s.sum())).abs().argsort().tolist().index(contents.index(g['name'])) + 1)
        return dict(mel_mae=float((mel[:, m] - truth[:, m]).abs().mean()), template_mae=float((speech_median[:, m] - truth[:, m]).abs().mean()),
                    zero_mae=float((zero_mel[:, m] - truth[:, m]).abs().mean()),
                    envelope_corr=pearson(mel[:, m].mean(0), truth[:, m].mean(0)), envelope_template_corr=pearson(speech_median[:, m].mean(0), truth[:, m].mean(0)),
                    envelope_zero_corr=pearson(zero_mel[:, m].mean(0), truth[:, m].mean(0)), rank=rank, rank_frames=rank_frames,
                    rank_fixed=rank_fixed, rank_duration=rank_duration)

    for c in contents:
        groups[c]['name'] = c
    results = {}
    for k in args.ks:
        rows = {'input': [], 'output': []}
        for c in contents:
            g = groups[c]; n = len(g['eeg']); size = n if k == 0 or k >= n else k
            repeats = 1 if size == n else args.repeats
            for _ in range(repeats):
                chosen = rng.choice(n, size=size, replace=False)
                averaged = torch.stack([g['eeg'][j] for j in chosen]).mean(0, keepdim=True)
                mel_in, z_in = decode(model, averaged, device)
                rows['input'].append(metrics(mel_in[0], z_in[0], g))
                mel_out = torch.stack([g['mel'][j] for j in chosen]).mean(0); z_out = torch.stack([g['z'][j] for j in chosen]).mean(0)
                rows['output'].append(metrics(mel_out, z_out, g))
        results[str(k)] = {}
        for mode, values in rows.items():
            ranks = np.array([v['rank'] for v in values]); frame_ranks = np.array([v['rank_frames'] for v in values])
            fixed_ranks = np.array([v['rank_fixed'] for v in values]); duration_ranks = np.array([v['rank_duration'] for v in values])
            results[str(k)][mode] = dict(pairs=len(values), mel_mae=float(np.mean([v['mel_mae'] for v in values])),
                                         template_mae=float(np.mean([v['template_mae'] for v in values])), zero_mae=float(np.mean([v['zero_mae'] for v in values])),
                                         envelope_corr=float(np.mean([v['envelope_corr'] for v in values])),
                                         envelope_template_corr=float(np.mean([v['envelope_template_corr'] for v in values])),
                                         envelope_zero_corr=float(np.mean([v['envelope_zero_corr'] for v in values])),
                                         retrieval_r1=float((ranks == 1).mean()), retrieval_top5=float((ranks <= 5).mean()), retrieval_mrr=float((1 / ranks).mean()),
                                         frame_retrieval_r1=float((frame_ranks == 1).mean()), frame_retrieval_top5=float((frame_ranks <= 5).mean()), frame_retrieval_mrr=float((1 / frame_ranks).mean()),
                                         fixed_window_retrieval_r1=float((fixed_ranks == 1).mean()), fixed_window_retrieval_top5=float((fixed_ranks <= 5).mean()), fixed_window_retrieval_mrr=float((1 / fixed_ranks).mean()),
                                         duration_only_retrieval_r1=float((duration_ranks == 1).mean()), duration_only_retrieval_top5=float((duration_ranks <= 5).mean()))
        label = 'all' if k == 0 else str(k)
        for mode in ('input', 'output'):
            r = results[str(k)][mode]
            print(f"k={label:>3s} {mode:6s} mel {r['mel_mae']:.4f} (template {r['template_mae']:.4f}, zero {r['zero_mae']:.4f}) | envelope r {r['envelope_corr']:.3f} (template {r['envelope_template_corr']:.3f}, zero {r['envelope_zero_corr']:.3f}) | pooled R1 {r['retrieval_r1']:.3f} | masked-frames R1 {r['frame_retrieval_r1']:.3f} top5 {r['frame_retrieval_top5']:.3f} | FIXED-1.5s R1 {r['fixed_window_retrieval_r1']:.3f} top5 {r['fixed_window_retrieval_top5']:.3f} MRR {r['fixed_window_retrieval_mrr']:.3f} | duration-only R1 {r['duration_only_retrieval_r1']:.3f} top5 {r['duration_only_retrieval_top5']:.3f}", flush=True)
    summary = dict(contract='group_decoding_v1', role=args.role, checkpoint=str(args.checkpoint), contents=len(contents), trials=len(data),
                   chance_r1=1 / len(contents), chance_top5=5 / len(contents), chance_mrr=chance_mrr(len(contents)), repeats=args.repeats, results=results)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / f'{args.role}.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(chance_r1=summary['chance_r1'], chance_top5=summary['chance_top5'], chance_mrr=summary['chance_mrr'])))


if __name__ == '__main__':
    main()

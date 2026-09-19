#!/usr/bin/env python3
"""S0(c): can EEG emit a *sequence of speech units* at all?  (alignment-free)

The regression route asked EEG to reproduce a 768-dimensional HuBERT sequence
frame by frame and produced a generic speech template.  This asks the smaller,
content-shaped question: predict the sequence of discrete units
(`scripts/build_speech_units.py`, k-means on train-fold HuBERT layer 9) with a
CTC head, which needs no frame alignment — exactly the property imagined or
loosely-timed speech would also need.

Two numbers decide whether an open-vocabulary route is worth building:

* **unit error rate** of the greedy decode against the sentence's own unit
  sequence, compared with the same model fed zero EEG and with a
  length-matched random sequence;
* **closed-set identification**: score every held-out sentence's unit sequence
  under the trial's CTC posteriors and rank them.  This is the "EEG decides
  what is said" decision the content-first pipeline would consume, and it is
  reported against zero-EEG and wrong-trial controls.

No audio is produced here.  Nothing in this file may be described as
reconstruction.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('ALIGNED_TARGET_CACHE_NAME', 'targets_adapted.h5')
import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src'))
import aligned_speech as legacy
import aligned_recovery as recovery
from aligned_recovery_model import TemporalResidual, augment_eeg
from eeg2speech.aligned import EEG_SAMPLES
from eeg2speech.losses import counterfactual_eeg

CONTRACT = 'unit_ctc_v1'
TOKEN_STRIDE = 4


class UnitCTC(nn.Module):
    """Recovery trunk + CTC head over discrete speech units (blank = index 0)."""
    def __init__(self, units: int, channels: int = 128, width: int = 128, subjects: int = 0, dropout: float = .1, positional: bool = True):
        super().__init__()
        self.channels, self.subjects, self.units = channels, int(subjects), units
        if self.subjects:
            self.subject_delta = nn.Parameter(torch.zeros(self.subjects, channels, channels))
        self.spatial = nn.Conv1d(channels, width, 1, bias=False)
        self.temporal = nn.Conv1d(width, width, 15, stride=TOKEN_STRIDE, padding=7)
        self.blocks = nn.Sequential(*[TemporalResidual(width, 2 ** i, dropout) for i in range(6)])
        self.output_norm = nn.LayerNorm(width)
        count = (EEG_SAMPLES + 2 * 7 - 15) // TOKEN_STRIDE + 1
        self.positional = positional
        if positional:
            self.position = nn.Parameter(torch.zeros(1, width, count))
        self.head = nn.Conv1d(width, units + 1, 1)

    def forward(self, eeg, subject=None):
        x = eeg
        if subject is not None and self.subjects:
            x = torch.bmm(torch.eye(self.channels, device=x.device, dtype=x.dtype) + self.subject_delta[subject], x)
        x = F.gelu(self.temporal(self.spatial(x)))
        if self.positional:
            x = x + self.position
        x = self.output_norm(self.blocks(x).transpose(1, 2)).transpose(1, 2)
        return self.head(x).transpose(1, 2)            # B, T, units+1


def load_units(path: Path):
    with h5py.File(path, 'r') as h5:
        clusters = int(h5.attrs['clusters'])
        sequences = {key: h5['units'][key]['collapsed'][:].astype(np.int64) + 1 for key in h5['units']}   # 0 is the CTC blank
    return clusters, sequences


def prior_only_gap(report: dict) -> float:
    """How much better the real-EEG decode is than the same model fed zero EEG.

    Negative means the head has settled on a prior-only solution: it emits the
    corpus' typical unit sequence and the EEG input only adds noise.  This is
    the failure mode of the 2026-09-16 run and is now a training-time signal,
    not something to be noticed afterwards in a log.
    """
    return report['zero']['unit_error_rate'] - report['correct']['unit_error_rate']


def edit_distance(a, b) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y))
        previous = current
    return previous[-1]


def greedy_decode(logits: torch.Tensor) -> list[int]:
    best = logits.argmax(-1).tolist()
    out = []
    for i, value in enumerate(best):
        if value != 0 and (i == 0 or value != best[i - 1]):
            out.append(value)
    return out


def ctc_loss(log_probabilities, targets, input_lengths, target_lengths):
    """CTC on the CPU: ``aten::_ctc_loss`` has no MPS kernel (torch 2.11).

    Only this operator moves; the forward and backward passes of the network
    stay on the accelerator, and gradients flow back through the transfer.
    """
    if log_probabilities.device.type == 'mps':
        loss = F.ctc_loss(log_probabilities.cpu(), targets.cpu(), input_lengths.cpu(), target_lengths.cpu(),
                          blank=0, reduction='none', zero_infinity=True)
        return loss.to(log_probabilities.device)
    return F.ctc_loss(log_probabilities, targets, input_lengths, target_lengths, blank=0, reduction='none', zero_infinity=True)


def sequence_scores(log_probabilities: torch.Tensor, candidates: list[np.ndarray]) -> np.ndarray:
    """Negative CTC loss of every candidate unit sequence under one trial's posteriors."""
    frames = log_probabilities.shape[0]
    targets = torch.cat([torch.from_numpy(c) for c in candidates]).to(log_probabilities.device)
    lengths = torch.tensor([len(c) for c in candidates], device=log_probabilities.device)
    inputs = log_probabilities[:, None, :].expand(-1, len(candidates), -1)
    loss = ctc_loss(inputs, targets, torch.full((len(candidates),), frames, dtype=torch.long, device=log_probabilities.device), lengths)
    return (-loss / lengths).detach().cpu().numpy()   # per-symbol log-likelihood


@torch.no_grad()
def evaluate(model, dataset, sequences, subjects, device, batch_size: int, controls=('zero', 'wrong_trial')) -> dict:
    from aligned_recovery_eval import matched_wrong_trial_indices
    model.eval()
    contents = sorted({str(c) for c in dataset.frame.content_group})
    key_of = dict(zip(dataset.frame.content_group.astype(str), dataset.frame.audio_key.astype(str)))
    candidates = [sequences[key_of[c]] for c in contents]
    wrong = matched_wrong_trial_indices(dataset.frame)
    results = {name: dict(errors=[], ranks=[]) for name in ('correct', *controls)}
    for offset in range(0, len(dataset), batch_size):
        ids = list(range(offset, min(len(dataset), offset + batch_size)))
        batch = legacy.move(torch.utils.data.default_collate([dataset[i] for i in ids]), device)
        swapped = legacy.move(torch.utils.data.default_collate([dataset[wrong[i]] for i in ids]), device)
        subject = torch.tensor([subjects[s] for s in batch['subject']], device=device) if subjects else None
        for name in results:
            if name == 'correct':
                eeg = batch['eeg']
            elif name == 'wrong_trial':
                eeg = swapped['eeg']
            else:
                eeg = counterfactual_eeg(batch['eeg'], name, time_mask=batch['time_mask'], channel_mask=batch['channel_mask'])
            logits = model(eeg * batch['channel_mask'][:, :, None], subject)
            log_probabilities = logits.log_softmax(-1)
            for i in range(len(ids)):
                target = sequences[str(batch['content'][i])] if str(batch['content'][i]) in sequences else sequences[key_of[str(batch['content'][i])]]
                decoded = greedy_decode(logits[i])
                results[name]['errors'].append(edit_distance(decoded, target.tolist()) / len(target))
                scores = sequence_scores(log_probabilities[i], candidates)          # (T, C) for this trial
                rank = int((np.argsort(-scores) == contents.index(str(batch['content'][i]))).nonzero()[0][0]) + 1
                results[name]['ranks'].append(rank)
    report = {}
    for name, values in results.items():
        ranks = np.array(values['ranks'])
        report[name] = dict(unit_error_rate=float(np.mean(values['errors'])), r1=float((ranks == 1).mean()),
                            top5=float((ranks <= 5).mean()), mrr=float((1 / ranks).mean()))
    report['candidates'] = len(contents); report['chance_r1'] = 1 / len(contents); report['chance_top5'] = min(1., 5 / len(contents))
    report['pairs'] = len(results['correct']['ranks'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/aligned_speech_local_v1.yaml'))
    parser.add_argument('--units', default=str(ROOT / 'artifacts/aligned_speech_local_v1/speech_units.h5'))
    parser.add_argument('--output', default=str(ROOT / 'outputs/unit_ctc/v1'))
    parser.add_argument('--updates', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--weight-decay', type=float, default=.05)
    parser.add_argument('--seed', type=int, default=31)
    parser.add_argument('--no-subject-layer', dest='subject_layer', action='store_false')
    parser.add_argument('--no-augment', dest='augment', action='store_false')
    parser.add_argument('--contrast-weight', type=float, default=1., help='weight of the beat-your-own-prior penalty (0 disables it)')
    parser.add_argument('--contrast-margin', type=float, default=.05, help='per-symbol margin the real-EEG decode must win by')
    parser.add_argument('--patience', type=int, default=4, help='evaluations with a non-positive EEG gap before stopping')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    legacy.seed_all(args.seed); torch.set_num_threads(4)
    device = legacy.device(args.device)
    cfg = legacy.config(args.config)
    train = legacy.dataset_for(cfg, 'train'); validation = legacy.dataset_for(cfg, 'validation')
    clusters, sequences = load_units(Path(args.units))
    key_of = dict(zip(train.frame.content_group.astype(str), train.frame.audio_key.astype(str)))
    key_of.update(dict(zip(validation.frame.content_group.astype(str), validation.frame.audio_key.astype(str))))
    subjects = {s: i for i, s in enumerate(sorted(set(train.frame.subject)))} if args.subject_layer else None
    model = UnitCTC(clusters, channels=train[0]['eeg'].shape[0], subjects=len(subjects) if subjects else 0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    history = []; best_gap = -float('inf'); started = time.monotonic(); collected = []
    for step in range(1, args.updates + 1):
        lr = args.lr * (min(1., step / args.warmup)) * (.1 + .9 * .5 * (1 + math.cos(math.pi * min(1., step / args.updates))))
        for group in optimizer.param_groups:
            group['lr'] = lr
        ids = rng.choice(len(train), size=args.batch_size, replace=False).tolist()
        batch = legacy.move(torch.utils.data.default_collate([train[i] for i in ids]), device)
        subject = torch.tensor([subjects[s] for s in batch['subject']], device=device) if subjects else None
        eeg = batch['eeg'] * batch['channel_mask'][:, :, None]
        if args.augment:
            eeg = augment_eeg(eeg, batch['channel_mask'])
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits = model(eeg, subject)
        targets = [torch.from_numpy(sequences[key_of[str(c)]]) for c in batch['content']]
        lengths = torch.tensor([len(t) for t in targets], device=device)
        frames = torch.full((len(ids),), logits.shape[1], dtype=torch.long, device=device)
        per_sequence = ctc_loss(logits.log_softmax(-1).transpose(0, 1), torch.cat(targets).to(device), frames, lengths)
        loss = (per_sequence / lengths).mean()            # per symbol, so long sentences do not dominate
        parts = dict(ctc=float(loss))
        if args.contrast_weight > 0:
            # The head must beat its own prior: the same targets scored under a
            # zero-EEG forward pass are treated as negatives.  Without this the
            # CTC optimum is simply the corpus' typical unit sequence.
            prior = model(torch.zeros_like(eeg), subject)
            prior_loss = ctc_loss(prior.log_softmax(-1).transpose(0, 1), torch.cat(targets).to(device), frames, lengths) / lengths
            margin = F.relu(args.contrast_margin + loss - prior_loss.mean())
            loss = loss + args.contrast_weight * margin
            parts.update(prior_ctc=float(prior_loss.mean()), margin=float(margin))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
        collected.append(parts)
        if step % 50 == 0:
            print(json.dumps(dict(update=step, lr=round(lr, 6), seconds=round(time.monotonic() - started, 1),
                                  **{k: round(float(np.mean([c[k] for c in collected])), 4) for k in collected[0]})), flush=True)
            collected = []
        if step % args.eval_every == 0 or step == args.updates:
            report = evaluate(model, validation, sequences, subjects, device, args.batch_size)
            report['update'] = step; report['prior_only_gap'] = prior_only_gap(report); history.append(report)
            print(json.dumps(dict(update=step, uer=round(report['correct']['unit_error_rate'], 4),
                                  uer_zero=round(report['zero']['unit_error_rate'], 4),
                                  gap=round(report['prior_only_gap'], 4),
                                  r1=round(report['correct']['r1'], 4), top5=round(report['correct']['top5'], 4),
                                  mrr=round(report['correct']['mrr'], 4), r1_zero=round(report['zero']['r1'], 4),
                                  r1_wrong=round(report['wrong_trial']['r1'], 4), chance_r1=round(report['chance_r1'], 4))), flush=True)
            (output / 'metrics.json').write_text(json.dumps(dict(contract=CONTRACT, clusters=clusters, history=history), indent=2) + '\n')
            # Selection is on the EEG-specific gap, never on the raw error rate:
            # a prior-only head can reach a low error rate while using no EEG.
            if report['prior_only_gap'] > best_gap:
                best_gap = report['prior_only_gap']
                torch.save(dict(contract=CONTRACT, model=model.state_dict(), clusters=clusters, update=step,
                                subjects=sorted(subjects) if subjects else [], report=report), output / 'best.pt')
            recent = [h['prior_only_gap'] for h in history[-args.patience:]]
            if len(recent) >= args.patience and max(recent) <= 0:
                print(json.dumps(dict(stopped='prior_only', update=step, gaps=[round(g, 4) for g in recent],
                                      reason='zero-EEG decodes at least as well as real EEG at every recent evaluation')), flush=True)
                break
    print(json.dumps(dict(best_prior_only_gap=best_gap, checkpoint=str(output / 'best.pt'),
                          verdict=('eeg_specific' if best_gap > 0 else 'prior_only'))))


if __name__ == '__main__':
    main()

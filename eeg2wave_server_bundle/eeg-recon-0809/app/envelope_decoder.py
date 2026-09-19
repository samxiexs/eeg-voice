#!/usr/bin/env python3
"""Train EEG -> speech envelope directly on DS004940, against the linear gate.

Every DS004940 model so far regressed the 768-dimensional HuBERT sequence or
the mel and converged on a generic speech template.  The linear gate says what
is actually recoverable is the **envelope**: a ridge model reaches r = 0.476
against a time-only prior of 0.469, i.e. an EEG-explained residual of 0.15
(17/17 participants above a circular-shift null).  No deep model has ever been
trained for that target here, so the obvious question was never asked: can a
network beat 0.15?

The prior is handed to the model explicitly (the train-fold median envelope
profile, resampled to the trial's speech region), so the network only has to
predict what the prior cannot — and the headline number is the **residual
correlation**, computed after removing the prior from both prediction and
target.  Raw correlation is reported too, but it is dominated by the prior and
must not be quoted alone.

Variants worth separating are exposed as flags: band-split input
(delta/theta/alpha/beta as extra channels), per-subject spatial filters instead
of a shared one, same-sentence trial mixing, and initialisation from the
Broderick envelope trunk.
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
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src'))
import aligned_recovery as recovery
from aligned_recovery_model import TemporalResidual, augment_eeg, average_eeg, same_content_partners
from eeg2speech.aligned import EEG_SAMPLES
from eeg2speech.losses import counterfactual_eeg

legacy = recovery.legacy
CONTRACT = 'envelope_decoder_v1'
TOKEN_STRIDE = 4
RATE = 256
BANDS = ((.5, 4.), (4., 8.), (8., 13.), (13., 45.))


def band_split(eeg: torch.Tensor, bands=BANDS, rate: int = RATE) -> torch.Tensor:
    """(B, C, T) -> (B, C*len(bands), T); delta/theta/alpha/beta as separate channels."""
    spectrum = torch.fft.rfft(eeg, dim=-1)
    freqs = torch.fft.rfftfreq(eeg.shape[-1], 1 / rate).to(eeg.device)
    parts = []
    for low, high in bands:
        mask = ((freqs >= low) & (freqs < high)).to(spectrum.dtype)
        parts.append(torch.fft.irfft(spectrum * mask, n=eeg.shape[-1], dim=-1))
    return torch.cat(parts, dim=1)


class EnvelopeDecoder(nn.Module):
    def __init__(self, channels=128, width=128, subjects=0, dropout=.1, bands=1, per_subject_spatial=False, positional=True):
        super().__init__()
        self.channels, self.subjects, self.bands = channels, int(subjects), bands
        inputs = channels * bands
        self.per_subject_spatial = per_subject_spatial and bool(subjects)
        if self.per_subject_spatial:
            # A full spatial filter per participant; the protocol is known-subject.
            self.spatial_weight = nn.Parameter(torch.randn(self.subjects, width, inputs) * (inputs ** -.5))
        else:
            self.spatial = nn.Conv1d(inputs, width, 1, bias=False)
            if self.subjects:
                self.subject_delta = nn.Parameter(torch.zeros(self.subjects, inputs, inputs))
        self.temporal = nn.Conv1d(width, width, 15, stride=TOKEN_STRIDE, padding=7)
        self.blocks = nn.Sequential(*[TemporalResidual(width, 2 ** i, dropout) for i in range(6)])
        self.output_norm = nn.LayerNorm(width)
        count = (EEG_SAMPLES + 2 * 7 - 15) // TOKEN_STRIDE + 1
        self.positional = positional
        if positional:
            self.position = nn.Parameter(torch.zeros(1, width, count))
        self.prior_gain = nn.Parameter(torch.ones(1))
        self.head = nn.Conv1d(width, 1, 1)
        nn.init.zeros_(self.head.bias); nn.init.normal_(self.head.weight, std=.01)

    def forward(self, eeg, subject=None, prior=None):
        x = band_split(eeg) if self.bands > 1 else eeg
        if self.per_subject_spatial:
            x = torch.bmm(self.spatial_weight[subject], x)
        else:
            if subject is not None and self.subjects:
                x = torch.bmm(torch.eye(x.shape[1], device=x.device, dtype=x.dtype) + self.subject_delta[subject], x)
            x = self.spatial(x)
        x = F.gelu(self.temporal(x))
        if self.positional:
            x = x + self.position
        x = self.output_norm(self.blocks(x).transpose(1, 2)).transpose(1, 2)
        residual = self.head(x).squeeze(1)
        return residual + self.prior_gain * prior if prior is not None else residual


def envelope_targets(dataset, token_times: np.ndarray):
    """Speech envelope (mean log-mel over bins) sampled at the token times, plus a speech mask."""
    import h5py
    with h5py.File(dataset.cache, 'r') as h5:
        mel_times = h5['mel_times'][:]
        out = {}
        for key in sorted(set(dataset.frame.audio_key)):
            mel = h5['targets'][key]['mel'][:]
            frames = int(h5['targets'][key].attrs['source_samples_16k']) // 256 + 1
            profile = mel[:, :frames].mean(0)
            values = np.interp(token_times, mel_times[:frames], profile, left=profile[0], right=profile[-1])
            mask = (token_times >= 0) & (token_times < mel_times[frames - 1])
            out[key] = (values.astype(np.float32), mask)
    return out


def standardize(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.float()
    mean = (values * weight).sum(1, keepdim=True) / weight.sum(1, keepdim=True)
    centred = (values - mean) * weight
    scale = centred.pow(2).sum(1, keepdim=True).div(weight.sum(1, keepdim=True)).sqrt() + 1e-6
    return centred / scale


def masked_correlation(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    a = standardize(prediction, mask); b = standardize(target, mask)
    weight = mask.float()
    return (a * b * weight).sum(1) / weight.sum(1)


def residualize(values: torch.Tensor, prior: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Standardised values with the prior regressed out, inside the mask.

    The same quantity the partial correlation is built from, kept as a vector so
    a batch of them can be compared against each other in one matrix product.
    """
    weight = mask.float()
    a = standardize(values, mask); p = standardize(prior, mask)
    beta = (a * p * weight).sum(1, keepdim=True) / weight.sum(1, keepdim=True)
    return standardize(a - beta * p, mask)


def contrastive_loss(prediction, target, prior, window, same, temperature: float = .07) -> torch.Tensor:
    """InfoNCE over the trials in the batch: predict *this* sentence, not speech.

    Correlation with the true envelope is a regression objective -- it is
    maximised by a prediction that is right about speech in general.  What the
    pooled sentence decision actually needs is a prediction that matches its own
    sentence *better than it matches the other candidates*, which is only
    trained by a discriminative term.  Trials sharing a sentence (17
    participants heard each one) are removed from the negatives.
    """
    a = residualize(prediction, prior, window); b = residualize(target, prior, window)
    weight = window.float()
    logits = (a * weight) @ (b * weight).T / weight.sum(1).clamp_min(1.)[:, None] / temperature
    logits = logits.masked_fill(same & ~torch.eye(len(a), dtype=torch.bool, device=a.device), -1e4)
    labels = torch.arange(len(a), device=a.device)
    return .5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def partial_correlation(prediction, target, prior, mask) -> torch.Tensor:
    """Correlation of prediction and target with the prior regressed out of both.

    Subtracting a common signal from both sides instead (z(pred) - z(prior) vs
    z(target) - z(prior)) leaves the shared -z(prior) term in each and reports
    a large correlation for a prediction that carries nothing: with an
    uninformative prediction that construction gives ~0.27 where the true
    correlation is ~0.01.  The partial correlation has no such term.
    """
    xy = masked_correlation(prediction, target, mask)
    xp = masked_correlation(prediction, prior, mask)
    yp = masked_correlation(target, prior, mask)
    return (xy - xp * yp) / ((1 - xp ** 2).clamp_min(1e-6) * (1 - yp ** 2).clamp_min(1e-6)).sqrt()


@torch.no_grad()
def evaluate(model, dataset, targets, prior, subjects, device, batch_size: int, candidates=None, window=None) -> dict:
    """Correlation metrics, plus single-trial retrieval among the role's sentences.

    Retrieval is scored exactly as `app/group_content.py` scores it -- partial
    correlation against every candidate over a fixed 1.5 s window, no oracle
    duration -- so a checkpoint can be selected for the decision the programme
    actually cares about rather than for correlation alone.
    """
    model.eval()
    conditions = ('correct', 'zero', 'wrong_trial')
    from aligned_recovery_eval import matched_wrong_trial_indices
    wrong = matched_wrong_trial_indices(dataset.frame)
    rows = {name: dict(raw=[], partial=[], rank=[]) for name in conditions}
    prior_raw = []
    keys_order = list(candidates[0]) if candidates is not None else []
    candidate_values = candidates[1] if candidates is not None else None
    for offset in range(0, len(dataset), batch_size):
        ids = list(range(offset, min(len(dataset), offset + batch_size)))
        batch = legacy.move(torch.utils.data.default_collate([dataset[i] for i in ids]), device)
        swapped = legacy.move(torch.utils.data.default_collate([dataset[wrong[i]] for i in ids]), device)
        keys = [dataset.frame.audio_key.iloc[i] for i in ids]
        target = torch.stack([torch.from_numpy(targets[k][0]) for k in keys]).to(device)
        mask = torch.stack([torch.from_numpy(targets[k][1]) for k in keys]).to(device)
        prior_batch = prior[None].expand(len(ids), -1).to(device)
        subject = torch.tensor([subjects[s] for s in batch['subject']], device=device) if subjects else None
        prior_raw.extend(masked_correlation(prior_batch, target, mask).cpu().tolist())
        for name in conditions:
            if name == 'correct':
                eeg = batch['eeg']
            elif name == 'wrong_trial':
                eeg = swapped['eeg']
            else:
                eeg = torch.zeros_like(batch['eeg'])
            prediction = model(eeg * batch['channel_mask'][:, :, None], subject, prior_batch)
            rows[name]['raw'].extend(masked_correlation(prediction, target, mask).cpu().tolist())
            rows[name]['partial'].extend(partial_correlation(prediction, target, prior_batch, mask).cpu().tolist())
            if candidate_values is not None:
                a = residualize(prediction, prior_batch, window.expand(len(ids), -1))
                b = residualize(candidate_values, prior[None].expand(len(candidate_values), -1).to(device),
                                window.expand(len(candidate_values), -1))
                weight = window.float()
                similarity = (a * weight) @ (b * weight).T / weight.sum().clamp_min(1.)
                truth = torch.tensor([keys_order.index(k) for k in keys], device=device)
                rank = (similarity.argsort(dim=1, descending=True) == truth[:, None]).float().argmax(dim=1) + 1
                rows[name]['rank'].extend(rank.cpu().tolist())
    report = {name: dict(raw=float(np.mean(v['raw'])), partial=float(np.mean(v['partial']))) for name, v in rows.items()}
    for name, v in rows.items():
        if v['rank']:
            rank = np.array(v['rank'])
            report[name] |= dict(retrieval_r1=float((rank == 1).mean()), retrieval_mrr=float((1 / rank).mean()))
    report['prior_raw'] = float(np.mean(prior_raw))
    # The EEG contribution is what the model adds over its own no-EEG output, on
    # the raw correlation; the partial correlation is reported for comparison
    # with the linear gate but is never the selection signal.
    # eeg_gain is measured against the model's *own* no-EEG output, so a model
    # that lets that output rot scores well without decoding anything better:
    # the band-split run reached +0.101 with its correct correlation up 0.015
    # and its zero output down 0.068.  zero_degradation makes that visible and
    # gain_over_prior, which is anchored to a fixed reference, is what selects.
    report['eeg_gain'] = report['correct']['raw'] - report['zero']['raw']
    report['zero_degradation'] = report['prior_raw'] - report['zero']['raw']
    report['eeg_partial_gain'] = report['correct']['partial'] - report['zero']['partial']
    report['gain_over_prior'] = report['correct']['raw'] - report['prior_raw']
    if 'retrieval_mrr' in report['correct']:
        report['retrieval_gain'] = report['correct']['retrieval_mrr'] - report['wrong_trial']['retrieval_mrr']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/aligned_speech_local_v1.yaml'))
    parser.add_argument('--output', default=str(ROOT / 'outputs/envelope_decoder/base'))
    parser.add_argument('--updates', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--weight-decay', type=float, default=.05)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--seed', type=int, default=31)
    parser.add_argument('--band-split', action='store_true', help='feed delta/theta/alpha/beta as separate channels')
    parser.add_argument('--per-subject-spatial', action='store_true', help='a full spatial filter per participant')
    parser.add_argument('--mix-same-content', type=float, default=0.)
    parser.add_argument('--contrastive', type=float, default=0.,
                        help='weight of the InfoNCE term that makes the prediction pick its own sentence')
    parser.add_argument('--select-on', choices=['gain_over_prior', 'eeg_gain', 'retrieval_mrr'],
                        default='gain_over_prior',
                        help='validation quantity the best checkpoint is kept on; eeg_gain is gameable '
                             '(see zero_degradation) and is kept only to reproduce earlier runs')
    parser.add_argument('--initialize-trunk', help='trunk checkpoint from app/broderick_pretrain.py')
    parser.add_argument('--no-augment', dest='augment', action='store_false')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    legacy.seed_all(args.seed); torch.set_num_threads(4)
    device = legacy.device(args.device); cfg = legacy.config(args.config)
    train = legacy.dataset_for(cfg, 'train'); validation = legacy.dataset_for(cfg, 'validation')
    count = (EEG_SAMPLES + 2 * 7 - 15) // TOKEN_STRIDE + 1
    token_times = -.25 + np.arange(count) * TOKEN_STRIDE / RATE
    train_targets = envelope_targets(train, token_times); validation_targets = envelope_targets(validation, token_times)
    stack = np.stack([v for v, _ in train_targets.values()]); masks = np.stack([m for _, m in train_targets.values()])
    prior = torch.from_numpy(np.where(masks.sum(0) > 0, np.nansum(stack * masks, 0) / np.maximum(masks.sum(0), 1), stack.mean(0)).astype(np.float32))
    subjects = {s: i for i, s in enumerate(sorted(set(train.frame.subject)))}
    window = torch.from_numpy((token_times >= 0) & (token_times < 1.5)).to(device)[None]
    validation_keys = sorted(validation_targets)
    validation_candidates = (validation_keys,
                             torch.stack([torch.from_numpy(validation_targets[k][0]) for k in validation_keys]).to(device))
    model = EnvelopeDecoder(channels=train[0]['eeg'].shape[0], width=args.width, subjects=len(subjects),
                            bands=len(BANDS) if args.band_split else 1,
                            per_subject_spatial=args.per_subject_spatial).to(device)
    if args.initialize_trunk:
        trunk = torch.load(Path(args.initialize_trunk), map_location='cpu', weights_only=False)['trunk']
        missing, unexpected = model.load_state_dict(trunk, strict=False)
        transferred = [k for k in trunk if k not in missing]
        print(json.dumps(dict(trunk=args.initialize_trunk, transferred=len(transferred), skipped=len(unexpected))), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    partners = same_content_partners(train.frame, args.mix_same_content, rng) if args.mix_same_content else {}
    history = []; best = -float('inf'); started = time.monotonic(); collected = []
    for step in range(1, args.updates + 1):
        lr = args.lr * min(1., step / args.warmup) * (.1 + .9 * .5 * (1 + math.cos(math.pi * min(1., step / args.updates))))
        for group in optimizer.param_groups:
            group['lr'] = lr
        ids = rng.choice(len(train), size=args.batch_size, replace=False).tolist()
        batch = legacy.move(torch.utils.data.default_collate([train[i] for i in ids]), device)
        eeg = batch['eeg'] * batch['channel_mask'][:, :, None]
        if partners:
            chosen = [i for i in ids if i in partners]
            if chosen:
                mixed = legacy.move(torch.utils.data.default_collate([train[partners.get(i, i)] for i in ids]), device)
                take = torch.tensor([i in partners for i in ids], device=device)[:, None, None]
                eeg = torch.where(take, average_eeg(eeg, mixed['eeg'], batch['channel_mask'], mixed['channel_mask']), eeg)
        if args.augment:
            eeg = augment_eeg(eeg, batch['channel_mask'])
        keys = [train.frame.audio_key.iloc[i] for i in ids]
        target = torch.stack([torch.from_numpy(train_targets[k][0]) for k in keys]).to(device)
        mask = torch.stack([torch.from_numpy(train_targets[k][1]) for k in keys]).to(device)
        prior_batch = prior[None].expand(len(ids), -1).to(device)
        subject = torch.tensor([subjects[s] for s in batch['subject']], device=device)
        model.train(); optimizer.zero_grad(set_to_none=True)
        prediction = model(eeg, subject, prior_batch)
        # Raw correlation with the target: the prior enters through the model's
        # own learned gain, so the head is only rewarded for what the prior
        # cannot explain, without the shared-term inflation of a hand-made
        # residual.  The partial correlation is added with a small weight to
        # push specifically on the EEG-explained part.
        loss = (1 - masked_correlation(prediction, target, mask)).mean() \
             + .3 * (1 - partial_correlation(prediction, target, prior_batch, mask)).mean()
        if args.contrastive:
            same = torch.tensor([[a == b for b in keys] for a in keys], device=device)
            loss = loss + args.contrastive * contrastive_loss(
                prediction, target, prior_batch, window.expand(len(ids), -1), same)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
        collected.append(float(loss))
        if step % 50 == 0:
            print(json.dumps(dict(update=step, loss=round(float(np.mean(collected)), 4), lr=round(lr, 6),
                                  seconds=round(time.monotonic() - started, 1))), flush=True); collected = []
        if step % args.eval_every == 0 or step == args.updates:
            report = evaluate(model, validation, validation_targets, prior, subjects, device, args.batch_size,
                              candidates=validation_candidates, window=window)
            report['update'] = step; history.append(report)
            print(json.dumps(dict(update=step, raw=round(report['correct']['raw'], 4),
                                  raw_zero=round(report['zero']['raw'], 4), raw_wrong=round(report['wrong_trial']['raw'], 4),
                                  prior=round(report['prior_raw'], 4), gain=round(report['eeg_gain'], 4),
                                  over_prior=round(report['gain_over_prior'], 4),
                                  partial=round(report['correct']['partial'], 4),
                                  partial_zero=round(report['zero']['partial'], 4),
                                  r1=round(report['correct'].get('retrieval_r1', float('nan')), 4),
                                  mrr=round(report['correct'].get('retrieval_mrr', float('nan')), 4),
                                  mrr_wrong=round(report['wrong_trial'].get('retrieval_mrr', float('nan')), 4),
                                  linear_gate=dict(raw=0.476, prior=0.469, partial=0.143))), flush=True)
            (output / 'metrics.json').write_text(json.dumps(dict(contract=CONTRACT, history=history), indent=2) + '\n')
            selection = (report['retrieval_gain'] if args.select_on == 'retrieval_mrr'
                         else report[args.select_on])
            if selection > best:
                best = selection
                torch.save(dict(contract=CONTRACT, model=model.state_dict(), update=step, report=report,
                                subjects=sorted(subjects), args=vars(args)), output / 'best.pt')
    print(json.dumps(dict(selected_on=args.select_on, best_selection=best, checkpoint=str(output / 'best.pt'),
                          best_eeg_gain=max(r['eeg_gain'] for r in history),
                          best_gain_over_prior=max(r['gain_over_prior'] for r in history),
                          worst_zero_degradation=max(r['zero_degradation'] for r in history),
                          best_retrieval_mrr=max((r['correct'].get('retrieval_mrr', 0.) for r in history), default=0.),
                          verdict=('eeg_specific' if best > 0 else 'prior_only'))))


if __name__ == '__main__':
    main()

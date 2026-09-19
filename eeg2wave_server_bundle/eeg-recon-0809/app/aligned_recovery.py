#!/usr/bin/env python3
"""Isolated recovery experiment (v3). No legacy checkpoint is modified.

All gate metrics are speech-frame-masked (see aligned_recovery_eval.py); the
template baseline is the median train mel; there is no variance penalty and
no 320-way bank classifier.  Full training starts from a fresh initialization
and only requires that a v3 M0 closure run has passed.
"""
from __future__ import annotations

import argparse
import copy
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

os.environ.setdefault('ALIGNED_TARGET_CACHE_NAME', 'targets_adapted.h5')
import numpy as np
import torch

import aligned_speech as legacy
from aligned_recovery_model import (RecoveryEEGModel, augment_eeg, average_eeg, diverse_batches, duration_fraction,
                                    recovery_loss, same_content_partners, speech_frame_masks)
from aligned_recovery_eval import evaluate_recovery, passes_validation_controls, train_templates

ROOT = legacy.ROOT
CONTRACT = 'aligned_recovery_v3'


def runtime_hash():
    files = [Path(__file__), Path(__file__).with_name('aligned_recovery_model.py'),
             Path(__file__).with_name('aligned_recovery_eval.py')]
    return hashlib.sha256(json.dumps([legacy.runtime_hash(), *[legacy.sha256(p) for p in files]]).encode()).hexdigest()


def subset(dataset, indices):
    result = copy.copy(dataset)
    result.frame = dataset.frame.iloc[indices].reset_index(drop=True)
    return result


def training_probe(dataset, per_subject=4):
    # Deterministic train-only monitoring cohort; never selects on test data.
    indices = []
    for _, rows in dataset.frame.groupby('subject'):
        distinct = rows.drop_duplicates('content_group')
        indices.extend(distinct.sample(n=min(per_subject, len(distinct)), random_state=322).index.tolist())
    return subset(dataset, sorted(indices))


def subject_lookup(dataset):
    return {s: i for i, s in enumerate(sorted(set(dataset.frame.subject)))}


def forward(model, batch, subject):
    return model(batch['eeg'], batch['channel_xyz'], batch['channel_mask'], batch['time_mask'], subject)


def compatible_runtimes():
    """Earlier runtime hashes whose checkpoints the current code may evaluate/resume.

    Registered in app/recovery_compatibility.json, keyed by the current hash,
    only for reviewed changes that leave the training forward/loss unchanged
    (evaluation code, control selection, reporting).
    """
    registry = Path(__file__).with_name('recovery_compatibility.json')
    table = json.loads(registry.read_text()) if registry.exists() else {}
    return [runtime_hash(), *table.get(runtime_hash(), {}).get('compatible_previous_runtimes', [])]


def load_checkpoint(path):
    value = torch.load(path, map_location='cpu', weights_only=False)
    if value.get('contract') != CONTRACT or value.get('runtime_hash') not in compatible_runtimes():
        raise ValueError('recovery checkpoint/code mismatch')
    return value


def dataset_signature(cfg, data):
    _, manifest, cache, norm = legacy.artifact_paths(cfg)
    return dict(manifest=legacy.sha256(manifest), cache=legacy.sha256(cache),
                normalizer=legacy.sha256(norm), teacher=data.teacher_sha256,
                trials=hashlib.sha256('\n'.join(data.frame.trial_id).encode()).hexdigest())


def metrics(model, cohort, train, target, batch, subjects, templates=None):
    return evaluate_recovery(model, cohort, train, target, batch, subjects, bootstrap=False, templates=templates)


def learning_rate(step, peak, updates, warmup):
    """Linear warmup then cosine decay to a tenth of the peak; a pure function of the step."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1., (step - warmup) / max(1, updates - warmup))
    return peak * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))


def train(args, cfg):
    legacy.seed_all(args.seed)
    target = legacy.device(args.device)
    full_train = legacy.dataset_for(cfg, 'train')
    data = legacy.dataset_for(cfg, 'train', m0=True) if args.mode == 'm0' else full_train
    validation = data if args.mode == 'm0' else legacy.dataset_for(cfg, 'validation')
    if args.mode == 'pilot':
        validation = training_probe(validation)
    probe = data if args.mode == 'm0' else training_probe(data)
    source = legacy.load_payload(Path(args.decoder))
    legacy.check_eeg_artifacts(source, cfg)
    if source['stage'] != 'adapt' or source['teacher_sha256'] != data.teacher_sha256:
        raise ValueError('matching train-fold adapted decoder required')
    decoder = legacy.decoder_from(source)
    # Subject indices come from the complete train fold so M0 and full runs share one table.
    subjects = subject_lookup(full_train)
    contents = {c: i for i, c in enumerate(sorted(set(full_train.frame.content_group)))}
    spec = dict(channels=data[0]['eeg'].shape[0], width=args.width, lag_ms=args.lag_ms,
                subjects=len(subjects) if args.subject_layer else 0, dropout=args.dropout,
                positional=bool(args.positional))
    model = RecoveryEEGModel(decoder, **spec).to(target)
    subject_index = subjects if args.subject_layer else None
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    weights = dict(contrastive_weight=args.contrastive_weight, sequence_weight=args.sequence_weight,
                   delta_weight=args.delta_weight, duration_weight=args.duration_weight, temperature=args.temperature)
    signature = dict(**dataset_signature(cfg, data), seed=args.seed, mode=args.mode,
                     batch_size=args.batch_size, updates=args.updates, lr=args.lr, warmup=args.warmup,
                     spec=spec, subjects=sorted(subjects), decoder=legacy.sha256(Path(args.decoder)),
                     eval_every=args.eval_every, weights=weights, augment=bool(args.augment),
                     mix_same_content=float(args.mix_same_content),
                     mel_warmup=args.mel_warmup, initialize=legacy.sha256(Path(args.initialize)) if args.initialize else None,
                     trunk=legacy.sha256(Path(args.initialize_trunk)) if args.initialize_trunk else None)
    if args.mode == 'full':
        if not args.m0_checkpoint:
            raise ValueError('full recovery training requires --m0-checkpoint from this architecture')
        gate = load_checkpoint(Path(args.m0_checkpoint))
        if (gate['signature']['mode'] != 'm0' or not gate['evaluation']['m0_passed']
                or gate['signature']['decoder'] != signature['decoder']
                or gate['signature']['manifest'] != signature['manifest']
                or gate['signature']['cache'] != signature['cache']
                or gate['signature']['spec']['width'] != args.width):
            raise ValueError('M0 gate/model/data mismatch')
        signature['m0_checkpoint'] = legacy.sha256(Path(args.m0_checkpoint))
    if args.initialize:
        initial = load_checkpoint(Path(args.initialize))
        for key in ('manifest', 'cache', 'normalizer', 'teacher', 'decoder', 'spec'):
            if initial['signature'][key] != signature[key]:
                raise ValueError('initialization mismatch: ' + key)
        if Path(args.initialize).resolve().parent == output.resolve():
            raise ValueError('use a separate output directory when initializing weights')
        model.load_state_dict(initial['model'])
    if args.initialize_trunk:
        # Trunk pretrained elsewhere (e.g. Broderick 2018 envelope tracking); head, decoder,
        # positional code, duration head and subject deltas start fresh.
        trunk = torch.load(Path(args.initialize_trunk), map_location='cpu', weights_only=False)['trunk']
        missing, unexpected = model.load_state_dict(trunk, strict=False)
        if unexpected or any(k.split('.')[0] in ('spatial', 'temporal', 'blocks', 'output_norm') for k in missing):
            raise ValueError('pretrained trunk does not match the encoder architecture')
        print(f'initialised {len(trunk)} trunk tensors from {args.initialize_trunk}', flush=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    step = epoch = next_batch = 0; best = best_passing = float('inf'); history = []
    progress = output / 'training_state.pt'
    if progress.exists():
        saved = load_checkpoint(progress)
        if signature != saved['signature']:
            raise ValueError('resume arguments/data changed; use a new output directory')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        for item in optimizer.state.values():
            for key, value in item.items():
                if torch.is_tensor(value):
                    item[key] = value.to(target)
        step, epoch, next_batch = saved['step'], saved['epoch'], saved['next_batch']
        best, best_passing, history = saved['best'], saved['best_passing'], saved['history']
        torch.set_rng_state(saved['torch_rng']); np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])
        if target.type == 'mps' and saved.get('mps_rng') is not None:
            torch.mps.set_rng_state(saved['mps_rng'])
        if target.type == 'cuda' and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        if saved['complete']:
            if saved.get('stop_reason') == 'collapse':
                raise RuntimeError('this recovery run stopped for collapse; inspect metrics')
            print('recovery training already complete'); return
        print(f'resuming recovery update {step}', flush=True)
    fetch = lru_cache(maxsize=256)(data.__getitem__)
    templates = train_templates(full_train)
    mel_frames = len(model.decoder.mel_times)
    stopping = [False]
    old_handler = signal.getsignal(signal.SIGINT)
    def interrupt(signum, frame):
        stopping[0] = True
        print('Finishing this update and saving recovery progress.', flush=True)
    signal.signal(signal.SIGINT, interrupt)
    def save(path, evaluation=None, complete=False, stop_reason=None):
        legacy.atomic_save(path, dict(contract=CONTRACT, runtime_hash=runtime_hash(), signature=signature,
            model=model.state_dict(), optimizer=optimizer.state_dict(), decoder_spec=source['decoder_spec'],
            decoder=model.decoder.state_dict(), step=step, epoch=epoch, next_batch=next_batch,
            best=best, best_passing=best_passing, history=history, evaluation=evaluation,
            complete=complete, stop_reason=stop_reason, torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
            python_rng=random.getstate(), mps_rng=torch.mps.get_rng_state() if target.type == 'mps' else None,
            cuda_rng=torch.cuda.get_rng_state_all() if target.type == 'cuda' else None))
    start = time.monotonic(); collected = []
    try:
        while step < args.updates:
            batches = diverse_batches(data.frame, args.batch_size, args.seed + epoch)
            partners = same_content_partners(data.frame, args.mix_same_content, np.random.default_rng(args.seed * 7919 + epoch))
            while next_batch < len(batches) and step < args.updates:
                ids = batches[next_batch]
                batch = legacy.move(torch.utils.data.default_collate([fetch(i) for i in ids]), target)
                if any(i in partners for i in ids):
                    # Same-sentence trial averaging: the mixed EEG keeps the
                    # anchor trial's subject index and every other field.
                    mixed = legacy.move(torch.utils.data.default_collate([fetch(partners.get(i, i)) for i in ids]), target)
                    chosen = torch.tensor([i in partners for i in ids], device=target)[:, None, None]
                    eeg = average_eeg(batch['eeg'], mixed['eeg'], batch['channel_mask'], mixed['channel_mask'])
                    batch = dict(batch, eeg=torch.where(chosen, eeg, batch['eeg']))
                subject = (torch.tensor([subjects[s] for s in batch['subject']], device=target)
                           if subject_index is not None else None)
                mel_mask, speech_mask = speech_frame_masks(batch['oracle_duration_frames'], model.decoder.mel_times,
                                                           model.decoder.speech_times)
                model.train(); model.decoder.eval(); optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate(step, args.lr, args.updates, args.warmup)
                if args.augment:
                    batch = dict(batch, eeg=augment_eeg(batch['eeg'], batch['channel_mask']))
                state = forward(model, batch, subject)
                indices = torch.tensor([contents[c] for c in batch['content']], device=target)
                loss, parts = recovery_loss(state, batch['teacher'], batch['mel'], model.decoder.normalizer,
                                            speech_mask, mel_mask, duration_fraction(batch['oracle_duration_frames'], mel_frames),
                                            indices, mel_weight=min(1., (step + 1) / max(1, args.mel_warmup)), **weights)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('nonfinite recovery loss')
                loss.backward()
                parts['grad_norm'] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True))
                parts['lr'] = optimizer.param_groups[0]['lr']
                optimizer.step(); step += 1; next_batch += 1; collected.append(parts)
                if step % 10 == 0:
                    print(json.dumps(dict(update=step, epoch=epoch + 1, seconds=round(time.monotonic() - start, 2),
                                          **{k: float(np.mean([v[k] for v in collected])) for k in parts})), flush=True)
                    collected = []
                if stopping[0]:
                    save(progress); return
                if step % args.eval_every == 0 or step == args.updates:
                    training = metrics(model, probe, full_train, target, args.batch_size, subject_index, templates)
                    report = training if args.mode == 'm0' else metrics(model, validation, full_train, target, args.batch_size, subject_index, templates)
                    row = dict(update=step, train_probe=training, validation=report,
                               evaluation_scope=args.mode, selection_role='train' if args.mode == 'm0' else 'validation')
                    history.append(row)
                    score = report['native_mel_mae']
                    if score < best:
                        best = score; save(output / 'best_metric.pt', report)
                    eligible = report['m0_passed'] if args.mode == 'm0' else passes_validation_controls(report)
                    if eligible and score < best_passing:
                        best_passing = score; save(output / 'best_passed.pt', report)
                    legacy.atomic_json(output / 'metrics.json', dict(contract=CONTRACT, signature=signature, history=history))
                    print(json.dumps(row), flush=True)
                    summary = {k: round(report[k], 4) for k in ('native_mel_mae', 'template_mae', 'template_improvement',
                               'retrieval_r1', 'chance_r1', 'zero_gain', 'wrong_trial_gain', 'time_block_shuffle_gain',
                               'envelope_corr', 'envelope_zero_gain', 'duration_corr', 'prediction_variance_ratio')}
                    print(json.dumps(dict(update=step, eligible=eligible, best=round(best, 4), **summary)), flush=True)
                    # Fail early on sustained collapse; do not consume the full budget silently.
                    collapsed = len(history) >= 5 and all(
                        h['train_probe']['prediction_variance_ratio'] < .001 and
                        h['train_probe']['retrieval_r1'] <= h['train_probe']['chance_r1'] + .02
                        for h in history[-3:])
                    save(progress, report, complete=(step >= args.updates or collapsed),
                         stop_reason='collapse' if collapsed else ('budget' if step >= args.updates else None))
                    if collapsed:
                        raise RuntimeError('recovery still collapsed at three evaluations; inspect metrics before another run')
                elif step % 10 == 0:
                    save(progress)
            if next_batch == len(batches):
                epoch += 1; next_batch = 0
        print(f'Finished. Review {output / "metrics.json"}; best_passed.pt exists only when objective gates pass.', flush=True)
    finally:
        signal.signal(signal.SIGINT, old_handler)
        if stopping[0]:
            raise SystemExit(130)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/aligned_speech_local_v1.yaml'))
    parser.add_argument('--mode', choices=['m0', 'pilot', 'full'], default='m0')
    parser.add_argument('--decoder', default=str(ROOT / 'outputs/aligned_speech_local_v1/adapt/best_checkpoint.pt'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--initialize'); parser.add_argument('--m0-checkpoint')
    parser.add_argument('--initialize-trunk', help='trunk checkpoint from app/broderick_pretrain.py')
    parser.add_argument('--seed', type=int, default=322)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--lag-ms', type=int, choices=[0, 100, 200, 300, 400], default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--weight-decay', type=float, default=.05)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--mel-warmup', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=.1)
    parser.add_argument('--contrastive-weight', type=float, default=1.)
    parser.add_argument('--sequence-weight', type=float, default=1.)
    parser.add_argument('--delta-weight', type=float, default=.2)
    parser.add_argument('--duration-weight', type=float, default=.5)
    parser.add_argument('--no-subject-layer', dest='subject_layer', action='store_false')
    parser.add_argument('--no-augment', dest='augment', action='store_false')
    parser.add_argument('--no-positional', dest='positional', action='store_false')
    parser.add_argument('--mix-same-content', type=float, default=0.,
                        help='probability of averaging a training trial with another trial of the same sentence')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if (args.batch_size < 2 or min(args.updates, args.eval_every, args.width, args.mel_warmup) < 1
            or not math.isfinite(args.lr) or args.lr <= 0 or args.temperature <= 0
            or not 0 <= args.mix_same_content <= 1):
        parser.error('require batch >= 2, positive sizes/updates, finite positive lr/temperature, mix probability in [0, 1]')
    torch.set_num_threads(4)
    train(args, legacy.config(args.config))


if __name__ == '__main__':
    main()

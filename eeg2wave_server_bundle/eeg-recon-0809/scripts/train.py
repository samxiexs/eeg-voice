"""Train and evaluate one stage of the plan (``configs/plan.yaml``).

    python scripts/train.py listen  --out outputs/listen
    python scripts/train.py imagery --fold 0 --out outputs/imagery/pretrained_f0 --init outputs/listen/model.pt
    python scripts/train.py imagery --fold 0 --out outputs/imagery/scratch_f0
    python scripts/train.py imagery --fold 0 --out outputs/imagery/imagined_only_f0 --only-target
    python scripts/train.py imagery --fold 0 --protocol within --out outputs/within/full_f0
    python scripts/train.py imagery --fold 0 --protocol within --set max_hz=45 --out outputs/within/le45_f0

Stage ``listen``: match-mismatch between EEG frames and speech features (1 matched + K
mismatched segments) plus InfoNCE between two people hearing the same stimulus window.
Stage ``imagery``: item cross-entropy on each dataset's head (non-target modalities at
``aux_weight``) plus supervised contrastive alignment across modalities.  Training runs a
fixed number of steps (no selection on held-out people); the held-out evaluation is
written to ``<out>/evaluation.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegspeech import ROOT                                                           # noqa: E402
from eegspeech.data import ListenSource, TrialSource, collate, subject_folds, within_fold   # noqa: E402
from eegspeech.evaluation import item_scores, listen_scores                          # noqa: E402
from eegspeech.losses import match_mismatch, pair_infonce, person_contrastive, supcon   # noqa: E402
from eegspeech.model import Model                                                    # noqa: E402
from eegspeech.store import open_store                                               # noqa: E402


def device_of(name):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def split(store, folds, fold):
    assignment = subject_folds(store.subjects, folds)
    train = [s for s in store.subjects if assignment[s] != fold]
    test = [s for s in store.subjects if assignment[s] == fold]
    return train, test


def schedule(step, total, warmup=300):
    if step < warmup:
        return (step + 1) / warmup
    return .5 * (1 + np.cos(np.pi * (step - warmup) / max(total - warmup, 1)))


def listen_step(model, source, cfg, device):
    windows, pairs = source.sample(cfg['batch'], cfg['partner_fraction'])
    b = collate(windows).to(device)
    frames, mask, _, _ = model.encoder(b.eeg, b.xyz, b.valid)
    n = cfg['batch']
    features = torch.as_tensor(np.stack([w['features'] for w in windows[:n]]), device=device)
    speech = model.speech(features.flatten(0, 1)).unflatten(0, features.shape[:2])
    loss, accuracy = match_mismatch(frames[:n], speech, mask[:n], cfg['temperature'])
    logs = dict(mm=float(loss.detach()), mm_acc=float(accuracy))
    if len(pairs) > 1:
        a, p = map(list, zip(*pairs))
        pair_loss, pair_acc = pair_infonce(frames[a], frames[p], mask[a], cfg['temperature'])
        loss = loss + cfg['pair_weight'] * pair_loss
        logs.update(pair=float(pair_loss.detach()), pair_acc=float(pair_acc))
    return loss, logs


def imagery_step(model, name, source, spec, cfg, device):
    windows = source.sample(cfg['batch'])
    person, person_loss = None, None
    if model.person is not None:
        subjects = sorted({w['subject'] for w in windows})
        sets = [source.windows_of(s, 2 * cfg['person_windows']) for s in subjects]
        flat = collate([w for s in sets for w in s]).to(device)
        vectors = model.person(flat.eeg, flat.xyz, flat.valid, [cfg['person_windows']] * (2 * len(subjects)))
        view_a, view_b = vectors[0::2], vectors[1::2]
        if len(subjects) > 1:
            person_loss = person_contrastive(view_a, view_b)
        lookup = {s: i for i, s in enumerate(subjects)}
        person = view_a[[lookup[w['subject']] for w in windows]]
    b = collate(windows, ('item', 'modality')).to(device)
    _, _, pooled, z = model.encoder(b.eeg, b.xyz, b.valid, b.lengths, person)
    item, modality = b.extra['item'].long(), b.extra['modality'].long()
    target = modality == source.store.modalities.index(spec['target'])
    weight = torch.where(target, 1., cfg['aux_weight'])
    logits = model.heads[name](pooled)
    ce = (F.cross_entropy(logits, item, reduction='none') * weight).sum() / weight.sum()
    contrast = supcon(z, item, cfg['temperature'])
    loss = ce + cfg['supcon_weight'] * contrast
    logs = dict(ce=float(ce.detach()), supcon=float(contrast.detach()))
    if target.any():
        logs['acc_target'] = float((logits.argmax(1) == item)[target].float().mean())
    if person_loss is not None:
        loss = loss + cfg['person_weight'] * person_loss
        logs["person"] = float(person_loss.detach())
    return loss, logs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['listen', 'imagery'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', default=str(ROOT / 'configs' / 'plan.yaml'))
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--protocol', choices=['cross', 'within'], default='cross',
                        help='imagery: cross = held-out people; within = held-out fifth of every person\'s trials '
                             '(contiguous in recording order; the person\'s other trials are in training)')
    parser.add_argument('--init', help='checkpoint whose encoder initialises this stage')
    parser.add_argument('--datasets', nargs='*', help='restrict to these datasets of the stage')
    parser.add_argument('--only-target', action='store_true', help='imagery: train on target-modality trials only')
    parser.add_argument('--steps', type=int, help='override the configured number of steps')
    parser.add_argument('--set', nargs='*', default=[], help='override stage keys, e.g. --set person_dim=32 align=false')
    parser.add_argument('--model-set', nargs='*', default=[], help='override model keys, e.g. --model-set branches=power')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    cfg = dict(config[args.stage])
    for item in args.set:
        key, value = item.split('=', 1)
        cfg[key] = yaml.safe_load(value)
    if args.steps:
        cfg['steps'] = args.steps
    for item in args.model_set:
        key, value = item.split('=', 1)
        config['model'][key] = yaml.safe_load(value)
    specs = {k: v for k, v in cfg['sources'].items() if not args.datasets or k in args.datasets}
    device = device_of(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fold = 0 if args.stage == 'listen' else args.fold
    rate, max_hz = int(cfg.get('rate', 128)), cfg.get('max_hz')
    train, test, splits = {}, {}, {}
    for name, spec in specs.items():
        try:
            store = open_store(name)
        except FileNotFoundError:
            print(f'skipping {name}: no store (run scripts/prepare.py {name})')
            continue
        common = dict(window_s=spec['window'], align=cfg['align'], rate=rate, max_hz=max_hz)
        if args.stage == 'listen':
            train_subjects, test_subjects = split(store, cfg['folds'], fold)
            splits[name] = dict(train=train_subjects, test=test_subjects)
            train[name] = ListenSource(store, train_subjects, mismatches=cfg['mismatches'], rng=rng, **common)
            test[name] = ListenSource(store, test_subjects, mismatches=cfg['mismatches'], rng=np.random.default_rng(1),
                                      **common)
        elif args.protocol == 'within':
            labelled = store.table[store.table.item >= 0]
            held = within_fold(labelled, cfg['folds'], fold)
            splits[name] = dict(protocol='within', held_out=int(len(held)))
            modalities = [spec['target']] if args.only_target else spec['modalities']
            train[name] = TrialSource(store, store.subjects, modalities=modalities, rng=rng,
                                      rows=labelled.index.difference(held), **common)
            test[name] = TrialSource(store, store.subjects, modalities=spec['modalities'], rng=np.random.default_rng(1),
                                     rows=held, **common)
        else:
            train_subjects, test_subjects = split(store, cfg['folds'], fold)
            splits[name] = dict(train=train_subjects, test=test_subjects)
            modalities = [spec['target']] if args.only_target else spec['modalities']
            train[name] = TrialSource(store, train_subjects, modalities=modalities, rng=rng, **common)
            test[name] = TrialSource(store, test_subjects, modalities=spec['modalities'], rng=np.random.default_rng(1),
                                     **common)
        print(f'{name}: {len(train[name])} training / {len(test[name])} test segments ({args.protocol})')
    if not train:
        raise SystemExit('no data')

    heads = {} if args.stage == 'listen' else {n: len(s.store.items) for n, s in train.items()}
    person_dim = cfg.get('person_dim', 0) if args.stage == 'imagery' else 0
    if args.init:
        model, _ = Model.load(args.init, heads=heads)
        if person_dim and model.person is None:
            raise SystemExit('--init model has no person encoder; train without --init or person_dim=0')
        if model.encoder.rate != rate:
            raise SystemExit(f'--init model runs at {model.encoder.rate} Hz, this stage at {rate} Hz')
    else:
        model = Model(heads, person_dim=person_dim, rate=rate, **config['model'])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    names = sorted(train)
    weights = np.array([specs[n]['weight'] for n in names], float)
    weights /= weights.sum()
    log = open(out / 'train.jsonl', 'w')
    running, start, skipped = {}, time.time(), 0
    for step in range(cfg['steps']):
        model.train()
        for group in optimizer.param_groups:
            group['lr'] = cfg['lr'] * schedule(step, cfg['steps'])
        name = names[rng.choice(len(names), p=weights)]
        if args.stage == 'listen':
            loss, logs = listen_step(model, train[name], cfg, device)
        else:
            loss, logs = imagery_step(model, name, train[name], specs[name], cfg, device)
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            skipped += 1
            if skipped > 20:
                raise SystemExit(f'step {step}: {skipped} non-finite losses ({name}); stopping')
            continue
        loss.backward()
        if not torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)):
            skipped += 1
            continue
        optimizer.step()
        for key, value in logs.items():
            running.setdefault(f'{name}/{key}', []).append(value)
        if (step + 1) % 100 == 0 or step + 1 == cfg['steps']:
            entry = dict(step=step + 1, seconds=round(time.time() - start), lr=optimizer.param_groups[0]['lr'], skipped=skipped,
                         **{k: round(float(np.mean(v)), 4) for k, v in running.items()})
            print(json.dumps(entry), flush=True)
            log.write(json.dumps(entry) + '\n'); log.flush()
            running = {}
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise SystemExit('non-finite weights after training; nothing saved')
    model.save(out / 'model.pt', stage=args.stage, fold=fold, splits=splits, cfg=cfg, init=args.init)

    report = dict(stage=args.stage, fold=fold, protocol=args.protocol, init=args.init, only_target=args.only_target,
                  cfg=cfg, results={})
    for name, source in test.items():
        if not len(source):
            continue
        if args.stage == 'listen':
            report['results'][name] = listen_scores(model, source, device)
        else:
            shots = () if args.protocol == 'within' else tuple(cfg['shots'])     # within: the head is already personal
            report['results'][name] = item_scores(model, source, name, specs[name]['target'], device, shots=shots)
        summary = {k: v for k, v in report['results'][name].items() if k != 'per_subject'}
        print(name, json.dumps(summary)[:600], flush=True)
    json.dump(report, open(out / 'evaluation.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

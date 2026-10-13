"""Do pretrained EEG foundation models read more imagined content than filter-bank log-power?

    python scripts/foundation.py embed    [--models reve cbramod labram] [--datasets ...]   # frozen features
    python scripts/foundation.py probe    --source fm fm+lp --fold 0 1 2 3 4 [--models ...] [--datasets ...]
    python scripts/foundation.py finetune --fold 0 1 2 3 4 [--models ...] [--datasets ...]
    python scripts/foundation.py report

Only the encoder is measured: whatever the speech side does, the reconstruction cannot carry more content
than the encoder reads (it renders the encoder's point of the CLIP space), so an encoder that is not better
here is not worth a reconstruction run.  Everything else is held fixed: the folds (whole blocks, cue
episodes, runs), the held-out trials, the inner cross-validation, the anchors (HuBERT of synthetic speech)
and the ensembles are those of ``reconstruct.py`` (items: Thinking Out Loud, BCI2020) and ``sentences.py``
(Chisco), so the measures pair trial by trial with the log-power encoders of ``outputs/reconstruct`` and
``outputs/sentences``.

Stage 1, probe (frozen): ``eegspeech.foundation`` embeds every trial once (inputs per the model's
pretraining; no fold-dependent statistic).  Its token embeddings, pooled (all tokens, per patch, per channel
reduced to the top directions of the person's training tokens), replace (``fm``) or join (``fm+lp``) the
aligned log-power features of the existing linear encoders: a personal linear CLIP map (items: InfoNCE,
L-BFGS; sentences: standardise, PCA, ridge onto the anchors), with layer, pooling, drift removal and weight
decay chosen by the same coordinate search on inner cross-validation, and the best 3 averaged.

Stage 2, finetune: the backbone (CBraMod, LaBraM: all of it; REVE: its top 4 layers) and personal heads
trained with cross-entropy of s * cos(z, a) against the anchors (items; or every training sentence:
CLIP with a frozen speech tower, like ``eegspeech.deep``), early stopped on each person's last inner block of
training trials; the temperature is calibrated there.

Measures per person (mean over held-out trials, then over folds): items: top-1 accuracy (chance 1/K) and the
log-loss of the calibrated posterior (uniform log K); sentences: rank percentile of the true sentence among
the run's sentences (chance 0.5) and top-10.  Each is compared with the log-power encoder on the same trials
(paired over people, Wilcoxon).
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegspeech import ROOT, clip, features, foundation                       # noqa: E402
from eegspeech.store import open_store                                       # noqa: E402

OUT = ROOT / 'outputs' / 'foundation'


def script(name):
    """One of the pipeline scripts as a module (their Targets, splits and devices)."""
    spec = importlib.util.spec_from_file_location(f'_{name}', ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def label(setting):
    return ','.join(f'{k}={v}' for k, v in setting.items())


# ----------------------------------------------------------------------------- frozen feature blocks
def blocks(fmf, subject, train_positions, k, lp=None):
    """{(layer tag, kind): (n, F)} pooled features of all of a person's trials (``Features`` order), the
    per-channel ones reduced on the training trials; ``lp``: (n, F) log-power block to add."""
    out = {}
    for tag, layer in (('mid', fmf.layers[0]), ('last', fmf.layers[-1])):
        mean = fmf.get(subject, layer, 'mean')
        out[tag, 'mean'] = mean
        out[tag, 'time'] = (fmf.get(subject, layer, 'time') - mean[:, None]).reshape(len(mean), -1)
        chan = fmf.file[subject][f'L{layer}/chan']
        basis = channel_basis(chan, np.sort(train_positions), k)
        out[tag, 'chan'] = np.concatenate([basis(chan[a:a + 512].astype(np.float32)) for a in range(0, len(chan), 512)])
    if lp is not None:
        out['lp'] = lp
    return out


def channel_basis(chan, positions, k, chunk=256):
    """``foundation.ChannelBasis`` fitted on the training trials, streamed (Chisco's tokens are large)."""
    total, outer, count = 0., 0., 0
    for a in range(0, len(positions), chunk):
        x = chan[positions[a:a + chunk]].astype(np.float64).reshape(-1, chan.shape[-1])
        total, outer, count = total + x.sum(0), outer + x.T @ x, count + len(x)
    mean = total / count
    values, vectors = np.linalg.eigh(outer / count - np.outer(mean, mean))
    return foundation.ChannelBasis(mean.astype(np.float32), vectors[:, ::-1][:, :k].astype(np.float32))


def design(person, setting, cache):
    """(n, F) features of a setting for all of a person's trials: the pooled blocks of its layer, the
    log-power block if any, minus a causal running baseline (``recenter``)."""
    key = (person['subject'], setting['layer'], setting['pool'], setting['recenter'])
    if key not in cache:
        b = person['blocks']
        parts = [b[setting['layer'], kind] for kind in setting['pool'].split('+')]
        if 'lp' in b:
            parts.append(b['lp'])
        x = np.concatenate(parts, 1).astype(np.float32)
        if setting['recenter']:
            out, modality = person['out'], person['modality']
            means = {m: x[(modality == m) & ~out].mean(0) for m in np.unique(modality)}
            x = x - features.local_baseline(x, person['run'], out, np.stack([means[m] for m in modality]),
                                            setting['recenter'])
        cache[key] = x
    return cache[key]


def coordinate_search(space, evaluate, rank):
    results = {}
    best = {k: v[0] for stage in space for k, v in stage.items()}
    for stage in space:
        for values in itertools.product(*stage.values()):
            setting = dict(best, **dict(zip(stage, values)))
            if label(setting) not in results:
                results[label(setting)] = (setting, evaluate(setting))
        best = max(results.values(), key=lambda r: rank(r[1]))[0]
    return sorted(results.values(), key=lambda r: rank(r[1]), reverse=True)


# ----------------------------------------------------------------------------- stage 1: items
def probe_items(name, spec, cfg, fold, key, source):
    rc = script('reconstruct')
    r, fc = cfg['reconstruct'], cfg['foundation']
    t = rc.Targets(name, spec)
    store = open_store(name)
    item_map = np.array([t.items.index(i) for i in store.items])
    labelled = features.labelled(store, spec['modalities'])
    held = set(features.within_fold(labelled, r['folds'], fold).tolist())
    fmf = foundation.Features(name, key)
    lp = features.load(name, held=held, align_by=spec.get('align_by', 'session'))[0] if 'lp' in source else None
    target = spec['target']
    people = {}
    for s in fmf.subjects:
        rows = fmf.rows(s)
        part = labelled[labelled.subject == s]
        train = part[~part.index.isin(held)]
        test = part[part.index.isin(held) & (part.modality_name == target)]
        if not len(test) or not (train.modality_name == target).any():
            continue
        position = pd.Series(np.arange(len(rows)), index=rows)
        lp_block = None
        if lp is not None:
            lrows, x, parts = lp[s]
            at = pd.Series(np.arange(len(lrows)), index=lrows)[rows].to_numpy()
            lp_block = np.concatenate([x[at], features.time_course(x[at], parts[at], 3)], 1)
        mine = store.table.loc[rows]
        people[s] = dict(subject=s, position=position, train=train, test=test,
                         blocks=blocks(fmf, s, position[train.index].to_numpy(), fc['channel_components'], lp_block),
                         out=np.isin(rows, list(held)), modality=mine.modality_name.to_numpy(),
                         run=(mine.session.astype(str) + '/' + mine.modality_name).to_numpy(),
                         inner=features.blocks(train, r['inner_folds']),
                         is_target=(train.modality_name == target).to_numpy())
    subjects = sorted(people)
    narrow = spec.get('encoder', {})                       # a dataset's restrictions (BCI2020: no drift removal)
    space = [{k: list(narrow.get(k, v)) for k, v in stage.items()} for stage in fc['items']['search']]
    if len(spec['modalities']) == 1:
        space = [{k: [0] if k == 'aux_weight' else v for k, v in stage.items()} for stage in space]
    cache = {}

    def take(s, index, setting):
        p = people[s]
        return design(p, setting, cache)[p['position'][index].to_numpy()]

    def persons(setting, j=None):
        out = []
        for s in subjects:
            p = people[s]
            keep = np.ones(len(p['train']), bool) if j is None else p['inner'] != j
            if setting['aux_weight'] == 0:
                keep &= p['is_target']
            out.append(clip.Person(take(s, p['train'].index[keep], setting), item_map[p['train']['item'].to_numpy()[keep]],
                                   np.where(p['is_target'][keep], 1., setting['aux_weight'])))
        return out

    enc = r['encoder']
    fit = lambda setting, ps, seed: clip.train(ps, t.anchor, weight_decay=setting['weight_decay'], steps=enc['steps'],
                                               seed=seed)

    def evaluate(setting):
        z, y = [], []
        for j in range(r['inner_folds']):
            model = fit(setting, persons(setting, j), j)
            for i, s in enumerate(subjects):
                p = people[s]
                rows = p['train'].index[(p['inner'] == j) & p['is_target']]
                if len(rows):
                    z.append(model.embed(i, take(s, rows, setting)))
                    y.append(item_map[p['train'].loc[rows, 'item'].to_numpy()])
        z, y = np.concatenate(z), np.concatenate(y)
        scale, nll = clip.calibrate(z, y, t.anchor)
        accuracy = float(np.mean((z @ t.anchor.T).argmax(1) == y))
        log(f'{name} {key} {source} f{fold} inner CV {label(setting)}: accuracy {accuracy:.3f}, log-loss {nll:.4f}')
        return dict(nll=nll, accuracy=accuracy, scale=scale, z=z, y=y)

    top = coordinate_search(space, evaluate, lambda res: -res['nll'])[:fc['items']['ensemble']]
    y = top[0][1]['y']
    scale, nll = clip.calibrate(np.mean([res['scale'] * res['z'] for _, res in top], 0), y, t.anchor)
    z = 0
    for setting, res in top:
        model = fit(setting, persons(setting), 0)
        z = z + res['scale'] * np.concatenate([model.embed(i, take(s, people[s]['test'].index, setting))
                                               for i, s in enumerate(subjects)]) / len(top)
    test = pd.concat([people[s]['test'] for s in subjects])
    posterior = clip.posterior(z, t.anchor, scale)
    item = item_map[test['item'].to_numpy()]
    trials = pd.DataFrame(dict(subject=test.subject.to_numpy(), row=test.index.to_numpy(), item=item,
                               decoded=posterior.argmax(1), p_true=posterior[np.arange(len(item)), item]))
    return trials, dict(settings=[s for s, _ in top], inner_nll=nll, chance=1 / t.K, K=t.K)


# ----------------------------------------------------------------------------- stage 1: sentences
def probe_sentences(name, spec, cfg, fold, key, source):
    sn = script('sentences')
    sc, fc = cfg['sentences'], cfg['foundation']
    pc = fc['sentences']
    t = sn.Targets(name)
    anchor = t.anchor(sc['space']['dims'], sc['space']['duration_weight'])
    table, held, trainable = sn.split(name, spec, sc, fold, t)
    fmf = foundation.Features(name, key)
    lp = None
    if 'lp' in source:
        lp = features.load(name, held=held, align_by=spec.get('align_by', 'subject'), modalities=spec['modalities'],
                           windows=sc['encoder']['windows'])[0]
    people = {}
    for s in fmf.subjects:
        all_rows = fmf.rows(s)
        keep = np.isin(all_rows, table.index)
        rows = all_rows[keep]
        out = np.isin(rows, list(held))
        if not out.any() or out.all():
            continue
        part = table.loc[rows]
        train = trainable.loc[rows].to_numpy()
        inner = np.full(len(rows), -1)
        inner[train] = features.blocks(part[train], sc['inner_folds'])
        lp_block = None
        if lp is not None:
            lrows, x, parts = lp[s]
            at = pd.Series(np.arange(len(lrows)), index=lrows)[rows].to_numpy()
            lp_block = np.concatenate([x[at], features.time_course(x[at], parts[at], 3)], 1)
        b = blocks(fmf, s, np.flatnonzero(keep)[train], fc['channel_components'])
        b = {k: v[keep] for k, v in b.items()}
        if lp_block is not None:
            b['lp'] = lp_block
        people[s] = dict(subject=s, rows=rows, blocks=b, out=out, train=train, inner=inner,
                         modality=part.modality_name.to_numpy(),
                         run=(part.session.astype(str) + '/' + part.modality_name).to_numpy(),
                         session=part.session.to_numpy(), text=part.text.to_numpy(),
                         sentence=np.array([t.index[text] for text in part.text]))
        log(f'{name} {key} f{fold} {s}: {len(rows)} trials, {sum(v.shape[1] for v in b.values())} features in all blocks')
    subjects = sorted(people)
    cache, reducers = {}, {}

    def fit(s, j, mask, setting):
        p = people[s]
        index = np.flatnonzero(mask)
        k = (s, j, setting['layer'], setting['pool'], setting['recenter'])
        if k not in reducers:
            if len(reducers) > 2 * len(subjects) * (sc['inner_folds'] + 1):
                reducers.clear()
            x = design(p, setting, cache)[index]
            reducer = clip.Reducer.fit(x, pc['components'])
            reducers[k] = (reducer, reducer(x))
        reducer, x = reducers[k]
        return reducer, clip.train_sentences(x, p['sentence'][index], p['session'][index], anchor, method='ridge',
                                             weight_decay=setting['weight_decay'], standardize=False)

    def embed(s, fitted, mask, setting):
        reducer, model = fitted
        index = np.flatnonzero(mask)
        return index, model.embed(reducer(design(people[s], setting, cache)[index]))

    def by_run(p, index, z):
        return [(index[trials], candidates, position, z[trials] @ anchor[candidates].T)
                for trials, candidates, position in clip.run_sets(p['sentence'][index], p['session'][index])]

    def evaluate(setting):
        sets = {s: [] for s in subjects}
        for j in range(sc['inner_folds']):
            for s in subjects:
                p = people[s]
                ask = p['inner'] == j
                if ask.any():
                    fit_mask = p['train'] & (p['inner'] != j) & ~np.isin(p['text'], p['text'][ask])
                    sets[s] += by_run(p, *embed(s, fit(s, j, fit_mask, setting), ask, setting))
        percentile = float(np.mean([np.mean(np.concatenate([clip.rank_percentile(S, pos) for _, _, pos, S in sets[s]]))
                                    for s in subjects]))
        flat = [x for s in subjects for x in sets[s]]
        scale, nll = clip.calibrate_sets([x[3] for x in flat], [x[2] for x in flat])
        log(f'{name} {key} {source} f{fold} inner CV {label(setting)}: rank percentile {percentile:.4f}, log-loss {nll:.4f}')
        return dict(percentile=percentile, nll=nll, scale=scale)

    top = coordinate_search([dict(stage) for stage in pc['search']], evaluate, lambda res: res['percentile'])
    top = top[:pc['ensemble']]
    lines = []
    for s in subjects:
        p = people[s]
        z = 0
        for setting, res in top:
            index, zr = embed(s, fit(s, 'all', p['train'], setting), p['out'], setting)
            z = z + res['scale'] * zr / len(top)
        for rows, candidates, position, S in by_run(p, index, z):
            rank = clip.rank_percentile(S, position)
            order = np.argsort(-S, 1)
            for i, row in enumerate(rows):
                lines.append(dict(subject=s, row=int(p['rows'][row]), run=int(p['session'][row]),
                                  candidates=len(candidates), percentile=float(rank[i]),
                                  top10=bool(position[i] in order[i, :10]), top1=bool(order[i, 0] == position[i])))
    return pd.DataFrame(lines), dict(settings=[s for s, _ in top], inner_percentile=top[0][1]['percentile'])


# ----------------------------------------------------------------------------- stage 2: fine-tuning
def finetune(name, spec, cfg, fold, key, kind, device):
    """Fine-tune one model for a dataset and fold; (trials, info) as the probes."""
    fc = cfg['foundation']['finetune']
    torch.manual_seed(fc['seed'] + fold)
    rng = np.random.default_rng(fc['seed'] + fold)
    if kind == 'items':
        rc, r = script('reconstruct'), cfg['reconstruct']
        t = rc.Targets(name, spec)
        store = open_store(name)
        item_map = np.array([t.items.index(i) for i in store.items])
        labelled = features.labelled(store, spec['modalities'])
        held = set(features.within_fold(labelled, r['folds'], fold).tolist())
        anchor, modalities, inner_folds = t.anchor, spec['modalities'], r['inner_folds']
        table = labelled
        target = item_map[table['item'].to_numpy()]
        weight = np.where(table.modality_name == spec['target'], 1., fc['aux_weight'])
        scored = (table.modality_name == spec['target']).to_numpy()
        trainable = ~table.index.isin(held)
    else:
        sn, sc = script('sentences'), cfg['sentences']
        t = sn.Targets(name)
        anchor = t.anchor(sc['space']['dims'], sc['space']['duration_weight'])
        table, held, trainable = sn.split(name, spec, sc, fold, t)
        trainable = trainable.to_numpy()
        modalities, inner_folds = spec['modalities'], sc['inner_folds']
        target = np.array([t.index[text] for text in table.text])
        weight, scored = np.ones(len(table)), np.ones(len(table), bool)
    out = table.index.isin(held)
    if key == 'reve':
        data = foundation.reve_lower(name, modalities, fc['reve_start'], device, log)
    else:
        data = foundation.inputs(name, key, modalities, cache=True)
    position = {}
    for s, (rows, x, info) in data.items():
        for i, row in enumerate(rows):
            position[int(row)] = (s, i)
    keep = np.array([int(row) in position for row in table.index])
    table, target, weight, scored, trainable, out = (v[keep] for v in (table, target, weight, scored, trainable, out))
    # validation: each person's last inner block of training trials; its sentences are kept out of the fit
    validation = np.zeros(len(table), bool)
    for s, part in table[trainable].groupby('subject'):
        inner = features.blocks(part, inner_folds)
        validation[table.index.get_indexer(part.index[inner == inner_folds - 1])] = True
    fit = trainable & ~validation
    if kind == 'sentences':
        fit &= ~table.text.isin(set(table.text[validation])).to_numpy()
    validation &= scored
    model = build_finetuner(key, data, anchor.shape[1], fc).to(device)
    with torch.no_grad():               # a few item anchors: start soft (random z at scale 10 gives a loss of ~5)
        model.logit_scale.fill_(math.log(1. if kind == 'items' else 10.))
    bank_ids = np.unique(target[fit]) if kind == 'sentences' else np.arange(len(anchor))
    bank = F.normalize(torch.as_tensor(anchor[bank_ids], dtype=torch.float32, device=device), dim=-1)
    bank_target = np.searchsorted(bank_ids, target)
    heads = [p for n, p in model.named_parameters() if not n.startswith('backbones')]
    optimizer = torch.optim.AdamW([dict(params=model.backbone_parameters(), lr=fc['lr']),
                                   dict(params=heads, lr=fc['head_lr'])], weight_decay=fc['weight_decay'])
    subjects_of = table.subject.to_numpy()
    index_of = np.array([position[int(row)][1] for row in table.index])

    def batches(mask, shuffle):
        out_ = []
        for s in np.unique(subjects_of[mask]):
            members = np.flatnonzero(mask & (subjects_of == s))
            if shuffle:
                members = rng.permutation(members)
            out_ += [(s, members[a:a + fc['batch']]) for a in range(0, len(members), fc['batch'])]
        return [out_[i] for i in rng.permutation(len(out_))] if shuffle else out_

    def tensor(s, members):
        x = data[s][1][index_of[members]]
        return torch.as_tensor(np.asarray(x, np.float32), device=device)

    @torch.no_grad()
    def embed(mask):
        model.eval()
        z = np.zeros((len(table), anchor.shape[1]), np.float32)
        for s, members in batches(mask, False):
            z[members] = F.normalize(model(s, tensor(s, members)), dim=-1).float().cpu().numpy()
        return z

    def score(z, mask):
        """items: (log-loss after calibration, scale); sentences: (-rank percentile within runs, scale)."""
        members = np.flatnonzero(mask)
        if kind == 'items':
            scale, nll = clip.calibrate(z[members], target[members], anchor / np.linalg.norm(anchor, axis=1, keepdims=True))
            return nll, scale
        sets = clip.run_sets(target[members], table.session.to_numpy()[members] * 1000 +
                             pd.factorize(subjects_of)[0][members])
        S = [z[members[trials]] @ anchor[candidates].T for trials, candidates, _ in sets]
        pct = float(np.mean(np.concatenate([clip.rank_percentile(s_, pos) for s_, (_, _, pos) in zip(S, sets)])))
        scale, _ = clip.calibrate_sets(S, [pos for _, _, pos in sets])
        return -pct, scale

    steps = fc['epochs'] * len(batches(fit, False))
    warmup, step = max(1, steps // 20), 0
    best, best_state, waited = np.inf, None, 0
    for epoch in range(fc['epochs']):
        started, losses = time.time(), []
        model.train()
        for s, members in batches(fit, True):
            for g, base in zip(optimizer.param_groups, (fc['lr'], fc['head_lr'])):
                g['lr'] = base * min(1., (step + 1) / warmup) * .5 * (1 + math.cos(math.pi * min(step, steps) / steps))
            z = F.normalize(model(s, tensor(s, members)), dim=-1)
            logits = model.logit_scale.exp() * z @ bank.T
            loss = (F.cross_entropy(logits, torch.as_tensor(bank_target[members], device=device), reduction='none')
                    * torch.as_tensor(weight[members], dtype=torch.float32, device=device)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            step += 1
            losses.append(float(loss.detach()))
        value, _ = score(embed(validation), validation)
        log(f'{name} {key} finetune f{fold} epoch {epoch + 1}: loss {np.mean(losses):.4f}, validation '
            f'{"log-loss" if kind == "items" else "-rank percentile"} {value:.4f} ({time.time() - started:.0f} s)')
        if value < best:
            best, best_state, waited = value, model.snapshot(), 0
        else:
            waited += 1
            if waited >= fc['patience']:
                break
    model.restore(best_state)
    _, scale = score(embed(validation), validation)
    z = embed(out)
    members = np.flatnonzero(out & scored)
    if kind == 'items':
        unit = anchor / np.linalg.norm(anchor, axis=1, keepdims=True)
        posterior = clip.posterior(z[members], unit, scale)
        item = target[members]
        trials = pd.DataFrame(dict(subject=subjects_of[members], row=table.index[members], item=item,
                                   decoded=posterior.argmax(1), p_true=posterior[np.arange(len(item)), item]))
        return trials, dict(validation_log_loss=best, chance=1 / len(anchor), K=len(anchor), epochs=epoch + 1)
    lines = []
    runs = table.session.to_numpy()
    for s in np.unique(subjects_of[members]):
        mine = members[subjects_of[members] == s]
        for trials_, candidates, position_ in clip.run_sets(target[mine], runs[mine]):
            S = z[mine[trials_]] @ anchor[candidates].T
            rank = clip.rank_percentile(S, position_)
            order = np.argsort(-S, 1)
            for i, m in enumerate(mine[trials_]):
                lines.append(dict(subject=s, row=int(table.index[m]), run=int(runs[m]), candidates=len(candidates),
                                  percentile=float(rank[i]), top10=bool(position_[i] in order[i, :10]),
                                  top1=bool(order[i, 0] == position_[i])))
    return pd.DataFrame(lines), dict(validation_percentile=-best, epochs=epoch + 1)


def build_finetuner(key, data, out, fc):
    """``foundation.Finetuner`` for the people of ``data`` (REVE: on cached tokens of the lower layers)."""
    if key == 'reve':
        people = {s: (len(info), None, info) for s, (rows, x, info) in data.items()}
        model = foundation.Finetuner.__new__(foundation.Finetuner)
        torch.nn.Module.__init__(model)
        shared = foundation.load_backbone('reve', 1, 200)
        model.key, model.start, model.depth = 'reve', fc['reve_start'], foundation.MODELS['reve']['depth']
        model.backbones = torch.nn.ModuleDict({s: foundation.Backbone('reve', c, 200, info, model=shared)
                                               for s, (c, _, info) in people.items()})
        model.heads = torch.nn.ModuleDict({s: foundation.Head(x.shape[1], foundation.MODELS['reve']['dim'], out,
                                                              fc['dropout'], fc.get('head', 'flat'), x.shape[1] // len(info))
                                           for s, (rows, x, info) in data.items()})
        model.logit_scale = torch.nn.Parameter(torch.tensor(math.log(10.)))
        for p in shared.parameters():
            p.requires_grad_(False)
        for layer in shared.transformer.layers[model.start:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        return model
    return foundation.Finetuner(key, {s: (x.shape[1], x.shape[2], info) for s, (rows, x, info) in data.items()},
                                out, fc['dropout'], head=fc.get('head', 'flat'))


# ----------------------------------------------------------------------------- report
def baseline_items(name, fold):
    path = ROOT / 'outputs' / 'reconstruct' / f'f{fold}' / name / 'trials.csv'
    return pd.read_csv(path)[['subject', 'row', 'item', 'decoded', 'p_true']] if path.exists() else None


def baseline_sentences(name, fold, deep=False):
    path = ROOT / 'outputs' / ('sentences_deep' if deep else 'sentences') / f'f{fold}' / name / 'trials.csv'
    if not path.exists():
        return None
    t = pd.read_csv(path)
    return t.assign(percentile=t.encoder_percentile, top10=t.encoder_top10)[['subject', 'row', 'percentile', 'top10']]


def per_person(trials, kind):
    g = trials.groupby('subject')
    if kind == 'items':
        return pd.DataFrame(dict(accuracy=g.apply(lambda d: (d.decoded == d.item).mean()),
                                 log_loss=g.apply(lambda d: -np.log(d.p_true.clip(1e-9)).mean())))
    return pd.DataFrame(dict(percentile=g.percentile.mean(), top10=g.top10.mean()))


def compare(ours, base, kind):
    """Means over people and the paired test against the log-power encoder on the same trials."""
    keys = ['subject', 'row']
    merged = ours.merge(base, on=keys, suffixes=('', '_base'))
    a = per_person(merged[[c for c in merged if not c.endswith('_base')]], kind)
    b = per_person(merged[keys + [c for c in merged if c.endswith('_base')]]
                   .rename(columns=lambda c: c[:-5] if c.endswith('_base') else c), kind)
    metric = 'accuracy' if kind == 'items' else 'percentile'
    diff = (a[metric] - b[metric]).to_numpy()
    p = float(stats.wilcoxon(diff).pvalue) if len(diff) > 1 and np.any(diff != 0) else float('nan')
    out = {m: dict(ours=float(a[m].mean()), baseline=float(b[m].mean())) for m in a.columns}
    out.update(people=len(a), trials=len(merged), better=int((diff > 0).sum()), p_wilcoxon=p,
               per_person={s: dict(ours=float(a.loc[s, metric]), baseline=float(b.loc[s, metric])) for s in a.index})
    return out


def report(cfg):
    rows, summary = [], {}
    for folder in sorted(OUT.glob('*_*')):
        if not folder.is_dir():
            continue
        for name in list(cfg['reconstruct']['datasets']) + list(cfg['sentences']['datasets']):
            kind = 'items' if name in cfg['reconstruct']['datasets'] else 'sentences'
            parts = [(f, pd.read_csv(folder / f'{name}_f{f}.csv')) for f in range(5) if (folder / f'{name}_f{f}.csv').exists()]
            if not parts:
                continue
            ours = pd.concat([p for _, p in parts])
            base = [baseline_items(name, f) if kind == 'items' else baseline_sentences(name, f) for f, _ in parts]
            if any(b is None for b in base):
                continue
            c = compare(ours, pd.concat(base), kind)
            c['folds'] = [f for f, _ in parts]
            if kind == 'sentences':
                deep = [baseline_sentences(name, f, deep=True) for f, _ in parts]
                if all(d is not None for d in deep):
                    c['deep_baseline'] = compare(ours, pd.concat(deep), kind)['percentile']['baseline']
            summary[f'{folder.name}/{name}'] = c
            metric = 'accuracy' if kind == 'items' else 'percentile'
            rows.append(f'| {folder.name} | {name} | {len(parts)} | {c[metric]["ours"]:.4f} | {c[metric]["baseline"]:.4f} | '
                        f'{c["better"]}/{c["people"]} | {c["p_wilcoxon"]:.3g} |' +
                        (f' {c["log_loss"]["ours"]:.4f} vs {c["log_loss"]["baseline"]:.4f} |' if kind == 'items'
                         else f' top-10 {c["top10"]["ours"]:.4f} vs {c["top10"]["baseline"]:.4f} |'))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(OUT / 'summary.json', 'w'), indent=1)
    table = ['| run | dataset | folds | encoder (FM) | log-power encoder | people better | p | other |',
             '|---|---|---|---|---|---|---|---|'] + rows
    (OUT / 'summary.md').write_text('Items: top-1 accuracy; sentences: rank percentile (chance 0.5).\n\n' + '\n'.join(table) + '\n')
    print('\n'.join(table))


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['embed', 'probe', 'finetune', 'report'])
    parser.add_argument('--models', nargs='*')
    parser.add_argument('--datasets', nargs='*')
    parser.add_argument('--source', nargs='*', default=['fm'], choices=['fm', 'fm+lp'])
    parser.add_argument('--fold', type=int, nargs='*', default=[0])
    parser.add_argument('--config', default=str(ROOT / 'configs' / 'plan.yaml'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--out', help='default: outputs/foundation')
    parser.add_argument('--head', choices=['flat', 'mean'], help="fine-tuning head (default: the config's, flat)")
    args = parser.parse_args()
    global OUT
    OUT = Path(args.out) if args.out else OUT
    cfg = yaml.safe_load(open(args.config))
    if args.stage == 'report':
        return report(cfg)
    device = script('reconstruct').device_of(args.device)
    specs = {**{n: ('items', s) for n, s in cfg['reconstruct']['datasets'].items()},
             **{n: ('sentences', s) for n, s in cfg['sentences']['datasets'].items()}}
    names = [n for n in specs if not args.datasets or n in args.datasets]
    fc = cfg['foundation']
    if args.head:
        fc['finetune']['head'] = args.head
    suffix = '' if fc['finetune'].get('head', 'flat') == 'flat' else '_' + fc['finetune']['head'] + 'head'
    for name in names:
        kind, spec = specs[name]
        if args.stage == 'embed':
            for key in args.models or fc['models']:
                if foundation.feature_path(name, key).exists():
                    log(f'{name} {key}: features exist')
                    continue
                foundation.embed(name, key, spec['modalities'], device, log)
            continue
        if args.stage == 'probe':
            for key, source, fold in itertools.product(args.models or fc['models'], args.source, args.fold):
                folder = OUT / f'probe_{key}_{source.replace("+", "_")}'
                if (folder / f'{name}_f{fold}.csv').exists():
                    continue
                started = time.time()
                run = probe_items if kind == 'items' else probe_sentences
                trials, info = run(name, spec, cfg, fold, key, source)
                folder.mkdir(parents=True, exist_ok=True)
                trials.to_csv(folder / f'{name}_f{fold}.csv', index=False)
                info['seconds'] = round(time.time() - started)
                json.dump(info, open(folder / f'{name}_f{fold}.json', 'w'), indent=1, default=str)
                log(f'{name} {key} {source} f{fold}: done ({info["seconds"]} s)')
        if args.stage == 'finetune':
            models = args.models or fc['finetune'][f'{kind}_models']
            for key, fold in itertools.product(models, args.fold):
                folder = OUT / f'finetune_{key}{suffix}'
                if (folder / f'{name}_f{fold}.csv').exists():
                    continue
                started = time.time()
                trials, info = finetune(name, spec, cfg, fold, key, kind, device)
                folder.mkdir(parents=True, exist_ok=True)
                trials.to_csv(folder / f'{name}_f{fold}.csv', index=False)
                info['seconds'] = round(time.time() - started)
                json.dump(info, open(folder / f'{name}_f{fold}.json', 'w'), indent=1, default=str)
                log(f'{name} {key} finetune f{fold}: done ({info["seconds"]} s)')
                if device.type == 'mps':
                    torch.mps.empty_cache()


if __name__ == '__main__':
    main()

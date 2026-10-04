"""Evaluation on held-out people.

Listening: 5-way match-mismatch accuracy (chance 20 %) and cross-person retrieval of
the same stimulus window.  Items: accuracy on the target modality (imagined
trials) of each held-out person, zero-shot from the dataset head and few-shot from
k labelled trials per item of that person (nearest class prototype in the
embedding: personalisation from the person's own data, no identity input).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .data import ListenSource, TrialSource, collate
from .losses import pair_infonce
from .metrics import binomial_p, summarize


@torch.no_grad()
def listen_scores(model, source: ListenSource, device, windows=512, batch=32, seed=1234):
    """Match-mismatch and cross-person pair accuracy on fixed random windows."""
    model.eval()
    state = source.rng
    source.rng = np.random.default_rng(seed)
    hits = total = pair_hits = pair_total = 0
    per_subject = {}
    for _ in range(max(windows // batch, 1)):
        items, pairs = source.sample(batch, partner_fraction=1.)
        anchors = items[:batch]
        b = collate(items).to(device)
        frames, mask, _, _ = model.encoder(b.eeg, b.xyz, b.valid)
        features = torch.as_tensor(np.stack([w['features'] for w in anchors]), device=device)
        speech = model.speech(features.flatten(0, 1)).unflatten(0, features.shape[:2])
        t = min(frames.shape[1], speech.shape[2])
        sims = F.normalize(frames[:batch, None, :t], dim=-1) * F.normalize(speech[:, :, :t], dim=-1)
        correct = (sims.sum(-1).mean(-1).argmax(1) == 0).cpu().numpy()
        for w, c in zip(anchors, correct):
            per_subject.setdefault(w['subject'], []).append(c)
        hits, total = hits + int(correct.sum()), total + len(correct)
        if len(pairs) > 1:
            a, p = zip(*pairs)
            _, acc = pair_infonce(frames[list(a)], frames[list(p)], mask[list(a)])
            pair_hits, pair_total = pair_hits + float(acc) * len(a), pair_total + len(a)
    source.rng = state
    chance = 1 / (1 + source.mismatches)
    report = summarize({s: float(np.mean(v)) for s, v in per_subject.items()}, chance)
    report.update(windows=total, accuracy=hits / max(total, 1), p_binomial=binomial_p(hits, total, chance),
                  pair_accuracy=pair_hits / max(pair_total, 1), pair_chance=1 / batch if pair_total else None)
    return report


@torch.no_grad()
def embed_trials(model, source: TrialSource, indices, device, crops=5, head=None, person=None):
    """Mean over evenly spaced crops: logits (from ``head``) and unit embeddings, per trial."""
    model.eval()
    logits, embeddings = [], []
    for i in indices:
        windows = source.crops(int(i), crops)
        b = collate(windows).to(device)
        p = None if person is None else person.expand(len(windows), -1)
        _, _, pooled, z = model.encoder(b.eeg, b.xyz, b.valid, b.lengths, p)
        if head is not None:
            logits.append(F.log_softmax(model.heads[head](pooled), -1).mean(0).cpu())
        embeddings.append(F.normalize(z.mean(0), dim=-1).cpu())
    return (torch.stack(logits) if logits else None), torch.stack(embeddings)


def person_vector(model, source: TrialSource, subject, device, windows=32, seed=7):
    """Person vector from the subject's own windows (any trial, labels unused)."""
    if model.person is None:
        return None
    state = source.rng
    source.rng = np.random.default_rng(seed)
    b = collate(source.windows_of(subject, windows)).to(device)
    source.rng = state
    with torch.no_grad():
        return model.person(b.eeg, b.xyz, b.valid, [len(b.eeg)])


def few_shot(embeddings, labels, shots, rounds=20, seed=0):
    """Nearest-prototype accuracy with ``shots`` support trials per item (rest are queries)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    items = np.unique(labels)
    if min((labels == k).sum() for k in items) <= shots:
        return float('nan')
    accuracies = []
    for _ in range(rounds):
        support = np.concatenate([rng.choice(np.flatnonzero(labels == k), shots, replace=False) for k in items])
        query = np.setdiff1d(np.arange(len(labels)), support)
        prototypes = torch.stack([F.normalize(embeddings[support[labels[support] == k]].mean(0), dim=-1) for k in items])
        predicted = items[(embeddings[query] @ prototypes.T).argmax(1).numpy()]
        accuracies.append(float((predicted == labels[query]).mean()))
    return float(np.mean(accuracies))


def item_scores(model, source: TrialSource, name, target, device, shots=(1, 5, 10), crops=5):
    """Per held-out subject accuracy on ``target``-modality trials: head (zero-shot) and few-shot prototypes."""
    rows = source.rows
    target_rows = rows[rows.modality_name == target]
    n_items = target_rows.item.nunique()
    chance = 1 / max(n_items, 1)
    results = {'zero_shot': {}, **{f'{k}_shot': {} for k in shots}}
    hits = total = 0
    for subject, part in target_rows.groupby('subject'):
        person = person_vector(model, source, subject, device)
        logits, embeddings = embed_trials(model, source, part.index, device, crops,
                                          head=name if name in model.heads else None, person=person)
        labels = part.item.to_numpy()
        if logits is not None:
            correct = (logits.argmax(1).numpy() == labels)
            results['zero_shot'][subject] = float(correct.mean())
            hits, total = hits + int(correct.sum()), total + len(correct)
        for k in shots:
            results[f'{k}_shot'][subject] = few_shot(embeddings, labels, k)
    report = {key: summarize(values, chance) for key, values in results.items() if values}
    report['per_subject'] = results
    report['trials'], report['items'] = int(len(target_rows)), int(n_items)
    if total:
        report['zero_shot']['p_binomial'] = binomial_p(hits, total, chance)
    return report

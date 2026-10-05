"""Imagined speech -> speech: personal CLIP encoders, a CLIP-conditioned mel diffusion, HiFi-GAN.

    python scripts/reconstruct.py targets [--datasets ...]              # macOS: synthesise the vocabulary
    python scripts/reconstruct.py decoder [--datasets ...]              # train each dataset's diffusion decoder
    python scripts/reconstruct.py check   [--datasets ...]              # decoder ceiling (oracle) and prior
    python scripts/reconstruct.py run --fold 0 1 2 3 4 [--datasets ...] # encoders, generation, measures, wavs
    python scripts/reconstruct.py report                                # pooled folds -> summary.json, index.html

Protocol: within person.  Fold k holds out the k-th contiguous fifth of every person's trials of every
modality (recording order); the person's other trials calibrate that person's encoder.  Nothing about
the held-out trials (labels, EEG statistics beyond the unlabelled alignment) is used before generation.

1 targets   every vocabulary item rendered with several voices and speaking rates (macOS say);
            SpeechT5 log-mel and HuBERT embeddings of every rendering (artifacts/audio/<dataset>.npz).
2 CLIP      anchors a_k = item speech embeddings (HuBERT, centred, PCA to K - 1 dims).  A person's
            encoder maps Euclidean-aligned full-band log-power to the CLIP space with InfoNCE in both
            directions (EEG <-> speech) and optionally supervised contrast across modalities.  Settings
            come from inner cross-validation on the training trials, whose held-out predictions are
            also the cross-fitted embeddings used to calibrate the posterior p(k | EEG).
3 decoder   mel diffusion conditioned on e = sum_k p_k a_k, trained on speech alone with synthetic
            posteriors p (Dirichlet; the spoken item is drawn from p): "say item k with probability
            p_k".  EEG never enters its training, so it cannot leak held-out trials.
4 generate  held-out trial -> p(k | EEG) -> e -> DDIM with classifier-free guidance -> log-mel ->
            HiFi-GAN -> 16 kHz speech.  The operating point (posterior sharpness, guidance scale) is
            chosen on cross-fitted training trials by how well the generated speech is identified.
5 measure   what the generated speech is heard as: Whisper forced choice among the vocabulary
            (listener), smallest DTW mel-cepstral distance to the items' renderings (mcd), nearest
            HuBERT item centroid (hubert); and the mel-cepstral distance to the true item.  Controls:
            another trial of the same person (wrong), no condition (prior) and the true item's anchor
            (oracle, the decoder's ceiling).  Wavs and an HTML page per fold and dataset for listening.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import html
import json
import math
from pathlib import Path
import re
import sys
import time
import unicodedata

import numpy as np
import pandas as pd
from scipy import stats
import soundfile as sf
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegspeech import ROOT, audio, clip, features                              # noqa: E402
from eegspeech.data import within_fold                                         # noqa: E402
from eegspeech.diffusion import EMA, MelDiffusion, MelScaler                   # noqa: E402
from eegspeech.metrics import summarize                                        # noqa: E402
from eegspeech.store import open_store                                         # noqa: E402

AUDIO = ROOT / 'artifacts' / 'audio'
OUT = ROOT / 'outputs' / 'reconstruct'
LEAD = 6                                   # frames of silence before each rendering (~0.1 s)
CONDITIONS = ('eeg', 'wrong', 'prior', 'oracle')
MEASURES = ('listener', 'mcd', 'hubert')


def device_of(name):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def slug(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()
    return re.sub(r'[^0-9A-Za-z]+', '_', text).strip('_')


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


# ----------------------------------------------------------------------------- 1 targets
def targets(name, spec, device):
    store = open_store(name)
    voices = audio.VOICES[store.language]
    jobs = [(k, voice, wpm, AUDIO / name / slug(item) / f'{slug(voice)}_{wpm}.wav', spec['text'][item])
            for k, item in enumerate(store.items) for voice in voices for wpm in audio.RATES_WPM]
    with ThreadPoolExecutor(8) as pool:                     # `say` runs as separate processes
        list(pool.map(lambda j: j[3].exists() or audio.synthesize(j[4], j[1], j[2], j[3]), jobs))
    waves = [audio.trim(audio.read(j[3])) for j in jobs]
    mels = [audio.log_mel(w) for w in waves]
    lengths = np.array([m.shape[1] for m in mels])
    frames = int(math.ceil((lengths.max() + LEAD + 4) / 8) * 8)
    hubert = audio.Hubert(device)(waves)                                          # R, layers, 768
    item = np.array([j[0] for j in jobs])
    voice = np.array([j[1] for j in jobs])
    fisher, held_voice = [], []
    for layer in range(hubert.shape[1]):
        h = hubert[:, layer]
        means = np.stack([h[item == k].mean(0) for k in range(len(store.items))])
        within = np.square(h - means[item]).sum()
        fisher.append(float(np.square(means[item] - h.mean(0)).sum() / within))
        hits = 0                                         # leave-one-voice-out nearest-centroid identification
        for v in np.unique(voice):
            rest = voice != v
            c = np.stack([h[rest & (item == k)].mean(0) for k in range(len(store.items))]) - h[rest].mean(0)
            q = h[~rest] - h[rest].mean(0)
            hits += ((q @ c.T / np.linalg.norm(c, axis=1)).argmax(1) == item[~rest]).sum()
        held_voice.append(hits / len(item))
    layer = int(np.argmax(np.array(held_voice) + 1e-3 * np.array(fisher) / max(fisher)))   # identification first
    np.savez(AUDIO / f'{name}.npz', items=np.array(store.items), item=item, voice=voice,
             wpm=np.array([j[2] for j in jobs]), mel=np.stack([audio.place(m, frames, LEAD) for m in mels]),
             length=lengths, hubert=hubert, layer=layer, fisher=np.array(fisher), language=store.language)
    log(f'{name}: {len(jobs)} renderings ({len(store.items)} items x {len(voices)} voices x {len(audio.RATES_WPM)} rates), '
        f'{lengths.min() / 62.5:.2f}-{lengths.max() / 62.5:.2f} s, canvas {frames} frames; HuBERT layer {layer} '
        f'(Fisher {fisher[layer]:.2f}; held-out-voice identification {held_voice[layer]:.3f}, '
        f'all layers {np.round(held_voice, 2).tolist()})')


class Targets:
    """Renderings of one dataset's vocabulary and the CLIP speech space built on them."""

    def __init__(self, name, spec):
        t = np.load(AUDIO / f'{name}.npz')
        self.items, self.item, self.wpm = list(t['items']), t['item'], t['wpm']
        self.mel, self.layer = t['mel'], int(t['layer'])
        self.language = str(t['language']) if 'language' in t.files else open_store(name).language
        self.texts = [spec['text'][i] for i in self.items]
        self.hubert = t['hubert'][:, self.layer]
        self.K = len(self.items)
        means = np.stack([self.hubert[self.item == k].mean(0) for k in range(self.K)])
        self.anchor = clip.anchors(means)
        self.centre = self.hubert.mean(0)
        centroids = np.stack([(self.hubert - self.centre)[self.item == k].mean(0) for k in range(self.K)])
        self.centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        middle = audio.RATES_WPM[len(audio.RATES_WPM) // 2]
        self.refs = [[audio.cepstra(self.mel[r]) for r in np.flatnonzero((self.item == k) & (self.wpm == middle))]
                     for k in range(self.K)]
        self.reference = [int(np.flatnonzero((self.item == k) & (self.wpm == middle))[0]) for k in range(self.K)]


class Judge:
    """What generated speech is heard as.  Scores (n, K), higher = more like item k:
    listener (Whisper log-likelihood of the item's text), mcd (minus the smallest DTW mel-cepstral
    distance to the item's renderings, dB) and hubert (cosine to the item's HuBERT centroid)."""

    def __init__(self, t: Targets, device, measures=MEASURES):
        self.t, self.vocoder = t, audio.Vocoder(device)
        self.listener = audio.Listener(t.language, device) if 'listener' in measures else None
        self.hubert = audio.Hubert(device) if 'hubert' in measures else None

    def __call__(self, mels, measures=MEASURES):
        waves = self.vocoder(mels)
        scores = {}
        if 'listener' in measures:
            scores['listener'] = self.listener.choose(list(waves), self.t.texts)
        if 'mcd' in measures:
            scores['mcd'] = -self.distances(mels)
        if 'hubert' in measures:
            h = self.hubert(waves)[:, self.t.layer] - self.t.centre
            scores['hubert'] = h @ self.t.centroids.T / np.linalg.norm(h, axis=1, keepdims=True).clip(1e-6)
        return waves, scores

    def distances(self, mels):
        """(n, K) smallest DTW mel-cepstral distance (dB) to each item's renderings at the middle rate."""
        cep = [audio.cepstra(m) for m in mels]
        refs = [(k, ref) for k in range(self.t.K) for ref in self.t.refs[k]]
        distance = audio.dtw_mcd([(c, ref) for c in cep for _, ref in refs]).reshape(len(cep), len(refs))
        owner = np.array([k for k, _ in refs])
        return np.stack([distance[:, owner == k].min(1) for k in range(self.t.K)], 1)


# ----------------------------------------------------------------------------- 3 decoder
def decoder(name, spec, cfg, device, seed=0):
    c = cfg['decoder']
    t = Targets(name, spec)
    mel = torch.from_numpy(t.mel)
    scaler = MelScaler.fit(mel)
    x_all = scaler.encode(mel)
    by_item = [np.flatnonzero(t.item == k) for k in range(t.K)]
    torch.manual_seed(seed)
    model = MelDiffusion(t.anchor.shape[1], frames=mel.shape[-1], hidden=c['hidden'], blocks=c['blocks']).to(device)
    ema = EMA(model, c['ema'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=.01)
    rng = np.random.default_rng(seed)
    anchor = torch.from_numpy(t.anchor)
    (OUT / 'decoders').mkdir(parents=True, exist_ok=True)
    started, running = time.time(), []
    for step in range(c['steps']):
        for group in optimizer.param_groups:
            group['lr'] = c['lr'] * min(1, (step + 1) / 500) * .5 * (1 + math.cos(math.pi * step / c['steps']))
        u = rng.random(c['batch'])
        null, oracle = u < c['null_fraction'], (u >= c['null_fraction']) & (u < c['null_fraction'] + c['oracle_fraction'])
        alpha = np.exp(rng.uniform(np.log(c['alpha'][0]), np.log(c['alpha'][1]), c['batch']))
        p = np.nan_to_num(np.stack([rng.dirichlet(np.full(t.K, a)) for a in alpha]))
        p[p.sum(1) < .5] = 1 / t.K                                                 # underflow of tiny alphas
        p[oracle] = np.eye(t.K)[rng.integers(0, t.K, oracle.sum())]
        p[null] = 1 / t.K
        k = np.array([rng.choice(t.K, p=q / q.sum()) for q in p])                 # the item to say, drawn from p
        r = np.array([rng.choice(by_item[i]) for i in k])
        shift = rng.integers(-3, 4, c['batch'])                                    # +-50 ms onset jitter
        x0 = torch.stack([torch.roll(x_all[i], int(s), -1) for i, s in zip(r, shift)])
        cond = torch.from_numpy(p.astype(np.float32)) @ anchor
        loss = model.loss(x0.to(device), cond.to(device), torch.from_numpy(null).to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        ema.update(model)
        running.append(float(loss.detach()))
        if (step + 1) % 200 == 0:
            log(json.dumps(dict(dataset=name, step=step + 1, loss=round(float(np.mean(running)), 4),
                                seconds=round(time.time() - started))))
            running = []
    path = OUT / 'decoders' / f'{name}.pt'
    torch.save(dict(config=ema.model.config, state=ema.model.state_dict(), scaler=scaler.state(), items=t.items,
                    anchor=t.anchor, layer=t.layer, steps=c['steps']), path)
    log(f'saved {path}')
    check_decoder(name, spec, cfg, device)


class Decoder:
    def __init__(self, name, t: Targets, device):
        path = OUT / 'decoders' / f'{name}.pt'
        payload = torch.load(path, map_location='cpu', weights_only=False)
        self.model = MelDiffusion(**payload['config'])
        try:
            self.model.load_state_dict(payload['state'])
        except RuntimeError as error:
            raise SystemExit(f'{path} is from another decoder version ({str(error)[:120]}...); '
                             f'retrain: python scripts/reconstruct.py decoder --datasets {name}')
        if payload['items'] != t.items or not np.allclose(payload['anchor'], t.anchor, atol=1e-5):
            raise SystemExit(f'{path} was trained on other targets; retrain it')
        self.model.to(device).eval()
        self.scaler = MelScaler(**payload['scaler'])
        self.device = device

    def mels(self, condition, null, guidance, steps, seed, batch=128):
        """Log-mels (n, 80, frames) for conditions (n, d); ``null`` rows unconditional."""
        out = []
        for i in range(0, len(condition), batch):
            generator = torch.Generator().manual_seed(seed + i)
            x = self.model.sample(torch.as_tensor(condition[i:i + batch], dtype=torch.float32, device=self.device),
                                  null=torch.as_tensor(null[i:i + batch], device=self.device), steps=steps,
                                  guidance=guidance, generator=generator)
            out.append(self.scaler.decode(x.cpu()))
        return torch.cat(out).numpy()


def check_decoder(name, spec, cfg, device, per_item=8):
    """The decoder's ceiling (oracle anchors, per guidance scale) and its prior (no condition)."""
    t = Targets(name, spec)
    d, judge, c = Decoder(name, t, device), Judge(t, device), cfg['decoder']
    items = np.repeat(np.arange(t.K), per_item)
    report = {}
    for guidance in c['guidance']:
        mels = d.mels(t.anchor[items], np.zeros(len(items), bool), guidance, c['sample_steps'], seed=0)
        _, scores = judge(mels)
        report[f'oracle_w{guidance:g}'] = {m: float((s.argmax(1) == items).mean()) for m, s in scores.items()}
        report[f'oracle_w{guidance:g}']['mcd_true_db'] = float(-scores['mcd'][np.arange(len(items)), items].mean())
    mels = d.mels(np.zeros((len(items), t.anchor.shape[1]), np.float32), np.ones(len(items), bool), 1.,
                  c['sample_steps'], seed=1)
    waves, scores = judge(mels, ('listener',))
    report['prior_heard_as'] = dict(zip(t.items, np.bincount(scores['listener'].argmax(1), minlength=t.K).tolist()))
    folder = OUT / 'decoders' / name
    folder.mkdir(parents=True, exist_ok=True)
    for i, w in enumerate(waves[::per_item]):
        sf.write(folder / f'prior_{i}.wav', w, audio.SAMPLE_RATE)
    guidance = c['guidance'][len(c['guidance']) // 2]
    for k, w in enumerate(judge.vocoder(d.mels(t.anchor, np.zeros(t.K, bool), guidance, c['sample_steps'], seed=2))):
        sf.write(folder / f'oracle_{slug(t.items[k])}.wav', w, audio.SAMPLE_RATE)
    json.dump(report, open(OUT / 'decoders' / f'{name}.json', 'w'), indent=1)
    log(f'{name} decoder check: {json.dumps(report)}')


# ----------------------------------------------------------------------------- 2 CLIP encoders
def encoders(name, spec, cfg, fold, t: Targets):
    """Inner-CV choice of the encoder settings, cross-fitted training embeddings and test embeddings."""
    store = open_store(name)
    feats = features.load(name)
    enc = cfg['encoder']
    item_map = np.array([t.items.index(i) for i in store.items])                  # store item -> anchor index
    table = store.table
    labelled = table[(table.item >= 0) & table.modality_name.isin(spec['modalities'])]
    held = set(within_fold(labelled, cfg['folds'], fold).tolist())
    target = spec['target']
    people = {}
    for s in store.subjects:
        if s not in feats:
            continue
        rows, x = feats[s]
        position = {int(r): i for i, r in enumerate(rows)}
        part = labelled[labelled.subject == s]
        train = part[~part.index.isin(held)]
        test = part[part.index.isin(held) & (part.modality_name == target)]
        if not len(test) or not (train.modality_name == target).any():
            continue
        inner = pd.Series(0, index=train.index)
        for _, group in train.groupby('modality_name'):
            for j, chunk in enumerate(np.array_split(group.index.to_numpy(), cfg['inner_folds'])):
                inner[chunk] = j
        people[s] = dict(x=x, position=position, train=train, inner=inner.to_numpy(), test=test,
                         is_target=(train.modality_name == target).to_numpy())
    subjects = sorted(people)

    def rows_x(s, rows):
        return people[s]['x'][[people[s]['position'][int(r)] for r in rows]]

    def persons(aux, j=None):
        out = []
        for s in subjects:
            p = people[s]
            keep = np.ones(len(p['train']), bool) if j is None else p['inner'] != j
            if aux == 0:
                keep &= p['is_target']
            rows = p['train'].index.to_numpy()[keep]
            out.append(clip.Person(rows_x(s, rows), item_map[p['train']['item'].to_numpy()[keep]],
                                   np.where(p['is_target'][keep], 1., aux)))
        return out

    fixed = dict(temperature=enc['temperature'], steps=enc['steps'], lr=enc['lr'])
    auxes = enc['aux_weight'] if len(spec['modalities']) > 1 else [0]
    grid = [(wd, sw, aux) for wd in enc['weight_decay'] for sw in enc['supcon_weight'] for aux in auxes]
    inner = {}
    for wd, sw, aux in grid:
        z, y, who, rows_all = [], [], [], []
        for j in range(cfg['inner_folds']):
            model = clip.train(persons(aux, j), t.anchor, weight_decay=wd, supcon_weight=sw, seed=j, **fixed)
            for i, s in enumerate(subjects):
                p = people[s]
                rows = p['train'].index.to_numpy()[(p['inner'] == j) & p['is_target']]
                if len(rows):
                    z.append(model.embed(i, rows_x(s, rows)))
                    y.append(item_map[p['train'].loc[rows, 'item'].to_numpy()])
                    who += [s] * len(rows)
                    rows_all.append(rows)
        z, y, who = np.concatenate(z), np.concatenate(y), np.array(who)
        hit = (z @ t.anchor.T).argmax(1) == y
        accuracy = float(np.mean([hit[who == s].mean() for s in subjects]))
        scale, nll = clip.calibrate(z, y, t.anchor)
        inner[(wd, sw, aux)] = dict(accuracy=accuracy, nll=nll, scale=scale, z=z, y=y, who=who,
                                    rows=np.concatenate(rows_all))
        log(f'{name} f{fold} inner CV wd={wd:g} supcon={sw:g} aux={aux:g}: accuracy {accuracy:.3f}, '
            f'log-loss {nll:.3f} (uniform {np.log(t.K):.3f}, scale {scale:.3g})')
    best = max(grid, key=lambda g: (round(inner[g]['accuracy'], 3), -inner[g]['nll']))
    chosen = inner[best]
    model = clip.train(persons(best[2]), t.anchor, weight_decay=best[0], supcon_weight=best[1], seed=0, **fixed)
    z_test, y_test, who_test, rows_test = [], [], [], []
    for i, s in enumerate(subjects):
        rows = people[s]['test'].index.to_numpy()
        z_test.append(model.embed(i, rows_x(s, rows)))
        y_test.append(item_map[people[s]['test']['item'].to_numpy()])
        who_test += [s] * len(rows)
        rows_test.append(rows)
    return dict(setting=dict(weight_decay=best[0], supcon_weight=best[1], aux_weight=best[2]),
                inner={f'wd={g[0]:g},supcon={g[1]:g},aux={g[2]:g}': dict(accuracy=v['accuracy'], nll=v['nll'], scale=v['scale'])
                       for g, v in inner.items()},
                scale=chosen['scale'], train=dict(z=chosen['z'], y=chosen['y'], who=chosen['who'], rows=chosen['rows']),
                test=dict(z=np.concatenate(z_test), y=np.concatenate(y_test), who=np.array(who_test),
                          rows=np.concatenate(rows_test)))


# ----------------------------------------------------------------------------- 4-5 run
def derangement(who, rng):
    """For each trial, another trial of the same person (a random permutation without fixed points)."""
    out = np.arange(len(who))
    for s in np.unique(who):
        idx = rng.permutation(np.flatnonzero(who == s))
        if len(idx) > 1:
            out[idx] = np.roll(idx, 1)
    return out


def operating_point(name, cfg, fold, t, e, d, judge):
    """Posterior sharpness x guidance scale chosen on cross-fitted training trials (never held-out ones)."""
    c = cfg['decoder']
    rng = np.random.default_rng(fold)
    tr = e['train']
    pick = np.sort(rng.choice(len(tr['y']), min(len(tr['y']), c['select_trials']), replace=False))
    scores = {}
    for sharpness in c['sharpness']:
        condition = clip.posterior(tr['z'][pick], t.anchor, e['scale'] * sharpness) @ t.anchor
        for guidance in c['guidance']:
            mels = d.mels(condition.astype(np.float32), np.zeros(len(pick), bool), guidance, c['sample_steps'],
                          seed=10_000 + fold)
            _, s = judge(mels, c['select_by'])
            scores[(sharpness, guidance)] = float(np.mean([(s[m].argmax(1) == tr['y'][pick]).mean() for m in c['select_by']]))
            log(f'{name} f{fold}: sharpness {sharpness:g} guidance {guidance:g} -> identified '
                f'{scores[(sharpness, guidance)]:.3f} ({"+".join(c["select_by"])}, cross-fitted training trials)')
    return max(scores, key=scores.get), scores


def listen_selection(trials, per_subject):
    """Up to ``per_subject`` trials per person, cycling through the items in recording order."""
    chosen = []
    for _, g in trials.groupby('subject'):
        order = g.groupby('item').cumcount()
        chosen += g.assign(order=order).sort_values(['order', 'row']).head(per_subject).index.tolist()
    return np.array(sorted(chosen), int)


def run(name, spec, cfg, fold, device):
    t = Targets(name, spec)
    d, judge, c = Decoder(name, t, device), Judge(t, device), cfg['decoder']
    folder = OUT / f'f{fold}' / name
    (folder / 'audio').mkdir(parents=True, exist_ok=True)
    started = time.time()
    e = encoders(name, spec, cfg, fold, t)
    log(f'{name} f{fold}: encoder settings {e["setting"]} ({time.time() - started:.0f} s)')
    (sharpness, guidance), choice = operating_point(name, cfg, fold, t, e, d, judge)
    log(f'{name} f{fold}: operating point sharpness {sharpness:g}, guidance {guidance:g}')

    te = e['test']
    n = len(te['y'])
    calibrated = clip.posterior(te['z'], t.anchor, e['scale'])
    condition = clip.posterior(te['z'], t.anchor, e['scale'] * sharpness) @ t.anchor
    other = derangement(te['who'], np.random.default_rng(1000 + fold))
    conditions = dict(eeg=condition, wrong=condition[other], prior=np.zeros_like(condition), oracle=t.anchor[te['y']])
    trials = pd.DataFrame(dict(subject=te['who'], row=te['rows'], item=te['y'], decoded=calibrated.argmax(1),
                               p_true=calibrated[np.arange(n), te['y']], wrong_item=te['y'][other]))
    listen = listen_selection(trials, cfg['listen_trials'])
    kept = {}
    for kind, value in conditions.items():
        mels = d.mels(value.astype(np.float32), np.full(n, kind == 'prior'), guidance, c['sample_steps'],
                      seed=fold * 1_000_000)
        waves, scores = judge(mels)
        for m, s in scores.items():
            trials[f'{kind}_{m}'] = s.argmax(1)
        trials[f'{kind}_mcd_true'] = -scores['mcd'][np.arange(n), te['y']]
        kept[kind] = waves[listen]
        log(f'{name} f{fold}: {kind:6s} identified as the true item '
            + ', '.join(f'{m} {np.mean(trials[f"{kind}_{m}"] == trials["item"]):.3f}' for m in MEASURES)
            + f'; MCD to the true item {trials[f"{kind}_mcd_true"].mean():.2f} dB')
    trials['item_name'] = [t.items[k] for k in trials['item']]
    trials['heard'] = ''
    trials.loc[listen, 'heard'] = judge.listener.transcribe(list(kept['eeg']))
    trials.to_csv(folder / 'trials.csv', index=False)
    np.savez(folder / 'encoder.npz', z_test=te['z'], p_test=calibrated, who_test=te['who'], rows_test=te['rows'],
             y_test=te['y'], z_train=e['train']['z'], y_train=e['train']['y'], who_train=e['train']['who'],
             rows_train=e['train']['rows'], anchor=t.anchor, scale=e['scale'])
    summary = dict(dataset=name, fold=fold, trials=n, items=t.K, chance=1 / t.K, encoder=e['setting'], inner=e['inner'],
                   scale=e['scale'], sharpness=sharpness, guidance=guidance,
                   operating_points={f'sharpness={k[0]:g},guidance={k[1]:g}': v for k, v in choice.items()},
                   results=measure(trials, t.K), seconds=round(time.time() - started))
    json.dump(summary, open(folder / 'summary.json', 'w'), indent=1)
    listen_page(name, fold, folder, trials, listen, kept, t, judge, summary)
    log(f'{name} f{fold}: done in {time.time() - started:.0f} s -> {folder / "listen.html"}')


def measure(trials, K):
    """Per-person identification accuracy of every condition and measure (vs chance), the paired
    advantage of the trial's own EEG over another trial's EEG, and the MCD to the true item."""
    per = trials.groupby('subject')
    accuracy = lambda column: {s: float((g[column] == g['item']).mean()) for s, g in per}
    out = dict(encoder=summarize(accuracy('decoded'), 1 / K))
    for kind in CONDITIONS:
        out[kind] = {m: summarize(accuracy(f'{kind}_{m}'), 1 / K) for m in MEASURES if f'{kind}_{m}' in trials}
        out[kind]['mcd_true_db'] = float(per[f'{kind}_mcd_true'].mean().mean())
    for m in MEASURES:
        if f'eeg_{m}' in trials:
            own, wrong = accuracy(f'eeg_{m}'), accuracy(f'wrong_{m}')
            diff = np.array([own[s] - wrong[s] for s in own])
            test = dict(mean_difference=float(diff.mean()), people_better=int((diff > 0).sum()), people=len(diff))
            if len(diff) >= 5 and np.any(diff != 0):
                test['p_wilcoxon'] = float(stats.wilcoxon(diff, alternative='greater', zero_method='zsplit').pvalue)
            out['eeg'][m]['vs_wrong'] = test
    return out


def player(path):
    return f'<audio controls preload="none" src="{html.escape(path)}"></audio>'


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>
body{{font:14px/1.45 system-ui,sans-serif;margin:16px;max-width:1400px}} table{{border-collapse:collapse;margin:8px 0}}
td,th{{border:1px solid #ccc;padding:4px 6px;vertical-align:top;text-align:left}} audio{{width:170px;height:30px}}
.num{{text-align:right}} .muted{{color:#666}}</style></head><body>{body}</body></html>"""


def listen_page(name, fold, folder, trials, listen, kept, t, judge, summary):
    """Reference, reconstruction and controls for a few trials per person, with the fold's numbers."""
    for k, w in enumerate(judge.vocoder(t.mel[t.reference])):
        sf.write(folder / 'audio' / f'ref_{slug(t.items[k])}.wav', w, audio.SAMPLE_RATE)
    rows = []
    for j, i in enumerate(listen):
        r = trials.loc[i]
        stem = f'{slug(r["subject"])}_{int(r["row"])}'
        for kind in CONDITIONS:
            sf.write(folder / 'audio' / f'{stem}_{kind}.wav', kept[kind][j], audio.SAMPLE_RATE)
        true = t.items[r['item']]
        rows.append(f'<tr><td>{html.escape(r["subject"])}</td><td>{int(r["row"])}</td>'
                    f'<td><b>{html.escape(true)}</b><br>{player(f"audio/ref_{slug(true)}.wav")}</td>'
                    f'<td>{html.escape(t.items[r["decoded"]])} <span class="muted">({r["p_true"]:.2f})</span></td>'
                    f'<td>{html.escape(t.items[r["eeg_listener"]])}<br><span class="muted">"{html.escape(r["heard"])}"</span></td>'
                    + ''.join(f'<td>{player(f"audio/{stem}_{kind}.wav")}</td>' for kind in CONDITIONS) + '</tr>')
    body = (f'<h2>{html.escape(name)}, fold {fold}: imagined speech reconstructed from EEG</h2>'
            + numbers_table({f'fold {fold}': summary['results']}, summary['chance'])
            + f'<p class="muted">Operating point: posterior sharpness {summary["sharpness"]:g}, guidance {summary["guidance"]:g}. '
              'Columns: the true item and its synthetic reference (vocoded); the encoder\'s decision and its probability for '
              'the true item; what Whisper picks among the vocabulary for the EEG reconstruction, and its free transcript; '
              'reconstructions from this trial\'s EEG, from another trial of the same person (wrong), without a condition '
              '(prior) and from the true item\'s anchor (oracle: the decoder\'s ceiling).</p>'
            + '<table><tr><th>person</th><th>trial</th><th>true item</th><th>decoded (p true)</th><th>heard as</th>'
              '<th>EEG</th><th>wrong trial</th><th>prior</th><th>oracle</th></tr>' + ''.join(rows) + '</table>')
    (folder / 'listen.html').write_text(PAGE.format(title=f'{html.escape(name)} fold {fold}', body=body))


def numbers_table(results, chance):
    """Identification accuracy per condition (mean over people) for each entry of ``results``."""
    head = ('<tr><th></th><th>encoder</th>' + ''.join(f'<th>{k}: listener / MCD / HuBERT</th>' for k in CONDITIONS)
            + '<th>EEG vs wrong (listener)</th><th>MCD to true item, EEG / oracle</th></tr>')
    lines = []
    for label, r in results.items():
        cells = [f'{r["encoder"].get("mean", float("nan")):.3f}']
        for kind in CONDITIONS:
            cells.append(' / '.join(f'{r[kind][m].get("mean", float("nan")):.3f}' for m in MEASURES if m in r[kind]))
        vs = r['eeg'].get('listener', {}).get('vs_wrong', {})
        cells.append(f'{vs.get("mean_difference", float("nan")):+.3f} (p {vs.get("p_wilcoxon", float("nan")):.3g})')
        cells.append(f'{r["eeg"]["mcd_true_db"]:.2f} / {r["oracle"]["mcd_true_db"]:.2f} dB')
        lines.append(f'<tr><td>{html.escape(label)}</td>' + ''.join(f'<td class="num">{c}</td>' for c in cells) + '</tr>')
    return (f'<p>Identification of the generated speech among the vocabulary, mean over people (chance {chance:.3f}).</p>'
            f'<table>{head}{"".join(lines)}</table>')


# ----------------------------------------------------------------------------- report
def report(names, cfg):
    out, sections = {}, []
    for name in names:
        parts = sorted(OUT.glob(f'f*/{name}/trials.csv'))
        if not parts:
            continue
        trials = pd.concat([pd.read_csv(p, keep_default_na=False).assign(fold=int(p.parts[-3][1:])) for p in parts],
                           ignore_index=True)
        K = len(Targets(name, cfg['datasets'][name]).items)
        folds = sorted(trials.fold.unique().tolist())
        out[name] = dict(folds=folds, trials=int(len(trials)), chance=1 / K, results=measure(trials, K))
        r = out[name]['results']
        line = f'{name:18s} folds {folds} chance {1 / K:.3f} | encoder {r["encoder"]["mean"]:.3f}'
        for kind in CONDITIONS:
            line += f' | {kind} ' + '/'.join(f'{r[kind][m]["mean"]:.3f}' for m in MEASURES if m in r[kind])
        print(line + f' | MCD eeg {r["eeg"]["mcd_true_db"]:.2f} dB')
        links = ' '.join(f'<a href="f{f}/{name}/listen.html">fold {f}</a>' for f in folds)
        sections.append(f'<h3>{html.escape(name)}</h3>' + numbers_table({'all folds': r}, 1 / K) + f'<p>Listen: {links}</p>')
    json.dump(out, open(OUT / 'summary.json', 'w'), indent=1)
    body = ('<h2>Imagined speech reconstructed from EEG (within person, held-out fifths)</h2>' + ''.join(sections))
    (OUT / 'index.html').write_text(PAGE.format(title='Reconstruction', body=body))
    print('identification listener/MCD/HuBERT per condition ->', OUT / 'summary.json', OUT / 'index.html')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['targets', 'decoder', 'check', 'run', 'report'])
    parser.add_argument('--datasets', nargs='*')
    parser.add_argument('--fold', type=int, nargs='*', default=[0])
    parser.add_argument('--config', default=str(ROOT / 'configs' / 'plan.yaml'))
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))['reconstruct']
    names = [n for n in cfg['datasets'] if not args.datasets or n in args.datasets]
    device = device_of(args.device)
    if args.stage == 'report':
        return report(names, cfg)
    for name in names:
        spec = cfg['datasets'][name]
        if args.stage == 'targets':
            targets(name, spec, device)
        elif args.stage == 'decoder':
            decoder(name, spec, cfg, device)
        elif args.stage == 'check':
            check_decoder(name, spec, cfg, device)
        else:
            for fold in args.fold:
                run(name, spec, cfg, fold, device)


if __name__ == '__main__':
    main()

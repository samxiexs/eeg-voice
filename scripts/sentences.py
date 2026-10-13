"""Imagined sentences -> speech in the person's voice, reconstructed through a CLIP speech space (Chisco).

    python scripts/sentences.py targets [--datasets chisco]     # macOS: every imagined sentence in 6 synthetic voices
    python scripts/sentences.py run --fold 0 1 2 3 4             # linear encoders, renderers, reconstruction, measures
    python scripts/sentences.py run --fold 0 1 2 3 4 --encoder deep      # the deep encoder (outputs/sentences_deep)
    python scripts/sentences.py report [--encoder deep]          # pooled folds -> summary.json, index.html
    python scripts/sentences.py run --fold 0 --modalities read --out outputs/sentences_read   # positive control:
                                                                 # the reading epochs (the sentence on screen)

The speech is reconstructed, not chosen: EEG -> a continuous embedding z in the CLIP speech space ->
a diffusion renderer that has never heard the sentence -> speech.  No step of the generation picks a
sentence; candidates only enter the measures.

Protocol: within person.  Fold k holds out the k-th fifth of every person's recording runs, whole and
contiguous in run order: a run presents its sentences in topic blocks (the next trial shares the topic 20 %
of the time, 3 % if shuffled), so trial-level splits would let time stand in for content (the published
Chisco results use random 8:2 trial splits).  A sentence imagined in a held-out run is dropped from
training, whoever imagined it: neither the encoders nor the fold's renderer ever see it.  Inner folds do
the same over the training runs.  The alignment never sees held-out runs, the drift baseline of a trial
only uses the trials before it.

1 targets   every sentence imagined in the dataset, spoken by 6 synthetic Mandarin voices (macOS say); per
            clip the SpeechT5 log-mel, the Mandarin HuBERT embedding of the layer that best identifies a
            sentence across voices, and the voiced duration.
2 space     anchors a_s: HuBERT embeddings averaged over voices, centred, PCA to ``space.dims``, whitened, plus
            the log duration, unit length.  Synthetic speech only, no EEG.  Fixed, as the renderer lives in it.
3 encoder   EEG -> z.  linear (default): per person, aligned filter-bank log-power of the imagery epoch
            (3.3 s), whole and its time course in 3 or 6 parts, PCA, then ridge or symmetric InfoNCE between
            a run's trials and its sentences (CLIP); settings by inner cross-validation, the best 3 averaged.
            deep: the same chain learned end to end (``eegspeech.deep``).
4 renderer  per fold, a mel diffusion model conditioned on a point of the space (noised during training,
            since EEG embeddings are far from any anchor) and on voice traits, trained on the speech of the
            fold's training sentences only.
5 speech    held-out trial -> z / |z| + the person's voice (a synthetic voice of the person's sex) -> DDIM ->
            log-mel -> SpeechT5 HiFi-GAN.
6 measure   of the generated speech: its HuBERT embedding projected into the space ranks the sentences of
            the run (rank percentile of the true one, chance 0.5; top-1/10), its duration against the true
            sentence's, Whisper's transcript (character accuracy, closest candidate), sex and pitch.  Controls,
            rendered the same way: another trial half a run away (wrong), no content (prior), the true
            sentence's anchor (oracle: the renderer's ceiling on sentences it never heard).  The encoder's
            own ranking of the run's sentences by z . a_s is reported as a diagnostic, with a per-person
            circular-shift test (each run's EEG sequence shifted against its sentences).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import itertools
import json
import math
from pathlib import Path
import shutil
import sys
import time

import h5py
import numpy as np
import pandas as pd
from scipy import stats
import soundfile as sf
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegspeech import ROOT, audio, clip, deep, features                        # noqa: E402
from eegspeech.diffusion import EMA, MelDiffusion, MelScaler                   # noqa: E402
from eegspeech.metrics import summarize                                        # noqa: E402
from eegspeech.store import open_store                                         # noqa: E402

AUDIO = ROOT / 'artifacts' / 'audio'
OUT = ROOT / 'outputs' / 'sentences'
RENDERERS = ROOT / 'outputs' / 'sentence_renderers'    # one per dataset and fold, shared by the encoders
CHUNK = 250                                # sentences per resumable part of the targets
LEAD = 6                                   # frames of silence before each clip on the renderer's canvas
FEMALE_F0 = 145.
KINDS = ('eeg', 'wrong', 'prior', 'oracle')


def device_of(name):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


# ----------------------------------------------------------------------------- 1 targets
def targets(name, cfg, device):
    """Every imagined sentence in every voice: log-mels, HuBERT embeddings of one layer, durations; resumable
    in parts of ``CHUNK`` sentences, merged into ``artifacts/audio/<name>_sentences.h5``."""
    store = open_store(name)
    texts = sorted(set(store.table.loc[store.table.modality_name == 'imagine', 'text']))
    voices = audio.VOICES[store.language]
    folder = AUDIO / name
    folder.mkdir(parents=True, exist_ok=True)
    order = np.random.default_rng(0).permutation(len(texts))            # part 0 is a random sample
    chunks = np.array_split(order, math.ceil(len(texts) / CHUNK))
    hubert = audio.Hubert(device, audio.MODELS / cfg['speech_model'])
    for c, chunk in enumerate(chunks):
        part = folder / f'part_{c:03d}.npz'
        if part.exists():
            continue
        started = time.time()
        tmp = folder / 'tmp'
        jobs = [(int(i), v, tmp / f'{i}_{v}.wav') for i in chunk for v in range(len(voices))]
        with ThreadPoolExecutor(8) as pool:                              # `say` runs as separate processes
            list(pool.map(lambda j: j[2].exists() or audio.synthesize(texts[j[0]], voices[j[1]], None, j[2]), jobs))
        waves = [audio.level(audio.trim(audio.read(path))) for *_, path in jobs]
        mels = [audio.log_mel(w) for w in waves]
        h = hubert(waves)                                                # clips, layers, 768
        extra = {}
        if c == 0:                                                       # the content layer and the voices' traits
            layer = choose_layer(h, np.array([j[0] for j in jobs]), np.array([j[1] for j in jobs]))
            json.dump(dict(layer=layer, model=cfg['speech_model']), open(folder / 'layer.json', 'w'))
            extra = dict(xvector=audio.Speaker(device)(waves), f0=np.array([audio.pitch(w) for w in waves]))
        layer = json.load(open(folder / 'layer.json'))['layer']
        np.savez(part, sentence=np.array([j[0] for j in jobs]), voice=np.array([j[1] for j in jobs]),
                 hubert=h[:, layer].astype(np.float16), duration=np.array([len(w) / audio.SAMPLE_RATE for w in waves]),
                 length=np.array([m.shape[1] for m in mels]), mel=np.concatenate(mels, 1).astype(np.float16), **extra)
        shutil.rmtree(tmp)
        log(f'{name}: part {c + 1}/{len(chunks)} ({len(jobs)} clips, {time.time() - started:.0f} s)')
    merge_targets(name, texts, voices)


def choose_layer(hidden, sentence, voice):
    """The HuBERT layer whose voice-averaged sentence means best identify held-out voices' clips."""
    scores = []
    for layer in range(hidden.shape[1]):
        h = hidden[:, layer].astype(np.float64)
        hits = 0
        for v in np.unique(voice):
            rest = voice != v
            names = np.unique(sentence[rest])
            centre = h[rest].mean(0)
            means = np.stack([h[rest & (sentence == s)].mean(0) for s in names]) - centre
            q = h[~rest] - centre
            hits += (names[(q @ means.T / np.linalg.norm(means, axis=1)).argmax(1)] == sentence[~rest]).sum()
        scores.append(hits / len(h))
    log('held-out-voice sentence identification per layer: ' + ' '.join(f'{s:.2f}' for s in scores))
    return int(np.argmax(scores))


def merge_targets(name, texts, voices):
    parts = sorted((AUDIO / name).glob('part_*.npz'))
    data = [dict(np.load(p)) for p in parts]
    length = np.concatenate([d['length'] for d in data])
    path = AUDIO / f'{name}_sentences.h5'
    with h5py.File(path.with_suffix('.tmp'), 'w') as f:
        f.attrs['voices'] = json.dumps(voices)
        f.attrs.update(json.load(open(AUDIO / name / 'layer.json')))
        f.create_dataset('texts', data=texts, dtype=h5py.string_dtype())
        for key in ('sentence', 'voice', 'hubert', 'duration'):
            f.create_dataset(key, data=np.concatenate([d[key] for d in data]))
        f.create_dataset('offset', data=np.concatenate([[0], np.cumsum(length)]))
        f.create_dataset('mel', data=np.concatenate([d['mel'] for d in data], 1), chunks=(80, 4096))
        first = data[0]
        f.create_dataset('voice_xvector', data=np.stack([first['xvector'][first['voice'] == v].mean(0)
                                                         for v in range(len(voices))]))
        f.create_dataset('voice_f0', data=np.array([np.nanmedian(first['f0'][first['voice'] == v])
                                                    for v in range(len(voices))]))
    path.with_suffix('.tmp').replace(path)
    log(f'{name}: {len(length)} clips of {len(texts)} sentences in {len(voices)} voices -> {path}')


class Targets:
    """The sentence speech of a dataset: the CLIP space, each sentence's log-mel in each voice, voice traits."""

    def __init__(self, name):
        self.file = h5py.File(AUDIO / f'{name}_sentences.h5', 'r')
        f = self.file
        self.texts = list(f['texts'].asstr()[:])
        self.index = {t: i for i, t in enumerate(self.texts)}
        self.voices = json.loads(f.attrs['voices'])
        self.layer = int(f.attrs['layer'])
        self.model = str(f.attrs['model'])
        self.sentence, self.voice = f['sentence'][:], f['voice'][:]
        self.clip_of = {(int(s), int(v)): i for i, (s, v) in enumerate(zip(self.sentence, self.voice))}
        self.offset = f['offset'][:]
        self.duration = f['duration'][:]
        x = f['voice_xvector'][:]
        self.voice_xvector = x / np.linalg.norm(x, axis=1, keepdims=True)
        self.voice_f0 = f['voice_f0'][:]
        self.voice_sex = np.array([audio.sex_of(v) for v in self.voices])
        self.traits = np.concatenate([self.voice_xvector, ((np.log(self.voice_f0) - np.log(150.)) / .5)[:, None]],
                                     1).astype(np.float32)
        K = len(self.texts)
        hubert = f['hubert'][:].astype(np.float64)
        count = np.bincount(self.sentence, minlength=K)[:, None]
        self.embedding = np.zeros((K, hubert.shape[1]))
        np.add.at(self.embedding, self.sentence, hubert)
        self.embedding /= np.maximum(count, 1)
        self.log_duration = np.zeros(K)
        np.add.at(self.log_duration, self.sentence, np.log(self.duration))
        self.log_duration /= np.maximum(count[:, 0], 1)
        self.centre = self.embedding.mean(0)
        _, self.singular, self.basis = np.linalg.svd(self.embedding - self.centre, full_matrices=False)
        self._anchors, self._mel = {}, None

    def project(self, embedding, log_duration, dims, duration_weight=1.):
        """(n, dims + 1) unit points of the space for HuBERT embeddings (n, 768) and log durations (n,)."""
        a = (np.asarray(embedding, np.float64) - self.centre) @ self.basis[:dims].T / (self.singular[:dims] / np.sqrt(len(self.embedding)))
        d = (np.asarray(log_duration, np.float64) - self.log_duration.mean()) / self.log_duration.std()
        a = np.concatenate([a, duration_weight * d[:, None]], 1)
        return (a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-9)).astype(np.float32)

    def anchor(self, dims, duration_weight=1.):
        """(K, dims + 1) unit anchors: whitened PCA of the voice-averaged HuBERT embeddings + log duration."""
        key = (dims, duration_weight)
        if key not in self._anchors:
            self._anchors[key] = self.project(self.embedding, self.log_duration, dims, duration_weight)
        return self._anchors[key]

    def clip_mel(self, i):
        """Log-mel (80, frames) of clip i (the whole mel array is read once)."""
        if self._mel is None:
            self._mel = self.file['mel'][:]
        return self._mel[:, self.offset[i]:self.offset[i + 1]].astype(np.float32)

    def mel(self, sentence, voice):
        return self.clip_mel(self.clip_of[(int(sentence), int(voice))])


def person_voices(t: Targets, sexes, subjects):
    """A synthetic voice of each person's sex, different voices in turn (F/M taken from the dataset)."""
    out, used = {}, {'F': 0, 'M': 0}
    for s in sorted(subjects):
        sex = sexes.get(s, 'F' if len(out) % 2 == 0 else 'M')
        pool = [v for v in range(len(t.voices)) if t.voice_sex[v] == sex] or list(range(len(t.voices)))
        out[s] = pool[used[sex] % len(pool)]
        used[sex] += 1
    return out


# ----------------------------------------------------------------------------- 3 encoders: EEG -> z
def split(name, spec, cfg, fold, t: Targets):
    """The trials with a target sentence, the fold's held-out rows, and the rows training may use: not held
    out and not a sentence imagined in the held-out runs (by anyone), so every test sentence is unseen."""
    table = features.labelled(open_store(name), spec['modalities'])
    table = table[table.text.isin(t.index)]
    held = set(features.within_fold(table, cfg['folds'], fold).tolist())
    out = table.index.isin(held)
    trainable = pd.Series(~out & ~table.text.isin(set(table.text[out])).to_numpy(), index=table.index)
    return table, held, trainable


def encoders(name, spec, cfg, fold, t: Targets):
    """Linear encoders: inner-CV choice of the settings, then per person and held-out run the trials'
    embeddings z (the best settings' calibrated embeddings averaged) and the encoder's scores of the run's
    sentences.  Features: the whole-trial log-power (drift-corrected) and, with ``time`` k, its time course
    in k parts; standardised and, above ``components`` features, projected on the training trials'
    principal directions."""
    enc = cfg['encoder']
    anchor = t.anchor(cfg['space']['dims'], cfg['space']['duration_weight'])
    table, held, trainable = split(name, spec, cfg, fold, t)
    feats, bands = features.load(name, held=held, align_by=spec.get('align_by', 'subject'),
                                 modalities=spec['modalities'], windows=enc['windows'])
    space = enc['search']
    windows = {w for stage in space for w in stage.get('recenter', [0])}
    people = {}
    for s in sorted(feats):
        rows, x, parts = feats[s]
        keep = np.isin(rows, table.index)
        rows, x, parts = rows[keep], x[keep], parts[keep]
        part = table.loc[rows]
        out = np.isin(rows, list(held))
        if not out.any() or out.all():
            continue
        modality = part.modality_name.to_numpy()
        means = {m: x[(modality == m) & ~out].mean(0) for m in np.unique(modality)}
        run = (part.session.astype(str) + '/' + part.modality_name).to_numpy()
        views = {0: x}
        for w in windows - {0}:
            views[w] = x - features.local_baseline(x, run, out, np.stack([means[m] for m in modality]), w)
        train = trainable.loc[rows].to_numpy()
        inner = np.full(len(rows), -1)
        inner[train] = features.blocks(part[train], cfg['inner_folds'])
        people[s] = dict(rows=rows, views=views, whole=x, windows=parts, test=out, train=train, inner=inner,
                         sentence=np.array([t.index[text] for text in part.text]), run=part.session.to_numpy(),
                         text=part.text.to_numpy(),
                         columns={hz: features.band_columns(bands, x.shape[1], hz)
                                  for hz in {hz for stage in space for hz in stage.get('min_band_hz', [0])}})
    subjects = sorted(people)

    def design(p, index, setting):
        columns = p['columns'][setting['min_band_hz']]
        x = p['views'][setting['recenter']][index][:, columns]
        k = setting.get('time', 0)
        if not k:
            return x
        return np.concatenate([x, features.time_course(p['whole'][index][:, columns],
                                                       p['windows'][index][:, :, columns], k)], 1)

    reducers = {}

    def fit(s, key, mask, setting):
        """(reducer, encoder) of person s on the masked trials; the reducer (standardisation, PCA) depends
        only on the trials and the features, so settings that differ elsewhere share it."""
        p = people[s]
        index = np.flatnonzero(mask)
        features_key = (s, key, setting.get('time', 0), setting['min_band_hz'], setting['recenter'])
        if features_key not in reducers:
            if len(reducers) > 64:
                reducers.clear()
            x = design(p, index, setting)
            reducer = clip.Reducer.fit(x, enc['components'])
            reducers[features_key] = (reducer, reducer(x))
        reducer, x = reducers[features_key]
        return reducer, clip.train_sentences(x, p['sentence'][index], p['run'][index], anchor,
                                             method=setting['method'], weight_decay=setting['weight_decay'],
                                             steps=enc['steps'], standardize=False)

    def embed(s, fitted, mask, setting):
        reducer, model = fitted
        index = np.flatnonzero(mask)
        return index, model.embed(reducer(design(people[s], index, setting)))

    def by_run(p, index, z):
        """[(rows, candidates, positions, scores, z)] per run, the run's sentences as candidates."""
        return [(index[trials], candidates, position, z[trials] @ anchor[candidates].T, z[trials])
                for trials, candidates, position in clip.run_sets(p['sentence'][index], p['run'][index])]

    def evaluate(setting):
        sets = {s: [] for s in subjects}
        for j in range(cfg['inner_folds']):
            for s in subjects:
                p = people[s]
                ask = p['inner'] == j
                if not ask.any():
                    continue
                fit_mask = p['train'] & (p['inner'] != j) & ~np.isin(p['text'], p['text'][ask])
                sets[s] += by_run(p, *embed(s, fit(s, j, fit_mask, setting), ask, setting))
        percentile = float(np.mean([np.mean(np.concatenate([clip.rank_percentile(S, pos) for _, _, pos, S, _ in sets[s]]))
                                    for s in subjects]))
        flat = [x for s in subjects for x in sets[s]]
        scale, nll = clip.calibrate_sets([x[3] for x in flat], [x[2] for x in flat])
        log(f'{name} f{fold} inner CV {label(setting)}: rank percentile {percentile:.4f}, log-loss {nll:.4f}')
        return dict(percentile=percentile, nll=nll, scale=scale)

    label = lambda g: ','.join(f'{k}={v}' for k, v in g.items())
    results = {}
    best = {k: v[0] for stage in space for k, v in stage.items()}
    for stage in space:                                                  # coordinate search
        for values in itertools.product(*stage.values()):
            setting = dict(best, **dict(zip(stage, values)))
            if label(setting) not in results:
                results[label(setting)] = (setting, evaluate(setting))
        best = max(results.values(), key=lambda r: r[1]['percentile'])[0]
    top = sorted(results.values(), key=lambda r: r[1]['percentile'], reverse=True)[:enc['ensemble']]
    test = {}
    for s in subjects:                                                   # calibrated embeddings, averaged
        p = people[s]
        z = 0
        for setting, r in top:
            index, zr = embed(s, fit(s, 'all', p['train'], setting), p['test'], setting)
            z = z + r['scale'] * zr / len(top)
        test[s] = [dict(rows=p['rows'][rows], run=int(p['run'][rows[0]]), candidates=c, position=pos,
                        scores=torch.log_softmax(torch.as_tensor(S), -1).numpy(), z=zz)
                   for rows, c, pos, S, zz in by_run(p, index, z)]
    return dict(setting=[setting for setting, _ in top], test=test, anchor=anchor,
                inner={key: dict(percentile=r['percentile'], nll=r['nll']) for key, (_, r) in results.items()})


def deep_encoders(name, spec, cfg, fold, t: Targets, device):
    """The deep encoder (``eegspeech.deep``): one model for every person of the fold (shared filter bank,
    personal spatial filters and readout), early-stopped on each person's last inner block of training
    runs, whose sentences are kept out of the fitted trials too."""
    c = cfg['deep']
    if len(spec['modalities']) != 1:
        raise SystemExit('the deep encoder takes one modality')
    table, held, trainable = split(name, spec, cfg, fold, t)
    cached, data = deep.epochs(name, spec['modalities'][0], c['rate'])
    table, trainable = table[table.index.isin(cached)], trainable[trainable.index.isin(cached)]
    out = table.index.isin(held)
    subjects = sorted(s for s, g in table.groupby('subject')
                      if out[(table.subject == s).to_numpy()].any() and trainable[g.index].any())
    position = pd.Series(np.arange(len(cached)), index=cached)
    person, sentence, run = (np.full(len(cached), -1) for _ in range(3))
    at = position[table.index].to_numpy()
    person[at] = table.subject.map({s: i for i, s in enumerate(subjects)}).fillna(-1).to_numpy()
    sentence[at] = [t.index[text] for text in table.text]
    run[at] = table.session.to_numpy()
    validation = np.zeros(len(table), bool)
    fit_rows = trainable.to_numpy() & (person[at] >= 0)
    inner = features.blocks(table[fit_rows], cfg['inner_folds'])
    validation[np.flatnonzero(fit_rows)[inner == cfg['inner_folds'] - 1]] = True
    fit_rows &= ~validation & ~table.text.isin(set(table.text[validation])).to_numpy()
    store = open_store(name)
    matrices = []
    for s in subjects:                                    # alignment from the person's training-period epochs
        valid = store.file['subjects'][s]['valid'][:].mean(0) > .5
        mine = at[(table.subject == s).to_numpy() & ~out]
        matrices.append(deep.alignment(data, mine, np.pad(valid, (0, data.shape[1] - len(valid)))))
    anchor = t.anchor(cfg['space']['dims'], cfg['space']['duration_weight'])
    model, examples, scale, best = deep.fit(data, matrices, person, sentence, run, anchor, at[fit_rows],
                                            at[validation], c, device, lambda m: log(f'{name} f{fold} {m}'),
                                            seed=c['seed'] + fold)
    test = {}
    for s in subjects:
        mine = at[(table.subject == s).to_numpy() & out]
        z = deep.embed(model, examples, mine)
        lookup = {int(p): i for i, p in enumerate(mine)}
        test[s] = [dict(rows=cached[pos], run=int(run[pos[0]]), candidates=cand, position=true,
                        scores=torch.log_softmax(torch.as_tensor(S), -1).numpy(),
                        z=z[[lookup[int(p)] for p in pos]])
                   for pos, cand, true, S in deep.run_scores(z, mine, sentence, run, anchor, scale)]
    setting = {k: c[k] for k in ('rate', 'crop', 'filters', 'depth', 'kernel', 'pool', 'stride', 'dropout')}
    return dict(setting=[dict(setting, encoder='deep')], test=test, anchor=anchor,
                inner=dict(validation_percentile=best, fitted_trials=int(fit_rows.sum()),
                           validation_trials=int(validation.sum())))


# ----------------------------------------------------------------------------- 4 renderer
class Renderer:
    """A fold's open-vocabulary mel diffusion renderer: p(mel | point of the speech space, voice traits)."""

    def __init__(self, payload, t: Targets, device):
        self.model = MelDiffusion(**payload['config'])
        self.model.load_state_dict(payload['state'])
        self.model.to(device).eval()
        self.scaler = MelScaler(**payload['scaler'])
        self.traits, self.device = t.traits, device

    def mels(self, condition, voices, null, guidance, steps, seed, batch=64):
        """Log-mels (n, 80, frames) for unit ``condition`` points in ``voices``; ``null`` rows: no content."""
        condition = np.asarray(condition, np.float32)
        out = []
        for i in range(0, len(condition), batch):
            as_tensor = lambda x, **kw: torch.as_tensor(x[i:i + batch], device=self.device, **kw)
            x = self.model.sample(as_tensor(condition, dtype=torch.float32), traits=as_tensor(self.traits[voices]),
                                  null=as_tensor(null), steps=steps, guidance=guidance,
                                  generator=torch.Generator().manual_seed(seed + i))
            out.append(self.scaler.decode(x.cpu()))
        return torch.cat(out).numpy()


def renderer(name, cfg, fold, t: Targets, sentences, device):
    """The fold's renderer, trained on the speech (every voice) of ``sentences`` only, the fold's training
    sentences; cached in ``outputs/sentence_renderers`` for every encoder of the fold.

    The condition of a clip is its sentence's anchor plus Gaussian noise of a random size up to
    ``noise`` (renormalised), so that EEG embeddings, far from any anchor, stay in the renderer's input
    distribution; a ``null_fraction`` of clips gets no content (classifier-free guidance)."""
    c, space = cfg['renderer'], cfg['space']
    anchor = t.anchor(space['dims'], space['duration_weight'])
    sentences = np.unique(sentences)
    key = hashlib.sha1(sentences.tobytes() + json.dumps([space, c], sort_keys=True).encode()).hexdigest()[:12]
    path = RENDERERS / f'{name}_f{fold}.pt'
    if path.exists():
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload.get('key') == key:
            return Renderer(payload, t, device)
    clips = np.flatnonzero(np.isin(t.sentence, sentences))
    rng = np.random.default_rng(fold)
    canvas = lambda i, lead=LEAD: audio.place(t.clip_mel(i), c['frames'], lead)
    scaler = MelScaler.fit(torch.from_numpy(np.stack([canvas(i) for i in rng.choice(clips, 2000)])))
    torch.manual_seed(fold)
    model = MelDiffusion(anchor.shape[1], frames=c['frames'], hidden=c['hidden'], blocks=c['blocks'],
                         trait_dim=t.traits.shape[1]).to(device)
    ema = EMA(model, c['ema'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=.01)
    traits = torch.from_numpy(t.traits)
    started, running = time.time(), []
    log(f'{name} f{fold}: renderer on {len(sentences)} training sentences ({len(clips)} clips)')
    for step in range(c['steps']):
        for group in optimizer.param_groups:
            group['lr'] = c['lr'] * min(1, (step + 1) / 500) * .5 * (1 + math.cos(math.pi * step / c['steps']))
        pick = rng.choice(clips, c['batch'])
        x0 = scaler.encode(torch.from_numpy(np.stack([canvas(i, LEAD + int(rng.integers(-3, 4))) for i in pick])))
        a = anchor[t.sentence[pick]]
        sigma = rng.uniform(0, c['noise'], (len(pick), 1))
        noisy = a + sigma * rng.standard_normal(a.shape) / math.sqrt(a.shape[1])
        cond = torch.from_numpy((noisy / np.linalg.norm(noisy, axis=1, keepdims=True)).astype(np.float32))
        null = torch.from_numpy(rng.random(len(pick)) < c['null_fraction'])
        loss = model.loss(x0.to(device), cond.to(device), null.to(device), traits[t.voice[pick]].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        ema.update(model)
        running.append(float(loss.detach()))
        if (step + 1) % 500 == 0:
            log(f'{name} f{fold} renderer step {step + 1}: loss {np.mean(running):.4f} ({time.time() - started:.0f} s)')
            running = []
    payload = dict(key=key, config=ema.model.config, state=ema.model.state_dict(), scaler=scaler.state(),
                   sentences=sentences, steps=c['steps'])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return Renderer(payload, t, device)


# ----------------------------------------------------------------------------- 5-6 speech, measures
def circular_test(score_sets, permutations, rng, guard=10):
    """One-sided p of the mean rank percentile against circular shifts of each run's EEG sequence."""
    observed = np.mean(np.concatenate([clip.rank_percentile(S, pos) for S, pos in score_sets]))
    ranks = [(np.argsort(np.argsort(-S, 1), 1), pos) for S, pos in score_sets]            # rank 0 = best
    null = np.zeros(permutations)
    for k in range(permutations):
        total, count = 0., 0
        for R, pos in ranks:
            n, C = R.shape
            shift = rng.integers(min(guard, n - 1), max(n - guard, min(guard, n - 1) + 1)) if n > 1 else 0
            r = R[(np.arange(n) + shift) % n, pos]
            total += (1 - r / max(C - 1, 1)).sum()
            count += n
        null[k] = total / count
    return float(observed), float((1 + (null >= observed).sum()) / (1 + permutations))


def topics(name, t: Targets):
    """Sentence (row of the targets) -> topic (store item index) of the dataset's imagined trials."""
    table = open_store(name).table
    rows = table[(table.modality_name == 'imagine') & (table.item >= 0) & table.text.isin(t.index)]
    return {t.index[text]: int(item) for text, item in zip(rows.text, rows.item)}


def reconstruct(trials, candidates, conditions, t: Targets, render: Renderer, cfg, device, folder):
    """Generate the speech of ``trials`` (a subset) for every kind of condition and judge it.

    ``candidates[(person, run)]``: every sentence of the person's held-out run (not only those of the subset).

    ``conditions[kind]``: (n, dims + 1) unit points (None for the null content of the prior).  Judges: the
    HuBERT embedding and voiced duration of the generated speech, projected into the space, rank the
    sentences of the trial's run (rank percentile, top-1/10 of the true one); duration against the true
    sentence's in the same voice; Whisper (the first ``transcribe_trials`` per person): transcript, its
    character accuracy and closest candidate; F0 (sex) and the nearest voice.  The first
    ``listen_trials`` per person are kept as wav, with the true sentence's synthetic reference."""
    c, space = cfg['renderer'], cfg['space']
    anchor = t.anchor(space['dims'], space['duration_weight'])
    vocoder, hubert = audio.Vocoder(device), audio.Hubert(device, audio.MODELS / t.model)
    listener, speaker = audio.Listener('zh', device), audio.Speaker(device)
    order = trials.groupby('subject').cumcount().to_numpy()
    transcribed, shown = order < cfg['transcribe_trials'], order < cfg['listen_trials']
    out = pd.DataFrame(dict(row=trials.row.to_numpy()))
    voices = trials.voice.to_numpy()
    for kind in KINDS:
        cond = conditions[kind]
        null = np.full(len(trials), cond is None)
        mels = render.mels(np.zeros((len(trials), anchor.shape[1]), np.float32) if cond is None else cond, voices,
                           null, c['guidance'], c['sample_steps'], seed=31 * len(trials))
        waves = list(vocoder(mels))
        lengths = np.array([len(audio.trim(w)) / audio.SAMPLE_RATE for w in waves]).clip(.05)
        point = t.project(hubert(waves)[:, t.layer], np.log(lengths), space['dims'], space['duration_weight'])
        rank, top1, top10 = [], [], []
        for p, s, r, k in zip(point, trials.subject, trials.run, trials.sentence):
            cand = candidates[(s, r)]
            scores = anchor[cand] @ p
            true = scores[list(cand).index(k)]
            above = (scores > true).sum()
            rank.append(1 - above / max(len(cand) - 1, 1))
            top1.append(above == 0)
            top10.append(above < 10)
        out[f'{kind}_rank'], out[f'{kind}_top1'], out[f'{kind}_top10'] = rank, top1, top10
        out[f'{kind}_duration'] = lengths
        out[f'{kind}_f0'] = [audio.pitch(w) for w in waves]
        out[f'{kind}_speaker'] = (speaker(waves) @ t.voice_xvector.T).argmax(1)
        heard = listener.transcribe([w for w, keep in zip(waves, transcribed) if keep])
        out[f'{kind}_heard'] = None
        out.loc[transcribed, f'{kind}_heard'] = heard
        sub = trials[transcribed]
        out.loc[transcribed, f'{kind}_char_accuracy'] = [1 - audio.cer(h, x) for h, x in zip(heard, sub.true_text)]
        out.loc[transcribed, f'{kind}_heard_as'] = [min(candidates[(s, r)], key=lambda q: audio.cer(h, t.texts[q]))
                                                     for h, s, r in zip(heard, sub.subject, sub.run)]
        for row, w, keep in zip(trials.row, waves, shown):
            if keep:
                sf.write(folder / 'audio' / f'{row}_{kind}.wav', w, audio.SAMPLE_RATE)
        if device.type == 'mps':
            torch.mps.empty_cache()
        log(f'{kind:6s} reconstruction ({len(waves)} trials, {np.mean([len(candidates[(s, r)]) for s, r in zip(trials.subject, trials.run)]):.0f} '
            f'candidates on average): rank percentile of the true sentence '
            f'{np.mean(rank):.4f}, top-10 {np.mean(top10):.3f}; Whisper heard the true sentence '
            f'{np.mean(out.loc[transcribed, f"{kind}_heard_as"].to_numpy() == sub.sentence.to_numpy()):.3f}')
    for row, k, v, keep in zip(trials.row, trials.sentence, voices, shown):            # the reference to compare with
        if keep:
            sf.write(folder / 'audio' / f'{row}_reference.wav', vocoder(t.mel(k, v)[None])[0], audio.SAMPLE_RATE)
    return out


def run(name, spec, cfg, fold, device, encoder='linear'):
    t = Targets(name)
    folder = OUT / f'f{fold}' / name
    (folder / 'audio').mkdir(parents=True, exist_ok=True)
    started = time.time()
    e = deep_encoders(name, spec, cfg, fold, t, device) if encoder == 'deep' else encoders(name, spec, cfg, fold, t)
    log(f'{name} f{fold}: encoder settings {e["setting"]} ({time.time() - started:.0f} s)')
    voice_of = person_voices(t, spec.get('sexes', {}), e['test'])
    topic = topics(name, t)
    rng = np.random.default_rng(fold)
    lines, embeddings, tests = [], {}, {}
    for s, sets in e['test'].items():
        tests[s] = circular_test([(r['scores'], r['position']) for r in sets], cfg['permutations'], rng)
        for r in sets:
            S, position, candidates, z = r['scores'], r['position'], r['candidates'], r['z']
            n = len(S)
            order = np.argsort(-S, 1)
            rank = clip.rank_percentile(S, position)
            wrong = (np.arange(n) + n // 2) % n                       # another trial half a run away
            kinds = np.array([topic.get(int(c), -1) for c in candidates])
            same_topic = kinds[position][:, None] == kinds[None]
            for i in range(n):
                row = int(r['rows'][i])
                embeddings[row] = (z[i], z[wrong[i]])
                lines.append(dict(subject=s, row=row, run=r['run'], candidates=len(candidates),
                                  sentence=int(candidates[position[i]]), encoder_top=int(candidates[order[i, 0]]),
                                  encoder_percentile=float(rank[i]),
                                  encoder_top10=bool(position[i] in order[i, :10]),
                                  encoder_wrong_percentile=float(clip.rank_percentile(S[wrong[i]][None],
                                                                                      position[i:i + 1])[0]),
                                  topic_chance=float(same_topic[i].mean()), voice=voice_of[s]))
    trials = pd.DataFrame(lines)
    trials['voice_sex'] = t.voice_sex[trials.voice.to_numpy()]
    trials['encoder_topic'] = [topic.get(a) == topic.get(b) for a, b in zip(trials.encoder_top, trials.sentence)]
    trials['true_text'] = [t.texts[k] for k in trials.sentence]
    trials['true_duration'] = [t.duration[t.clip_of[(int(k), int(v))]] for k, v in zip(trials.sentence, trials.voice)]
    held_sentences = set(trials.sentence)
    render = renderer(name, cfg, fold, t, [k for k in range(len(t.texts)) if k not in held_sentences], device)
    chosen = np.concatenate([g.index[np.unique(np.linspace(0, len(g) - 1, cfg['render_trials']).round().astype(int))]
                             for _, g in trials.groupby('subject')])
    sub = trials.loc[chosen]
    unit = lambda z: (z / np.linalg.norm(z, axis=1, keepdims=True).clip(1e-9)).astype(np.float32)
    conditions = dict(eeg=unit(np.stack([embeddings[r][0] for r in sub.row])),
                      wrong=unit(np.stack([embeddings[r][1] for r in sub.row])), prior=None,
                      oracle=e['anchor'][sub.sentence.to_numpy()])
    candidates = {key: g.sentence.unique() for key, g in trials.groupby(['subject', 'run'])}     # whole runs
    assert all(len(candidates[(s, r)]) == n for s, r, n in zip(sub.subject, sub.run, sub.candidates))
    trials = trials.merge(reconstruct(sub, candidates, conditions, t, render, cfg, device, folder), on='row', how='left')
    trials.to_csv(folder / 'trials.csv', index=False)
    summary = dict(dataset=name, fold=fold, encoder_kind=encoder, trials=len(trials), rendered=len(sub),
                   settings=e['setting'], inner=e['inner'], voices={s: t.voices[v] for s, v in voice_of.items()},
                   circular_shift_test={s: dict(percentile=v[0], p=v[1]) for s, v in tests.items()},
                   results=measure(trials), seconds=round(time.time() - started))
    json.dump(summary, open(folder / 'summary.json', 'w'), indent=1, default=float)
    listen_page(name, f'fold {fold}', folder, trials, t, summary['results'])
    r = summary['results']['reconstruction']
    log(f'{name} f{fold}: reconstructed speech, rank percentile of the true sentence ' +
        ', '.join(f'{k} {r[k]["percentile"].get("mean", float("nan")):.4f}' for k in KINDS) +
        f'; encoder {summary["results"]["encoder"]["percentile"]["mean"]:.4f}')


def measure(trials):
    """Per person, then summarised over people.  reconstruction: the measures of the generated speech for
    every condition, and the paired advantage of the trial's own EEG over another trial's (per person:
    mean difference of the rank percentile; across people: Wilcoxon).  encoder: the encoder's own ranking
    of the run's sentences (a diagnostic; it produces no speech)."""
    mean = lambda frame, column: {s: float(g[column].mean()) for s, g in frame.groupby('subject') if g[column].notna().any()}
    chance = lambda k, frame: float((np.minimum(k, frame.candidates) / frame.candidates).mean())
    out = dict(encoder=dict(percentile=summarize(mean(trials, 'encoder_percentile'), .5),
                            wrong_percentile=summarize(mean(trials, 'encoder_wrong_percentile'), .5),
                            top10=summarize(mean(trials, 'encoder_top10'), chance(10, trials)),
                            topic=summarize(mean(trials, 'encoder_topic'), float(trials.topic_chance.mean()))))
    rendered = trials[trials.eeg_rank.notna()]
    reconstruction = {}
    for kind in KINDS:
        f0 = rendered[f'{kind}_f0']
        sex = (np.where(f0 >= FEMALE_F0, 'F', 'M') == rendered.voice_sex)[f0.notna().to_numpy()]
        heard = rendered[rendered[f'{kind}_heard_as'].notna()]
        reconstruction[kind] = dict(
            percentile=summarize(mean(rendered, f'{kind}_rank'), .5),
            top1=summarize(mean(rendered, f'{kind}_top1'), chance(1, rendered)),
            top10=summarize(mean(rendered, f'{kind}_top10'), chance(10, rendered)),
            duration_r=summarize({s: float(np.corrcoef(g[f'{kind}_duration'], g.true_duration)[0, 1])
                                  for s, g in rendered.groupby('subject')}, 0.),
            char_accuracy=float(heard[f'{kind}_char_accuracy'].mean()),
            heard_as_true=float((heard[f'{kind}_heard_as'] == heard.sentence).mean()),
            sex=float(sex.mean()) if len(sex) else float('nan'), voice=float((rendered[f'{kind}_speaker'] == rendered.voice).mean()),
            trials=len(rendered))
    diff = {s: float((g.eeg_rank - g.wrong_rank).mean()) for s, g in rendered.groupby('subject')}
    values = np.array(list(diff.values()))
    reconstruction['eeg_vs_wrong'] = dict(per_person=diff, mean_difference=float(values.mean()),
                                          people_better=int((values > 0).sum()), people=len(values))
    if len(values) >= 5 and np.any(values != 0):
        reconstruction['eeg_vs_wrong']['p_wilcoxon'] = float(stats.wilcoxon(values, alternative='greater').pvalue)
    out['reconstruction'] = reconstruction
    return out


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>
body{{font:14px/1.45 system-ui,sans-serif;margin:16px;max-width:1500px}} table{{border-collapse:collapse;margin:8px 0}}
td,th{{border:1px solid #ccc;padding:4px 6px;vertical-align:top;text-align:left}} audio{{width:160px;height:30px}}
.muted{{color:#666}} .num{{text-align:right}}</style></head><body>{body}</body></html>"""


def numbers(results):
    """Tables of the reconstructed speech (per condition) and of the encoder's diagnostic ranking."""
    r = results['reconstruction']
    cell = lambda v, f='.4f': f'{v.get("mean", float("nan")):{f}}' if isinstance(v, dict) else f'{v:{f}}'
    rows = ''.join(f'<tr><td>{k}</td><td class="num">{cell(r[k]["percentile"])}</td>'
                   f'<td class="num">{r[k]["percentile"].get("above_chance", "")}/{r[k]["percentile"].get("subjects", "")}</td>'
                   f'<td class="num">{cell(r[k]["top10"], ".3f")} ({r[k]["top10"].get("chance", float("nan")):.3f})</td>'
                   f'<td class="num">{cell(r[k]["top1"], ".4f")} ({r[k]["top1"].get("chance", float("nan")):.4f})</td>'
                   f'<td class="num">{cell(r[k]["duration_r"], ".3f")}</td><td class="num">{r[k]["heard_as_true"]:.3f}</td>'
                   f'<td class="num">{r[k]["char_accuracy"]:.3f}</td><td class="num">{r[k]["sex"]:.2f}</td></tr>'
                   for k in KINDS)
    vs = r['eeg_vs_wrong']
    e = results['encoder']
    return ('<table><tr><th>reconstructed speech</th><th>rank percentile of the true sentence (chance 0.5)</th>'
            '<th>people above 0.5</th><th>top-10 (chance)</th><th>top-1 (chance)</th><th>duration r with the true '
            'sentence</th><th>Whisper: heard as the true sentence</th><th>Whisper: character accuracy</th>'
            f'<th>intended sex</th></tr>{rows}</table>'
            f'<p>Own EEG vs another trial\'s (rank percentile of the generated speech): {vs["mean_difference"]:+.4f}, '
            f'{vs["people_better"]}/{vs["people"]} people better, Wilcoxon p {vs.get("p_wilcoxon", float("nan")):.3g}.</p>'
            f'<p class="muted">Encoder diagnostic (z ranks the run\'s sentences; no speech): rank percentile '
            f'{e["percentile"].get("mean", float("nan")):.4f} (another trial {e["wrong_percentile"].get("mean", float("nan")):.4f}), '
            f'top-10 {e["top10"].get("mean", float("nan")):.3f} (chance {e["top10"].get("chance", float("nan")):.3f}), '
            f'topic {e["topic"].get("mean", float("nan")):.3f} (chance {e["topic"].get("chance", float("nan")):.3f}).</p>')


def player(path):
    return f'<audio controls preload="none" src="{html.escape(path)}"></audio>'


def listen_page(name, label, folder, trials, t: Targets, results, prefix=''):
    rows = []
    for _, r in trials[trials.eeg_rank.notna()].iterrows():
        if not (folder / 'audio' / f'{int(r.row)}_eeg.wav').exists():
            continue
        cell = lambda kind: (f'<td>{player(f"{prefix}audio/{int(r.row)}_{kind}.wav")}<br><span class="muted">'
                             f'rank {r[f"{kind}_rank"]:.2f}, {r[f"{kind}_duration"]:.1f} s; heard: '
                             f'{html.escape(str(r[f"{kind}_heard"]))}</span></td>')
        rows.append(f'<tr><td>{html.escape(r.subject)}<br><span class="muted">{html.escape(t.voices[int(r.voice)])}, '
                    f'run {int(r.run)}</span></td><td>{html.escape(r.true_text)}<br>'
                    f'{player(f"{prefix}audio/{int(r.row)}_reference.wav")}<br><span class="muted">{r.true_duration:.1f} s</span></td>'
                    + ''.join(cell(kind) for kind in KINDS) + '</tr>')
    body = (f'<h2>{html.escape(name)}, {html.escape(label)}: imagined sentences reconstructed from EEG</h2>'
            '<p>Every clip except the reference is generated by the fold\'s diffusion renderer, which never heard the '
            'held-out sentences: from the EEG embedding of the trial (EEG), of another trial (wrong), from no content '
            '(prior) and from the true sentence\'s point of the speech space (oracle, the renderer\'s ceiling). '
            '"rank": where the generated speech puts the true sentence among the run\'s sentences (1 best, 0.5 chance).</p>'
            + numbers(results)
            + '<table><tr><th>person</th><th>true sentence (synthetic reference)</th><th>from EEG</th><th>wrong trial</th>'
              '<th>prior</th><th>oracle</th></tr>' + ''.join(rows) + '</table>')
    (folder / 'listen.html').write_text(PAGE.format(title=f'{html.escape(name)} {html.escape(label)}', body=body))


# ----------------------------------------------------------------------------- report
def report(names, cfg):
    out, sections = {}, []
    for name in names:
        parts = sorted(OUT.glob(f'f*/{name}/trials.csv'))
        if not parts:
            continue
        trials = pd.concat([pd.read_csv(p, keep_default_na=False, na_values=['']).assign(fold=int(p.parts[-3][1:]))
                            for p in parts], ignore_index=True)
        folds = sorted(trials.fold.unique().tolist())
        tests = {f: json.load(open(OUT / f'f{f}' / name / 'summary.json'))['circular_shift_test'] for f in folds}
        out[name] = dict(folds=folds, trials=int(len(trials)), rendered=int(trials.eeg_rank.notna().sum()),
                         results=measure(trials), circular_shift_test=tests)
        r = out[name]['results']
        print(f'{name}: folds {folds} | reconstructed speech, rank percentile of the true sentence: ' +
              ', '.join(f'{k} {r["reconstruction"][k]["percentile"].get("mean", float("nan")):.4f}' for k in KINDS) +
              f' | encoder {r["encoder"]["percentile"]["mean"]:.4f}')
        links = ' '.join(f'<a href="f{f}/{name}/listen.html">fold {f}</a>' for f in folds)
        sections.append(f'<h3>{html.escape(name)}</h3>' + numbers(r) + f'<p>Listen: {links}</p>')
    json.dump(out, open(OUT / 'summary.json', 'w'), indent=1, default=float)
    (OUT / 'index.html').write_text(PAGE.format(title='Sentences', body='<h2>Imagined sentences reconstructed from EEG '
                                                '(within person, held-out runs)</h2>' + ''.join(sections)))
    print('->', OUT / 'summary.json', OUT / 'index.html')


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['targets', 'run', 'report'])
    parser.add_argument('--datasets', nargs='*')
    parser.add_argument('--fold', type=int, nargs='*', default=[0])
    parser.add_argument('--config', default=str(ROOT / 'configs' / 'plan.yaml'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--encoder', choices=['linear', 'deep'], default='linear')
    parser.add_argument('--out', help='default: outputs/sentences (linear), outputs/sentences_deep (deep)')
    parser.add_argument('--modalities', nargs='*', help="instead of the configured ones, e.g. 'read' (positive control)")
    args = parser.parse_args()
    OUT = Path(args.out) if args.out else OUT.with_name('sentences' + ('_deep' if args.encoder == 'deep' else ''))
    cfg = yaml.safe_load(open(args.config))['sentences']
    if args.modalities:
        for spec in cfg['datasets'].values():
            spec['modalities'] = args.modalities
    names = [n for n in cfg['datasets'] if not args.datasets or n in args.datasets]
    if args.stage == 'report':
        return report(names, cfg)
    device = device_of(args.device)
    for name in names:
        if args.stage == 'targets':
            targets(name, cfg, device)
        else:
            for fold in args.fold:
                run(name, cfg['datasets'][name], cfg, fold, device, args.encoder)


if __name__ == '__main__':
    main()

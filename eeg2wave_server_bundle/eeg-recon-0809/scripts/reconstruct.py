"""Imagined speech -> speech in the person's voice: the content comes from EEG, the voice from the person.

    python scripts/reconstruct.py targets [--datasets ...]              # item speech: synthetic and own voices
    python scripts/reconstruct.py decoder [--datasets ...]              # train each dataset's diffusion decoder
    python scripts/reconstruct.py check   [--datasets ...]              # decoder ceiling per voice
    python scripts/reconstruct.py run --fold 0 1 2 3 4 [--datasets ...] # encoders, generation, measures, wavs
    python scripts/reconstruct.py report                                # pooled folds -> summary.json, index.html

Protocol: within person.  Fold k holds out the k-th contiguous fifth of every person's trials of every
modality (recording order); the person's other trials calibrate that person's encoder.  Nothing about
the held-out trials (labels, EEG statistics beyond the unlabelled alignment) is used before generation.

1 targets   every vocabulary item spoken by several synthetic voices at three rates (macOS say) and,
            where the dataset recorded it (KaraOne), by every participant; per clip the SpeechT5 log-mel,
            HuBERT embeddings, a WavLM speaker embedding and the median F0.
2 content   CLIP anchors a_k = item speech embeddings (HuBERT of the synthetic clips, centred, PCA to
            K - 1 dims).  A person's encoder maps Euclidean-aligned full-band log-power to the CLIP space
            (InfoNCE EEG <-> speech, optionally supervised contrast across modalities); settings come from
            inner cross-validation, whose held-out predictions calibrate the posterior p(k | EEG).
3 voice     the traits of a voice are its mean speaker embedding (timbre, sex) and its median log F0
            (pitch).  A person speaks in their own recorded voice (KaraOne) or, where neither voice nor sex
            was recorded (BCI2020, Thinking Out Loud), in a synthetic voice assigned to them.  The traits
            come from the person's speech, never from an identity input or from EEG.
4 decoder   mel diffusion conditioned on the content e = sum_k p_k a_k (whitened) and on voice traits,
            trained on speech alone: synthetic posteriors p (Dirichlet; the spoken item is drawn from p),
            each clip with its own voice's traits.  Classifier-free guidance acts on the content only.
5 generate  held-out trial -> content from EEG + the person's voice -> DDIM -> log-mel -> HiFi-GAN.  The
            operating point (posterior sharpness, guidance) is chosen on cross-fitted training trials.
6 measure   content: Whisper forced choice among the vocabulary (listener), smallest DTW mel-cepstral
            distance to the person's voice saying each item (mcd), nearest HuBERT item centroid; voice:
            nearest voice by speaker embedding, sex implied by the F0, pitch error.  Controls: another
            trial's EEG (wrong), no content (prior), the true item (oracle, the decoder's ceiling); prior
            and oracle do not depend on the trial and are generated for a subset of trials.
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
LEAD = 6                                   # frames of silence before each clip (~0.1 s)
CONDITIONS = ('eeg', 'wrong', 'prior', 'oracle')
MEASURES = ('listener', 'mcd', 'hubert')
FEMALE_F0 = 145.                           # Hz: our male voices sit at 86-125 Hz, female ones at 163-276 Hz


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
    clips = [(k, voice, wpm, audio.trim(audio.read(path))) for k, voice, wpm, path, _ in jobs]
    if spec.get('voices') == 'own':                         # each participant's own recordings
        for done in sorted((AUDIO / name / 'persons').glob('*/labels.txt')):
            labels = [line.strip() for line in done.read_text().splitlines() if line.strip()]
            for i, label in enumerate(labels):
                wav = done.parent / f'{i}.wav'
                if wav.exists() and label in store.items:   # room recordings: trim more tightly
                    clips.append((store.items.index(label), f'person:{done.parent.name}', 0,
                                  audio.trim(audio.read(wav), threshold_db=-25.)))
    waves = [audio.level(w) for *_, w in clips]
    mels = [audio.log_mel(w) for w in waves]
    lengths = np.array([m.shape[1] for m in mels])
    frames = int(math.ceil((min(lengths.max(), 150) + LEAD + 4) / 8) * 8)
    item = np.array([c[0] for c in clips])
    voice = np.array([c[1] for c in clips])
    wpm = np.array([c[2] for c in clips])
    log(f'{name}: {len(clips)} clips ({int((wpm > 0).sum())} synthetic, {int((wpm == 0).sum())} own voices); '
        f'HuBERT, speaker embeddings and F0 ...')
    hubert = audio.Hubert(device)(waves)                                          # clips, layers, 768
    xvector = audio.Speaker(device)(waves)                                        # clips, 512
    f0 = np.array([audio.pitch(w) for w in waves])
    synthetic = wpm > 0
    fisher, held_voice = [], []
    for layer in range(hubert.shape[1]):                     # the content layer, chosen on the synthetic clips
        h, it, vo = hubert[synthetic, layer], item[synthetic], voice[synthetic]
        means = np.stack([h[it == k].mean(0) for k in range(len(store.items))])
        fisher.append(float(np.square(means[it] - h.mean(0)).sum() / np.square(h - means[it]).sum()))
        hits = 0                                             # leave-one-voice-out nearest-centroid identification
        for v in np.unique(vo):
            rest = vo != v
            c = np.stack([h[rest & (it == k)].mean(0) for k in range(len(store.items))]) - h[rest].mean(0)
            q = h[~rest] - h[rest].mean(0)
            hits += ((q @ c.T / np.linalg.norm(c, axis=1)).argmax(1) == it[~rest]).sum()
        held_voice.append(hits / len(it))
    layer = int(np.argmax(np.array(held_voice) + 1e-3 * np.array(fisher) / max(fisher)))   # identification first
    np.savez(AUDIO / f'{name}.npz', items=np.array(store.items), item=item, voice=voice, wpm=wpm,
             mel=np.stack([audio.place(m, frames, LEAD) for m in mels]), length=lengths, hubert=hubert, layer=layer,
             fisher=np.array(fisher), language=store.language, xvector=xvector, f0=f0)
    log(f'{name}: {len(clips)} clips, {lengths.min() / 62.5:.2f}-{lengths.max() / 62.5:.2f} s, canvas {frames} '
        f'frames; HuBERT layer {layer} (held-out-voice identification {held_voice[layer]:.3f})')


class Targets:
    """Item speech of one dataset: the CLIP content space and the voices it can be spoken in."""

    def __init__(self, name, spec):
        t = np.load(AUDIO / f'{name}.npz')
        if 'xvector' not in t.files:
            raise SystemExit(f'{name}: targets predate the voice traits; rerun: python scripts/reconstruct.py targets')
        self.items, self.item, self.wpm = list(t['items']), t['item'], t['wpm']
        self.mel, self.layer = t['mel'], int(t['layer'])
        self.language = str(t['language'])
        self.texts = [spec['text'][i] for i in self.items]
        self.K = len(self.items)
        synthetic = self.wpm > 0
        hubert = t['hubert'][:, self.layer]
        means = np.stack([hubert[synthetic & (self.item == k)].mean(0) for k in range(self.K)])
        self.anchor = clip.anchors(means)
        self.centre = hubert[synthetic].mean(0)
        centroids = np.stack([(hubert - self.centre)[synthetic & (self.item == k)].mean(0) for k in range(self.K)])
        self.centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        # voices: identity (mean speaker embedding), pitch (median F0) and sex; their traits condition the decoder
        self.voices = sorted(set(t['voice']))
        self.voice = np.array([self.voices.index(v) for v in t['voice']])
        x = np.stack([t['xvector'][self.voice == v].mean(0) for v in range(len(self.voices))])
        self.voice_xvector = x / np.linalg.norm(x, axis=1, keepdims=True)
        self.voice_f0 = np.array([np.nanmedian(t['f0'][self.voice == v]) for v in range(len(self.voices))])
        self.voice_sex = np.array([('F' if f >= FEMALE_F0 else 'M') if v.startswith('person:') else audio.sex_of(v)
                                   for v, f in zip(self.voices, self.voice_f0)])
        self.traits = np.concatenate([self.voice_xvector, ((np.log(self.voice_f0) - np.log(150.)) / .5)[:, None]],
                                     1).astype(np.float32)
        # what every voice says for every item: references for the cepstral distance and one clip to play
        middle = audio.RATES_WPM[len(audio.RATES_WPM) // 2]
        self.refs, self.reference = {}, {}
        for v in range(len(self.voices)):
            for k in range(self.K):
                own = np.flatnonzero((self.voice == v) & (self.item == k))
                if len(own):
                    self.refs[(v, k)] = [audio.cepstra(self.mel[r]) for r in own]
                    self.reference[(v, k)] = int(min(own, key=lambda r: abs(int(self.wpm[r]) - middle) if self.wpm[r] else 0))

    def label(self, v):
        """'Samantha (F, 177 Hz)'"""
        return f'{self.voices[v].replace("person:", "")} ({self.voice_sex[v]}, {self.voice_f0[v]:.0f} Hz)'


class Judge:
    """What generated speech says and who seems to say it.

    Content scores (n, K), higher = more like item k: listener (Whisper log-likelihood of the item's
    text), mcd (minus the smallest DTW mel-cepstral distance to the intended voice saying the item) and
    hubert (cosine to the item's HuBERT centroid).  Voice: the nearest voice by speaker embedding, the
    cosine to the intended voice and the median F0."""

    def __init__(self, t: Targets, device, measures=MEASURES + ('voice',)):
        self.t, self.vocoder = t, audio.Vocoder(device)
        self.listener = audio.Listener(t.language, device) if 'listener' in measures else None
        self.hubert = audio.Hubert(device) if 'hubert' in measures else None
        self.speaker = audio.Speaker(device) if 'voice' in measures else None

    def __call__(self, mels, voices, measures=MEASURES + ('voice',)):
        waves = self.vocoder(mels)
        scores, voice = {}, {}
        if 'listener' in measures:
            scores['listener'] = self.listener.choose(list(waves), self.t.texts)
        if 'mcd' in measures:
            scores['mcd'] = -self.distances(mels, voices)
        if 'hubert' in measures:
            h = self.hubert(waves)[:, self.t.layer] - self.t.centre
            scores['hubert'] = h @ self.t.centroids.T / np.linalg.norm(h, axis=1, keepdims=True).clip(1e-6)
        if 'voice' in measures:
            x = self.speaker(waves)
            voice = dict(speaker=(x @ self.t.voice_xvector.T).argmax(1),
                         similarity=np.sum(x * self.t.voice_xvector[voices], 1),
                         f0=np.array([audio.pitch(w) for w in waves]))
        return waves, scores, voice

    def distances(self, mels, voices):
        """(n, K) smallest DTW mel-cepstral distance (dB) to the intended voice saying each item."""
        cep = [audio.cepstra(m) for m in mels]
        pairs, owner = [], []
        for i, (c, v) in enumerate(zip(cep, voices)):
            for k in range(self.t.K):
                refs = self.t.refs.get((int(v), k)) or [ref for (w, j), rs in self.t.refs.items() if j == k for ref in rs]
                pairs += [(c, ref) for ref in refs]
                owner += [(i, k)] * len(refs)
        out = np.full((len(cep), self.t.K), np.inf)
        for (i, k), d in zip(owner, audio.dtw_mcd(pairs)):
            out[i, k] = min(out[i, k], d)
        return out


# ----------------------------------------------------------------------------- 4 decoder
def whitening(anchor):
    """Symmetric whitening of the anchors' second moment, applied to the decoder's content condition.

    Near-homophones (KaraOne /tiy/ vs /piy/: anchor cosine 0.81) would otherwise reach the decoder as
    almost the same condition; whitened, every item direction has unit scale.  The CLIP space and the
    encoders are unchanged.
    """
    w, v = np.linalg.eigh(anchor.T.astype(np.float64) @ anchor / len(anchor))
    return ((v / np.sqrt(np.maximum(w, 1e-6))) @ v.T).astype(np.float32)


def decoder(name, spec, cfg, device, seed=0):
    c = cfg['decoder']
    steps = spec.get('decoder_steps', c['steps'])
    t = Targets(name, spec)
    mel = torch.from_numpy(t.mel)
    scaler = MelScaler.fit(mel)
    x_all = scaler.encode(mel)
    clips = {(v, k): np.flatnonzero((t.voice == v) & (t.item == k)) for v in range(len(t.voices)) for k in range(t.K)}
    speakers = [[v for v in range(len(t.voices)) if len(clips[(v, k)])] for k in range(t.K)]
    torch.manual_seed(seed)
    model = MelDiffusion(t.anchor.shape[1], frames=mel.shape[-1], hidden=c['hidden'], blocks=c['blocks'],
                         trait_dim=t.traits.shape[1]).to(device)
    ema = EMA(model, c['ema'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=.01)
    rng = np.random.default_rng(seed)
    white = whitening(t.anchor)
    anchor, traits = torch.from_numpy(t.anchor @ white), torch.from_numpy(t.traits)
    (OUT / 'decoders').mkdir(parents=True, exist_ok=True)
    started, running = time.time(), []
    for step in range(steps):
        for group in optimizer.param_groups:
            group['lr'] = c['lr'] * min(1, (step + 1) / 500) * .5 * (1 + math.cos(math.pi * step / steps))
        u = rng.random(c['batch'])
        null, oracle = u < c['null_fraction'], (u >= c['null_fraction']) & (u < c['null_fraction'] + c['oracle_fraction'])
        alpha = np.exp(rng.uniform(np.log(c['alpha'][0]), np.log(c['alpha'][1]), c['batch']))
        p = np.nan_to_num(np.stack([rng.dirichlet(np.full(t.K, a)) for a in alpha]))
        p[p.sum(1) < .5] = 1 / t.K                                                 # underflow of tiny alphas
        p[oracle] = np.eye(t.K)[rng.integers(0, t.K, oracle.sum())]
        p[null] = 1 / t.K
        k = np.array([rng.choice(t.K, p=q / q.sum()) for q in p])                 # the item to say, drawn from p
        v = np.array([rng.choice(speakers[i]) for i in k])                         # a voice that says it
        r = np.array([rng.choice(clips[(a, b)]) for a, b in zip(v, k)])
        shift = rng.integers(-3, 4, c['batch'])                                    # +-50 ms onset jitter
        x0 = torch.stack([torch.roll(x_all[i], int(s), -1) for i, s in zip(r, shift)])
        cond = torch.from_numpy(p.astype(np.float32)) @ anchor
        loss = model.loss(x0.to(device), cond.to(device), torch.from_numpy(null).to(device),
                          traits[torch.from_numpy(v)].to(device))
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
                    voices=t.voices, anchor=t.anchor, whiten=white, layer=t.layer, steps=steps), path)
    log(f'saved {path}')
    check_decoder(name, spec, cfg, device)


class Decoder:
    def __init__(self, name, t: Targets, device):
        path = OUT / 'decoders' / f'{name}.pt'
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload['items'] != t.items or payload.get('voices') != t.voices or \
                not np.allclose(payload['anchor'], t.anchor, atol=1e-5):
            raise SystemExit(f'{path} was trained on other targets or voices; '
                             f'retrain: python scripts/reconstruct.py decoder --datasets {name}')
        self.model = MelDiffusion(**payload['config'])
        self.model.load_state_dict(payload['state'])
        self.model.to(device).eval()
        self.scaler = MelScaler(**payload['scaler'])
        self.whiten, self.traits, self.device = payload['whiten'], t.traits, device

    def mels(self, condition, voices, null, guidance, steps, seed, batch=128):
        """Log-mels (n, 80, frames) saying CLIP-space ``condition`` in ``voices``; ``null`` rows have no content."""
        condition = np.asarray(condition, np.float32) @ self.whiten
        out = []
        for i in range(0, len(condition), batch):
            as_tensor = lambda x, **kw: torch.as_tensor(x[i:i + batch], device=self.device, **kw)
            x = self.model.sample(as_tensor(condition, dtype=torch.float32), traits=as_tensor(self.traits[voices]),
                                  null=as_tensor(null), steps=steps, guidance=guidance,
                                  generator=torch.Generator().manual_seed(seed + i))
            out.append(self.scaler.decode(x.cpu()))
        return torch.cat(out).numpy()


def check_decoder(name, spec, cfg, device, per_item=2):
    """The decoder's ceiling: every voice says every item from the item's anchor (oracle)."""
    t = Targets(name, spec)
    d, judge, c = Decoder(name, t, device), Judge(t, device), cfg['decoder']
    pairs = np.array(sorted(t.reference))                                       # (voice, item) that exist
    pairs = np.repeat(pairs, per_item, axis=0)
    voices, items = pairs[:, 0], pairs[:, 1]
    report = {}
    for guidance in c['guidance']:
        mels = d.mels(t.anchor[items], voices, np.zeros(len(items), bool), guidance, c['sample_steps'], seed=0)
        _, scores, voice = judge(mels, voices)
        report[f'oracle_w{guidance:g}'] = {m: float((s.argmax(1) == items).mean()) for m, s in scores.items()}
        report[f'oracle_w{guidance:g}'].update(voice_measures(voice, voices, t))
    own = [v for v in range(len(t.voices)) if t.voices[v].startswith('person:')]
    if own:                                                                     # the participants' own voices
        mask = np.isin(voices, own)
        mels = d.mels(t.anchor[items[mask]], voices[mask], np.zeros(mask.sum(), bool), c['guidance'][1],
                      c['sample_steps'], seed=1)
        _, scores, voice = judge(mels, voices[mask])
        report['own_voices'] = {m: float((s.argmax(1) == items[mask]).mean()) for m, s in scores.items()}
        report['own_voices'].update(voice_measures(voice, voices[mask], t))
    folder = OUT / 'decoders' / name
    folder.mkdir(parents=True, exist_ok=True)
    shown = [v for v in range(len(t.voices)) if t.voice_sex[v] == 'F'][:1] + \
            [v for v in range(len(t.voices)) if t.voice_sex[v] == 'M'][:1] + own[:2]
    for v in shown:                                                             # listen: reference vs generated
        keys = [(v, k) for k in range(t.K) if (v, k) in t.reference]
        refs = judge.vocoder(t.mel[[t.reference[key] for key in keys]])
        mels = d.mels(t.anchor[[k for _, k in keys]], np.full(len(keys), v), np.zeros(len(keys), bool),
                      c['guidance'][1], c['sample_steps'], seed=2)
        for (_, k), ref, gen in zip(keys, refs, judge.vocoder(mels)):
            stem = f'{slug(t.voices[v])}_{slug(t.items[k])}'
            sf.write(folder / f'{stem}_ref.wav', ref, audio.SAMPLE_RATE)
            sf.write(folder / f'{stem}_oracle.wav', gen, audio.SAMPLE_RATE)
    json.dump(report, open(OUT / 'decoders' / f'{name}.json', 'w'), indent=1)
    log(f'{name} decoder check: {json.dumps(report)}')


def voice_measures(voice, voices, t):
    """Speaker identification, sex match and pitch error of generated speech against the intended voices."""
    f0 = voice['f0']
    sex = np.where(f0 >= FEMALE_F0, 'F', 'M')
    voiced = np.isfinite(f0)
    return dict(speaker=float((voice['speaker'] == voices).mean()), similarity=float(voice['similarity'].mean()),
                sex=float((sex == t.voice_sex[voices])[voiced].mean()) if voiced.any() else float('nan'),
                pitch_error_st=semitones(f0, t.voice_f0[voices]))


def semitones(f0, target):
    """Mean absolute pitch difference in semitones over the voiced clips (NaN if none is voiced)."""
    ratio = np.asarray(f0, float) / np.asarray(target, float)
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    return float(np.mean(np.abs(12 * np.log2(ratio)))) if len(ratio) else float('nan')


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


# ----------------------------------------------------------------------------- 5-6 generate, measure
def derangement(who, rng):
    """For each trial, another trial of the same person (a random permutation without fixed points)."""
    out = np.arange(len(who))
    for s in np.unique(who):
        idx = rng.permutation(np.flatnonzero(who == s))
        if len(idx) > 1:
            out[idx] = np.roll(idx, 1)
    return out


def person_voices(t: Targets, spec, subjects):
    """The voice each person speaks in: their own recording, or a synthetic voice assigned in turn."""
    if spec.get('voices') == 'own':
        missing = [s for s in subjects if f'person:{s}' not in t.voices]
        if missing:
            raise SystemExit(f'no recordings of {missing}: python scripts/prepare.py karaone_voices, then targets')
        return {s: t.voices.index(f'person:{s}') for s in subjects}
    synthetic = [v for v in audio.VOICES[t.language] if v in t.voices]           # listed order mixes the sexes
    return {s: t.voices.index(synthetic[i % len(synthetic)]) for i, s in enumerate(sorted(subjects))}


def operating_point(name, cfg, fold, t, e, d, judge, voice_of):
    """Posterior sharpness x guidance scale chosen on cross-fitted training trials (never held-out ones)."""
    c = cfg['decoder']
    rng = np.random.default_rng(fold)
    tr = e['train']
    pick = np.sort(rng.choice(len(tr['y']), min(len(tr['y']), c['select_trials']), replace=False))
    voices = np.array([voice_of[s] for s in tr['who'][pick]])
    scores = {}
    for sharpness in c['sharpness']:
        condition = clip.posterior(tr['z'][pick], t.anchor, e['scale'] * sharpness) @ t.anchor
        for guidance in c['guidance']:
            mels = d.mels(condition, voices, np.zeros(len(pick), bool), guidance, c['sample_steps'], seed=10_000 + fold)
            _, s, _ = judge(mels, voices, c['select_by'])
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


def control_selection(trials, per_item):
    """The first ``per_item`` trials of every person and item (prior and oracle ignore the trial's EEG)."""
    return np.array(sorted(trials.groupby(['subject', 'item']).head(per_item).index), int)


def run(name, spec, cfg, fold, device):
    t = Targets(name, spec)
    d, judge, c = Decoder(name, t, device), Judge(t, device), cfg['decoder']
    folder = OUT / f'f{fold}' / name
    (folder / 'audio').mkdir(parents=True, exist_ok=True)
    started = time.time()
    e = encoders(name, spec, cfg, fold, t)
    log(f'{name} f{fold}: encoder settings {e["setting"]} ({time.time() - started:.0f} s)')
    voice_of = person_voices(t, spec, sorted(set(e['train']['who']) | set(e['test']['who'])))
    (sharpness, guidance), choice = operating_point(name, cfg, fold, t, e, d, judge, voice_of)
    log(f'{name} f{fold}: operating point sharpness {sharpness:g}, guidance {guidance:g}')

    te = e['test']
    n = len(te['y'])
    voices = np.array([voice_of[s] for s in te['who']])
    calibrated = clip.posterior(te['z'], t.anchor, e['scale'])
    condition = clip.posterior(te['z'], t.anchor, e['scale'] * sharpness) @ t.anchor
    other = derangement(te['who'], np.random.default_rng(1000 + fold))
    trials = pd.DataFrame(dict(subject=te['who'], row=te['rows'], item=te['y'], decoded=calibrated.argmax(1),
                               p_true=calibrated[np.arange(n), te['y']], wrong_item=te['y'][other], voice=voices,
                               voice_sex=t.voice_sex[voices], voice_f0=t.voice_f0[voices]))
    listen = listen_selection(trials, cfg['listen_trials'])
    controls = np.union1d(control_selection(trials, cfg['control_trials']), listen)
    every = np.arange(n)
    plan = dict(eeg=(every, condition), wrong=(every, condition[other]),
                prior=(controls, np.zeros_like(condition)), oracle=(controls, t.anchor[te['y']]))
    kept, shown = {}, set(listen.tolist())
    for kind, (rows, value) in plan.items():
        mels = d.mels(value[rows], voices[rows], np.full(len(rows), kind == 'prior'), guidance, c['sample_steps'],
                      seed=fold * 1_000_000)
        waves, scores, voice = judge(mels, voices[rows])
        for m, s in scores.items():
            trials.loc[rows, f'{kind}_{m}'] = s.argmax(1)
        trials.loc[rows, f'{kind}_mcd_true'] = -scores['mcd'][np.arange(len(rows)), te['y'][rows]]
        trials.loc[rows, f'{kind}_speaker'] = voice['speaker']
        trials.loc[rows, f'{kind}_similarity'] = voice['similarity']
        trials.loc[rows, f'{kind}_f0'] = voice['f0']
        kept[kind] = {int(r): w for r, w in zip(rows, waves) if int(r) in shown}
        heard = (trials.loc[rows, f'{kind}_listener'] == trials.loc[rows, 'item']).mean()
        log(f'{name} f{fold}: {kind:6s} ({len(rows)} trials) identified as the true item listener {heard:.3f}, '
            f'mcd {np.mean(trials.loc[rows, f"{kind}_mcd"] == trials.loc[rows, "item"]):.3f}; voice '
            + ', '.join(f'{k} {v:.2f}' for k, v in voice_measures(voice, voices[rows], t).items()))
    trials['item_name'] = [t.items[k] for k in trials['item']]
    trials['voice_name'] = [t.voices[v] for v in trials['voice']]
    trials['heard'] = ''
    trials.loc[listen, 'heard'] = judge.listener.transcribe([kept['eeg'][int(r)] for r in listen])
    trials.to_csv(folder / 'trials.csv', index=False)
    np.savez(folder / 'encoder.npz', z_test=te['z'], p_test=calibrated, who_test=te['who'], rows_test=te['rows'],
             y_test=te['y'], z_train=e['train']['z'], y_train=e['train']['y'], who_train=e['train']['who'],
             rows_train=e['train']['rows'], anchor=t.anchor, scale=e['scale'])
    summary = dict(dataset=name, fold=fold, trials=n, items=t.K, chance=1 / t.K, encoder=e['setting'], inner=e['inner'],
                   scale=e['scale'], sharpness=sharpness, guidance=guidance,
                   voices={s: t.label(v) for s, v in voice_of.items()},
                   operating_points={f'sharpness={k[0]:g},guidance={k[1]:g}': v for k, v in choice.items()},
                   results=measure(trials, t), seconds=round(time.time() - started))
    json.dump(summary, open(folder / 'summary.json', 'w'), indent=1)
    listen_page(name, fold, folder, trials, listen, kept, t, judge, summary)
    log(f'{name} f{fold}: done in {time.time() - started:.0f} s -> {folder / "listen.html"}')


def measure(trials, t: Targets):
    """Per person: content identification of every condition (vs chance), voice (speaker identification,
    sex, pitch error), the paired advantage of the trial's own EEG over another trial's EEG, and the
    cepstral distance to the person's voice saying the true item."""
    per = trials.groupby('subject')

    def rate(column, truth='item'):
        out = {}
        for s, g in per:
            seen = g[column].notna()
            if seen.any():
                out[s] = float((g.loc[seen, column] == g.loc[seen, truth]).mean())
        return out

    out = dict(encoder=summarize(rate('decoded'), 1 / t.K))
    for kind in CONDITIONS:
        r = {m: summarize(rate(f'{kind}_{m}'), 1 / t.K) for m in MEASURES if f'{kind}_{m}' in trials}
        r['mcd_true_db'] = float(per[f'{kind}_mcd_true'].mean().mean())
        r['speaker'] = summarize(rate(f'{kind}_speaker', 'voice'), 1 / len(t.voices))
        f0 = trials[f'{kind}_f0']
        sex = pd.Series(np.where(f0 >= FEMALE_F0, 'F', 'M'), index=trials.index).where(f0.notna())
        r['sex'] = summarize({s: float((sex[g.index] == g['voice_sex'])[sex[g.index].notna()].mean())
                              for s, g in per if sex[g.index].notna().any()}, .5)
        r['pitch_error_st'] = semitones(f0, trials['voice_f0'])
        out[kind] = r
    for m in MEASURES:
        own, wrong = rate(f'eeg_{m}'), rate(f'wrong_{m}')
        diff = np.array([own[s] - wrong[s] for s in own])
        test = dict(mean_difference=float(diff.mean()), people_better=int((diff > 0).sum()), people=len(diff))
        if len(diff) >= 5 and np.any(diff != 0):
            test['p_wilcoxon'] = float(stats.wilcoxon(diff, alternative='greater', zero_method='zsplit').pvalue)
        out['eeg'][m]['vs_wrong'] = test
    return out


def player(path):
    return f'<audio controls preload="none" src="{html.escape(path)}"></audio>'


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>
body{{font:14px/1.45 system-ui,sans-serif;margin:16px;max-width:1500px}} table{{border-collapse:collapse;margin:8px 0}}
td,th{{border:1px solid #ccc;padding:4px 6px;vertical-align:top;text-align:left}} audio{{width:160px;height:30px}}
.num{{text-align:right}} .muted{{color:#666}}</style></head><body>{body}</body></html>"""


def listen_page(name, fold, folder, trials, listen, kept, t, judge, summary):
    """For a few trials per person: the person's voice saying the true item, the reconstruction and controls."""
    keys = sorted({(int(trials.loc[i, 'voice']), int(trials.loc[i, 'item'])) for i in listen})
    references = judge.vocoder(t.mel[[t.reference[key] for key in keys]])
    for (v, k), w in zip(keys, references):
        sf.write(folder / 'audio' / f'ref_{slug(t.voices[v])}_{slug(t.items[k])}.wav', w, audio.SAMPLE_RATE)
    rows = []
    for i in listen:
        r = trials.loc[i]
        stem = f'{slug(r["subject"])}_{int(r["row"])}'
        for kind in CONDITIONS:
            sf.write(folder / 'audio' / f'{stem}_{kind}.wav', kept[kind][int(i)], audio.SAMPLE_RATE)
        v, k = int(r['voice']), int(r['item'])
        rows.append(f'<tr><td>{html.escape(r["subject"])}<br><span class="muted">{html.escape(t.label(v))}</span></td>'
                    f'<td><b>{html.escape(t.items[k])}</b><br>{player(f"audio/ref_{slug(t.voices[v])}_{slug(t.items[k])}.wav")}</td>'
                    f'<td>{html.escape(t.items[int(r["decoded"])])} <span class="muted">({r["p_true"]:.2f})</span></td>'
                    f'<td>{html.escape(t.items[int(r["eeg_listener"])])}<br><span class="muted">"{html.escape(r["heard"])}", '
                    f'{r["eeg_f0"]:.0f} Hz</span></td>'
                    + ''.join(f'<td>{player(f"audio/{stem}_{kind}.wav")}</td>' for kind in CONDITIONS) + '</tr>')
    body = (f'<h2>{html.escape(name)}, fold {fold}: imagined speech reconstructed from EEG, in the person\'s voice</h2>'
            + numbers_table({f'fold {fold}': summary['results']}, summary['chance'])
            + f'<p class="muted">Operating point: posterior sharpness {summary["sharpness"]:g}, guidance {summary["guidance"]:g}. '
              'Columns: the person and the voice they speak in (own recording, or an assigned synthetic voice when the '
              'dataset recorded neither voice nor sex); the true item said in that voice (vocoded reference); the '
              'encoder\'s decision and its probability for the true item; what Whisper picks for the EEG '
              'reconstruction, its free transcript and pitch; reconstructions from this trial\'s EEG, from another '
              'trial of the same person (wrong), without content (prior) and from the true item (oracle: the '
              'decoder\'s ceiling).</p>'
            + '<table><tr><th>person, voice</th><th>true item</th><th>decoded (p true)</th><th>heard as</th>'
              '<th>EEG</th><th>wrong trial</th><th>prior</th><th>oracle</th></tr>' + ''.join(rows) + '</table>')
    (folder / 'listen.html').write_text(PAGE.format(title=f'{html.escape(name)} fold {fold}', body=body))


def numbers_table(results, chance):
    """Content identification and voice measures per condition (means over people)."""
    head = ('<tr><th></th><th>encoder</th>' + ''.join(f'<th>{k}: listener / MCD / HuBERT</th>' for k in CONDITIONS)
            + '<th>voice (EEG / oracle): speaker id, sex, pitch error</th><th>EEG vs wrong (listener)</th>'
              '<th>MCD to true item, EEG / oracle</th></tr>')
    lines = []
    for label, r in results.items():
        cells = [f'{r["encoder"].get("mean", float("nan")):.3f}']
        for kind in CONDITIONS:
            cells.append(' / '.join(f'{r[kind][m].get("mean", float("nan")):.3f}' for m in MEASURES if m in r[kind]))
        cells.append(' | '.join(f'{r[k]["speaker"].get("mean", float("nan")):.2f}, {r[k]["sex"].get("mean", float("nan")):.2f}, '
                                f'{r[k]["pitch_error_st"]:.1f} st' for k in ('eeg', 'oracle')))
        vs = r['eeg'].get('listener', {}).get('vs_wrong', {})
        cells.append(f'{vs.get("mean_difference", float("nan")):+.3f} (p {vs.get("p_wilcoxon", float("nan")):.3g})')
        cells.append(f'{r["eeg"]["mcd_true_db"]:.2f} / {r["oracle"]["mcd_true_db"]:.2f} dB')
        lines.append(f'<tr><td>{html.escape(label)}</td>' + ''.join(f'<td class="num">{c}</td>' for c in cells) + '</tr>')
    return (f'<p>Identification of the generated speech among the vocabulary, mean over people (chance {chance:.3f}); '
            f'voice: share identified as the intended voice, share with the intended sex, mean pitch error in semitones.</p>'
            f'<table>{head}{"".join(lines)}</table>')


# ----------------------------------------------------------------------------- report
def report(names, cfg):
    out, sections = {}, []
    for name in names:
        parts = sorted(OUT.glob(f'f*/{name}/trials.csv'))
        if not parts:
            continue
        trials = pd.concat([pd.read_csv(p, keep_default_na=False, na_values=['']).assign(fold=int(p.parts[-3][1:]))
                            for p in parts], ignore_index=True)
        trials['heard'] = trials['heard'].fillna('')
        t = Targets(name, cfg['datasets'][name])
        folds = sorted(trials.fold.unique().tolist())
        out[name] = dict(folds=folds, trials=int(len(trials)), chance=1 / t.K, results=measure(trials, t))
        r = out[name]['results']
        line = f'{name:18s} folds {folds} chance {1 / t.K:.3f} | encoder {r["encoder"]["mean"]:.3f}'
        for kind in CONDITIONS:
            line += f' | {kind} ' + '/'.join(f'{r[kind][m]["mean"]:.3f}' for m in MEASURES if m in r[kind])
        for kind in ('eeg', 'oracle'):
            v = r[kind]
            line += (f' | voice {kind} {v["speaker"].get("mean", np.nan):.2f}/{v["sex"].get("mean", np.nan):.2f}/'
                     f'{v["pitch_error_st"]:.1f}st')
        print(line)
        links = ' '.join(f'<a href="f{f}/{name}/listen.html">fold {f}</a>' for f in folds)
        sections.append(f'<h3>{html.escape(name)}</h3>' + numbers_table({'all folds': r}, 1 / t.K) + f'<p>Listen: {links}</p>')
    json.dump(out, open(OUT / 'summary.json', 'w'), indent=1)
    body = ('<h2>Imagined speech reconstructed from EEG in the person\'s voice (within person, held-out fifths)</h2>'
            + ''.join(sections))
    (OUT / 'index.html').write_text(PAGE.format(title='Reconstruction', body=body))
    print('content listener/MCD/HuBERT and voice speaker/sex/pitch per condition ->', OUT / 'summary.json',
          OUT / 'index.html')


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['targets', 'decoder', 'check', 'run', 'report'])
    parser.add_argument('--datasets', nargs='*')
    parser.add_argument('--fold', type=int, nargs='*', default=[0])
    parser.add_argument('--config', default=str(ROOT / 'configs' / 'plan.yaml'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--out', default=str(OUT), help='folder of decoders, folds and report')
    args = parser.parse_args()
    OUT = Path(args.out)
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

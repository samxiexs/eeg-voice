"""Two-tower soft alignment of EEG and speech, sized for one laptop night (Apple M2 Pro).

EEG and audio each get their own encoder into one 128-d space at 25 Hz. Nothing is
regressed onto a fixed audio feature: a contrastive loss decides what the two share.

* Electrodes are pooled into K learnable scalp regions, initialised by k-means on the
  3-D positions of every montage, so 60-, 63-, 64- and 128-channel caps share one input.
  Each region then gets its own bank of temporal filters (space x band x time).
* An audio frame is compared with EEG from 80 ms before to 480 ms after it, with a soft
  maximum over that lag window and a learnable lag prior per person.
* EEG of different people (and repeated presentations) at the same audio time are
  positives for each other, which pulls out the stimulus-driven part of the signal.

A decoder turns the audio-side embedding back into a Mel spectrogram; a bridge produces
that embedding from EEG; frozen Vocos renders the waveform.
"""
from __future__ import annotations
from collections import defaultdict
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import IterableDataset, get_worker_info
from ..utils import atomic_json, append_jsonl, read_json

DATASETS = ('ds004408', 'brennan2019', 'etard', 'kymata_english')
FRAME = .04; PRE = 12; SEG = 75; WIN = 112; LAGS = tuple(range(-2, 13)); EE_LAGS = tuple(range(-2, 3))
RAW = 200; SAMPLES = WIN*8; GUARD = 4.0; MEL_RATE = 24000/256; MEL_FRAMES = 281; LAYERS = (3, 6, 9, 12)
PRE_S, POST_S = PRE*FRAME, (WIN-PRE)*FRAME
# full: everything; synchronous: one fixed 200 ms lag and no EEG-EEG term; no-region-prior: electrodes pooled without the k-means geometry.
VARIANTS = {'full': {'lags': LAGS, 'ee_weight': .5, 'geometric': True},
            'synchronous': {'lags': (5,), 'ee_weight': 0., 'geometric': True},
            'no-region-prior': {'lags': LAGS, 'ee_weight': .5, 'geometric': False}}


def wall(text):
    """Today's or tomorrow's HH:MM as a Unix time."""
    hour, minute = map(int, text.split(':')); now = time.localtime()
    target = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, hour, minute, 0, 0, 0, -1))
    return target if target > time.time()-3600 else target+86400


def status(folder, **value):
    path = Path(folder)/'status.json'; old = read_json(path) if path.exists() else {}
    atomic_json(path, {**old, **value, 'updated': time.strftime('%Y-%m-%d %H:%M:%S')})


# ----------------------------------------------------------------------------- cache

def build_cache(cfg, out, contents_per_dataset=None, device='cpu'):
    """Frozen speech features of every clean canonical stimulus on the 25 Hz frame grid, plus the
    native Mel for decoding, the broadband envelope, the electrode k-means and the record index."""
    import soundfile as sf
    from scipy.signal import resample_poly
    from scipy.interpolate import interp1d
    from sklearn.cluster import KMeans
    from transformers import AutoFeatureExtractor, WavLMModel
    from ..perceived_audio import FrozenVocos
    out = Path(out); (out/'audio').mkdir(parents=True, exist_ok=True); registry = cfg.registry()
    processor = AutoFeatureExtractor.from_pretrained(cfg.model_dir('wavlm'), local_files_only=True)
    wavlm = WavLMModel.from_pretrained(cfg.model_dir('wavlm'), local_files_only=True).eval()
    vocos = FrozenVocos(cfg.model_dir('vocos')); index = {}; positions = []
    for dataset in DATASETS:
        folder = cfg.dir('processed')/dataset; manifest = read_json(folder/'manifest.json'); splits = registry['content_splits'][dataset]
        records = [r for r in manifest['records'].values() if r['task'] == 'perceived' and r['metadata'].get('condition', 'clean') == 'clean']
        used = sorted({r['audio'] for r in records})[:contents_per_dataset]; (out/'audio'/dataset).mkdir(exist_ok=True)
        for content in used:
            stem = out/'audio'/dataset/content
            if (stem.with_suffix('.mel.npy')).exists(): continue
            wave, rate = sf.read(folder/manifest['contents'][content]['path'], dtype='float32'); seconds = len(wave)/rate
            wave16 = resample_poly(wave, 2, 3).astype(np.float32); frames = int(seconds/FRAME); grid = (np.arange(frames)+.5)*FRAME
            times, feats = [], []
            for start in np.arange(0, seconds, 20.):
                lo = max(0, round((start-2)*16000)); hi = min(len(wave16), round((start+22)*16000))
                inputs = processor([wave16[lo:hi]], sampling_rate=16000, return_tensors='pt')
                with torch.no_grad(): hidden = wavlm(**inputs, output_hidden_states=True).hidden_states
                t = lo/16000+(199.5+320*np.arange(hidden[0].shape[1]))/16000; keep = (t >= start) & (t < start+20)
                times.append(t[keep]); feats.append(torch.stack([hidden[k][0] for k in LAYERS], 1).numpy()[keep])
            times = np.concatenate(times); feats = np.concatenate(feats)
            layered = interp1d(times, feats, axis=0, bounds_error=False, fill_value=(feats[0], feats[-1]))(grid)
            np.save(stem.with_suffix('.wavlm.npy'), layered.astype(np.float16))
            # Broadband 25 ms RMS at 64 Hz and its positive change, as in the TRF front end (computed directly; the filter bank is not needed).
            env = np.sqrt(np.square(np.lib.stride_tricks.sliding_window_view(np.pad(wave, 300, mode='reflect'), 600)[::375]).mean(1)+1e-12)
            onset = np.maximum(np.diff(env, prepend=env[0]), 0); t64 = (np.arange(len(env))+.5)/64
            features = np.stack([np.log(env+1e-4), np.log(onset+1e-4)], 1)
            np.save(stem.with_suffix('.env.npy'), interp1d(t64, features, axis=0, bounds_error=False, fill_value=(features[0], features[-1]))(grid).astype(np.float32))
            with torch.no_grad(): mel = vocos.features(torch.from_numpy(wave)[None])[0].numpy()
            np.save(stem.with_suffix('.mel.npy'), mel.astype(np.float16))
            print(json.dumps({'cache': dataset, 'content': content, 'seconds': round(seconds, 1)}), flush=True)
        index[dataset] = []
        for r in records:
            if r['audio'] not in used: continue
            index[dataset].append({'key': f"{dataset}:{r['subject']}", 'path': str(folder/r['path']), 'audio': r['audio'], 'start': float(r['audio_start']),
                                   'duration': float(r['duration']), 'split': splits[r['content_group']], 'channels': len(r['names'])})
            positions.append(np.asarray(r['positions'], np.float32))
        np.save(out/f'{dataset}.positions.npy', np.stack([np.asarray(r['positions'], np.float32) for r in records if r['audio'] in used]))
        np.save(out/f'{dataset}.good.npy', np.stack([np.asarray(r['good_channels'], bool) for r in records if r['audio'] in used]))
    unique = np.unique(np.round(np.concatenate(positions), 4), axis=0)
    centroids = KMeans(32, n_init=10, random_state=0).fit(unique).cluster_centers_.astype(np.float32); np.save(out/'centroids.npy', centroids)
    # Normalisation of the decoder target and the envelope input from training content only.
    mel, env = [], []
    for dataset in DATASETS:
        spans = {(r['audio'], r['start'], r['duration']) for r in index[dataset] if r['split'] == 'train'}
        for content, start, seconds in spans:
            stem = out/'audio'/dataset/content
            env.append(np.load(stem.with_suffix('.env.npy'))[int(start/FRAME):int((start+seconds)/FRAME):5])
            mel.append(np.load(stem.with_suffix('.mel.npy'))[:, int(start*MEL_RATE):int((start+seconds)*MEL_RATE):20].astype(np.float32).T)
    env, mel = np.concatenate(env), np.concatenate(mel)
    atomic_json(out/'index.json', {'records': index, 'stats': {'env_mean': env.mean(0).tolist(), 'env_std': env.std(0).tolist(),
                'mel_mean': mel.mean(0).tolist(), 'mel_std': (mel.std(0)+1e-3).tolist()}, 'layers': LAYERS, 'frame_seconds': FRAME})
    return out


class Cache:
    """Audio arrays memory-mapped once; records with their montages; subject vocabulary of the training set."""
    def __init__(self, folder, train_subjects):
        self.folder = Path(folder); value = read_json(self.folder/'index.json'); self.records = value['records']; self.stats = value['stats']
        self.positions = {d: np.load(self.folder/f'{d}.positions.npy') for d in self.records}
        self.good = {d: np.load(self.folder/f'{d}.good.npy') for d in self.records}
        self.subjects = sorted(train_subjects); self.subject_index = {s: i for i, s in enumerate(self.subjects)}
        self.audio = {}

    def arrays(self, dataset, content):
        key = (dataset, content)
        if key not in self.audio:
            stem = self.folder/'audio'/dataset/content
            self.audio[key] = tuple(np.load(stem.with_suffix(f'.{kind}.npy'), mmap_mode='r') for kind in ('wavlm', 'env', 'mel'))
        return self.audio[key]

    def audio_batch(self, anchors):
        """[M,75,4,768] speech-model layers and [M,75,2] standardized envelope / onset of each anchor."""
        wv, en = [], []
        mean, std = np.asarray(self.stats['env_mean'], np.float32), np.asarray(self.stats['env_std'], np.float32)
        for dataset, content, a in anchors:
            w, e, _ = self.arrays(dataset, content); k = round(a/FRAME)
            wv.append(np.asarray(w[k:k+SEG], np.float32)); en.append((np.asarray(e[k:k+SEG])-mean)/std)
        return torch.from_numpy(np.stack(wv)), torch.from_numpy(np.stack(en).astype(np.float32))

    def mel_batch(self, anchors):
        mean, std = np.asarray(self.stats['mel_mean'], np.float32)[:, None], np.asarray(self.stats['mel_std'], np.float32)[:, None]
        out = []
        for dataset, content, a in anchors:
            _, _, m = self.arrays(dataset, content); k = round(a*MEL_RATE); out.append((np.asarray(m[:, k:k+MEL_FRAMES], np.float32)-mean)/std)
        return torch.from_numpy(np.stack(out))


def covers(record, a):
    return record['start']+GUARD+PRE_S <= a+1e-6 and a+POST_S <= record['start']+record['duration']-GUARD+1e-6


def eeg_window(record, a, good, train=False, rng=None):
    """Canonical [a-0.48, a+4.0) at 100 Hz, z-scored per channel; flat / over-range / annotated bad channels masked."""
    x = np.load(Path(record['path'])/'eeg.npy', mmap_mode='r'); first = round((a-PRE_S-record['start'])*RAW)
    w = np.asarray(x[:, first:first+SAMPLES:2], np.float32)
    if w.shape[1] != SAMPLES//2: raise ValueError('EEG window outside its record')
    sd = w.std(1); mask = good & (sd >= .1) & (np.ptp(w, 1) <= 400)
    if train and rng is not None:
        drop = rng.random(len(mask)) < .15
        if (mask & ~drop).sum() >= 8: mask = mask & ~drop
    w = np.clip((w-w.mean(1, keepdims=True))/np.maximum(sd, 1e-6)[:, None], -10, 10); w[~mask] = 0
    return w, mask


class Stream(IterableDataset):
    """Batches of M audio anchors from one dataset, each seen through up to S people / repetitions."""
    def __init__(self, cache, subjects, split, anchors=32, views=4, seed=11, train=True):
        self.cache, self.split, self.anchors, self.views, self.seed, self.train = cache, split, anchors, views, seed, train
        self.allowed = set(subjects); self.records = {}
        for d, rows in cache.records.items():
            keep = [i for i, r in enumerate(rows) if r['key'] in self.allowed and r['split'] == split]
            if keep: self.records[d] = keep

    def batch(self, dataset, rng):
        rows = self.cache.records[dataset]; pool = self.records[dataset]
        usable = np.array([max(0., rows[i]['duration']-2*GUARD-PRE_S-POST_S) for i in pool]); p = usable/usable.sum()
        by_audio = defaultdict(list)
        for i in pool: by_audio[rows[i]['audio']].append(i)
        chosen = []; tries = 0
        while len(chosen) < self.anchors and tries < self.anchors*30:
            tries += 1; r = rows[pool[rng.choice(len(pool), p=p)]]
            lo, hi = r['start']+GUARD+PRE_S, r['start']+r['duration']-GUARD-POST_S
            if hi <= lo: continue
            a = round(rng.uniform(lo, hi)/FRAME)*FRAME
            if any(c == r['audio'] and abs(a-b) < SEG*FRAME for _, c, b, _ in chosen): continue
            candidates = [i for i in by_audio[r['audio']] if covers(rows[i], a)]
            people = defaultdict(list)
            for i in candidates: people[rows[i]['key']].append(i)
            order = list(people); rng.shuffle(order); picked = []
            while len(picked) < self.views and any(people.values()):
                for key in order:
                    if people[key] and len(picked) < self.views: picked.append(people[key].pop(int(rng.integers(len(people[key])))))
            if picked: chosen.append((dataset, r['audio'], a, picked))
        eeg, mask, pos, subject, anchor = [], [], [], [], []
        for m, (_, _, a, picked) in enumerate(chosen):
            for i in picked:
                w, ok = eeg_window(rows[i], a, self.cache.good[dataset][i], self.train, rng)
                eeg.append(w); mask.append(ok); pos.append(self.cache.positions[dataset][i])
                subject.append(self.cache.subject_index.get(rows[i]['key'], len(self.cache.subjects))); anchor.append(m)
        return {'eeg': torch.from_numpy(np.stack(eeg)), 'mask': torch.from_numpy(np.stack(mask)), 'positions': torch.from_numpy(np.stack(pos)),
                'subject': torch.tensor(subject), 'anchor': torch.tensor(anchor), 'dataset': dataset,
                'anchors': [(d, c, a) for d, c, a, _ in chosen]}

    def __iter__(self):
        info = get_worker_info(); worker = info.id if info else 0
        rng = np.random.default_rng(self.seed+7919*worker+int(time.time()) % 100000); names = sorted(self.records); step = worker
        while True:
            yield self.batch(names[step % len(names)], rng); step += 1


# ----------------------------------------------------------------------------- models

class Residual(nn.Module):
    def __init__(self, width, dilation=1, kernel=3, dropout=.1):
        super().__init__()
        self.conv = nn.Conv1d(width, width, kernel, dilation=dilation, padding=dilation*(kernel-1)//2)
        self.norm = nn.GroupNorm(8, width); self.mix = nn.Conv1d(width, width, 1); self.drop = nn.Dropout(dropout)
    def forward(self, x): return x+self.mix(self.drop(F.gelu(self.norm(self.conv(x)))))


class RegionPool(nn.Module):
    """Soft clustering of electrodes into K scalp regions: Gaussian around a learnable centroid plus
    a learnable correction from Fourier features of position; invalid electrodes never contribute."""
    def __init__(self, centroids, fourier=32, geometric=True):
        super().__init__(); centroids = torch.as_tensor(centroids, dtype=torch.float32); self.geometric = geometric
        self.centroids = nn.Parameter(centroids.clone()); self.log_sigma = nn.Parameter(torch.full((len(centroids),), math.log(.025)))
        self.register_buffer('freq', torch.randn(3, fourier, generator=torch.Generator().manual_seed(0))*20)
        self.correction = nn.Linear(2*fourier, len(centroids)); nn.init.zeros_(self.correction.bias)
        if geometric: nn.init.zeros_(self.correction.weight)

    def weights(self, positions, mask):
        d2 = (positions[:, None]-self.centroids[None, :, None]).square().sum(-1)
        phase = positions@self.freq; extra = self.correction(torch.cat((phase.sin(), phase.cos()), -1)).transpose(1, 2)
        logits = (-d2/(2*self.log_sigma.exp()[None, :, None]**2) if self.geometric else 0)+extra
        return logits.masked_fill(~mask[:, None], -1e4).softmax(-1)

    def forward(self, x, positions, mask):
        w = self.weights(positions, mask); return torch.einsum('bkc,bct->bkt', w, x), w


class EEGTower(nn.Module):
    def __init__(self, centroids, subjects, width=160, dim=128, bands=4, geometric=True):
        super().__init__(); k = len(centroids); self.subjects = subjects
        self.pool = RegionPool(centroids, geometric=geometric)
        self.subject_delta = nn.Parameter(torch.zeros(subjects, k, k))
        self.bank = nn.Conv1d(k, k*bands, 25, padding=12, groups=k)
        self.mix = nn.Conv1d(k*bands, width, 1)
        self.at100 = nn.Sequential(*[Residual(width, d) for d in (1, 2, 4, 8)]); self.down1 = nn.Conv1d(width, width, 4, stride=2, padding=1)
        self.at50 = nn.Sequential(*[Residual(width, d) for d in (1, 2)]); self.down2 = nn.Conv1d(width, width, 4, stride=2, padding=1)
        self.at25 = nn.Sequential(*[Residual(width, d) for d in (1, 2, 4)]); self.out = nn.Conv1d(width, dim, 1)

    def subject_matrix(self, index):
        eye = torch.eye(self.subject_delta.shape[1], device=index.device)
        full = torch.cat((self.subject_delta, torch.zeros_like(self.subject_delta[:1])), 0)
        return eye+full[index]

    def forward(self, x, positions, mask, subject, drop_region=None):
        r, _ = self.pool(x, positions, mask)
        r = torch.einsum('bkj,bjt->bkt', self.subject_matrix(subject), r)
        if drop_region is not None: r = r.clone(); r[:, drop_region] = 0
        h = self.at100(self.mix(F.gelu(self.bank(r))))
        h = self.at25(self.down2(self.at50(self.down1(h))))
        return F.normalize(self.out(h).transpose(1, 2), dim=-1)


class AudioTower(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(len(LAYERS))); self.norm = nn.LayerNorm(768)
        self.proj = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Dropout(.1)); self.env = nn.Linear(2, 256)
        self.conv = nn.Conv1d(256, 256, 3, padding=1); self.out = nn.Linear(256, dim)

    def forward(self, layers, env):
        h = (layers*self.layer_logits.softmax(0)[None, None, :, None]).sum(2)
        h = self.proj(self.norm(h))+self.env(env)
        h = h+F.gelu(self.conv(h.transpose(1, 2))).transpose(1, 2)
        return F.normalize(self.out(h), dim=-1)


class Aligner(nn.Module):
    """Both towers, the per-person lag prior and the temperatures of the soft maxima."""
    def __init__(self, centroids, subjects, variant='full'):
        super().__init__(); spec = VARIANTS[variant]; self.variant, self.lags, self.ee_weight = variant, tuple(spec['lags']), spec['ee_weight']
        self.eeg = EEGTower(centroids, subjects, geometric=spec['geometric']); self.audio = AudioTower()
        self.lag_logits = nn.Parameter(torch.zeros(subjects, len(self.lags)))
        self.log_beta = nn.Parameter(torch.tensor(math.log(10.))); self.log_scale = nn.Parameter(torch.tensor(math.log(1/.07)))
        self.log_scale_ee = nn.Parameter(torch.tensor(math.log(1/.07)))

    def lag_prior(self, subject):
        full = torch.cat((self.lag_logits, self.lag_logits.mean(0, keepdim=True)), 0)
        return full[subject].log_softmax(-1)

    def beta(self): return self.log_beta.exp().clamp(1, 50)

    def scores(self, E, A, subject, lags=None):
        """[B,M] mean over audio frames of a soft maximum over EEG lags; also the per-lag scores."""
        lags = tuple(lags or self.lags)
        shifted = torch.stack([E[:, PRE+d:PRE+d+SEG] for d in lags], 1)
        s = torch.einsum('bltd,mtd->bmlt', shifted, A)
        prior = self.lag_prior(subject) if lags == self.lags else torch.zeros(len(E), len(lags), device=E.device)
        beta = self.beta()
        soft = (torch.logsumexp(beta*s+prior[:, None, :, None], 2)-math.log(len(lags)))/beta
        return soft.mean(-1), s

    def ee_scores(self, E):
        core = E[:, PRE:PRE+SEG]; shifted = torch.stack([E[:, PRE+d:PRE+d+SEG] for d in EE_LAGS], 1); beta = self.beta()
        s = torch.einsum('btd,jstd->bjst', core, shifted)
        return ((torch.logsumexp(beta*s, 2)-math.log(len(EE_LAGS)))/beta).mean(-1)

    def compensated(self, E, subject):
        """EEG embedding re-timed to audio frames with the learned lag prior; no audio needed."""
        weight = self.lag_prior(subject).exp()
        shifted = torch.stack([E[:, PRE+d:PRE+d+SEG] for d in self.lags], 1)
        return F.normalize((weight[:, :, None, None]*shifted).sum(1), dim=-1)


def multi_positive(logits, positive):
    """-log of the total probability of the positives in each row; rows without a positive are ignored."""
    has = positive.any(1)
    if not has.any(): return logits.sum()*0
    top = torch.logsumexp(logits.masked_fill(~positive, -1e4), 1); return (torch.logsumexp(logits, 1)-top)[has].mean()


def alignment_losses(model, E, A, batch):
    anchor = batch['anchor'].to(E.device); subject = batch['subject'].to(E.device)
    S, _ = model.scores(E, A, subject); logits = S*model.log_scale.exp().clamp(max=100)
    target = F.one_hot(anchor, len(A)).bool()
    e2a = F.cross_entropy(logits, anchor); a2e = multi_positive(logits.T, target.T)
    EE = model.ee_scores(E)*model.log_scale_ee.exp().clamp(max=100); eye = torch.eye(len(E), dtype=torch.bool, device=E.device)
    same = (anchor[:, None] == anchor[None]) & ~eye
    ee = multi_positive(EE.masked_fill(eye, -1e4), same) if model.ee_weight else logits.sum()*0
    return e2a+a2e+model.ee_weight*ee, {'eeg_to_audio': e2a.item(), 'audio_to_eeg': a2e.item(), 'eeg_to_eeg': ee.item(),
                           'batch_top1': float((logits.argmax(1) == anchor).float().mean()), 'chance': 1/len(A)}


class MelDecoder(nn.Module):
    def __init__(self, dim=128, width=256):
        super().__init__()
        self.inp = nn.Linear(dim, width)
        self.low = nn.Sequential(*[Residual(width, d, 5) for d in (1, 2, 1)])
        self.high = nn.Sequential(*[Residual(width, d, 5) for d in (1, 2, 4, 1)]); self.out = nn.Conv1d(width, 100, 1)

    def forward(self, z, frames=MEL_FRAMES):
        h = self.low(self.inp(z).transpose(1, 2))
        return self.out(self.high(F.interpolate(h, size=frames, mode='linear', align_corners=False)))


class Bridge(nn.Module):
    """Audio-side embedding predicted from the whole EEG window, starting from the lag-compensated EEG."""
    def __init__(self, dim=128, width=256, heads=4, layers=3):
        super().__init__()
        self.inp = nn.Linear(dim, width); self.pos = nn.Parameter(torch.randn(WIN, width)*.02); self.query = nn.Parameter(torch.randn(SEG, width)*.02)
        layer = nn.TransformerEncoderLayer(width, heads, 4*width, dropout=.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers); self.out = nn.Linear(width, dim); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, E, compensated):
        h = self.encoder(torch.cat((self.inp(E)+self.pos, self.inp(compensated)+self.query), 1))[:, WIN:]
        return F.normalize(compensated+self.out(h), dim=-1)


# ----------------------------------------------------------------------------- training

def to(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def fixed_set(cache, subjects, split, step=SEG*FRAME):
    """Every person's EEG at a fixed 3 s grid of anchors on the given content; identical anchors across people."""
    out = []
    for dataset, rows in cache.records.items():
        for i, r in enumerate(rows):
            if r['key'] not in subjects or r['split'] != split: continue
            first = math.ceil((r['start']+GUARD+PRE_S)/step)*step
            for a in np.arange(first, r['start']+r['duration']-GUARD-POST_S+1e-6, step):
                out.append((dataset, i, round(float(a)/FRAME)*FRAME))
    return out


@torch.no_grad()
def identification(model, cache, items, device, drop_region=None, lag_sets=None, limit=None):
    """Rank of each EEG window's own audio among all anchors of the same dataset (chance rank percentile 0.5)."""
    model.eval(); result = {}; lag_sets = lag_sets or {'model lags': model.lags}
    for dataset in sorted({d for d, _, _ in items}):
        rows = cache.records[dataset]; chosen = [(i, a) for d, i, a in items if d == dataset][:limit]
        anchors = sorted({(rows[i]['audio'], a) for i, a in chosen}); where = {k: n for n, k in enumerate(anchors)}
        A = []
        for n in range(0, len(anchors), 64):
            wv, en = cache.audio_batch([(dataset, c, a) for c, a in anchors[n:n+64]]); A.append(model.audio(wv.to(device), en.to(device)))
        A = torch.cat(A); per = {name: [] for name in lag_sets}; people = []
        for n in range(0, len(chosen), 64):
            part = chosen[n:n+64]; win = [eeg_window(rows[i], a, cache.good[dataset][i]) for i, a in part]
            x = torch.from_numpy(np.stack([w for w, _ in win])).to(device); mask = torch.from_numpy(np.stack([m for _, m in win])).to(device)
            pos = torch.from_numpy(np.stack([cache.positions[dataset][i] for i, _ in part])).to(device)
            subject = torch.tensor([cache.subject_index.get(rows[i]['key'], len(cache.subjects)) for i, _ in part], device=device)
            E = model.eeg(x, pos, mask, subject, drop_region); own = torch.tensor([where[(rows[i]['audio'], a)] for i, a in part], device=device)
            for name, lags in lag_sets.items():
                S, _ = model.scores(E, A, subject, lags); per[name].append((S > S.gather(1, own[:, None])).sum(1).cpu())
            people += [rows[i]['key'] for i, _ in part]
        result[dataset] = {'windows': len(chosen), 'candidates': len(anchors)}
        for name in lag_sets:
            better = torch.cat(per[name]).numpy(); pct = better/max(len(anchors)-1, 1)
            by = defaultdict(list)
            for p, v in zip(people, pct): by[p].append(v)
            result[dataset][name] = {'top1': float(np.mean(better == 0)), 'top5': float(np.mean(better < 5)), 'top10': float(np.mean(better < 10)),
                                     'chance_top1': 1/len(anchors), 'chance_top10': min(10, len(anchors))/len(anchors), 'mean_rank_percentile': float(pct.mean()),
                                     'people_better_than_chance': int(sum(np.mean(v) < .5 for v in by.values())), 'people': len(by)}
    first = next(iter(lag_sets)); result['mean_rank_percentile'] = float(np.mean([v[first]['mean_rank_percentile'] for k, v in result.items() if k in DATASETS]))
    return result


def lr_at(fraction, step, peak=1e-3, warmup=300):
    if step < warmup: return peak*(step+1)/warmup
    return peak*(.05+.95*.5*(1+math.cos(math.pi*min(1., max(0., (fraction-.6)/.4))))) if fraction > .6 else peak


def train_alignment(cfg, cache, folder, deadline, device='cpu', threads=8, workers=2, validate_minutes=20, seed=11, max_steps=None, variant='full'):
    from torch.utils.data import DataLoader
    torch.manual_seed(seed); torch.set_num_threads(threads); folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    train_people = set(cache.subjects); dev = set(cfg.fold_subjects('P1', 'F0', 'validation'))
    model = Aligner(np.load(cache.folder/'centroids.npy'), len(cache.subjects), variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    loader = DataLoader(Stream(cache, train_people, 'train', seed=seed), batch_size=None, num_workers=workers, persistent_workers=workers > 0, prefetch_factor=4 if workers else None)
    dev_items = fixed_set(cache, dev, 'validation'); began = time.time(); last_check = began; best = 1.; step = 0
    status(folder, stage='alignment', variant=variant, device=device, began=time.strftime('%H:%M'), deadline=time.strftime('%H:%M', time.localtime(deadline)))
    for batch in loader:
        now = time.time(); fraction = (now-began)/max(deadline-began, 1)
        if now >= deadline or (max_steps and step >= max_steps): break
        for group in optimizer.param_groups: group['lr'] = lr_at(fraction, step)
        model.train(); batch = to(batch, device)
        wv, en = cache.audio_batch(batch['anchors'])
        E = model.eeg(batch['eeg'], batch['positions'], batch['mask'], batch['subject']); A = model.audio(wv.to(device), en.to(device))
        loss, parts = alignment_losses(model, E, A, batch)
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step(); step += 1
        if step % 25 == 0:
            append_jsonl(folder/'alignment.jsonl', {'step': step, 'minutes': round((now-began)/60, 1), 'dataset': batch['dataset'], 'loss': float(loss), **parts,
                                                    'beta': float(model.beta()), 'lr': optimizer.param_groups[0]['lr']})
            status(folder, step=step, last={'loss': float(loss), **parts})
        if now-last_check >= validate_minutes*60 or (max_steps and step == max_steps):
            last_check = now; score = identification(model, cache, dev_items, device)
            append_jsonl(folder/'validation.jsonl', {'step': step, 'minutes': round((now-began)/60, 1), **score})
            torch.save({'model': model.state_dict(), 'step': step, 'subjects': cache.subjects, 'validation': score, 'variant': model.variant}, folder/'latest.pt')
            if score['mean_rank_percentile'] < best:
                best = score['mean_rank_percentile']; torch.save({'model': model.state_dict(), 'step': step, 'subjects': cache.subjects, 'validation': score, 'variant': model.variant}, folder/'best.pt')
            status(folder, validated_step=step, validation=score['mean_rank_percentile'], best=best)
    score = identification(model, cache, dev_items, device)
    append_jsonl(folder/'validation.jsonl', {'step': step, 'minutes': round((time.time()-began)/60, 1), 'final': True, **score})
    torch.save({'model': model.state_dict(), 'step': step, 'subjects': cache.subjects, 'validation': score, 'variant': model.variant}, folder/'latest.pt')
    if score['mean_rank_percentile'] < best or not (folder/'best.pt').exists():
        torch.save({'model': model.state_dict(), 'step': step, 'subjects': cache.subjects, 'validation': score, 'variant': model.variant}, folder/'best.pt')
    status(folder, stage='alignment-done', step=step)
    return score


def load_aligner(cache, folder, device):
    saved = torch.load(Path(folder)/'best.pt', map_location=device, weights_only=False)
    model = Aligner(np.load(cache.folder/'centroids.npy'), len(cache.subjects), saved.get('variant', 'full')).to(device); model.load_state_dict(saved['model']); model.eval().requires_grad_(False)
    return model, saved


def audio_anchor_sampler(cache, split, rng, count):
    """Audio-only anchors on content of one split, for the decoder; no EEG or people involved."""
    spans = {}
    for d, rows in cache.records.items():
        for r in rows:
            if r['split'] == split: spans[(d, r['audio'], r['start'])] = r['duration']
    keys = sorted(spans); weight = np.array([spans[k] for k in keys]); weight = weight/weight.sum(); out = []
    while len(out) < count:
        d, c, start = keys[rng.choice(len(keys), p=weight)]; hi = start+spans[(d, c, start)]-SEG*FRAME-.1
        if hi <= start: continue
        out.append((d, c, round(rng.uniform(start, hi)/FRAME)*FRAME))
    return out


def train_decoder(cache, folder, deadline, device='cpu', threads=4, seed=11, max_steps=None):
    torch.manual_seed(seed); torch.set_num_threads(threads); folder = Path(folder); rng = np.random.default_rng(seed)
    model, _ = load_aligner(cache, folder, device); decoder = MelDecoder().to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-3, weight_decay=.01); step = 0; began = time.time()
    while time.time() < deadline and not (max_steps and step >= max_steps):
        anchors = audio_anchor_sampler(cache, 'train', rng, 32); wv, en = cache.audio_batch(anchors); target = cache.mel_batch(anchors).to(device)
        with torch.no_grad(): z = model.audio(wv.to(device), en.to(device))
        if rng.random() < .5: z = F.normalize(z+.15*torch.randn_like(z), dim=-1)
        for group in optimizer.param_groups: group['lr'] = lr_at((time.time()-began)/max(deadline-began, 1), step)
        loss = F.l1_loss(decoder(z), target); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); step += 1
        if step % 50 == 0: append_jsonl(folder/'decoder.jsonl', {'step': step, 'minutes': round((time.time()-began)/60, 1), 'l1': float(loss)})
        if step % 500 == 0: torch.save({'decoder': decoder.state_dict(), 'step': step}, folder/'decoder.pt')
    torch.save({'decoder': decoder.state_dict(), 'step': step}, folder/'decoder.pt')
    return step


def train_bridge(cfg, cache, folder, deadline, device='cpu', threads=5, workers=1, seed=11, max_steps=None):
    from torch.utils.data import DataLoader
    torch.manual_seed(seed); torch.set_num_threads(threads); folder = Path(folder)
    model, _ = load_aligner(cache, folder, device); bridge = Bridge().to(device)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=5e-4, weight_decay=.01); step = 0; began = time.time()
    loader = DataLoader(Stream(cache, set(cache.subjects), 'train', seed=seed+1), batch_size=None, num_workers=workers, persistent_workers=workers > 0)
    for batch in loader:
        if time.time() >= deadline or (max_steps and step >= max_steps): break
        batch = to(batch, device); wv, en = cache.audio_batch(batch['anchors'])
        with torch.no_grad():
            E = model.eeg(batch['eeg'], batch['positions'], batch['mask'], batch['subject']); A = model.audio(wv.to(device), en.to(device))
            comp = model.compensated(E, batch['subject'])
        bridge.train(); z = bridge(E, comp); target = A[batch['anchor']]
        cosine = (1-(z*target).sum(-1)).mean(); logits = torch.einsum('btd,mtd->bm', z, A)/SEG*10
        loss = cosine+.5*F.cross_entropy(logits, batch['anchor'])
        for group in optimizer.param_groups: group['lr'] = lr_at((time.time()-began)/max(deadline-began, 1), step, peak=5e-4)
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.); optimizer.step(); step += 1
        if step % 25 == 0: append_jsonl(folder/'bridge.jsonl', {'step': step, 'minutes': round((time.time()-began)/60, 1), 'cosine_loss': float(cosine), 'loss': float(loss),
                                                                 'batch_top1': float((logits.argmax(1) == batch['anchor']).float().mean()), 'chance': 1/len(A)})
        if step % 250 == 0: torch.save({'bridge': bridge.state_dict(), 'step': step}, folder/'bridge.pt')
    torch.save({'bridge': bridge.state_dict(), 'step': step}, folder/'bridge.pt')
    return step


# ----------------------------------------------------------------------------- evaluation

def _wave(cache_cfg, dataset, content, a, seconds=SEG*FRAME):
    import soundfile as sf
    folder = cache_cfg.dir('processed')/dataset; manifest = read_json(folder/'manifest.json')
    with sf.SoundFile(folder/manifest['contents'][content]['path']) as f:
        f.seek(round(a*24000)); return f.read(round(seconds*24000), dtype='float32'), manifest['contents'][content].get('words', [])


def _envelope_r(x, y):
    from .neural import AcousticFrontend
    front = AcousticFrontend(); a = front.numpy(x)[0][:, 0]; b = front.numpy(y)[0][:, 0]; n = min(len(a), len(b))
    return float(np.corrcoef(a[:n], b[:n])[0, 1])


@torch.no_grad()
def evaluate(cfg, cache, folder, out, device='cpu', demos_per_dataset=3, threads=8, audio=True):
    """Identification on unseen speech (seen and unseen people), learned lags and regions, and test audio."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import soundfile as sf
    from ..perceived_audio import ASR, FrozenVocos
    from ..metrics import word_errors
    from .data import reference
    torch.set_num_threads(threads); folder, out = Path(folder), Path(out); (out/'audio').mkdir(parents=True, exist_ok=True)
    model, saved = load_aligner(cache, folder, device); summary = {'variant': model.variant, 'alignment_step': saved['step'], 'checkpoint_validation': saved['validation']['mean_rank_percentile']}
    seen, unseen, dev = set(cache.subjects), set(cfg.fold_subjects('P1', 'F0', 'test')), set(cfg.fold_subjects('P1', 'F0', 'validation'))
    lag_sets = {'soft lag window': LAGS, 'fixed 0 ms': (0,), 'fixed 200 ms': (5,)}
    status(folder, stage='evaluation: identification')
    summary['seen_people_unseen_speech'] = identification(model, cache, fixed_set(cache, seen, 'test'), device, lag_sets=lag_sets)
    summary['unseen_people_unseen_speech'] = identification(model, cache, fixed_set(cache, unseen, 'test'), device, lag_sets=lag_sets)
    dev_items = fixed_set(cache, dev, 'validation'); base = identification(model, cache, dev_items, device, limit=240)
    status(folder, stage='evaluation: regions')
    importance = []
    for k in range(model.eeg.pool.centroids.shape[0]):
        importance.append(identification(model, cache, dev_items, device, drop_region=k, limit=240)['mean_rank_percentile']-base['mean_rank_percentile'])
    summary['region_importance'] = importance
    lags_ms = [d*FRAME*1000 for d in model.lags]; prior = model.lag_logits.softmax(-1).cpu().numpy(); profiles = {}
    for dataset in DATASETS:
        rows = [cache.subject_index[s] for s in cache.subjects if s.startswith(dataset+':')]
        if rows: profiles[dataset] = prior[rows].mean(0).tolist()
    summary['lag_prior'] = {'lags_ms': lags_ms, 'by_dataset': profiles}
    summary['audio_layer_weights'] = dict(zip([f'WavLM layer {k}' for k in LAYERS], model.audio.layer_logits.softmax(0).tolist()))
    # Figures: learned lag prior, region map, training curves.
    figure, axis = plt.subplots(figsize=(6, 3.2))
    for dataset, p in profiles.items(): axis.plot(lags_ms, p, marker='o', label=dataset)
    axis.set_xlabel('EEG lag after audio (ms)'); axis.set_ylabel('learned lag prior'); axis.legend(fontsize=8); figure.tight_layout(); figure.savefig(out/'lag_prior.png', dpi=150); plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    pos = torch.from_numpy(cache.positions['ds004408'][:1]).to(device); good = torch.ones(1, pos.shape[1], dtype=torch.bool, device=device)
    owner = model.eeg.pool.weights(pos, good)[0].argmax(0).cpu().numpy(); p = pos[0].cpu().numpy(); c = model.eeg.pool.centroids.detach().cpu().numpy()
    axes[0].scatter(p[:, 0], p[:, 1], c=owner, cmap='tab20', s=40); axes[0].scatter(c[:, 0], c[:, 1], marker='x', c='k', s=20)
    axes[0].set_title('electrode clusters (128-channel cap, top view)'); axes[0].set_aspect('equal'); axes[0].axis('off')
    scale = np.clip(np.asarray(importance), 0, None); axes[1].scatter(c[:, 0], c[:, 1], s=40+4000*scale, c=importance, cmap='viridis')
    for k, (x, y) in enumerate(c[:, :2]): axes[1].annotate(str(k), (x, y), fontsize=7)
    axes[1].set_title('region importance (loss of identification when removed)'); axes[1].set_aspect('equal'); axes[1].axis('off')
    figure.tight_layout(); figure.savefig(out/'regions.png', dpi=150); plt.close(figure)
    curve = [json.loads(l) for l in open(folder/'alignment.jsonl')] if (folder/'alignment.jsonl').exists() else []
    check = [json.loads(l) for l in open(folder/'validation.jsonl')] if (folder/'validation.jsonl').exists() else []
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    if curve: axes[0].plot([r['minutes'] for r in curve], [r['loss'] for r in curve], lw=.6); axes[0].set_xlabel('minutes'); axes[0].set_ylabel('training loss')
    if check: axes[1].plot([r['minutes'] for r in check], [r['mean_rank_percentile'] for r in check], marker='o'); axes[1].axhline(.5, ls='--', c='gray')
    axes[1].set_xlabel('minutes'); axes[1].set_ylabel('dev mean rank percentile (chance 0.5)'); figure.tight_layout(); figure.savefig(out/'training.png', dpi=150); plt.close(figure)
    # Test audio: the decoder from the audio side (upper bound) and from EEG through the bridge, with a wrong-time control.
    if audio and (folder/'decoder.pt').exists() and (folder/'bridge.pt').exists():
        status(folder, stage='evaluation: audio')
        decoder = MelDecoder().to(device); decoder.load_state_dict(torch.load(folder/'decoder.pt', map_location=device)['decoder']); decoder.eval()
        bridge = Bridge().to(device); bridge.load_state_dict(torch.load(folder/'bridge.pt', map_location=device)['bridge']); bridge.eval()
        vocos = FrozenVocos(cfg.model_dir('vocos')).to(device); asr = ASR(cfg, 'cpu')
        mean = torch.tensor(cache.stats['mel_mean'], device=device)[:, None]; std = torch.tensor(cache.stats['mel_std'], device=device)[:, None]
        render = lambda z: vocos(decoder(z)*std+mean)[0].cpu().numpy()
        items = fixed_set(cache, seen, 'test'); rows_out = []
        for dataset in DATASETS:
            rows = cache.records[dataset]; mine = [(i, a) for d, i, a in items if d == dataset]
            if not mine: continue
            anchors = sorted({(rows[i]['audio'], a) for i, a in mine}); picks = [anchors[int(n)] for n in np.linspace(0, len(anchors)-1, demos_per_dataset).round()]
            for content, a in picks:
                people = sorted({rows[i]['key'] for i, b in mine if b == a and rows[i]['audio'] == content})[:2]
                original, words = _wave(cfg, dataset, content, a); text = reference(words, a, SEG*FRAME)
                stem = f'{dataset}_{content}_{a:07.2f}'; sf.write(out/'audio'/f'{stem}_original.wav', original, 24000)
                wv, en = cache.audio_batch([(dataset, content, a)]); z = model.audio(wv.to(device), en.to(device)); oracle = render(z)
                sf.write(out/'audio'/f'{stem}_from_audio_embedding.wav', oracle, 24000)
                for key in people:
                    i = next(i for i, b in mine if b == a and rows[i]['key'] == key and rows[i]['audio'] == content)
                    other = [(j, b) for j, b in mine if rows[j]['key'] == key and (rows[j]['audio'] != content or abs(b-a) >= 10)]
                    result = {'dataset': dataset, 'content': content, 'start': a, 'person': key, 'reference': text}
                    for label, (j, b) in (('eeg', (i, a)), ('wrong_time_eeg', other[0] if other else (i, a))):
                        w, m = eeg_window(rows[j], b, cache.good[dataset][j])
                        subject = torch.tensor([cache.subject_index.get(key, len(cache.subjects))], device=device)
                        E = model.eeg(torch.from_numpy(w)[None].to(device), torch.from_numpy(cache.positions[dataset][j])[None].to(device), torch.from_numpy(m)[None].to(device), subject)
                        audio = render(bridge(E, model.compensated(E, subject)))
                        result[f'{label}_envelope_r'] = _envelope_r(audio, original)
                        if label == 'eeg':
                            sf.write(out/'audio'/f'{stem}_{key.split(":")[1]}_from_eeg.wav', audio, 24000); result['eeg_transcript'] = asr(audio)
                        else:
                            sf.write(out/'audio'/f'{stem}_{key.split(":")[1]}_wrong_time.wav', audio, 24000)
                    result['oracle_envelope_r'] = _envelope_r(oracle, original); result['oracle_transcript'] = asr(oracle); result['original_transcript'] = asr(original)
                    ref = asr.normalize(text).strip()
                    if ref:
                        for label in ('original', 'oracle', 'eeg'): result[f'{label}_wer'] = word_errors(ref, result[f'{label}_transcript'])['wer']
                    rows_out.append(result); append_jsonl(out/'audio_results.jsonl', result)
        summary['audio'] = rows_out
    atomic_json(out/'summary.json', summary); write_report(summary, out); status(folder, stage='done')
    return summary


def write_report(summary, out):
    lines = [f"# 两塔软对齐：一夜测试结果（{summary['variant']}）", '', f"对齐模型训练到第 {summary['alignment_step']} 步。指标是平均排名百分位（0 = 总排第一，0.5 = 偶然）和 top-1 / top-10。", '']
    for name, title in (('seen_people_unseen_speech', '训练过的人，没听过的语音'), ('unseen_people_unseen_speech', '没见过的人，没听过的语音')):
        lines += [f'## {title}', '', '| 数据集 | 候选 | 窗口 | 软时延窗口 | 固定 0 ms | 固定 200 ms | top-1（偶然） | top-10（偶然） | 好于偶然的人 |', '|---|---|---|---|---|---|---|---|---|']
        for d in DATASETS:
            v = summary[name].get(d)
            if not v: continue
            s = v['soft lag window']
            lines.append(f"| {d} | {v['candidates']} | {v['windows']} | {s['mean_rank_percentile']:.3f} | {v['fixed 0 ms']['mean_rank_percentile']:.3f} | {v['fixed 200 ms']['mean_rank_percentile']:.3f} | "
                         f"{s['top1']:.3f}（{s['chance_top1']:.3f}） | {s['top10']:.3f}（{s['chance_top10']:.3f}） | {s['people_better_than_chance']} / {s['people']} |")
        lines.append('')
    lines += ['## 学到的结构', '', f"- 音频侧各层权重：{', '.join(f'{k} {v:.2f}' for k, v in summary['audio_layer_weights'].items())}",
              '- 时延先验见 `lag_prior.png`，电极聚类与区域重要性见 `regions.png`，训练曲线见 `training.png`。', '']
    if summary.get('audio'):
        lines += ['## 测试音频', '', '| 数据集 | 起点 | 被试 | 从音频嵌入还原的包络 r | 从 EEG 生成的包络 r | 错位 EEG 的包络 r | 原音转写 | 从 EEG 生成的转写 |', '|---|---|---|---|---|---|---|---|']
        for r in summary['audio']:
            lines.append(f"| {r['dataset']} | {r['start']:.2f} | {r['person'].split(':')[1]} | {r['oracle_envelope_r']:.2f} | {r['eeg_envelope_r']:.2f} | {r['wrong_time_eeg_envelope_r']:.2f} | {r['original_transcript']} | {r['eeg_transcript']} |")
        lines += ['', '音频文件在 `audio/`：`_original` 原音，`_from_audio_embedding` 从音频侧嵌入还原（上限），`_from_eeg` 从 EEG 生成，`_wrong_time` 用另一时间的 EEG 生成（对照）。', '']
    (Path(out)/'report.md').write_text('\n'.join(lines))

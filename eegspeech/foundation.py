"""Pretrained EEG foundation models as encoders of imagined speech: REVE, CBraMod, LaBraM (via braindecode).

    reve     El Ouahidi et al. 2025 (NeurIPS): 69M parameters, 22 layers, 512 dims, masked autoencoding on
             60k h / 25k people; 4D (x, y, z, t) Fourier position encoding, so any montage; 1 s patches with
             0.1 s overlap; pretrained on 0.5-99.5 Hz at 200 Hz, z-scored, clipped at 15 SD
    cbramod  Wang et al. 2025 (ICLR): 4.9M parameters, 12 criss-cross (spatial | temporal) layers, 200 dims;
             convolutional position encoding (any channel count); 1 s patches; TUEG 0.3-75 Hz, 200 Hz, 100 uV units
    labram   Jiang et al. 2024 (ICLR): 5.8M parameters, 12 layers, 200 dims; learned embedding per 10-05
             electrode name (128 names); 1 s patches; 0.1-75 Hz, 200 Hz, 100 uV units

Every model sees a trial as (channels, 1 s patches) tokens.  Inputs follow each model's pretraining:
resampled to 200 Hz, band-passed, line frequency notched, and scaled (REVE: each trial and channel
z-scored, as REVE z-scores its recordings; CBraMod and LaBraM: units of 100 uV).  None of this uses any
other trial, so the inputs and the frozen features do not depend on the fold and nothing of a held-out
trial enters any statistic (the folds' Euclidean alignment is not applied: the models were pretrained on
unaligned EEG).

Channels: those valid in most of a person's trials (invalid ones in a trial are zeroed).  REVE takes the
stored electrode positions (metres, MNE head frame, as its position bank); LaBraM takes 10-05 names, so
a channel without one (Biosemi A1..D32, half positions) is given the nearest free canonical electrode
within 2 cm, or dropped; CBraMod takes the channels in store order.

Frozen features (``embed``) of every trial, for a few layers: the token embeddings (C, P, E) pooled as
``mean`` (E), ``time`` (P, E: mean over channels) and ``chan`` (C, E: mean over patches).  ``ChannelBasis``
reduces ``chan`` to the top principal directions of a person's training tokens.

Fine-tuning (``Finetuner``): the backbone (CBraMod, LaBraM: all of it; REVE: its top layers, from cached
lower-layer tokens) and a personal head (LayerNorm per token, flatten, linear map to the CLIP space)
trained with cross-entropy of s * cos(z, a) against the anchors (items, or every training sentence: CLIP
with a frozen speech tower), early stopped on the person's last inner block.
"""
from __future__ import annotations

import json
import math
import time

import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import ROOT
from .signal import bandpass, notch, positions, resample
from .store import open_store

MODELS = {
    'reve': dict(weights='reve_base', band=(.5, 99.5), scale='trial', step=180, multiple=False, dim=512,
                 depth=22, layers=(11, 22), batch=8),
    'cbramod': dict(weights='cbramod_pretrained', band=(.3, 75.), scale=.01, step=200, multiple=True, dim=200,
                    depth=12, layers=(6, 12), batch=32),
    'labram': dict(weights='labram_pretrained', band=(.3, 75.), scale=.01, step=200, multiple=True, dim=200,
                   depth=12, layers=(6, 12), batch=32),
}
RATE = 200
PATCH = 200
LINE = {'thinking_out_loud': 50., 'bci2020': 60., 'chisco': 50.}   # mains frequency of the recording site
MODELS_DIR = ROOT / 'models'
CACHE = ROOT / 'artifacts' / 'features'
LABRAM_RADIUS = .02              # m: a channel without a 10-05 name takes the nearest free one this close


# ----------------------------------------------------------------------------- inputs
def prepare(x, rate, name, key):
    """(C, T') model input of one trial (C, T) in uV: 200 Hz, band, notch, crop, scale."""
    spec = MODELS[key]
    y = resample(np.asarray(x, np.float64), rate, RATE)
    y = bandpass(y, RATE, *spec['band'])
    y = notch(y, RATE, LINE.get(name, 50.))
    if spec['multiple']:                                   # whole patches: the centre of the trial
        keep = y.shape[-1] // PATCH * PATCH
        start = (y.shape[-1] - keep) // 2
        y = y[:, start:start + keep]
    if spec['scale'] == 'trial':
        std = y.std(-1, keepdims=True)
        y = np.where(std > 0, (y - y.mean(-1, keepdims=True)) / np.where(std > 0, std, 1), 0).clip(-15, 15)
    else:
        y = y * spec['scale']
    return y.astype(np.float32)


def montage(store, subject, key):
    """(channel indices into the store, model channel info) of a person: positions (C, 3) for REVE,
    10-05 names for LaBraM, None for CBraMod."""
    valid = store.file['subjects'][subject]['valid'][:].mean(0) > .5
    index = np.flatnonzero(valid)
    if key == 'reve':
        return index, store.xyz(subject)[index].astype(np.float32)
    if key == 'cbramod':
        return index, None
    from braindecode.models.labram import LABRAM_CHANNEL_ORDER
    canonical = [c.upper() for c in LABRAM_CHANNEL_ORDER]
    where, found = positions(canonical)
    names = [str(c).strip().upper() for c in store.channels(subject)]
    xyz = store.xyz(subject)
    taken, chosen = {}, {}
    for i in index:                                       # exact names at the right place first (Biosemi 'A1'
        j = canonical.index(names[i]) if names[i] in canonical else -1          # is not the ear electrode A1)
        if j >= 0 and found[j] and np.linalg.norm(xyz[i] - where[j]) <= LABRAM_RADIUS and names[i] not in taken:
            taken[names[i]], chosen[i] = i, names[i]
    pairs = sorted((float(np.linalg.norm(xyz[i] - where[j])), i, canonical[j])
                   for i in index if i not in chosen for j in np.flatnonzero(found))
    for distance, i, name in pairs:                       # then the nearest free canonical electrode
        if distance > LABRAM_RADIUS:
            break
        if i not in chosen and name not in taken:
            taken[name], chosen[i] = i, name
    index = np.array(sorted(chosen), int)
    return index, [chosen[i] for i in index]


def inputs(name, key, modalities, rows=None, cache=False):
    """{subject: (table rows, (n, C, T) float16 inputs, channel info)} of the labelled trials of these
    modalities (optionally only ``rows``).  ``cache``: kept on disk (``artifacts/features/fm_input_*``) and
    returned memory-mapped, for fine-tuning on Chisco."""
    store = open_store(name)
    table = store.table[(store.table.item >= 0) & store.table.modality_name.isin(modalities)]
    if rows is not None:
        table = table[table.index.isin(rows)]
    out = {}
    for subject, part in table.groupby('subject'):
        index, info = montage(store, subject, key)
        stem = CACHE / f'fm_input_{key}_{name}_{"-".join(sorted(modalities))}_{subject}'
        if cache and stem.with_suffix('.npy').exists() and rows is None:
            saved = np.load(stem.with_suffix('.rows.npy'))
            if np.array_equal(saved, part.index.to_numpy()):
                out[subject] = (saved, np.load(stem.with_suffix('.npy'), mmap_mode='r'), info)
                continue
        valid = store.file['subjects'][subject]['valid'][:]
        x = None
        for i, row in enumerate(part.itertuples()):
            eeg = store.segment(row)
            eeg[~valid[int(row.segment)]] = 0
            y = prepare(eeg[index], store.rate, name, key)
            if x is None:
                x = np.zeros((len(part),) + y.shape, np.float16)
            x[i] = y
        if cache and rows is None:
            CACHE.mkdir(parents=True, exist_ok=True)
            np.save(stem.with_suffix('.tmp.npy'), x)
            stem.with_suffix('.tmp.npy').replace(stem.with_suffix('.npy'))
            np.save(stem.with_suffix('.rows.npy'), part.index.to_numpy())
            x = np.load(stem.with_suffix('.npy'), mmap_mode='r')
        out[subject] = (part.index.to_numpy(), x, info)
    return out


# ----------------------------------------------------------------------------- backbones
def load_backbone(key, n_chans, n_times):
    """A braindecode model with the pretrained weights (only the classification head is new)."""
    from safetensors.torch import load_file
    from braindecode.models import CBraMod, Labram, REVE
    state = load_file(str(MODELS_DIR / MODELS[key]['weights'] / 'model.safetensors'))
    if key == 'reve':
        model = REVE(n_outputs=1, n_chans=n_chans, n_times=n_times, sfreq=RATE)
        state.pop('cls_query_token', None)
    elif key == 'cbramod':
        model = CBraMod(n_outputs=1, n_chans=n_chans, n_times=n_times, sfreq=RATE)
    else:
        config = json.load(open(MODELS_DIR / MODELS[key]['weights'] / 'config.json'))
        model = Labram(n_outputs=1, n_times=n_times, chs_info=config['chs_info'])
        state['temporal_embedding'] = state['temporal_embedding'][:, :model.temporal_embedding.shape[1]]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith('final_layer') for k in missing):
        raise RuntimeError(f'{key}: weights do not fit: missing {missing}, unexpected {unexpected}')
    return model


class Backbone(nn.Module):
    """Token embeddings (B, C, P, E) of the requested layers (1 .. depth) of a pretrained model."""

    def __init__(self, key, n_chans, n_times, info, model=None):
        super().__init__()
        self.key, self.model = key, model if model is not None else load_backbone(key, n_chans, n_times)
        self.depth = MODELS[key]['depth']
        if key == 'reve':
            self.register_buffer('pos', torch.as_tensor(info, dtype=torch.float32))
        self.names = info if key == 'labram' else None

    def lower(self, x, upto):
        """REVE: the tokens (B, C * P, E) after ``upto`` layers (0: the embedded patches)."""
        m = self.model
        patches = x.unfold(2, m.patch_size, m.patch_size - m.patch_overlap)
        b, c, p, _ = patches.shape
        from braindecode.models.reve import FourierEmb4D
        pos = FourierEmb4D.add_time_patch(self.pos[None].expand(b, -1, -1), p)
        h = m.to_patch_embedding(patches).reshape(b, c * p, -1) + m.ln(m.fourier4d(pos) + m.mlp4d(pos))
        for attn, ff in m.transformer.layers[:upto]:
            h = attn(h) + h
            h = ff(h) + h
        return h, c, p

    def upper(self, h, start, layers):
        """REVE: continue from the tokens after ``start`` layers; outputs of ``layers`` (> start)."""
        out = {}
        for i, (attn, ff) in enumerate(self.model.transformer.layers[start:], start + 1):
            h = attn(h) + h
            h = ff(h) + h
            if i in layers:
                out[i] = h
        return out

    def forward(self, x, layers):
        """{layer: (B, C, P, E)} for the requested layers."""
        layers = set(layers)
        if self.key == 'reve':
            h, c, p = self.lower(x, 0)
            return {i: v.reshape(len(x), c, p, -1) for i, v in self.upper(h, 0, layers).items()}
        if self.key == 'cbramod':
            m = self.model
            h = m.patch_embedding(m.rearrange(x))
            out = {}
            for i, layer in enumerate(m.encoder.layers, 1):
                h = layer(h)
                if i in layers:
                    out[i] = h
            return out
        captured = {}
        hooks = [self.model.blocks[i - 1].register_forward_hook(
            lambda _m, _i, o, i=i: captured.__setitem__(i, o)) for i in layers if i < self.depth]
        try:
            last = self.model(x, return_features=True, ch_names=self.names)['features']   # final norm applied
        finally:
            for h in hooks:
                h.remove()
        b, c = len(x), x.shape[1]
        out = {i: v[:, 1:].reshape(b, c, -1, v.shape[-1]) for i, v in captured.items()}
        if self.depth in layers:
            out[self.depth] = last.reshape(b, c, -1, last.shape[-1])
        return out


# ----------------------------------------------------------------------------- frozen features
def feature_path(name, key):
    return CACHE / f'fm_{key}_{name}.h5'


@torch.no_grad()
def embed(name, key, modalities, device, log=print):
    """Frozen pooled features of every labelled trial of these modalities (``feature_path``)."""
    spec = MODELS[key]
    path = feature_path(name, key)
    tmp = path.with_suffix('.tmp.h5')
    path.parent.mkdir(parents=True, exist_ok=True)
    store = open_store(name)
    started = time.time()
    with h5py.File(tmp, 'w') as f:
        f.attrs.update(model=key, layers=json.dumps(list(spec['layers'])), modalities=json.dumps(list(modalities)))
        for subject in store.subjects:
            data = inputs(name, key, modalities, rows=store.table.index[store.table.subject == subject])
            if subject not in data:
                continue
            rows, x, info = data[subject]
            net = Backbone(key, x.shape[1], x.shape[2], info).to(device).eval()
            g = f.create_group(subject)
            g.create_dataset('rows', data=rows)
            g.attrs['channels'] = len(x[0])
            sets = {}
            for a in range(0, len(x), spec['batch']):
                out = net(torch.from_numpy(np.asarray(x[a:a + spec['batch']], np.float32)).to(device), spec['layers'])
                for layer, h in out.items():
                    h = h.float()
                    pooled = dict(mean=h.mean((1, 2)), time=h.mean(1), chan=h.mean(2))
                    for kind, v in pooled.items():
                        v = v.cpu().numpy()
                        if (layer, kind) not in sets:
                            sets[layer, kind] = g.create_dataset(
                                f'L{layer}/{kind}', (len(x),) + v.shape[1:], np.float32 if kind == 'mean' else np.float16)
                        sets[layer, kind][a:a + len(v)] = v
            log(f'{name} {key} {subject}: {len(x)} trials, {x.shape[1]} channels, {x.shape[2] / RATE:.1f} s '
                f'({time.time() - started:.0f} s)')
            del net
            if device.type == 'mps':
                torch.mps.empty_cache()
    tmp.replace(path)
    return path


class Features:
    """Read access to ``embed``'s cache: per person, the table rows and pooled features of each layer."""

    def __init__(self, name, key):
        self.file = h5py.File(feature_path(name, key), 'r')
        self.layers = json.loads(self.file.attrs['layers'])
        self.subjects = sorted(self.file)

    def rows(self, subject):
        return self.file[subject]['rows'][:]

    def get(self, subject, layer, kind, index=None):
        d = self.file[subject][f'L{layer}/{kind}']
        if index is None:
            return d[:].astype(np.float32)
        order = np.argsort(index)
        out = np.empty((len(index),) + d.shape[1:], np.float32)
        out[order] = d[np.asarray(index)[order]]
        return out


class ChannelBasis:
    """Top principal directions (E, k) of a person's per-channel token embeddings (C, E) over training
    trials (unsupervised); ``__call__`` maps (n, C, E) to (n, C * k)."""

    def __init__(self, mean, basis):
        self.mean, self.basis = mean, basis

    @classmethod
    def fit(cls, chan, k):
        x = np.asarray(chan, np.float64).reshape(-1, chan.shape[-1])
        mean = x.mean(0)
        cov = (x - mean).T @ (x - mean) / len(x)
        values, vectors = np.linalg.eigh(cov)
        return cls(mean.astype(np.float32), vectors[:, ::-1][:, :k].astype(np.float32))

    def __call__(self, chan):
        return ((np.asarray(chan, np.float32) - self.mean) @ self.basis).reshape(len(chan), -1)


# ----------------------------------------------------------------------------- fine-tuning
class Head(nn.Module):
    """Personal readout: LayerNorm per token, then ``flat`` (all tokens flattened) or ``mean`` (the mean token
    and the mean over channels of each patch: far fewer weights), dropout, linear map to the CLIP space."""

    def __init__(self, n_tokens, dim, out, dropout, pool='flat', patches=1):
        super().__init__()
        self.pool, self.patches = pool, patches
        self.norm, self.drop = nn.LayerNorm(dim), nn.Dropout(dropout)
        n_in = n_tokens * dim if pool == 'flat' else (1 + patches) * dim
        self.linear = nn.Linear(n_in, out)
        nn.init.normal_(self.linear.weight, std=1 / math.sqrt(n_in))
        nn.init.zeros_(self.linear.bias)

    def forward(self, h):                      # (B, C, P, E) or (B, C * P, E)
        h = self.norm(h)
        if self.pool == 'flat':
            return self.linear(self.drop(h.flatten(1)))
        h = h.reshape(len(h), -1, self.patches, h.shape[-1])
        return self.linear(self.drop(torch.cat([h.mean((1, 2)), h.mean(1).flatten(1)], 1)))


class Finetuner(nn.Module):
    """Shared pretrained backbone + personal heads; ``start`` > 0 (REVE) takes cached tokens after that
    many layers and trains only the layers above."""

    def __init__(self, key, people, out, dropout, start=0, head='flat'):
        """``people``: {subject: (n_chans, n_times, info)}."""
        super().__init__()
        self.key, self.start, self.depth = key, start, MODELS[key]['depth']
        first = next(iter(people.values()))
        shared = load_backbone(key, first[0], first[1])     # one set of weights; the montage is per person
        self.backbones = nn.ModuleDict({s: Backbone(key, c, t, info, model=shared) for s, (c, t, info) in people.items()})
        step = MODELS[key]['step']
        patches = lambda t: len(range(0, t - PATCH + 1, step)) if key == 'reve' else t // PATCH
        self.heads = nn.ModuleDict({s: Head(c * patches(t), MODELS[key]['dim'], out, dropout, head, patches(t))
                                    for s, (c, t, info) in people.items()})
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.)))
        if start:
            for p in shared.parameters():
                p.requires_grad_(False)
            for layer in shared.transformer.layers[start:]:
                for p in layer.parameters():
                    p.requires_grad_(True)

    def forward(self, s, x):
        b = self.backbones[s]
        if self.start:
            h = b.upper(x, self.start, {self.depth})[self.depth]
        else:
            h = b(x, [self.depth])[self.depth]
        return self.heads[s](h)

    def shared(self):
        return next(iter(self.backbones.values())).model

    def snapshot(self):
        """Copies of the trained weights (the shared backbone once, or only its trained layers)."""
        trained = self.shared().transformer.layers[self.start:] if self.start else self.shared()
        copy = lambda m: {k: v.detach().clone() for k, v in m.state_dict().items()}
        return dict(backbone=copy(trained), heads=copy(self.heads), scale=self.logit_scale.detach().clone())

    def restore(self, state):
        (self.shared().transformer.layers[self.start:] if self.start else self.shared()).load_state_dict(state['backbone'])
        self.heads.load_state_dict(state['heads'])
        with torch.no_grad():
            self.logit_scale.copy_(state['scale'])

    def backbone_parameters(self):
        return [p for p in next(iter(self.backbones.values())).model.parameters() if p.requires_grad]


@torch.no_grad()
def reve_lower(name, modalities, upto, device, log=print):
    """{subject: (rows, (n, C * P, E) float16 tokens after ``upto`` frozen REVE layers, positions)}."""
    out = {}
    for subject, (rows, x, info) in inputs(name, 'reve', modalities).items():
        net = Backbone('reve', x.shape[1], x.shape[2], info).to(device).eval()
        h = np.concatenate([net.lower(torch.from_numpy(np.asarray(x[a:a + 16], np.float32)).to(device), upto)[0]
                            .half().cpu().numpy()
                            for a in range(0, len(x), 16)])
        out[subject] = (rows, h, info)
        log(f'{name} reve tokens after layer {upto} {subject}: {h.shape}')
        del net
    return out

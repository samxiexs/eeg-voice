"""End-to-end CLIP encoder of imagined sentences: the learned counterpart of time-resolved filter-bank log-power.

    temporal  a bank of band-pass filters shared by every person, initialised as log-spaced windowed-sinc
              band-passes and learned: ShallowConvNet is a deep filter-bank CSP (Schirrmeister et al. 2017)
    spatial   per person, ``depth`` spatial filters for every temporal filter (depthwise, EEGNet: Lawhern et
              al. 2018); the montage- and head-specific part is personal (the subject layer of Défossez et
              al. 2023)
    power     square, average over ``pool`` s every ``stride`` s, log, normalised: the time course of band
              power, where the Chisco information is (``features``: time-resolved 0.512 vs whole trial 0.503)
    readout   per person, a linear map of that time course to the sentence CLIP space; a trial's logits are
              scale * cos(z, a_s) over sentence anchors

Training is CLIP with a frozen speech tower: each training trial against the anchors of every training
sentence (cross-entropy over the bank), with a random crop of the epoch, early stopped on validation runs
by the measure of the test (rank of the true sentence among the sentences of its run).

``epochs`` caches a store's imagery epochs once, unaligned, at ``rate`` (anti-aliased) as float16; the
Euclidean alignment of a fold (per person, without its held-out trials) is applied batch by batch.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import ROOT, STORE
from .clip import calibrate_sets, rank_percentile, run_sets
from .signal import resample, whitening
from .store import open_store

CACHE = ROOT / 'artifacts' / 'features'


def epochs(name, modality, rate):
    """(table rows, (n, C, T) float16 memmap) of a store's labelled ``modality`` epochs at ``rate``, cut to
    the shortest epoch; cached in ``artifacts/features/<name>_<modality>_<rate>hz.npy``."""
    path = CACHE / f'{name}_{modality}_{rate:g}hz.npy'
    index = path.with_suffix('.rows.npy')
    if not path.exists() or path.stat().st_mtime < (STORE / f'{name}.h5').stat().st_mtime:
        store = open_store(name)
        rows = store.table[(store.table.modality_name == modality) & (store.table.item >= 0)]
        samples = int(rows.length.min() * rate / store.rate)
        channels = max(len(store.channels(s)) for s in rows.subject.unique())
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + '.tmp.npy')
        out = np.lib.format.open_memmap(tmp, 'w+', np.float16, (len(rows), channels, samples))
        for i, row in enumerate(rows.itertuples()):
            x = resample(store.segment(row), store.rate, rate)
            out[i] = 0
            out[i, :len(x)] = x[:, :samples]
        out.flush()
        del out
        np.save(index, rows.index.to_numpy())
        tmp.replace(path)
    return np.load(index), np.load(path, mmap_mode='r')


def alignment(data, positions, valid, shrink=.1, chunk=512):
    """(C, C) Euclidean-alignment matrix from the epochs ``data[positions]`` (one person, no held-out trials)."""
    total, count = 0., 0
    for start in range(0, len(positions), chunk):
        x = torch.from_numpy(np.asarray(data[np.sort(positions[start:start + chunk])], np.float32))
        x = x - x.mean(-1, keepdim=True)
        total = total + torch.einsum('bct,bdt->cd', x, x).double().numpy()
        count += x.shape[0] * x.shape[-1]
    return whitening(total / count * np.outer(valid, valid), shrink, valid)


def sinc_bank(filters, taps, rate, low=1.):
    """(filters, taps) Hamming-windowed sinc band-passes with log-spaced edges from ``low`` to 0.45 rate."""
    edges = np.geomspace(low, .45 * rate, filters + 1)
    t = (np.arange(taps) - (taps - 1) / 2) / rate
    bank = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        h = (2 * hi * np.sinc(2 * hi * t) - 2 * lo * np.sinc(2 * lo * t)) * np.hamming(taps)
        bank.append(h / np.sqrt(np.square(h).sum()))
    return np.stack(bank).astype(np.float32)


class SentenceNet(nn.Module):
    """Shared learned filter bank, personal depthwise spatial filters, log-power time course, personal readout."""

    def __init__(self, channels, persons, samples, dim, rate, filters=32, depth=2, kernel=.25, pool=.25,
                 stride=.125, dropout=.5):
        super().__init__()
        taps = int(round(kernel * rate)) | 1
        maps = filters * depth
        self.depth, self.pool, self.stride = depth, int(round(pool * rate)), int(round(stride * rate))
        bins = (samples - taps + 1 - self.pool) // self.stride + 1
        self.temporal = nn.Parameter(torch.from_numpy(sinc_bank(filters, taps, rate)))
        self.spatial = nn.Parameter(torch.randn(persons, maps, channels) / math.sqrt(channels))
        self.norm = nn.BatchNorm1d(maps)
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Parameter(torch.randn(persons, maps * bins, dim) / math.sqrt(maps * bins))
        self.scale = nn.Parameter(torch.tensor(math.log(10.)))

    def forward(self, x, person):
        """x (B, C, T) aligned epochs, person (B,) -> unit embeddings (B, dim)."""
        s = torch.einsum('bct,bmc->bmt', x, self.spatial[person])                   # spatial projections
        bank = self.temporal.repeat_interleave(self.depth, 0)[:, None]               # each map: its filter
        y = F.conv1d(s, bank, groups=s.shape[1])
        p = self.norm(F.avg_pool1d(y.square(), self.pool, self.stride).clamp_min(1e-8).log())
        z = torch.einsum('bf,bfd->bd', self.dropout(p.flatten(1)), self.readout[person])
        return F.normalize(z, dim=-1)


class Examples:
    """Aligned (B, C, crop) examples from the epoch cache: a random crop for training, the centre otherwise."""

    def __init__(self, data, matrices, person, crop, device):
        self.data, self.person, self.crop, self.device = data, person, crop, device
        self.matrices = torch.as_tensor(np.stack(matrices), dtype=torch.float32, device=device)

    def __call__(self, positions, rng=None):
        order = np.argsort(positions)                                               # memmap reads in order
        x = np.asarray(self.data[positions[order]], np.float32)[np.argsort(order)]
        samples = x.shape[-1]
        starts = (rng.integers(0, samples - self.crop + 1, len(x)) if rng is not None
                  else np.full(len(x), (samples - self.crop) // 2))
        x = np.stack([e[:, a:a + self.crop] for e, a in zip(x, starts)])
        person = torch.as_tensor(self.person[positions], device=self.device)
        x = torch.einsum('bcd,bdt->bct', self.matrices[person], torch.from_numpy(x).to(self.device))
        return x, person


def embed(model, examples, positions, batch=256):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(*examples(positions[i:i + batch])).cpu()
                          for i in range(0, len(positions), batch)]).numpy()


def run_scores(z, positions, sentence, run, anchor, scale=1.):
    """[(positions, candidates, true column, scores)] per run, the run's sentences as candidates."""
    out = []
    for trials, candidates, position in run_sets(sentence[positions], run[positions]):
        out.append((positions[trials], candidates, position, scale * z[trials] @ anchor[candidates].T))
    return out


def fit(data, matrices, person, sentence, run, anchor, fit_positions, validation, cfg, device, log, seed=0):
    """Train on ``fit_positions`` (epoch-cache positions), early-stop on ``validation`` positions.

    ``person``, ``sentence`` (row of ``anchor``) and ``run`` are indexed by cache position.  Returns the
    model, its examples and the temperature calibrated on the validation runs."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    crop = int(round(cfg['crop'] * cfg['rate']))
    examples = Examples(data, matrices, person, crop, device)
    model = SentenceNet(data.shape[1], len(matrices), crop, anchor.shape[1], cfg['rate'], cfg['filters'], cfg['depth'],
                        cfg['kernel'], cfg['pool'], cfg['stride'], cfg['dropout']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    bank_ids = np.unique(sentence[fit_positions])                                    # the training sentences
    bank = torch.as_tensor(anchor[bank_ids], device=device)
    target = np.searchsorted(bank_ids, sentence)
    best, best_state, waited = -np.inf, None, 0
    for epoch in range(cfg['epochs']):
        started, model_loss = time.time(), []
        model.train()
        order = rng.permutation(fit_positions)
        for i in range(0, len(order), cfg['batch']):
            chosen = order[i:i + cfg['batch']]
            x, p = examples(chosen, rng)
            logits = model.scale.exp() * model(x, p) @ bank.T
            loss = F.cross_entropy(logits, torch.as_tensor(target[chosen], device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model_loss.append(float(loss.detach()))
        z = embed(model, examples, validation)
        score = float(np.mean(np.concatenate([rank_percentile(S, pos) for *_, pos, S in
                                              run_scores(z, validation, sentence, run, anchor)])))
        log(f'deep epoch {epoch + 1}: loss {np.mean(model_loss):.4f}, validation rank percentile {score:.4f} '
            f'({time.time() - started:.0f} s)')
        if score > best:
            best, best_state, waited = score, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            waited += 1
            if waited >= cfg['patience']:
                break
    model.load_state_dict(best_state)
    z = embed(model, examples, validation)
    sets = run_scores(z, validation, sentence, run, anchor)
    scale, _ = calibrate_sets([S for *_, S in sets], [pos for _, _, pos, _ in sets])
    return model, examples, scale, best

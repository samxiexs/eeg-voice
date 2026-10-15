"""EEG -> speech reconstruction on top of the two-tower alignment (generation, not classification).

Data are small, so the EEG side is kept small and is supervised by two losses only.

1. Audio side, no EEG. A Mel decoder learns to render the native Vocos Mel from a synthesis code of the heard
   audio: the 128-d shared embedding of the aligned audio tower ('embedding', after the projection) or the frozen
   WavLM layers ('wavlm', before it), so the loss of the projection is measured at an equal budget. Datasets are
   drawn equally often, no noise is added by default, and the checkpoint is chosen on validation content.
2. EEG side. The aligned EEG tower stays frozen; a small bridge maps the 4.5 s EEG window to the 3 s code sequence,
   which the frozen decoder renders. Losses: the code frame by frame, and a word-level contrast (the code averaged
   over each word's time span must be closer to that word's true code than to the other words of the batch).
   Words are cut on the audio time line only, never on the EEG: every output frame can attend to the whole EEG
   window, so overlapping responses and latencies are left to the bridge.

Sentences are reconstructed by sliding the 3 s window in 1 s steps and averaging the overlapping codes before
decoding. Evaluation at three scales (frames, words, sentences), always against the output for another time of
the same person (wrong-time) and for no EEG at all (null).
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
from ..utils import atomic_json, append_jsonl, read_json
from .softalign import (DATASETS, SEG, FRAME, WIN, MEL_RATE, LAYERS, MelDecoder, Stream, audio_anchor_sampler, covers, eeg_window, fixed_set, load_aligner,
                        lr_at, status, tie_rank, two_way_bootstrap, permutation_plan, permutation_null, to, _wave)

WORDS_PER_WINDOW = 16


# ----------------------------------------------------------------------------- measures on Mel

def descriptor(mel):
    """Time-varying part of a standardized Mel at 2 bins x 4 frames (about 43 ms), as a unit vector:
    the dot product of two descriptors is the Pearson correlation of their spectro-temporal dynamics."""
    x = F.avg_pool2d(mel[:, None].float(), (2, 4))[:, 0]; x = (x-x.mean(-1, keepdim=True)).flatten(1)
    return F.normalize(x-x.mean(1, keepdim=True), dim=1)


def natural(mel, stats):
    return mel.float()*torch.as_tensor(stats['mel_std'], device=mel.device)[:, None]+torch.as_tensor(stats['mel_mean'], device=mel.device)[:, None]


def envelope(mel, stats):
    """Broadband log energy per Mel frame (from the de-standardized natural-log Mel), z-scored over time."""
    e = torch.logsumexp(natural(mel, stats), 1); return (e-e.mean(-1, keepdim=True))/(e.std(-1, keepdim=True)+1e-6)


def detail(mel, stats):
    """Mean absolute log-Mel difference between neighbouring bins: harmonic and formant structure (smooth -> small)."""
    return natural(mel, stats).diff(dim=1).abs().mean((1, 2))


def spread(x):
    """Mean over Mel cells (pooled) of the standard deviation across windows."""
    return F.avg_pool2d(x[:, None].float(), (2, 4)).flatten(1).std(0).mean()


def mel_frames(code_frames):
    return round(code_frames*FRAME*MEL_RATE)


# ----------------------------------------------------------------------------- words on the audio time line

def word_spans(cfg, cache):
    """{(dataset, content): (starts, ends, texts)} from the processed manifests."""
    out = {}
    for dataset, rows in cache.records.items():
        contents = read_json(cfg.dir('processed')/dataset/'manifest.json')['contents']
        for c in {r['audio'] for r in rows}:
            w = contents[c].get('words', [])
            out[dataset, c] = (np.array([float(x['start']) for x in w]), np.array([float(x['end']) for x in w]), [str(x.get('word', x.get('text', ''))) for x in w])
    return out


def pooling(spans, anchors, frames=SEG):
    """Averaging weights [M, K, frames] of the words lying wholly inside each anchor's segment, their validity [M, K]
    and their word indices [M, K] (-1 when empty)."""
    W = np.zeros((len(anchors), WORDS_PER_WINDOW, frames), np.float32); index = np.full((len(anchors), WORDS_PER_WINDOW), -1)
    for m, (d, c, a) in enumerate(anchors):
        starts, ends, _ = spans[d, c]
        for k, w in enumerate(np.flatnonzero((starts >= a-1e-6) & (ends <= a+frames*FRAME+1e-6))[:WORDS_PER_WINDOW]):
            s = int(math.floor((starts[w]-a)/FRAME+1e-6)); e = min(frames, max(s+1, int(math.ceil((ends[w]-a)/FRAME-1e-6))))
            W[m, k, s:e] = 1/(e-s); index[m, k] = w
    return W, index >= 0, index


def pooled(code, W):
    """[M, K, D]: the code averaged over each word's frames."""
    return torch.bmm(torch.as_tensor(W, device=code.device), code)


# ----------------------------------------------------------------------------- models

class AudioDecoder(nn.Module):
    """Synthesis code -> native Mel. 'embedding': the 128-d shared embedding of the aligned audio tower (after the
    projection). 'wavlm': the frozen WavLM layers 3/6/9/12 (before it), mixed with learnable weights."""
    def __init__(self, route, width=256):
        super().__init__(); self.route = route; self.dim = 768 if route == 'wavlm' else 128
        if route == 'wavlm': self.layer_logits = nn.Parameter(torch.zeros(len(LAYERS))); self.norm = nn.LayerNorm(768)
        self.decoder = MelDecoder(dim=self.dim, width=width)

    def code(self, aligner, layers, env):
        if self.route == 'embedding': return aligner.audio(layers, env)
        return self.norm((layers*self.layer_logits.softmax(0)[None, None, :, None]).sum(2))

    def forward(self, code, frames=None): return self.decoder(code, frames or mel_frames(code.shape[1]))


class CodeBridge(nn.Module):
    """Whole EEG window (WIN frames of the aligned embedding) -> SEG frames of the synthesis code. Small on purpose.
    For the shared embedding it starts from the latency-compensated EEG embedding (identity at initialisation)."""
    def __init__(self, code_dim, residual, dim=128, width=192, heads=4, layers=2):
        super().__init__(); self.residual = residual
        self.inp = nn.Linear(dim, width); self.pos = nn.Parameter(torch.randn(WIN, width)*.02); self.query = nn.Parameter(torch.randn(SEG, width)*.02)
        layer = nn.TransformerEncoderLayer(width, heads, 2*width, dropout=.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False); self.out = nn.Linear(width, code_dim)
        if residual: nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, E, compensated):
        h = self.encoder(torch.cat((self.inp(E)+self.pos, self.inp(compensated)+self.query), 1))[:, WIN:]
        return F.normalize(compensated+self.out(h), dim=-1) if self.residual else self.out(h)


class Generator(nn.Module):
    """Frozen aligned EEG tower -> bridge -> synthesis code -> frozen audio-side Mel decoder."""
    def __init__(self, aligner, audio):
        super().__init__(); self.aligner, self.audio_decoder = aligner, audio
        self.bridge = CodeBridge(audio.dim, residual=audio.route == 'embedding')

    def code(self, x, positions, mask, subject):
        E = self.aligner.eeg(x, positions, mask, subject); return self.bridge(E, self.aligner.compensated(E, subject))

    def forward(self, x, positions, mask, subject):
        z = self.code(x, positions, mask, subject); return self.audio_decoder(z), z


def load_audio_decoder(folder, device, checkpoint='audio_decoder.pt'):
    saved = torch.load(Path(folder)/checkpoint, map_location=device, weights_only=False)
    audio = AudioDecoder(saved['route']).to(device); audio.load_state_dict(saved['decoder']); return audio, saved


def build_generator(cache, align_folder, audio_folder, device, init=None):
    aligner, _ = load_aligner(cache, align_folder, device); audio, _ = load_audio_decoder(audio_folder, device)
    gen = Generator(aligner, audio).to(device)
    if init: gen.load_state_dict(torch.load(init, map_location=device, weights_only=False)['generator'])
    aligner.requires_grad_(False).eval(); audio.requires_grad_(False).eval()
    return gen


def frame_loss(z, target, route):
    """The code frame by frame: cosine for the unit-length shared embedding, mean squared error for WavLM features."""
    return 1-F.cosine_similarity(z, target, dim=-1).mean() if route == 'embedding' else F.mse_loss(z, target)


# ----------------------------------------------------------------------------- stage 1: audio side

@torch.no_grad()
def audio_metrics(audio, aligner, cache, anchors, device):
    """Decoder on the true code of audio anchors: L1, dynamic Mel r, envelope r, detail ratio (1 = as sharp as real)."""
    audio.eval(); per = defaultdict(lambda: defaultdict(list))
    for n in range(0, len(anchors), 64):
        part = anchors[n:n+64]; wv, en = cache.audio_batch(part); true = cache.mel_batch(part).to(device)
        pred = audio(audio.code(aligner, wv.to(device), en.to(device)), true.shape[-1])
        values = {'l1': (pred-true).abs().mean((1, 2)), 'dynamic_mel_r': (descriptor(pred)*descriptor(true)).sum(1),
                  'envelope_r': (envelope(pred, cache.stats)*envelope(true, cache.stats)).mean(1), 'detail_ratio': detail(pred, cache.stats)/detail(true, cache.stats)}
        for k, v in values.items():
            for (d, _, _), x in zip(part, v.cpu().tolist()): per[d][k].append(x)
    result = {d: {k: float(np.mean(v)) for k, v in m.items()} for d, m in per.items()}
    result['pooled'] = {k: float(np.mean([result[d][k] for d in per])) for k in ('l1', 'dynamic_mel_r', 'envelope_r', 'detail_ratio')}
    return result


def train_audio_decoder(cache, align_folder, folder, route, deadline, device='cpu', threads=6, seed=11, max_steps=None, validate_minutes=5, noise=0., peak=1e-3):
    """Stage 1: Mel decoder from the synthesis code of real audio (L1 on the Mel and on its differences along time and
    frequency, which keeps transitions and harmonics). Balanced datasets; best checkpoint on validation content.
    `noise` (off by default) is the expected length of a perturbation relative to the code's own length."""
    folder = Path(folder)
    if (folder/'audio_decoder.pt').exists(): raise ValueError(f'{folder} already holds an audio decoder; choose another --run')
    folder.mkdir(parents=True, exist_ok=True); torch.manual_seed(seed); torch.set_num_threads(threads); rng = np.random.default_rng(seed)
    aligner, _ = load_aligner(cache, align_folder, device); audio = AudioDecoder(route).to(device)
    optimizer = torch.optim.AdamW(audio.parameters(), lr=peak, weight_decay=.01)
    dev = audio_anchor_sampler(cache, 'validation', np.random.default_rng(0), 600, balance=True, conditions=('clean',))
    began = time.time(); last = began; best = -math.inf; step = 0
    status(folder, stage='audio decoder', route=route, noise=noise, align=str(align_folder), began=time.strftime('%H:%M'), deadline=time.strftime('%H:%M', time.localtime(deadline)))

    def validate(final=False):
        nonlocal best
        score = audio_metrics(audio, aligner, cache, dev, device); minutes = round((time.time()-began)/60, 1)
        append_jsonl(folder/'audio_validation.jsonl', {'step': step, 'minutes': minutes, 'final': final, **score})
        saved = {'decoder': audio.state_dict(), 'route': route, 'step': step, 'minutes': minutes, 'validation': score, 'noise': noise, 'align': str(align_folder)}
        torch.save(saved, folder/'audio_decoder_latest.pt')
        if score['pooled']['dynamic_mel_r'] > best: best = score['pooled']['dynamic_mel_r']; torch.save(saved, folder/'audio_decoder.pt')
        status(folder, validated_step=step, validation=score['pooled'], best_dynamic_mel_r=best); audio.train()

    audio.train()
    while time.time() < deadline and not (max_steps and step >= max_steps):
        anchors = audio_anchor_sampler(cache, 'train', rng, 32, balance=True)
        wv, en = cache.audio_batch(anchors); target = cache.mel_batch(anchors).to(device)
        if route == 'embedding':
            with torch.no_grad(): code = audio.code(aligner, wv.to(device), en.to(device))
        else: code = audio.code(aligner, wv.to(device), en.to(device))
        if noise:
            code = code+noise*code.norm(dim=-1, keepdim=True)/math.sqrt(code.shape[-1])*torch.randn_like(code)
            if route == 'embedding': code = F.normalize(code, dim=-1)
        pred = audio(code, target.shape[-1])
        loss = (pred-target).abs().mean()+(pred.diff(dim=-1)-target.diff(dim=-1)).abs().mean()+(pred.diff(dim=1)-target.diff(dim=1)).abs().mean()
        for group in optimizer.param_groups: group['lr'] = lr_at((time.time()-began)/max(deadline-began, 1), step, peak=peak)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); step += 1
        if step % 50 == 0: append_jsonl(folder/'audio_training.jsonl', {'step': step, 'minutes': round((time.time()-began)/60, 1), 'loss': loss.item()})
        if time.time()-last >= validate_minutes*60: last = time.time(); validate()
    validate(final=True); status(folder, stage='audio decoder done', step=step)


# ----------------------------------------------------------------------------- stage 2: EEG bridge

def train_generator(cfg, cache, align_folder, audio_folder, folder, deadline, device='cpu', threads=6, workers=2, seed=11, word_weight=1.,
                    validate_minutes=10, max_steps=None, peak=5e-4, limit=300, init=None):
    """Only the bridge learns. Loss = code frame by frame + word_weight x word contrast. Every `validate_minutes` the
    development reconstruction is measured; the checkpoint whose words are identified best is kept, unless collapsed."""
    from torch.utils.data import DataLoader
    folder = Path(folder)
    if (folder/'latest.pt').exists(): raise ValueError(f'{folder} already holds a run; choose another --run')
    folder.mkdir(parents=True, exist_ok=True); torch.manual_seed(seed); torch.set_num_threads(threads)
    gen = build_generator(cache, align_folder, audio_folder, device, init); route = gen.audio_decoder.route; spans = word_spans(cfg, cache)
    optimizer = torch.optim.AdamW(gen.bridge.parameters(), lr=peak, weight_decay=.05)
    loader = DataLoader(Stream(cache, set(cache.subjects), 'train', seed=seed), batch_size=None, num_workers=workers,
                        persistent_workers=workers > 0, prefetch_factor=4 if workers else None)
    dev_items = fixed_set(cache, set(cfg.fold_subjects('P1', 'F0', 'validation')), 'validation')
    began = time.time(); last_check = began; best = math.inf; step = 0
    status(folder, stage='bridge', route=route, word_weight=word_weight, align=str(align_folder), audio=str(audio_folder),
           began=time.strftime('%H:%M'), deadline=time.strftime('%H:%M', time.localtime(deadline)), smoke=cache.smoke)

    def validate(final=False):
        nonlocal best
        frames = reconstruction_metrics(gen, cache, dev_items, device, limit=limit)['pooled']
        words = word_identification(gen, cache, spans, dev_items, device, limit=limit)['pooled']; minutes = round((time.time()-began)/60, 1)
        score = {**frames, **{f'word_{k}': v for k, v in words.items()}}; collapsed = score['spread_ratio'] < .2
        append_jsonl(folder/'validation.jsonl', {'step': step, 'minutes': minutes, 'final': final, 'collapsed': collapsed, **score})
        saved = {'generator': gen.state_dict(), 'step': step, 'minutes': minutes, 'validation': score, 'route': route, 'word_weight': word_weight,
                 'align': str(align_folder), 'audio': str(audio_folder)}
        torch.save(saved, folder/'latest.pt')
        if (not collapsed and score['word_rank_percentile'] < best) or not (folder/'best.pt').exists():
            best = score['word_rank_percentile'] if not collapsed else best; torch.save(saved, folder/'best.pt')
        status(folder, validated_step=step, validation=score, best_word_rank_percentile=best if best < math.inf else None, collapse_warning=collapsed); gen.bridge.train()

    gen.bridge.train()
    for batch in loader:
        now = time.time()
        if now >= deadline or (max_steps and step >= max_steps): break
        for group in optimizer.param_groups: group['lr'] = lr_at((now-began)/max(deadline-began, 1), step, peak=peak)
        batch = to(batch, device); anchor = batch['anchor']; wv, en = cache.audio_batch(batch['anchors'])
        with torch.no_grad(): code = gen.audio_decoder.code(gen.aligner, wv.to(device), en.to(device))
        z = gen.code(batch['eeg'], batch['positions'], batch['mask'], batch['subject'])
        loss_frames = frame_loss(z, code[anchor], route)
        # Words: pooled on the audio time line; positives are the same word instance, negatives every other word of the batch.
        W, valid, _ = pooling(spans, batch['anchors']); flat = np.flatnonzero(valid.reshape(-1))
        loss_words = z.sum()*0; hit = float('nan')
        if len(flat) > 1:
            truth = F.normalize(pooled(code, W).reshape(-1, code.shape[-1])[torch.as_tensor(flat, device=z.device)], dim=-1)
            a = anchor.cpu().numpy(); mine = valid[a].reshape(-1); slot = (a[:, None]*WORDS_PER_WINDOW+np.arange(WORDS_PER_WINDOW)[None]).reshape(-1)[mine]
            guess = F.normalize(pooled(z, W[a]).reshape(-1, z.shape[-1])[torch.as_tensor(np.flatnonzero(mine), device=z.device)], dim=-1)
            target = torch.as_tensor(np.searchsorted(flat, slot), device=z.device); logits = guess@truth.T/.1
            loss_words = F.cross_entropy(logits, target); hit = float((logits.argmax(1) == target).float().mean())
        loss = loss_frames+word_weight*loss_words
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(gen.bridge.parameters(), 1.); optimizer.step(); step += 1
        if step % 25 == 0:
            parts = {'frames': loss_frames.item(), 'words': loss_words.item(), 'batch_word_top1': hit, 'words_in_batch': int(len(flat))}
            append_jsonl(folder/'training.jsonl', {'step': step, 'minutes': round((now-began)/60, 1), 'dataset': batch['dataset'], 'loss': loss.item(), **parts})
            status(folder, step=step, last=parts)
        if now-last_check >= validate_minutes*60: last_check = now; validate()
    validate(final=True); status(folder, stage='bridge done', step=step)
    return read_json(folder/'status.json')


def zero_shot(cfg, cache, align_folder, audio_folder, folder, device='cpu', limit=300):
    """No training on EEG: the audio-side decoder applied to the latency-compensated EEG embedding (the bridge is the
    identity at initialisation). Only for the shared-embedding route."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    gen = build_generator(cache, align_folder, audio_folder, device)
    if gen.audio_decoder.route != 'embedding': raise ValueError('zero-shot needs the shared-embedding route')
    items = fixed_set(cache, set(cfg.fold_subjects('P1', 'F0', 'validation')), 'validation')
    score = {**reconstruction_metrics(gen, cache, items, device, limit=limit)['pooled'],
             **{f'word_{k}': v for k, v in word_identification(gen, cache, word_spans(cfg, cache), items, device, limit=limit)['pooled'].items()}}
    append_jsonl(folder/'validation.jsonl', {'step': 0, 'minutes': 0, 'final': True, 'collapsed': score['spread_ratio'] < .2, **score})
    torch.save({'generator': gen.state_dict(), 'step': 0, 'minutes': 0, 'validation': score, 'route': 'embedding', 'word_weight': 'zero-shot',
                'align': str(align_folder), 'audio': str(audio_folder)}, folder/'best.pt')
    status(folder, stage='zero-shot ready', validation=score)
    return score


# ----------------------------------------------------------------------------- evaluation

def _eeg_batch(cache, dataset, part, device, null=False):
    rows = cache.records[dataset]; win = [eeg_window(rows[i], a, cache.good[dataset][i]) for i, a in part]
    x = torch.from_numpy(np.stack([w for w, _ in win])).to(device)
    if null: x = torch.zeros_like(x)
    mask = torch.from_numpy(np.stack([m for _, m in win])).to(device); pos = torch.from_numpy(np.stack([cache.positions[dataset][i] for i, _ in part])).to(device)
    subject = torch.tensor([cache.subject_id(rows[i]['key']) for i, _ in part], device=device)
    return x, pos, mask, subject


@torch.no_grad()
def reconstruction_metrics(gen, cache, items, device, limit=None, permutations=0, bootstrap=0, seed=11):
    """Frames. Generated Mel of every 3 s window against its true Mel and against the other windows of the same content
    block (wrong-time null): dynamic Mel r, envelope r, identification of the true segment from the generated Mel,
    spread of the outputs across windows, and the same r for the output without EEG (no-EEG null)."""
    gen.eval(); rng = np.random.default_rng(seed); result, pooled_boot = {}, defaultdict(list)
    for dataset in sorted({d for d, _, _ in items}):
        rows = cache.records[dataset]; chosen = [(i, a) for d, i, a in items if d == dataset][:limit]
        keys = sorted({(rows[i]['audio'], a) for i, a in chosen}); where = {k: n for n, k in enumerate(keys)}
        td, te, tsum = [], [], []
        for n in range(0, len(keys), 64):
            mel = cache.mel_batch([(dataset, c, a) for c, a in keys[n:n+64]]).to(device)
            td.append(descriptor(mel).cpu()); te.append(envelope(mel, cache.stats).cpu()); tsum.append(F.avg_pool2d(mel[:, None], (2, 4)).flatten(1).cpu())
        td, te, tsum = torch.cat(td), torch.cat(te), torch.cat(tsum)
        pd, pe, psum, nd, ne = [], [], [], [], []
        for n in range(0, len(chosen), 64):
            part = chosen[n:n+64]
            pred, _ = gen(*_eeg_batch(cache, dataset, part, device)); pd.append(descriptor(pred).cpu()); pe.append(envelope(pred, cache.stats).cpu())
            psum.append(F.avg_pool2d(pred[:, None], (2, 4)).flatten(1).cpu())
            empty, _ = gen(*_eeg_batch(cache, dataset, part, device, null=True)); nd.append(descriptor(empty).cpu()); ne.append(envelope(empty, cache.stats).cpu())
        pd, pe, psum, nd, ne = torch.cat(pd), torch.cat(pe), torch.cat(psum), torch.cat(nd), torch.cat(ne)
        own = np.array([where[(rows[i]['audio'], a)] for i, a in chosen]); idx = torch.as_tensor(own)
        people = [rows[i]['key'] for i, _ in chosen]; blocks = [cache.block(dataset, rows[i]['audio'], a) for i, a in chosen]
        scores = (pd@td.T).numpy(); env = (pe@te.T/pe.shape[1]).numpy()
        timeline = defaultdict(list)
        for (c, a), n in where.items(): timeline[cache.block(dataset, c, a)].append(n)
        timeline = {b: sorted(v, key=lambda n: keys[n][1]) for b, v in timeline.items()}
        rows_n = np.arange(len(chosen)); same = np.zeros_like(scores, dtype=bool)
        for w, b in enumerate(blocks): same[w, timeline[b]] = True
        same[rows_n, own] = False; others = same.sum(1).clip(min=1)
        mel_r, env_r = scores[rows_n, own], env[rows_n, own]
        mel_null, env_null = (scores*same).sum(1)/others, (env*same).sum(1)/others
        rank, pct = tie_rank(scores, own); n_cand = len(keys)
        value = {'windows': len(chosen), 'candidates': n_cand, 'people': len(set(people)),
                 'dynamic_mel_r': float(mel_r.mean()), 'dynamic_mel_r_null': float(mel_null.mean()),
                 'envelope_r': float(env_r.mean()), 'envelope_r_null': float(env_null.mean()),
                 'no_eeg_dynamic_mel_r': float((nd*td[idx]).sum(1).mean()), 'no_eeg_envelope_r': float((ne*te[idx]).mean(1).mean()),
                 'identification_rank_percentile': float(pct.mean()), 'top10': float(np.mean(rank < 9.5)), 'chance_top10': min(10, n_cand)/n_cand,
                 'spread_ratio': float(psum.std(0).mean()/tsum.std(0).mean().clamp(min=1e-6))}
        if permutations:
            null = permutation_null(scores, permutation_plan(own, people, blocks, timeline), rng, permutations)
            value.update(identification_permutation_mean=float(null.mean()), p_identification=float((1+np.sum(null <= pct.mean()))/(1+permutations)))
        if bootstrap:
            for name, diff in (('dynamic_mel_r_minus_null', mel_r-mel_null), ('envelope_r_minus_null', env_r-env_null)):
                boot = two_way_bootstrap(diff, people, blocks, rng, bootstrap); value[name] = float(diff.mean())
                value[name+'_ci95'] = [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]; pooled_boot[name].append(boot)
        result[dataset] = value
    found = [d for d in DATASETS if d in result]
    keys = ('dynamic_mel_r', 'dynamic_mel_r_null', 'envelope_r', 'envelope_r_null', 'no_eeg_dynamic_mel_r', 'no_eeg_envelope_r',
            'identification_rank_percentile', 'top10', 'spread_ratio')
    summary = {k: float(np.mean([result[d][k] for d in found])) for k in keys} if found else {}
    for name, boots in pooled_boot.items():
        b = np.mean(boots, 0); summary[name] = float(np.mean([result[d][name] for d in found])); summary[name+'_ci95'] = [float(np.quantile(b, .025)), float(np.quantile(b, .975))]
    result['pooled'] = summary
    return result


@torch.no_grad()
def word_identification(gen, cache, spans, items, device, limit=None, bootstrap=0, seed=11):
    """Words. The generated code averaged over each word's time span (inside the 3 s windows) is ranked against the
    true code of every word of the same content split (0.5 = chance); also for the output without EEG."""
    gen.eval(); rng = np.random.default_rng(seed); result, boots = {}, []
    for dataset in sorted({d for d, _, _ in items}):
        rows = cache.records[dataset]; chosen = [(i, a) for d, i, a in items if d == dataset][:limit]
        keys = sorted({(rows[i]['audio'], a) for i, a in chosen}); truth, ids = [], []
        for n in range(0, len(keys), 64):
            part = [(dataset, c, a) for c, a in keys[n:n+64]]; wv, en = cache.audio_batch(part)
            W, valid, index = pooling(spans, part); code = gen.audio_decoder.code(gen.aligner, wv.to(device), en.to(device))
            truth.append(F.normalize(pooled(code, W), dim=-1)[torch.as_tensor(valid, device=device)].cpu())
            ids += [(c, int(w)) for (_, c, _), row, ok in zip(part, index, valid) for w in row[ok]]
        if not ids: continue
        truth = torch.cat(truth); where = {}
        for n, key in enumerate(ids): where.setdefault(key, n)
        guess, empty, own, people, blocks = [], [], [], [], []
        for n in range(0, len(chosen), 64):
            part = chosen[n:n+64]; anchors = [(dataset, rows[i]['audio'], a) for i, a in part]; W, valid, index = pooling(spans, anchors)
            for null, out in ((False, guess), (True, empty)):
                z = gen.code(*_eeg_batch(cache, dataset, part, device, null)); out.append(F.normalize(pooled(z, W), dim=-1)[torch.as_tensor(valid, device=device)].cpu())
            for (i, a), row, ok in zip(part, index, valid):
                for w in row[ok]: own.append(where[(rows[i]['audio'], int(w))]); people.append(rows[i]['key']); blocks.append(cache.block(dataset, rows[i]['audio'], a))
        guess, empty, own = torch.cat(guess), torch.cat(empty), np.array(own)
        rank, pct = tie_rank((guess@truth.T).numpy(), own); _, pct_empty = tie_rank((empty@truth.T).numpy(), own)
        value = {'words': len(own), 'candidates': len(truth), 'rank_percentile': float(pct.mean()), 'no_eeg_rank_percentile': float(pct_empty.mean()),
                 'top10': float(np.mean(rank < 9.5)), 'chance_top10': min(10, len(truth))/len(truth)}
        if bootstrap:
            boot = two_way_bootstrap(pct, people, blocks, rng, bootstrap); boots.append(boot); value['rank_percentile_ci95'] = [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
        result[dataset] = value
    found = [d for d in DATASETS if d in result]
    result['pooled'] = {k: float(np.mean([result[d][k] for d in found])) for k in ('rank_percentile', 'no_eeg_rank_percentile', 'top10')} if found else {}
    if boots: b = np.mean(boots, 0); result['pooled']['rank_percentile_ci95'] = [float(np.quantile(b, .025)), float(np.quantile(b, .975))]
    return result


def sentences(spans, gap=.4, shortest=1.5, longest=8.):
    """(start, end, text) of sentence-like stretches of one content: breaks at final punctuation or at pauses >= gap,
    long stretches split at their largest pause, short ones dropped."""
    starts, ends, texts = spans; out, current = [], []
    for k in range(len(starts)):
        current.append(k)
        if k == len(starts)-1 or texts[k].rstrip().endswith(('.', '?', '!')) or starts[k+1]-ends[k] >= gap: out.append(current); current = []
    done = []
    while out:
        s = out.pop()
        if ends[s[-1]]-starts[s[0]] > longest and len(s) > 1:
            cut = 1+int(np.argmax([starts[s[j+1]]-ends[s[j]] for j in range(len(s)-1)])); out += [s[:cut], s[cut:]]
        elif ends[s[-1]]-starts[s[0]] >= shortest: done.append((float(starts[s[0]]), float(ends[s[-1]]), ' '.join(texts[k] for k in s)))
    return sorted(done)


@torch.no_grad()
def stitch(gen, cache, dataset, i, start, end, shift=0., null=False, hop=1.):
    """Code of [start, end) from overlapping 3 s windows (1 s hop) of record i, averaged with a smooth window weight.
    `shift` moves the EEG (not the audio) in time: the wrong-time control."""
    rows = cache.records[dataset]; first = math.floor((start-1.)/FRAME)*FRAME; n = int(round((end-first)/FRAME))+1
    anchors = [round((first+k*hop)/FRAME)*FRAME for k in range(int(math.ceil((end-first-SEG*FRAME)/hop))+1)]
    total = torch.zeros(n+SEG, gen.audio_decoder.dim); weight = torch.zeros(n+SEG, 1); taper = torch.sin(math.pi*(torch.arange(SEG)+.5)/SEG)[:, None]
    for a in anchors:
        if not covers(rows[i], a+shift): return None
        z = gen.code(*_eeg_batch(cache, dataset, [(i, a+shift)], 'cpu', null))[0]; k = int(round((a-first)/FRAME))
        total[k:k+SEG] += z*taper; weight[k:k+SEG] += taper
    k0, k1 = int(round((start-first)/FRAME)), int(round((end-first)/FRAME))
    code = total[k0:k1]/weight[k0:k1].clamp(min=1e-6)
    return F.normalize(code, dim=-1) if gen.audio_decoder.route == 'embedding' else code


@torch.no_grad()
def listen(cfg, cache, gen, audio, aligner, people, out, per_dataset=3, files=2, seed=5):
    """Sentences. Whole sentence-like stretches of the test content: original, Vocos ceiling (true Mel), audio-side
    reference (decoder on the true code), and, with a generator, EEG, wrong-time EEG (shifted by 20 s) and no EEG,
    each transcribed; word error rate and the share of reference words found in the transcript."""
    import soundfile as sf
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ..perceived_audio import ASR, FrozenVocos
    from ..metrics import word_errors
    (out/'audio').mkdir(parents=True, exist_ok=True); (out/'figures').mkdir(exist_ok=True)
    vocos = FrozenVocos(cfg.model_dir('vocos')); asr = ASR(cfg, 'cpu'); spans = word_spans(cfg, cache); rng = np.random.default_rng(seed)
    render = lambda m: vocos(natural(m, cache.stats)[None])[0].numpy(); rows_out = []
    for dataset in DATASETS:
        rows = cache.records.get(dataset, []); records = [i for i, r in enumerate(rows) if r['key'] in people and r['split'] == 'test' and r.get('condition', 'clean') == 'clean']
        # Margins so that every sliding EEG window, and the one shifted for the wrong-time control, lies inside the record.
        pool = [(i, s) for i in records for s in sentences(spans[dataset, rows[i]['audio']])
                if rows[i]['start']+6 <= s[0] and s[1] <= rows[i]['start']+rows[i]['duration']-7]
        if not pool: continue
        for k in rng.choice(len(pool), min(per_dataset, len(pool)), replace=False):
            i, (start, end, text) = pool[k]; content, key = rows[i]['audio'], rows[i]['key']
            frames = int(round((end-start)/FRAME)); a0 = round(start/FRAME)*FRAME
            wv, en = cache.audio_batch([(dataset, content, a0)], frames); true = cache.mel_batch([(dataset, content, a0)], mel_frames(frames))[0]
            mels = {'true': true, 'audio_reference': audio(audio.code(aligner, wv, en), true.shape[-1])[0]}
            if gen is not None:
                for label, shift, null in (('from_eeg', 0., False), ('wrong_time_eeg', 20., False), ('no_eeg', 0., True)):
                    code = stitch(gen, cache, dataset, i, a0, a0+frames*FRAME, shift, null)
                    if code is None and label == 'wrong_time_eeg': code = stitch(gen, cache, dataset, i, a0, a0+frames*FRAME, -20., null)
                    if code is not None: mels[label] = audio(code[None], true.shape[-1])[0]
            original, _ = _wave(cfg, dataset, content, a0, frames*FRAME)
            waves = {'original': original, 'vocos_ceiling': render(true), **{label: render(m) for label, m in mels.items() if label != 'true'}}
            row = {'dataset': dataset, 'content': content, 'start': a0, 'seconds': round(frames*FRAME, 2), 'person': key, 'reference': text}
            ref = asr.normalize(text).strip(); words = ref.split()
            for label, m in mels.items():
                if label == 'true': continue
                row[f'{label}_dynamic_mel_r'] = (descriptor(m[None])@descriptor(true[None]).T).item()
                row[f'{label}_envelope_r'] = (envelope(m[None], cache.stats)@envelope(true[None], cache.stats).T/true.shape[-1]).item()
            for label, wave in waves.items():
                said = asr(wave); row[f'{label}_transcript'] = said
                row[f'{label}_wer'] = word_errors(ref, said)['wer'] if ref else None
                row[f'{label}_word_hits'] = float(np.mean([w in said.split() for w in words])) if words else None
            if len([r for r in rows_out if r['dataset'] == dataset]) < files:
                stem = f"{dataset}_{content}_{a0:07.2f}_{key.split(':')[1]}"
                for label, wave in waves.items(): sf.write(out/'audio'/f'{stem}_{label}.wav', wave, 24000)
                figure, axes = plt.subplots(len(mels), 1, figsize=(10, 1.5*len(mels)), sharex=True)
                for axis, (label, m) in zip(axes, mels.items()):
                    axis.imshow(m.numpy(), aspect='auto', origin='lower', cmap='magma', vmin=-3, vmax=3); axis.set_ylabel(label, fontsize=7); axis.set_yticks([])
                axes[-1].set_xlabel('Mel frame (10.7 ms)'); figure.suptitle(f'{stem}  “{text}”', fontsize=8); figure.tight_layout()
                figure.savefig(out/'figures'/f'{stem}.png', dpi=110); plt.close(figure); row['files'] = stem
            rows_out.append(row); append_jsonl(out/'listening.jsonl', row)
    labels = [k[:-len('_wer')] for k in rows_out[0] if k.endswith('_wer')] if rows_out else []
    summary = {label: {m: float(np.mean([r[f'{label}_{m}'] for r in rows_out if r.get(f'{label}_{m}') is not None]))
                       for m in ('wer', 'word_hits')} for label in labels}
    return rows_out, summary


def evaluate_audio_decoder(cfg, cache, folder, out, run, device='cpu', sentences_per_dataset=5, threads=6):
    """Stage-1 report on test content: decoder on the true code (no EEG), with metrics and listening clips."""
    torch.set_num_threads(threads); folder, out = Path(folder), Path(out); out.mkdir(parents=True, exist_ok=True)
    audio, saved = load_audio_decoder(folder, device); aligner, _ = load_aligner(cache, saved['align'], device)
    test = audio_metrics(audio, aligner, cache, audio_anchor_sampler(cache, 'test', np.random.default_rng(1), 600, balance=True, conditions=('clean',)), device)
    rows, heard = listen(cfg, cache, None, audio.to('cpu').eval(), aligner.to('cpu'), set(cache.subjects), out, sentences_per_dataset)
    summary = {'run': run, 'route': saved['route'], 'step': saved['step'], 'minutes': saved['minutes'], 'noise': saved['noise'],
               'validation': saved['validation'], 'test': test, 'listening': heard, 'clips': rows}
    atomic_json(out/'audio_reference.json', summary)
    lines = [f"# 音频侧参照：{run}（{'128 维共享嵌入，投影之后' if saved['route'] == 'embedding' else 'WavLM 特征，投影之前'}）", '',
             f"不经过 EEG：把真实音频的合成代码送进解码器再经 Vocos。训练 {saved['step']} 步、{saved['minutes']} 分钟，噪声 {saved['noise']}，按验证内容挑检查点。", '',
             '| 数据集 | L1 | 动态 Mel r | 包络 r | 细节比（1 = 和真实一样锐利） |', '|---|---|---|---|---|']
    for d in [*DATASETS, 'pooled']:
        if d in test: v = test[d]; lines.append(f"| {d} | {v['l1']:.3f} | {v['dynamic_mel_r']:.3f} | {v['envelope_r']:.3f} | {v['detail_ratio']:.2f} |")
    lines += ['', '整句试听与转写：', '', '| 输出 | 词错误率 | 参考词出现在转写里的比例 |', '|---|---|---|']
    lines += [f"| {label} | {v['wer']:.2f} | {v['word_hits']:.2f} |" for label, v in heard.items()]
    lines += ['', '文件在 `audio/`，频谱图在 `figures/`，每句的转写在 `listening.jsonl`。']
    (out/'audio_reference.md').write_text('\n'.join(lines)+'\n')
    return summary


@torch.no_grad()
def evaluate_reconstruction(cfg, cache, folder, out, run, device='cpu', checkpoint='best.pt', threads=6, permutations=1000, bootstrap=2000, sentences_per_dataset=5):
    """Test content for trained-on and never-seen people at three scales, against the wrong-time and no-EEG nulls."""
    torch.set_num_threads(threads); folder, out = Path(folder), Path(out); out.mkdir(parents=True, exist_ok=True)
    saved = torch.load(folder/checkpoint, map_location=device, weights_only=False)
    gen = build_generator(cache, saved['align'], saved['audio'], device); gen.load_state_dict(saved['generator']); gen.eval(); spans = word_spans(cfg, cache)
    summary = {'run': run, 'checkpoint': checkpoint, 'step': saved['step'], 'minutes': saved['minutes'], 'route': saved['route'], 'word_weight': saved['word_weight'],
               'smoke': cache.smoke, 'permutations': permutations, 'bootstrap': bootstrap}
    groups = (('seen_people_unseen_speech', set(cache.subjects)), ('unseen_people_unseen_speech', set(cfg.fold_subjects('P1', 'F0', 'test'))))
    for name, people in groups:
        status(folder, stage=f'reconstruction evaluation: {name}'); items = fixed_set(cache, people, 'test')
        summary[name] = {'frames': reconstruction_metrics(gen, cache, items, device, permutations=permutations, bootstrap=bootstrap),
                         'words': word_identification(gen, cache, spans, items, device, bootstrap=bootstrap)}
    status(folder, stage='reconstruction evaluation: sentences'); gen = gen.to('cpu')
    summary['sentences'] = {}
    for name, people in groups:
        rows, heard = listen(cfg, cache, gen, gen.audio_decoder, gen.aligner, people, out/name.split('_')[0], sentences_per_dataset if name.startswith('seen') else 2)
        summary['sentences'][name] = {'summary': heard, 'clips': rows}
    atomic_json(out/'reconstruction.json', summary); write_report(summary, out, folder); status(folder, stage='reconstruction evaluation done')
    return summary


def write_report(summary, out, folder):
    nan = float('nan'); kind = '冒烟测试：只检查代码能否运行，不是实验结果' if summary.get('smoke') else '完整数据'
    lines = [f"# EEG → 语音重建：{summary['run']}（{'WavLM 代码' if summary['route'] == 'wavlm' else '128 维共享嵌入'}，{summary['checkpoint']}）", '',
             f"{kind}。只训练桥接（EEG 塔和解码器冻结），{summary['step']} 步、{summary['minutes']} 分钟；损失 = 逐帧代码 + {summary['word_weight']} × 词级对比。", '',
             '全部是没参与训练的测试内容。对照：错时（同一个人另一时刻）与无 EEG（EEG 输入置零）。三个尺度：帧（3 秒窗口的 Mel）、词（按词时间段平均的代码在全部测试词里认词，0.5 = 偶然）、'
             '句子（1 秒步长滑动窗口拼成整句，Whisper 转写）。', '']
    for name, title in (('seen_people_unseen_speech', '训练过的人，没听过的语音'), ('unseen_people_unseen_speech', '没见过的人，没听过的语音')):
        f = summary[name]['frames']; w = summary[name]['words']; p = f['pooled']; q = w['pooled']; ci = p.get('dynamic_mel_r_minus_null_ci95', [nan, nan])
        lines += [f'## {title}', '',
                  f"- 帧：动态 Mel r {p.get('dynamic_mel_r', nan):.3f}（错时 {p.get('dynamic_mel_r_null', nan):.3f}，无 EEG {p.get('no_eeg_dynamic_mel_r', nan):.3f}，与错时之差区间 {ci[0]:+.3f}–{ci[1]:+.3f}）；"
                  f"包络 r {p.get('envelope_r', nan):.3f}（错时 {p.get('envelope_r_null', nan):.3f}，无 EEG {p.get('no_eeg_envelope_r', nan):.3f}）；输出离散度 {p.get('spread_ratio', nan):.2f}",
                  f"- 词：认词排名百分位 {q.get('rank_percentile', nan):.3f}（区间 {q.get('rank_percentile_ci95', [nan, nan])[0]:.3f}–{q.get('rank_percentile_ci95', [nan, nan])[1]:.3f}；"
                  f"无 EEG {q.get('no_eeg_rank_percentile', nan):.3f}）；top-10 {q.get('top10', nan):.3f}", '',
                  '| 数据集 | 帧：动态 Mel r（错时 / 无 EEG） | 帧：包络 r（错时 / 无 EEG） | 段落认片段 | p | 词数 | 候选词 | 认词排名百分位（无 EEG） | 认词 top-10（偶然） |',
                  '|---|---|---|---|---|---|---|---|---|']
        for d in DATASETS:
            if d not in f: continue
            v, u = f[d], w.get(d, {})
            lines.append(f"| {d} | {v['dynamic_mel_r']:.3f}（{v['dynamic_mel_r_null']:.3f} / {v['no_eeg_dynamic_mel_r']:.3f}） | {v['envelope_r']:.3f}（{v['envelope_r_null']:.3f} / {v['no_eeg_envelope_r']:.3f}） | "
                         f"{v['identification_rank_percentile']:.3f} | {v.get('p_identification', nan):.4f} | {u.get('words', 0)} | {u.get('candidates', 0)} | "
                         f"{u.get('rank_percentile', nan):.3f}（{u.get('no_eeg_rank_percentile', nan):.3f}） | {u.get('top10', nan):.3f}（{u.get('chance_top10', nan):.3f}） |")
        heard = summary['sentences'][name]['summary']
        if heard:
            lines += ['', '整句（Whisper）：', '', '| 输出 | 词错误率 | 参考词出现在转写里的比例 |', '|---|---|---|']
            lines += [f"| {label} | {v['wer']:.2f} | {v['word_hits']:.2f} |" for label, v in heard.items()]
            lines += ['', f"音频在 `{name.split('_')[0]}/audio/`，频谱图在 `{name.split('_')[0]}/figures/`。"]
        lines.append('')
    check = [json.loads(l) for l in open(Path(folder)/'validation.jsonl')] if (Path(folder)/'validation.jsonl').exists() else []
    if check:
        lines += ['## 训练中的开发集检查', '', '| 分钟 | 步数 | 动态 Mel r（错时 / 无 EEG） | 包络 r | 认词（无 EEG） | 离散度 | 坍缩 |', '|---|---|---|---|---|---|---|']
        for r in check:
            lines.append(f"| {r['minutes']} | {r['step']} | {r['dynamic_mel_r']:.3f}（{r['dynamic_mel_r_null']:.3f} / {r.get('no_eeg_dynamic_mel_r', nan):.3f}） | "
                         f"{r['envelope_r']:.3f} | {r.get('word_rank_percentile', nan):.3f}（{r.get('word_no_eeg_rank_percentile', nan):.3f}） | {r['spread_ratio']:.2f} | {'是' if r['collapsed'] else '否'} |")
    (Path(out)/'reconstruction.md').write_text('\n'.join(lines)+'\n')

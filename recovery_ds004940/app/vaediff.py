#!/usr/bin/env python3
"""EEG -> speech with a VAE latent and a diffusion decoder on DS004940 (``vaediff``).

Stages (each writes into ``outputs/vaediff/<run>``):

  audio     stage 1, no EEG: audio posterior q_a(z | WavLM) + mel diffusion decoder p(mel | z)
  eeg       stage 2: EEG posterior q_e(z | EEG, subject) + text head, decoder fine-tuned;
            loss w1 * L_voice + w2 * L_text + w3 * L_diff + beta * KL(q_e || widened q_a),
            weights adjusted every --tune-every updates by WeightController (control.json overrides it)
  evaluate  test sentences, seen and unseen people: latent retrieval with permutation nulls and two-way
            bootstrap; generation (only if retrieval beats its null, unless --force-audio) under the
            counterfactual conditions; Vocos waveforms, STOI / envelope / MCD / timbre / identification, ASR
  status    the last tuning rounds of a stage-2 run (what to read before writing control.json)

The model and losses are in vaediff_model.py, the data in vaediff_data.py; README_VAEDIFF.md has the design.
Nothing here starts on its own: every stage is run by hand (scripts/run_vaediff.sh).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src'))
from aligned_recovery_model import augment_eeg, diverse_batches
import vaediff_data as vd
from vaediff_model import (AudioPosterior, EEGPosterior, LatentMelDiffusion, MelScaler, WeightController, chunk_infonce,
                           chunk_pool, gaussian_kl, latent_times, product_of_experts, rank_percentile, reparameterize,
                           text_infonce, timbre_distance, widened)

OUTPUT = ROOT / 'outputs/vaediff'
CONTRACT = 'vaediff_v1'


def log(**row):
    print(json.dumps(row, default=lambda v: round(float(v), 5) if isinstance(v, (float, np.floating)) else str(v)), flush=True)


def device_of(name):
    if name != 'auto':
        return torch.device(name)
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def atomic_save(path: Path, value):
    tmp = path.with_suffix('.part')
    torch.save(value, tmp); tmp.replace(path)


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


class EMA:
    def __init__(self, module, decay=.999):
        self.shadow = copy.deepcopy(module).eval().requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, module):
        for s, p in zip(self.shadow.parameters(), module.parameters()):
            s.lerp_(p.detach(), 1 - self.decay)
        for s, b in zip(self.shadow.buffers(), module.buffers()):
            s.copy_(b)


def lr_at(step, peak, updates, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * (.05 + .95 * .5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, updates - warmup)))))


# ----------------------------------------------------------------------------- data
class Store:
    """Processed DS004940 (vaediff_data.py output): EEG memmaps, audio tables, roles and subject indices."""
    def __init__(self, root: Path, *, eeg_window=(-.25, 3.6), seconds=3.6, max_peak=30.):
        import pandas as pd
        self.root = Path(root)
        audio = self.root / 'audio'
        self.sentences = json.loads((audio / 'sentences.json').read_text())
        mel_times = np.load(audio / 'mel_times.npy')
        self.frames = int((mel_times < seconds).sum())
        self.mel = torch.from_numpy(np.load(audio / 'mel.npy')[:, :, :self.frames].astype(np.float32))
        self.mel_times = torch.from_numpy(mel_times[:self.frames].astype(np.float32))
        wavlm_times = np.load(audio / 'wavlm_times.npy')
        keep = int((wavlm_times < seconds + .2).sum())
        self.wavlm = torch.from_numpy(np.ascontiguousarray(np.load(audio / 'wavlm.npy', mmap_mode='r')[:, :, :keep]))
        self.wavlm_times = torch.from_numpy(wavlm_times[:keep].astype(np.float32))
        self.text = torch.from_numpy(np.load(audio / 'text.npy'))
        self.wave = np.load(audio / 'wave24.npy', mmap_mode='r')
        self.seconds = float(seconds)
        self.latent_times = latent_times(seconds)
        self.duration = torch.tensor([min(s['duration'], seconds) for s in self.sentences], dtype=torch.float32)
        pairs = sorted({s['pair'] for s in self.sentences})
        self.pair = torch.tensor([pairs.index(s['pair']) for s in self.sentences])
        splits = json.loads((self.root / 'splits.json').read_text())
        self.sentence_role = np.array([splits['pair_role'][s['pair']] for s in self.sentences])
        trials = json.loads((self.root / 'trials.json').read_text())
        self.removed = sum(t['peak'] > max_peak for t in trials)
        self.trials = [t for t in trials if t['peak'] <= max_peak]
        self.files = {f: np.load(self.root / 'eeg' / f'{f}.npy', mmap_mode='r') for f in sorted({t['file'] for t in self.trials})}
        self.start = int(round((eeg_window[0] - vd.EPOCH_START) * vd.EEG_RATE))
        self.samples = int(round((eeg_window[1] - eeg_window[0]) * vd.EEG_RATE))
        if self.start < 0 or self.start + self.samples > vd.EPOCH_SAMPLES:
            raise ValueError('EEG window outside the stored epoch')
        self.eeg_window = tuple(eeg_window)
        self.role = np.array([t['role'] for t in self.trials])
        self.sentence_of = np.array([t['sentence'] for t in self.trials])
        seen = sorted({t['subject'] for t in self.trials if t['role'] in ('train', 'dev', 'test_seen')})
        self.subjects = {s: i for i, s in enumerate(seen)}
        self.subject_index = np.array([self.subjects.get(t['subject'], -1) for t in self.trials])
        self.subject_name = np.array([t['subject'] for t in self.trials])
        self.task = np.array([t['task'] for t in self.trials])
        self.frame = pd.DataFrame(dict(content_group=self.sentence_of, subject=self.subject_name))
        self.signature = dict(trials=hashlib.sha256('\n'.join(f"{t['file']}:{t['row']}" for t in self.trials).encode()).hexdigest(),
                              splits=splits['seed'], eeg_window=list(eeg_window), seconds=seconds, max_peak=max_peak)

    def ids(self, role):
        return np.flatnonzero(self.role == role)

    def sentences_in(self, role):
        return np.flatnonzero(self.sentence_role == role)

    def eeg(self, ids, device):
        x = np.stack([self.files[self.trials[i]['file']][self.trials[i]['row'], :, self.start:self.start + self.samples] for i in ids])
        x = torch.from_numpy(x.astype(np.float32)).to(device)
        return x, x.abs().amax(-1) > 0

    def wavlm_of(self, sentences, device):
        return self.wavlm[torch.as_tensor(np.asarray(sentences))].to(device)

    def speech_mask(self, sentences, times, device):
        return times.to(device)[None] < self.duration[torch.as_tensor(sentences)].to(device)[:, None]

    def wrong_trials(self, ids):
        """Per trial: the same person and task, another sentence frame of the same role, closest in duration."""
        out = []
        for i in ids:
            same = np.flatnonzero((self.subject_name == self.subject_name[i]) & (self.task == self.task[i]) & (self.role == self.role[i])
                                  & (self.pair.numpy()[self.sentence_of] != int(self.pair[self.sentence_of[i]])))
            if not len(same):
                raise ValueError(f'no wrong trial for {i}')
            gap = np.abs(self.duration.numpy()[self.sentence_of[same]] - float(self.duration[self.sentence_of[i]]))
            out.append(int(same[np.lexsort((same, gap))[0]]))
        return np.array(out)


class Frozen:
    """Times on the device, shared by every stage."""
    def __init__(self, store: Store, device):
        self.lat = store.latent_times.to(device)
        self.mel = store.mel_times.to(device)
        self.wavlm = store.wavlm_times.to(device)


# ----------------------------------------------------------------------------- shared measures
@torch.no_grad()
def identification(generated, truth, gen_mask, candidate_mask, pair_gen, pair_cand, own):
    """Rank percentile of the own sentence when the generated mel is compared with every candidate's true mel.

    Score: Pearson correlation of the standardised mel over the frames where both are speech.  Candidates
    sharing the sentence frame (NPC/NPI partner) are excluded; 0 = best, 0.5 = chance.
    """
    scores = torch.full((len(generated), len(truth)), -2., device=generated.device)
    for j in range(len(truth)):
        both = (gen_mask & candidate_mask[j][None]).float()[:, None]                      # B, 1, F
        n = both.sum((1, 2)).clamp_min(1.)
        g = generated * both; t = truth[j][None] * both
        g = g - g.sum((1, 2), keepdim=True) / n[:, None, None]; t = t - t.sum((1, 2), keepdim=True) / n[:, None, None]
        scores[:, j] = ((g * t * both).sum((1, 2)) / ((g.square() * both).sum((1, 2)).sqrt() * (t.square() * both).sum((1, 2)).sqrt() + 1e-8))
    partner = (pair_gen[:, None] == pair_cand[None]) & (torch.arange(len(truth), device=generated.device)[None] != own[:, None])
    scores = scores.masked_fill(partner, -3.)
    keep = ~partner
    ranks = []
    for i in range(len(generated)):
        row = scores[i][keep[i]]
        position = int(keep[i][:own[i] + 1].sum()) - 1
        ranks.append(float(rank_percentile(row[None], row[position][None])[0]))
    return np.array(ranks)


@torch.no_grad()
def voice_rank_matrix(mu_e, mu_a, trial_mask, sentence_mask, pair_trial, pair_sentence, size):
    """R[i, s]: chunk-level rank percentile of candidate sentence s for trial i (mean over chunks where s is speech).

    Candidates at chunk k are the sentences that are speech at k, minus those sharing the frame with s.
    """
    pe, _ = chunk_pool(mu_e, trial_mask, size)
    pa, speech = chunk_pool(mu_a, sentence_mask, size)
    pe, pa = F.normalize(pe, dim=1), F.normalize(pa, dim=1)
    sim = torch.einsum('idk,sdk->isk', pe, pa)                                             # I, S, K
    S = pa.shape[0]
    R = torch.full((pe.shape[0], S), float('nan'), device=mu_e.device)
    same_frame = pair_sentence[:, None] == pair_sentence[None]
    for s in range(S):
        values = []
        for k in range(speech.shape[1]):
            if not bool(speech[s, k]):
                continue
            candidates = speech[:, k] & (~same_frame[s] | (torch.arange(S, device=mu_e.device) == s))
            if int(candidates.sum()) < 2:
                continue
            row = sim[:, candidates, k]
            own = sim[:, s, k]
            values.append(rank_percentile(row, own))
        if values:
            R[:, s] = torch.stack(values).mean(0)
    return R


def text_rank_matrix(pred, text, pair_sentence):
    """R[i, s]: rank percentile of sentence s's text among the candidates (frame partner of s excluded)."""
    sim = pred @ text.T
    S = text.shape[0]
    R = torch.zeros(len(pred), S, device=pred.device)
    for s in range(S):
        candidates = (pair_sentence != pair_sentence[s]) | (torch.arange(S, device=pred.device) == s)
        R[:, s] = rank_percentile(sim[:, candidates], sim[:, s])
    return R


# ----------------------------------------------------------------------------- stage 1: audio
def audio_stage(args):
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    device = device_of(args.device)
    store = Store(args.root, eeg_window=tuple(args.eeg_window), seconds=args.seconds)
    times = Frozen(store, device)
    train, val = store.sentences_in('train'), store.sentences_in('val')
    scaler = MelScaler(store.mel.shape[1]).fit(store.mel[train]).to(device)
    posterior = AudioPosterior(layers=store.wavlm.shape[1], latent=args.latent, hidden=args.audio_hidden).to(device)
    decoder = LatentMelDiffusion(args.latent, store.mel.shape[1], store.frames, args.hidden, args.blocks, dropout=args.dropout).to(device)
    model = nn.ModuleDict(dict(audio=posterior, decoder=decoder))
    ema = EMA(model, args.ema)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01, betas=(.9, .99))
    out = Path(args.outputs) / args.run; out.mkdir(parents=True, exist_ok=True)
    config = dict(stage='audio', latent=args.latent, hidden=args.hidden, blocks=args.blocks, audio_hidden=args.audio_hidden,
                  dropout=args.dropout, lr=args.lr, batch=args.batch, beta_audio=args.beta_audio, free_bits=args.free_bits,
                  p_null=args.p_null, seed=args.seed, store=store.signature)
    log(stage='audio', run=args.run, parameters=sum(p.numel() for p in model.parameters()), train_sentences=len(train),
        val_sentences=len(val), frames=store.frames, latent_frames=len(times.lat), device=str(device))
    val_ids = val[:min(len(val), 64)]
    history, best, step, started, running = [], float('inf'), 0, time.monotonic(), []
    deadline = started + 60 * args.minutes
    while step < args.updates and time.monotonic() < deadline:
        ids = rng.choice(train, size=min(args.batch, len(train)), replace=False)
        wavlm = store.wavlm_of(ids, device)
        x0 = scaler.encode(store.mel[ids].to(device))
        for group in optimizer.param_groups:
            group['lr'] = lr_at(step, args.lr, args.updates, args.warmup)
        model.train(); optimizer.zero_grad(set_to_none=True)
        mu, logvar = posterior(wavlm, times.wavlm, times.lat)
        c = decoder.condition(reparameterize(mu, logvar), times.lat, times.mel)
        null = torch.from_numpy(rng.random(len(ids)) < args.p_null).to(device)
        c = torch.where(null[:, None, None], decoder.null_stream(len(ids)), c)
        diff, _, _ = decoder.loss(x0, c)
        kl = gaussian_kl(mu, logvar, torch.zeros_like(mu), torch.zeros_like(logvar),
                         torch.ones(len(ids), len(times.lat), dtype=torch.bool, device=device), args.free_bits)
        loss = diff.mean() + args.beta_audio * kl
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite stage-1 loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step(); ema.update(model); step += 1
        running.append(dict(loss=float(loss), diff=float(diff.mean()), kl=float(kl)))
        if step % 50 == 0:
            log(update=step, minutes=round((time.monotonic() - started) / 60, 2), **{k: np.mean([r[k] for r in running]) for k in running[0]})
            running = []
        if step % args.eval_every == 0:
            report = audio_check(ema.shadow, scaler, store, times, val_ids, device, args)
            report.update(update=step, minutes=round((time.monotonic() - started) / 60, 2))
            history.append(report); log(validation=report)
            payload = dict(contract=CONTRACT, config=config, model=ema.shadow.state_dict(), scaler=scaler.state_dict(), update=step, history=history)
            if report['val_loss_audio'] < best:
                best = report['val_loss_audio']; atomic_save(out / 'audio_best.pt', payload)
            atomic_save(out / 'audio_last.pt', payload)
            (out / 'audio_metrics.json').write_text(json.dumps(dict(config=config, history=history), indent=1))
    log(finished=step, best_val_loss_audio=best, output=str(out))


@torch.no_grad()
def fixed_loss(decoder, x0, c, steps=(100, 300, 500, 700, 900)):
    generator = torch.Generator().manual_seed(1234)
    noise = torch.randn(x0.shape, generator=generator).to(x0.device)
    return float(np.mean([float(decoder.loss(x0, c, torch.full((len(x0),), s, device=x0.device), noise)[0].mean()) for s in steps]))


def shared_noise(keys, shape, device):
    return torch.stack([torch.randn(shape, generator=torch.Generator().manual_seed(stable_seed(k))) for k in keys]).to(device)


@torch.no_grad()
def audio_check(model, scaler, store, times, ids, device, args):
    """Validation sentences through the audio route: denoising loss vs null, and quick samples (no EEG)."""
    model.eval()
    posterior, decoder = model['audio'], model['decoder']
    mu, logvar = posterior(store.wavlm_of(ids, device), times.wavlm, times.lat)
    x0 = scaler.encode(store.mel[ids].to(device))
    c = decoder.condition(mu, times.lat, times.mel)
    report = dict(val_loss_audio=fixed_loss(decoder, x0, c), val_loss_null=fixed_loss(decoder, x0, decoder.null_stream(len(ids))))
    report['timbre_db_x0'] = one_step_timbre(decoder, scaler, c, store.mel[ids].to(device).clamp_min(scaler.floor),
                                             store.speech_mask(ids, times.mel, device))
    k = ids[:args.quick_trials]
    noise = shared_noise([f'audio:{i}' for i in k], (store.mel.shape[1], store.frames), device)
    mel = scaler.decode(decoder.sample(c[:len(k)], decoder.null_stream(len(k)), steps=args.quick_steps, guidance=args.guidance, noise=noise))
    truth = store.mel[k].to(device).clamp_min(scaler.floor); mask = store.speech_mask(k, times.mel, device)
    report['timbre_db'] = float(timbre_distance(mel, truth, mask).mean())
    report['envelope_r'] = float(np.mean(mel_envelope_r(mel, truth, mask)))
    report['identification'] = float(identification(scaler.encode(mel), scaler.encode(store.mel[ids].to(device)), mask,
                                                    store.speech_mask(ids, times.mel, device), store.pair[k].to(device),
                                                    store.pair[ids].to(device), torch.arange(len(k), device=device)).mean())
    report['kl_audio'] = float(gaussian_kl(mu, logvar, torch.zeros_like(mu), torch.zeros_like(logvar),
                                           torch.ones(len(ids), len(times.lat), dtype=torch.bool, device=device)))
    report['latent_spread'] = float(mu.std(0).mean())
    return report


@torch.no_grad()
def one_step_timbre(decoder, scaler, c, truth, mask, step=200):
    """Timbre distance (dB) of the decoder's one-step clean estimate at a fixed step and noise: the controller's measure."""
    x0 = scaler.encode(truth)
    noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(99)).to(x0.device)
    _, x0_hat, _ = decoder.loss(x0, c, torch.full((len(x0),), step, device=x0.device), noise)
    return float(timbre_distance(scaler.decode(x0_hat), truth, mask).mean())


def mel_envelope_r(mel, truth, mask):
    """Per trial correlation of the frame-mean log-mel (a loudness envelope) over speech frames."""
    out = []
    for i in range(len(mel)):
        m = mask[i]
        a, b = mel[i][:, m].mean(0), truth[i][:, m].mean(0)
        a, b = a - a.mean(), b - b.mean()
        out.append(float(a @ b / (a.norm() * b.norm() + 1e-8)))
    return out


@torch.no_grad()
def audio_codes(audio, store, times, device):
    """Frozen audio posterior of every sentence: (mu, logvar), each (S, latent, T_latent)."""
    audio.eval(); mus, lvs = [], []
    for chunk in torch.arange(len(store.sentences)).split(32):
        mu, lv = audio(store.wavlm_of(chunk.numpy(), device), times.wavlm, times.lat)
        mus.append(mu); lvs.append(lv)
    return torch.cat(mus), torch.cat(lvs)


def load_stage1(path, device):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    c = payload['config']
    if payload.get('contract') != CONTRACT or c['stage'] != 'audio':
        raise ValueError(f'{path} is not a vaediff stage-1 checkpoint')
    return payload, c


# ----------------------------------------------------------------------------- stage 2: EEG
def eeg_stage(args):
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    device = device_of(args.device)
    store = Store(args.root, eeg_window=tuple(args.eeg_window), seconds=args.seconds, max_peak=args.max_peak)
    times = Frozen(store, device)
    stage1, c1 = load_stage1(Path(args.stage1), device)
    if c1['store']['splits'] != store.signature['splits'] or c1['store']['seconds'] != store.signature['seconds']:
        raise ValueError('stage-1 checkpoint was trained on other splits or another audio window')
    scaler = MelScaler(store.mel.shape[1]); scaler.load_state_dict(stage1['scaler']); scaler.to(device)
    audio = AudioPosterior(layers=store.wavlm.shape[1], latent=c1['latent'], hidden=c1['audio_hidden'])
    decoder = LatentMelDiffusion(c1['latent'], store.mel.shape[1], store.frames, c1['hidden'], c1['blocks'], dropout=c1['dropout'])
    holder = nn.ModuleDict(dict(audio=audio, decoder=decoder)); holder.load_state_dict(stage1['model'])
    audio.to(device).eval().requires_grad_(False); decoder.to(device)
    codes_mu, codes_lv = audio_codes(audio, store, times, device)
    if args.max_timbre_db <= 0:      # auto: 1.5 x the audio-route timbre distance of the stage-1 checkpoint
        args.max_timbre_db = max(1., 1.5 * min(h['timbre_db_x0'] for h in stage1['history']))
    posterior = EEGPosterior(latent=c1['latent'], subjects=len(store.subjects), rate=vd.EEG_RATE, start=store.eeg_window[0],
                             samples=store.samples, width=args.width, dropout=args.dropout, text_dim=store.text.shape[1]).to(device)
    model = nn.ModuleDict(dict(eeg=posterior, decoder=decoder))
    groups = [dict(params=list(posterior.parameters()), lr=args.lr, weight_decay=.05, base=args.lr)]
    if args.freeze_decoder:
        decoder.requires_grad_(False)
    else:
        groups.append(dict(params=list(decoder.parameters()), lr=args.decoder_lr, weight_decay=.01, base=args.decoder_lr))
    optimizer = torch.optim.AdamW(groups, betas=(.9, .99))
    ema = EMA(model, args.ema)
    out = Path(args.outputs) / args.run; out.mkdir(parents=True, exist_ok=True)
    controller = WeightController(dict(voice=args.w_voice, text=args.w_text, diff=args.w_diff),
                                  dict(voice=dict(voice_rank=args.max_voice_rank, timbre_db=args.max_timbre_db),
                                       text=dict(text_rank=args.max_text_rank)),
                                  eta=args.eta, bound=args.bound, mode=args.tune_mode, control=out / 'control.json')
    config = dict(stage='eeg', stage1=str(args.stage1), latent=c1['latent'], width=args.width, dropout=args.dropout, lr=args.lr,
                  decoder_lr=args.decoder_lr, freeze_decoder=args.freeze_decoder, batch=args.batch, beta=args.beta,
                  free_bits=args.free_bits, spread=args.spread, chunk=args.chunk, routes=args.routes, augment=args.augment,
                  seed=args.seed, store=store.signature, subjects=store.subjects, eeg_window=list(store.eeg_window),
                  initial_weights=dict(controller.weights), constraints=copy.deepcopy(controller.constraints))
    train = store.ids('train')
    dev = rng.choice(store.ids('dev'), size=min(256, len(store.ids('dev'))), replace=False)
    val_sentences = store.sentences_in('val')
    log(stage='eeg', run=args.run, parameters=sum(p.numel() for p in posterior.parameters()), train_trials=len(train),
        dev_trials=len(dev), subjects=len(store.subjects), removed_by_peak=store.removed, device=str(device))
    state_path = out / 'training_state.pt'
    step = epoch = cursor = 0; history, tuning, best = [], [], float('inf')
    if state_path.exists():
        saved = torch.load(state_path, map_location='cpu', weights_only=False)
        if saved['config'] != config:
            raise ValueError('resume arguments changed; use a new --run')
        model.load_state_dict(saved['model']); ema.shadow.load_state_dict(saved['ema']); optimizer.load_state_dict(saved['optimizer'])
        controller.load(saved['controller']); step, epoch, cursor = saved['step'], saved['epoch'], saved['cursor']
        history, tuning, best = saved['history'], saved['tuning'], saved['best']
        rng = np.random.default_rng(args.seed + step)
        log(resumed=step)

    def save(path, extra=None):
        atomic_save(path, dict(contract=CONTRACT, config=config, model=model.state_dict(), ema=ema.shadow.state_dict(),
                               audio=audio.state_dict(), scaler=scaler.state_dict(), optimizer=optimizer.state_dict(),
                               controller=controller.state(), step=step, epoch=epoch, cursor=cursor, history=history,
                               tuning=tuning, best=best, extra=extra))

    routes = np.array(['eeg', 'audio', 'null']); p_routes = np.array(args.routes) / np.sum(args.routes)
    started = time.monotonic(); deadline = started + 60 * args.minutes; running = []
    batches = diverse_batches(store.frame.iloc[train].reset_index(drop=True), args.batch, args.seed + epoch)
    while step < args.updates and time.monotonic() < deadline:
        if cursor >= len(batches):
            epoch += 1; cursor = 0
            batches = diverse_batches(store.frame.iloc[train].reset_index(drop=True), args.batch, args.seed + epoch)
        ids = train[np.array(batches[cursor])]; cursor += 1
        sentences = store.sentence_of[ids]
        eeg, mask = store.eeg(ids, device)
        if args.augment:
            eeg = augment_eeg(eeg, mask, span_max=32)                # spans of at most 250 ms at 128 Hz
        subject = torch.from_numpy(store.subject_index[ids]).to(device)
        lat_mask = store.speech_mask(sentences, times.lat, device)
        mel_mask = store.speech_mask(sentences, times.mel, device)
        pair = store.pair[sentences].to(device)
        mu_a, lv_a = codes_mu[sentences], codes_lv[sentences]
        for i, group in enumerate(optimizer.param_groups):
            group['lr'] = lr_at(step, group['base'], args.updates, args.warmup)
        model.train(); optimizer.zero_grad(set_to_none=True)
        mu_e, lv_e = posterior(eeg, mask, subject, times.lat)
        z_e = reparameterize(mu_e, lv_e)
        voice_nce, voice_acc = chunk_infonce(mu_e, mu_a, lat_mask, pair[:, None] == pair[None], size=args.chunk)
        text_loss, text_acc = text_infonce(posterior.text(z_e, lat_mask), store.text[sentences].to(device))
        route = rng.choice(routes, size=len(ids), p=p_routes)
        c = torch.where(torch.from_numpy(route == 'eeg').to(device)[:, None, None], decoder.condition(z_e, times.lat, times.mel),
                        decoder.condition(reparameterize(mu_a, lv_a), times.lat, times.mel))
        c = torch.where(torch.from_numpy(route == 'null').to(device)[:, None, None], decoder.null_stream(len(ids)), c)
        truth = store.mel[sentences].to(device).clamp_min(scaler.floor)
        diff, x0_hat, ab = decoder.loss(scaler.encode(truth), c)
        trusted = torch.from_numpy(route == 'eeg').to(device) & (ab > .5)
        if bool(trusted.any()):
            timbre = timbre_distance(scaler.decode(x0_hat[trusted]), truth[trusted], mel_mask[trusted])
            timbre_loss = (timbre * ab[trusted]).sum() / ab[trusted].sum()
        else:
            timbre_loss = diff.sum() * 0
        kl = gaussian_kl(mu_e, lv_e, mu_a, widened(lv_a, args.spread), torch.ones_like(lat_mask), args.free_bits)
        voice_loss = voice_nce + timbre_loss / args.max_timbre_db
        w = controller.weights
        loss = w['voice'] * voice_loss + w['text'] * text_loss + w['diff'] * diff.mean() + args.beta * kl
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite stage-2 loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step(); ema.update(model); step += 1
        running.append(dict(loss=float(loss), voice_nce=float(voice_nce), voice_acc=voice_acc, timbre_db=float(timbre_loss),
                            text=float(text_loss), text_acc=text_acc, diff=float(diff.mean()), kl=float(kl),
                            spread=float(mu_e.std(0).mean())))
        if step % args.tune_every == 0:
            measures = dev_measures(posterior, decoder, codes_mu, scaler, store, times, dev, val_sentences, device, args)
            measures.update({f'train_{k}': float(np.mean([r[k] for r in running])) for k in running[0]})
            row = controller.step(step, measures)
            row['minutes'] = round((time.monotonic() - started) / 60, 2)
            tuning.append(row)
            with (out / 'tune.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            log(tune=step, minutes=row['minutes'], voice_rank=measures['voice_rank'], text_rank=measures['text_rank'],
                timbre_db=measures['timbre_db'], diff_advantage=measures['diff_advantage'],
                weights={k: round(v, 4) for k, v in controller.weights.items()}, violated=row['violated'],
                **({'override': row['override'].get('note', True)} if 'override' in row else {}))
            running = []
        if step % args.eval_every == 0 or step >= args.updates or time.monotonic() >= deadline:
            report = sampled_check(ema.shadow, scaler, store, times, dev[:args.check_trials], val_sentences, device, args)
            report.update(update=step, minutes=round((time.monotonic() - started) / 60, 2), weights=dict(controller.weights))
            history.append(report); log(check=report)
            if report['identification'] < best:
                best = report['identification']; save(out / 'best.pt', report)
            save(out / 'last.pt', report); save(state_path)
            (out / 'metrics.json').write_text(json.dumps(dict(config=config, history=history), indent=1))
    log(finished=step, best_identification=best, output=str(out))


@torch.no_grad()
def dev_measures(posterior, decoder, codes_mu, scaler, store, times, dev, val_sentences, device, args):
    """The controller's inputs: latent voice and text retrieval among the dev sentences, timbre and EEG advantage."""
    was = posterior.training; posterior.eval(); decoder.eval()
    mus, preds = [], []
    for chunk in np.array_split(dev, max(1, len(dev) // 64)):
        eeg, mask = store.eeg(chunk, device)
        sentences = store.sentence_of[chunk]
        mu, _ = posterior(eeg, mask, torch.from_numpy(store.subject_index[chunk]).to(device), times.lat)
        mus.append(mu); preds.append(posterior.text(mu, store.speech_mask(sentences, times.lat, device)))
    mu_e, pred = torch.cat(mus), torch.cat(preds)
    sentences = store.sentence_of[dev]
    mu_a = codes_mu[val_sentences]
    column = {s: j for j, s in enumerate(val_sentences)}
    own = torch.tensor([column[s] for s in sentences], device=device)
    pair_v = store.pair[val_sentences].to(device)
    R = voice_rank_matrix(mu_e, mu_a, store.speech_mask(sentences, times.lat, device), store.speech_mask(val_sentences, times.lat, device),
                          store.pair[sentences].to(device), pair_v, args.chunk)
    T = text_rank_matrix(pred, store.text[val_sentences].to(device), pair_v)
    rows = torch.arange(len(dev), device=device)
    k = dev[:32]; ks = store.sentence_of[k]
    eeg, mask = store.eeg(k, device)
    mu, _ = posterior(eeg, mask, torch.from_numpy(store.subject_index[k]).to(device), times.lat)
    truth = store.mel[ks].to(device).clamp_min(scaler.floor); x0 = scaler.encode(truth)
    c = decoder.condition(mu, times.lat, times.mel)
    timbre = one_step_timbre(decoder, scaler, c, truth, store.speech_mask(ks, times.mel, device))
    advantage = fixed_loss(decoder, x0, decoder.null_stream(len(k))) - fixed_loss(decoder, x0, c)
    posterior.train(was); decoder.train(was)
    return dict(voice_rank=float(R[rows, own].nanmean()), text_rank=float(T[rows, own].mean()),
                timbre_db=timbre, diff_advantage=advantage, latent_spread=float(mu_e.std(0).mean()))


@torch.no_grad()
def sampled_check(model, scaler, store, times, ids, candidates, device, args):
    """Checkpoint selection: identification of the generated mel among the dev sentences, real vs wrong-trial EEG."""
    model.eval(); posterior, decoder = model['eeg'], model['decoder']
    wrong = store.wrong_trials(ids)
    sentences = store.sentence_of[ids]
    noise = shared_noise([f'check:{i}' for i in ids], (store.mel.shape[1], store.frames), device)
    column = {s: j for j, s in enumerate(candidates)}
    truth_c = scaler.encode(store.mel[candidates].to(device)); mask_c = store.speech_mask(candidates, times.mel, device)
    report = {}
    for name, rows in (('real', ids), ('wrong', wrong)):
        eeg, mask = store.eeg(rows, device)
        mu, lv = posterior(eeg, mask, torch.from_numpy(store.subject_index[rows]).to(device), times.lat)
        c = decoder.condition(mu, times.lat, times.mel)
        mel = scaler.decode(decoder.sample(c, decoder.null_stream(len(rows)), steps=args.quick_steps, guidance=args.guidance, noise=noise))
        own_mask = store.speech_mask(sentences, times.mel, device)
        ident = identification(scaler.encode(mel), truth_c, own_mask, mask_c, store.pair[sentences].to(device),
                               store.pair[candidates].to(device), torch.tensor([column[s] for s in sentences], device=device))
        truth = store.mel[sentences].to(device).clamp_min(scaler.floor)
        report[name] = dict(identification=float(ident.mean()), envelope_r=float(np.mean(mel_envelope_r(mel, truth, own_mask))),
                            timbre_db=float(timbre_distance(mel, truth, own_mask).mean()))
    report['identification'] = report['real']['identification']
    report['identification_gain_over_wrong'] = report['wrong']['identification'] - report['real']['identification']
    report['envelope_gain_over_wrong'] = report['real']['envelope_r'] - report['wrong']['envelope_r']
    return report


# ----------------------------------------------------------------------------- evaluate
def load_stage2(path, store, device):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    c = payload['config']
    if payload.get('contract') != CONTRACT or c['stage'] != 'eeg':
        raise ValueError(f'{path} is not a vaediff stage-2 checkpoint')
    _, c1 = load_stage1(Path(c['stage1']), device)
    scaler = MelScaler(store.mel.shape[1]); scaler.load_state_dict(payload['scaler'])
    audio = AudioPosterior(layers=store.wavlm.shape[1], latent=c1['latent'], hidden=c1['audio_hidden']); audio.load_state_dict(payload['audio'])
    posterior = EEGPosterior(latent=c1['latent'], subjects=len(c['subjects']), rate=vd.EEG_RATE, start=c['eeg_window'][0],
                             samples=store.samples, width=c['width'], dropout=c['dropout'], text_dim=store.text.shape[1])
    decoder = LatentMelDiffusion(c1['latent'], store.mel.shape[1], store.frames, c1['hidden'], c1['blocks'], dropout=c1['dropout'])
    model = nn.ModuleDict(dict(eeg=posterior, decoder=decoder)); model.load_state_dict(payload['ema'])
    if c['subjects'] != store.subjects:
        raise ValueError('subject table differs from the checkpoint')
    for m in (scaler, audio, model):
        m.to(device).eval()
    return payload, scaler, audio, posterior, decoder


def bootstrap_two_way(values, people, sentences, rng, repeats=2000):
    """95 % interval of the mean, resampling people and sentences independently (crossed design)."""
    values = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(values)
    values, people, sentences = values[ok], np.asarray(people)[ok], np.asarray(sentences)[ok]
    up, ip = np.unique(people, return_inverse=True); us, is_ = np.unique(sentences, return_inverse=True)
    means = []
    for _ in range(repeats):
        wp = np.bincount(rng.integers(len(up), size=len(up)), minlength=len(up))
        ws = np.bincount(rng.integers(len(us), size=len(us)), minlength=len(us))
        weight = wp[ip] * ws[is_]
        if weight.sum() > 0:
            means.append(float((weight * values).sum() / weight.sum()))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def permutation_null(R, own, groups, rng, repeats=1000):
    """Mean rank when each person x task's sentence labels are shuffled among their own trials; p (one-sided, lower)."""
    observed = float(np.nanmean(R[np.arange(len(own)), own]))
    null = []
    for _ in range(repeats):
        shuffled = own.copy()
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            shuffled[idx] = own[rng.permutation(idx)]
        null.append(float(np.nanmean(R[np.arange(len(own)), shuffled])))
    null = np.array(null)
    return dict(observed=observed, null_mean=float(null.mean()), null_95=float(np.percentile(null, 5)),
                p=float((1 + (null <= observed).sum()) / (1 + repeats)))


@torch.no_grad()
def posteriors(posterior, store, ids, times, device, zero=False):
    mus, lvs, preds = [], [], []
    for chunk in np.array_split(ids, max(1, int(math.ceil(len(ids) / 64)))):
        eeg, mask = store.eeg(chunk, device)
        if zero:
            eeg = torch.zeros_like(eeg)
        mu, lv = posterior(eeg, mask, torch.from_numpy(store.subject_index[chunk]).to(device), times.lat)
        mus.append(mu); lvs.append(lv); preds.append(posterior.text(mu, store.speech_mask(store.sentence_of[chunk], times.lat, device)))
    return torch.cat(mus), torch.cat(lvs), torch.cat(preds)


def evaluate(args):
    rng = np.random.default_rng(args.seed); device = device_of(args.device)
    payload0 = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    c = payload0['config']
    store = Store(args.root, eeg_window=tuple(c['eeg_window']), seconds=c['store']['seconds'], max_peak=c['store']['max_peak'])
    if store.signature != c['store']:
        raise ValueError('processed data differ from the ones the checkpoint was trained on')
    times = Frozen(store, device)
    payload, scaler, audio, posterior, decoder = load_stage2(args.checkpoint, store, device)
    out = Path(args.output or Path(args.checkpoint).parent / f'evaluate_{Path(args.checkpoint).stem}'); out.mkdir(parents=True, exist_ok=True)
    test_sentences = store.sentences_in('test')
    column = {s: j for j, s in enumerate(test_sentences)}
    codes_mu, codes_lv = audio_codes(audio, store, times, device)
    mu_a, lv_a = codes_mu[test_sentences], codes_lv[test_sentences]
    pair_t = store.pair[test_sentences].to(device)
    summary = dict(contract=CONTRACT, checkpoint=str(args.checkpoint), update=payload['step'], smoke=args.smoke,
                   test_sentences=len(test_sentences), removed_by_peak=store.removed, roles={})
    for role in ('test_seen', 'test_unseen'):
        ids = store.ids(role)
        if not len(ids):
            continue
        sentences = store.sentence_of[ids]
        own = np.array([column[s] for s in sentences])
        mu_e, lv_e, pred = posteriors(posterior, store, ids, times, device)
        R = voice_rank_matrix(mu_e, mu_a, store.speech_mask(sentences, times.lat, device), store.speech_mask(test_sentences, times.lat, device),
                              store.pair[sentences].to(device), pair_t, c['chunk']).cpu().numpy()
        T = text_rank_matrix(pred, store.text[test_sentences].to(device), pair_t).cpu().numpy()
        groups = np.array([f'{a}|{b}' for a, b in zip(store.subject_name[ids], store.task[ids])])
        people = store.subject_name[ids]
        result = dict(trials=len(ids), people=len(set(people)))
        for name, M in (('voice', R), ('text', T)):
            per = M[np.arange(len(ids)), own]
            result[f'{name}_retrieval'] = dict(**permutation_null(M, own, groups, rng, args.permutations),
                                               ci95=bootstrap_two_way(per, people, sentences, rng, args.bootstrap))
        gate = any(result[f'{n}_retrieval']['p'] < .05 for n in ('voice', 'text'))
        result['gate_passed'] = gate
        log(role=role, voice=result['voice_retrieval'], text=result['text_retrieval'], gate=gate)
        if gate or args.force_audio:
            result['generation'] = generate(role, ids, mu_e, lv_e, mu_a, lv_a, test_sentences, column, posterior, decoder,
                                            scaler, store, times, device, out, rng, args, forced=not gate)
        else:
            result['generation'] = 'skipped: neither voice nor text retrieval beats its permutation null (use --force-audio to override)'
        summary['roles'][role] = result
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
    write_report(summary, out)
    log(evaluated=str(out))


def stratified(ids, people, limit, rng):
    if limit <= 0 or len(ids) <= limit:
        return np.arange(len(ids))
    order = []
    by = {p: list(rng.permutation(np.flatnonzero(people == p))) for p in np.unique(people)}
    while len(order) < limit:
        for p in list(by):
            if by[p] and len(order) < limit:
                order.append(by[p].pop())
    return np.array(sorted(order))


@torch.no_grad()
def generate(role, ids, mu_e, lv_e, mu_a, lv_a, test_sentences, column, posterior, decoder, scaler, store, times, device, out, rng, args, forced):
    """Counterfactual conditions on the same trials and the same noise; Vocos waveforms; metrics; ASR; listening files."""
    import soundfile as sf
    import audio_comparison as ac
    from scipy.signal import resample_poly
    people = store.subject_name[ids]
    pick = stratified(ids, people, args.limit, rng)
    trial_ids = ids[pick]; sentences = store.sentence_of[trial_ids]
    index_of = {int(t): k for k, t in enumerate(ids)}
    wrong = store.wrong_trials(trial_ids)
    in_role = {int(t) for t in ids}
    wrong_mu = torch.stack([mu_e[index_of[int(w)]] for w in wrong]); wrong_lv = torch.stack([lv_e[index_of[int(w)]] for w in wrong])
    zero_mu, zero_lv, _ = posteriors(posterior, store, trial_ids, times, device, zero=True)
    by_sentence = {}
    for k, t in enumerate(ids):
        by_sentence.setdefault(int(store.sentence_of[t]), []).append(k)
    def pooled(s):
        rows = by_sentence[int(s)]
        return product_of_experts(mu_e[rows], lv_e[rows])
    pooled_own = [pooled(s) for s in sentences]
    pooled_wrong = [pooled(store.sentence_of[w]) for w in wrong]
    eps = shared_noise([f'z:{int(t)}' for t in trial_ids], tuple(mu_e.shape[1:]), device)
    noise = shared_noise([f'x:{int(t)}' for t in trial_ids], (store.mel.shape[1], store.frames), device)
    own_cols = [column[s] for s in sentences]
    sample = lambda mu, lv: mu + eps * (.5 * lv).exp()
    sel = torch.as_tensor(pick, device=device)
    codes = dict(real=sample(mu_e[sel], lv_e[sel]), wrong_trial=sample(wrong_mu, wrong_lv), zero_eeg=sample(zero_mu, zero_lv),
                 pooled=sample(torch.stack([p[0] for p in pooled_own]), torch.stack([p[1] for p in pooled_own])),
                 pooled_wrong=sample(torch.stack([p[0] for p in pooled_wrong]), torch.stack([p[1] for p in pooled_wrong])),
                 audio_ceiling=sample(mu_a[own_cols], lv_a[own_cols]))
    truth = store.mel[sentences].to(device)
    mels = dict(vocos_ceiling=truth)
    for name, z in codes.items():
        parts = []
        for chunk in torch.arange(len(z), device=device).split(args.batch):
            c = decoder.condition(z[chunk], times.lat, times.mel)
            parts.append(scaler.decode(decoder.sample(c, decoder.null_stream(len(chunk)), steps=args.steps, guidance=args.guidance, noise=noise[chunk])))
        mels[name] = torch.cat(parts)
    parts = []
    for chunk in torch.arange(len(trial_ids), device=device).split(args.batch):
        null = decoder.null_stream(len(chunk))
        parts.append(scaler.decode(decoder.sample(null, null, steps=args.steps, guidance=1., noise=noise[chunk])))
    mels['null'] = torch.cat(parts)
    vocos = vd.vocos_frontend().to(device)
    mask = store.speech_mask(sentences, times.mel, device)
    truth_c = scaler.encode(store.mel[test_sentences].to(device)); mask_c = store.speech_mask(test_sentences, times.mel, device)
    own_t = torch.tensor(own_cols, device=device)
    floored = truth.clamp_min(scaler.floor)
    per = {}
    waves = {}
    for name, mel in mels.items():
        wave = []
        for chunk in torch.arange(len(mel), device=device).split(16):
            wave.append(vocos.head(vocos.backbone(mel[chunk].float())).cpu().numpy())
        waves[name] = np.concatenate(wave)
        ident = identification(scaler.encode(mel), truth_c, mask, mask_c, store.pair[sentences].to(device), store.pair[test_sentences].to(device), own_t)
        rows = []
        for k, t in enumerate(trial_ids):
            n = int(float(store.duration[sentences[k]]) * vd.AUDIO_RATE)
            ref = resample_poly(np.asarray(store.wave[sentences[k]][:n], dtype=np.float32), 2, 3)
            deg = resample_poly(waves[name][k][:n], 2, 3)
            rows.append(dict(identification=float(ident[k]), stoi=ac.stoi(ref, deg), mcd=ac.mcd(ref, deg),
                             envelope_r=ac.correlation(ac.envelope(ref), ac.envelope(deg)),
                             timbre_db=float(timbre_distance(mel[k:k + 1], floored[k:k + 1], mask[k:k + 1])[0])))
        per[name] = rows
    measures = ['identification', 'stoi', 'mcd', 'envelope_r', 'timbre_db']
    people_t = store.subject_name[trial_ids]
    table = {name: {m: float(np.nanmean([r[m] for r in rows])) for m in measures} for name, rows in per.items()}
    paired = {}
    for a, b in (('real', 'wrong_trial'), ('real', 'zero_eeg'), ('real', 'null'), ('pooled', 'pooled_wrong')):
        paired[f'{a}-{b}'] = {}
        for m in measures:
            d = np.array([x[m] - y[m] for x, y in zip(per[a], per[b])])
            ok = np.isfinite(d)
            paired[f'{a}-{b}'][m] = dict(mean=float(d[ok].mean()), ci95=bootstrap_two_way(d[ok], people_t[ok], sentences[ok], rng, args.bootstrap))
    result = dict(forced=forced, trials=len(trial_ids), people=len(set(people_t)), steps=args.steps, guidance=args.guidance,
                  conditions=table, paired=paired)
    if args.asr_trials > 0:
        result['asr'] = asr_scores(waves, sentences, store, args, rng)
    listen = out / 'listening' / role
    for k in range(min(args.listen, len(trial_ids))):
        folder = listen / f'{k:02d}_{store.sentences[sentences[k]]["stim_file"].removesuffix(".wav")}_{people_t[k]}'
        folder.mkdir(parents=True, exist_ok=True)
        n = int(float(store.duration[sentences[k]]) * vd.AUDIO_RATE) + vd.AUDIO_RATE // 4
        sf.write(folder / '0_original.wav', np.asarray(store.wave[sentences[k]][:n], dtype=np.float32), vd.AUDIO_RATE)
        for j, name in enumerate(('vocos_ceiling', 'audio_ceiling', 'real', 'pooled', 'wrong_trial', 'zero_eeg', 'null'), start=1):
            sf.write(folder / f'{j}_{name}.wav', waves[name][k][:n], vd.AUDIO_RATE)
        (folder / 'transcript.txt').write_text(store.sentences[sentences[k]]['text'] + '\n')
    return result


def word_error_rate(reference: str, hypothesis: str) -> float:
    r, h = reference.split(), hypothesis.split()
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return float(d[len(h)] / max(len(r), 1))


def asr_scores(waves, sentences, store, args, rng):
    """Whisper small.en on the generated audio: WER against the transcript and text-embedding distance."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from whisper.normalizers import EnglishTextNormalizer
    from scipy.signal import resample_poly
    folder = vd.MODELS / 'whisper-small.en'
    processor = WhisperProcessor.from_pretrained(folder, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(folder, local_files_only=True).eval()
    model.generation_config.forced_decoder_ids = None
    normalize = EnglishTextNormalizer()
    count = min(args.asr_trials, len(sentences))
    picks = np.sort(rng.choice(len(sentences), size=count, replace=False))
    out = {}
    names = ['original', 'vocos_ceiling', 'audio_ceiling', 'real', 'pooled', 'wrong_trial']
    for name in names:
        texts = []
        for k in picks:
            n = int(float(store.duration[sentences[k]]) * vd.AUDIO_RATE) + vd.AUDIO_RATE // 4
            source = np.asarray(store.wave[sentences[k]][:n], dtype=np.float32) if name == 'original' else waves[name][k][:n]
            features = processor(resample_poly(source, 2, 3), sampling_rate=16000, return_tensors='pt').input_features
            tokens = model.generate(features, do_sample=False, num_beams=1)
            texts.append(normalize(processor.batch_decode(tokens, skip_special_tokens=True)[0]).strip())
        refs = [normalize(store.sentences[sentences[k]]['text']) for k in picks]
        embedded = vd.text_embeddings([t or '.' for t in texts], vd.text_model_dir())
        truth = store.text[sentences[picks]].numpy()
        out[name] = dict(wer=float(np.mean([word_error_rate(r, h) for r, h in zip(refs, texts)])),
                         text_distance=float(np.mean(1 - (embedded * truth).sum(1))),
                         examples=[dict(reference=r, heard=h) for r, h in list(zip(refs, texts))[:8]])
    return out


def write_report(summary, out: Path):
    smoke = summary['smoke']
    title = '冒烟测试：只检查代码能否跑通，不是实验结果' if smoke else 'vaediff 测试集评估'
    lines = [f'# {title}', '', f"检查点 `{summary['checkpoint']}`（第 {summary['update']} 步），测试句子 {summary['test_sentences']} 个，"
             f"按峰值剔除的试次 {summary['removed_by_peak']} 个。排名百分位 0 = 最好，0.5 = 随机。", '']
    for role, r in summary['roles'].items():
        lines += [f"## {'见过的人' if role == 'test_seen' else '没见过的人'}（{role}）：{r['trials']} 个试次，{r['people']} 人", '',
                  '| 潜空间检索 | 观测 | 95% 区间（人 × 句子） | 置换零分布均值 | p |', '|---|---|---|---|---|']
        for name in ('voice', 'text'):
            v = r[f'{name}_retrieval']
            lines.append(f"| {'音色/内容（块）' if name == 'voice' else '文本'} | {v['observed']:.3f} | {v['ci95'][0]:.3f}–{v['ci95'][1]:.3f} | {v['null_mean']:.3f} | {v['p']:.3f} |")
        lines.append('')
        g = r['generation']
        if isinstance(g, str):
            lines += [f'生成：{g}', '']; continue
        if g['forced']:
            lines += ['**注意：检索没有超过零分布，这里的生成是用 --force-audio 强制做的。**', '']
        lines += [f"生成：{g['trials']} 个试次，DDIM {g['steps']} 步，引导 {g['guidance']}。", '',
                  '| 条件 | 辨认排名 | STOI | MCD | 包络 r | 音色差 (dB) |', '|---|---|---|---|---|---|']
        for name, m in g['conditions'].items():
            lines.append(f"| {name} | {m['identification']:.3f} | {m['stoi']:.3f} | {m['mcd']:.2f} | {m['envelope_r']:.3f} | {m['timbre_db']:.2f} |")
        lines += ['', '| 配对差 | 辨认排名 | 包络 r | 音色差 (dB) |', '|---|---|---|---|']
        for name, p in g['paired'].items():
            cell = lambda m: f"{p[m]['mean']:+.3f} [{p[m]['ci95'][0]:+.3f}, {p[m]['ci95'][1]:+.3f}]"
            lines.append(f"| {name} | {cell('identification')} | {cell('envelope_r')} | {cell('timbre_db')} |")
        if 'asr' in g:
            lines += ['', '| 条件 | Whisper WER | 文本嵌入距离 |', '|---|---|---|']
            for name, a in g['asr'].items():
                lines.append(f"| {name} | {a['wer']:.2f} | {a['text_distance']:.3f} |")
        lines.append('')
    (out / 'report.md').write_text('\n'.join(lines) + '\n')


# ----------------------------------------------------------------------------- status
def status(args):
    folder = Path(args.outputs) / args.run
    path = folder / 'tune.jsonl'
    if not path.exists():
        raise SystemExit(f'no tuning log at {path}')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    print(f'{folder}: {len(rows)} tuning rounds; control file: {"present" if (folder / "control.json").exists() else "none"}')
    print(f"{'update':>7} {'min':>6} {'voice':>6} {'text':>6} {'timbre':>7} {'advant':>7} {'w_voice':>8} {'w_text':>7} {'w_diff':>7}  violated")
    for r in rows[-args.last:]:
        m, w = r['measures'], r['weights']
        print(f"{r['update']:7d} {r.get('minutes', 0):6.1f} {m['voice_rank']:6.3f} {m['text_rank']:6.3f} {m['timbre_db']:7.2f} "
              f"{m['diff_advantage']:7.4f} {w['voice']:8.3f} {w['text']:7.3f} {w['diff']:7.3f}  {','.join(r['violated']) or '-'}"
              + ('  [override]' if 'override' in r else ''))
    metrics = folder / 'metrics.json'
    if metrics.exists():
        h = json.loads(metrics.read_text())['history']
        if h:
            print('last sampled check:', json.dumps({k: h[-1][k] for k in ('update', 'identification', 'identification_gain_over_wrong', 'envelope_gain_over_wrong')}))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=['audio', 'eeg', 'evaluate', 'status'])
    p.add_argument('--root', default=str(vd.OUT)); p.add_argument('--outputs', default=str(OUTPUT)); p.add_argument('--run', default='main')
    p.add_argument('--device', default='auto'); p.add_argument('--seed', type=int, default=11)
    p.add_argument('--eeg-window', type=float, nargs=2, default=(-.25, 3.6))
    p.add_argument('--seconds', type=float, default=3.6, help='audio window (s); also the latent and mel length')
    p.add_argument('--max-peak', type=float, default=30., help='drop trials whose stored epoch exceeds this |MAD units|')
    p.add_argument('--minutes', type=float, default=60.); p.add_argument('--updates', type=int, default=10000)
    p.add_argument('--batch', type=int, default=32); p.add_argument('--warmup', type=int, default=300)
    p.add_argument('--ema', type=float, default=.999); p.add_argument('--dropout', type=float, default=.1)
    p.add_argument('--eval-every', type=int, default=500); p.add_argument('--quick-steps', type=int, default=20)
    p.add_argument('--quick-trials', type=int, default=16); p.add_argument('--guidance', type=float, default=2.)
    # stage 1
    p.add_argument('--latent', type=int, default=32); p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--blocks', type=int, default=6); p.add_argument('--audio-hidden', type=int, default=256)
    p.add_argument('--lr', type=float, default=None); p.add_argument('--beta-audio', type=float, default=.01)
    p.add_argument('--free-bits', type=float, default=.05); p.add_argument('--p-null', type=float, default=.1)
    # stage 2
    p.add_argument('--stage1', default=None, help='stage-1 checkpoint (default <outputs>/audio/audio_best.pt)')
    p.add_argument('--width', type=int, default=128); p.add_argument('--decoder-lr', type=float, default=5e-5)
    p.add_argument('--freeze-decoder', action='store_true'); p.add_argument('--no-augment', dest='augment', action='store_false')
    p.add_argument('--routes', type=float, nargs=3, default=(.6, .3, .1), metavar=('EEG', 'AUDIO', 'NULL'))
    p.add_argument('--beta', type=float, default=.05, help='weight of KL(q_e || widened q_a)')
    p.add_argument('--spread', type=float, default=.3, help='allowed person-to-person spread s around the audio code')
    p.add_argument('--chunk', type=int, default=10, help='latent frames per contrast chunk (10 = 0.4 s)')
    p.add_argument('--w-voice', type=float, default=.5); p.add_argument('--w-text', type=float, default=.2); p.add_argument('--w-diff', type=float, default=1.)
    p.add_argument('--max-voice-rank', type=float, default=.42)
    p.add_argument('--max-timbre-db', type=float, default=0., help='timbre threshold in dB; <= 0: 1.5 x the stage-1 audio-route value')
    p.add_argument('--max-text-rank', type=float, default=.45)
    p.add_argument('--tune-every', type=int, default=50); p.add_argument('--tune-mode', choices=['auto', 'hold'], default='auto')
    p.add_argument('--eta', type=float, default=.5); p.add_argument('--bound', type=float, default=4.)
    p.add_argument('--check-trials', type=int, default=48)
    # evaluate / status
    p.add_argument('--checkpoint'); p.add_argument('--output')
    p.add_argument('--steps', type=int, default=50); p.add_argument('--limit', type=int, default=120)
    p.add_argument('--permutations', type=int, default=1000); p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--asr-trials', type=int, default=40); p.add_argument('--listen', type=int, default=6)
    p.add_argument('--force-audio', action='store_true'); p.add_argument('--smoke', action='store_true')
    p.add_argument('--last', type=int, default=20)
    args = p.parse_args(argv)
    if args.stage1 is None:
        args.stage1 = str(Path(args.outputs) / 'audio/audio_best.pt')
    if args.lr is None:
        args.lr = 2e-4 if args.stage == 'audio' else 3e-4
    torch.set_num_threads(4)
    if args.stage == 'audio':
        audio_stage(args)
    elif args.stage == 'eeg':
        eeg_stage(args)
    elif args.stage == 'evaluate':
        if not args.checkpoint:
            p.error('evaluate needs --checkpoint')
        evaluate(args)
    else:
        status(args)


if __name__ == '__main__':
    main()

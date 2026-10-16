"""Generative (flow-matching) Mel decoder for EEG -> speech reconstruction.

Conditional flow matching is the diffusion-family model of the original plan: a small network learns the velocity
that carries a starting point to the real Mel along a straight path, and sampling integrates it in a few Euler steps.

* Start: the output of the deterministic v3 decoder for the same code, plus Gaussian noise (sigma0). The flow only
  has to add what the deterministic decoder smooths away (harmonics, sharp transitions).
* Condition: the synthesis code (WavLM features or the shared embedding) and that deterministic Mel; dropped in
  10 % of the training steps, so classifier-free guidance can be used when sampling.
* One loss: mean squared error of the velocity.
* Stage 'audio': conditioned on true codes of real audio (no EEG). Stage 'eeg': conditioned on the codes the frozen
  v3 bridge predicts from EEG (optionally mixed with true codes), with the real Mel as target.

Sampling is seeded, so an evaluation is reproducible; the agreement between two seeds measures how much of the
output the model invents rather than takes from its condition.
"""
from __future__ import annotations
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..utils import atomic_json, append_jsonl
from .softalign import Residual, Stream, audio_anchor_sampler, fixed_set, load_aligner, lr_at, status, to
from .reconstruct import (build_generator, descriptor, load_audio_decoder, mel_frames, reconstruction_metrics, audio_metrics, word_identification,
                          word_spans, listen, _eeg_batch)
from . import recon_eval


def timestep_embedding(t, dim=64):
    half = dim//2; freq = torch.exp(-math.log(1000.)*torch.arange(half, device=t.device)/half)
    angle = t[:, None]*freq[None]*1000; return torch.cat((angle.sin(), angle.cos()), -1)


class VelocityNet(nn.Module):
    """Dilated residual convolutions over Mel frames, conditioned on the code and the deterministic Mel, FiLM on time."""
    def __init__(self, code_dim, width=256, dilations=(1, 2, 4, 8, 1, 2, 4, 8)):
        super().__init__()
        self.x_in = nn.Conv1d(100, width, 3, padding=1); self.mu_in = nn.Conv1d(100, width, 3, padding=1); self.code_in = nn.Linear(code_dim, width)
        self.time = nn.Sequential(nn.Linear(64, width), nn.SiLU(), nn.Linear(width, 2*width*len(dilations)))
        self.blocks = nn.ModuleList([Residual(width, d, 5) for d in dilations]); self.out = nn.Conv1d(width, 100, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, t, code, mu):
        c = F.interpolate(self.code_in(code).transpose(1, 2), size=x.shape[-1], mode='linear', align_corners=False)
        h = self.x_in(x)+self.mu_in(mu)+c; film = self.time(timestep_embedding(t)).chunk(2*len(self.blocks), -1)
        for k, block in enumerate(self.blocks): h = block(h*(1+film[2*k][:, :, None])+film[2*k+1][:, :, None])
        return self.out(h)


class FlowDecoder(nn.Module):
    """Drop-in replacement for the deterministic AudioDecoder: same `code`, `route`, `dim` (and `norm`), but forward()
    samples the Mel by integrating the learned flow from (deterministic Mel + noise)."""
    def __init__(self, base, sigma=.5, steps=8, guidance=1., seed=0):
        super().__init__(); self.base = base.requires_grad_(False).eval(); self.route, self.dim = base.route, base.dim
        if base.route == 'wavlm': self.norm = base.norm
        self.net = VelocityNet(base.dim); self.sigma, self.steps, self.guidance, self.seed = sigma, steps, guidance, seed

    def code(self, aligner, layers, env): return self.base.code(aligner, layers, env)

    def start(self, mu, seed):
        g = torch.Generator().manual_seed(seed); return mu+self.sigma*torch.randn(mu.shape, generator=g).to(mu.device)

    def loss(self, code, target, uncond=.1):
        """Flow-matching loss on a batch: straight path from (deterministic Mel + noise) to the real Mel."""
        with torch.no_grad(): mu = self.base(code, target.shape[-1])
        x0 = mu+self.sigma*torch.randn_like(mu); t = torch.rand(len(target), device=target.device)
        xt = (1-t[:, None, None])*x0+t[:, None, None]*target
        keep = (torch.rand(len(target), device=target.device) >= uncond).float()
        v = self.net(xt, t, code*keep[:, None, None], mu*keep[:, None, None])
        return F.mse_loss(v, target-x0)

    @torch.no_grad()
    def sample(self, code, frames, steps=None, guidance=None, seed=None):
        steps, guidance = steps or self.steps, self.guidance if guidance is None else guidance
        mu = self.base(code, frames); x = self.start(mu, self.seed if seed is None else seed)
        for k in range(steps):
            t = torch.full((len(x),), k/steps, device=x.device); v = self.net(x, t, code, mu)
            if guidance != 1.: free = self.net(x, t, code*0, mu*0); v = free+guidance*(v-free)
            x = x+v/steps
        return x

    def forward(self, code, frames=None): return self.sample(code, frames or mel_frames(code.shape[1]))


def build_flow(audio_folder, device, checkpoint=None, steps=8, guidance=1.):
    base, _ = load_audio_decoder(audio_folder, device); flow = FlowDecoder(base, steps=steps, guidance=guidance).to(device)
    if checkpoint: saved = torch.load(checkpoint, map_location=device, weights_only=False); flow.net.load_state_dict(saved['net']); flow.sigma = saved['sigma']
    return flow


def train_flow(cfg, cache, align_folder, audio_folder, folder, deadline, device='cpu', threads=6, workers=2, seed=11, stage='audio', bridge=None,
               init=None, eeg_share=1., validate_minutes=10, max_steps=None, peak=5e-4, limit=200, steps=8):
    """Stage 'audio': true codes of real audio. Stage 'eeg': codes predicted from EEG by the frozen bridge of `bridge`
    (a v3 EEG-side run), mixed with true codes in 1 - eeg_share of the batches. Best checkpoint on development data:
    dynamic Mel r of sampled Mel (audio stage on validation audio, EEG stage on development EEG)."""
    from torch.utils.data import DataLoader
    folder = Path(folder)
    if (folder/'flow_latest.pt').exists(): raise ValueError(f'{folder} already holds a run; choose another --run')
    folder.mkdir(parents=True, exist_ok=True); torch.manual_seed(seed); torch.set_num_threads(threads); rng = np.random.default_rng(seed)
    flow = build_flow(audio_folder, device, init, steps); aligner, _ = load_aligner(cache, align_folder, device)
    optimizer = torch.optim.AdamW(flow.net.parameters(), lr=peak, weight_decay=.01)
    gen = None
    if stage == 'eeg':
        gen = build_generator(cache, align_folder, audio_folder, device, init=Path(bridge)/'best.pt'); gen.requires_grad_(False).eval()
        loader = iter(DataLoader(Stream(cache, set(cache.subjects), 'train', seed=seed), batch_size=None, num_workers=workers,
                                 persistent_workers=workers > 0, prefetch_factor=4 if workers else None))
        dev_items = fixed_set(cache, set(cfg.fold_subjects('P1', 'F0', 'validation')), 'validation')
    else: dev = audio_anchor_sampler(cache, 'validation', np.random.default_rng(0), 600, balance=True, conditions=('clean',))
    began = time.time(); last = began; best = -math.inf; step = 0
    status(folder, stage=f'flow ({stage})', audio=str(audio_folder), bridge=str(bridge), init=str(init), eeg_share=eeg_share, sigma=flow.sigma,
           began=time.strftime('%H:%M'), deadline=time.strftime('%H:%M', time.localtime(deadline)))

    def validate(final=False):
        nonlocal best
        flow.eval()
        if stage == 'eeg':
            decoder, gen.audio_decoder = gen.audio_decoder, flow
            score = reconstruction_metrics(gen, cache, dev_items, device, limit=limit)['pooled']; gen.audio_decoder = decoder
        else: score = audio_metrics(flow, aligner, cache, dev, device)['pooled']
        minutes = round((time.time()-began)/60, 1); append_jsonl(folder/'flow_validation.jsonl', {'step': step, 'minutes': minutes, 'final': final, **score})
        saved = {'net': flow.net.state_dict(), 'sigma': flow.sigma, 'stage': stage, 'audio': str(audio_folder), 'align': str(align_folder), 'bridge': str(bridge) if bridge else None,
                 'step': step, 'minutes': minutes, 'validation': score}
        torch.save(saved, folder/'flow_latest.pt')
        if score['dynamic_mel_r'] > best: best = score['dynamic_mel_r']; torch.save(saved, folder/'flow.pt')
        status(folder, validated_step=step, validation=score, best_dynamic_mel_r=best); flow.net.train()

    flow.net.train()
    while time.time() < deadline and not (max_steps and step >= max_steps):
        if stage == 'eeg':
            batch = to(next(loader), device); anchor = batch['anchor']; wv, en = cache.audio_batch(batch['anchors'])
            target = cache.mel_batch(batch['anchors']).to(device)[anchor]
            with torch.no_grad():
                code = gen.code(batch['eeg'], batch['positions'], batch['mask'], batch['subject'])
                if eeg_share < 1:
                    true = flow.code(aligner, wv.to(device), en.to(device))[anchor]; swap = torch.rand(len(code), device=device) >= eeg_share
                    code = torch.where(swap[:, None, None], true, code)
        else:
            anchors = audio_anchor_sampler(cache, 'train', rng, 32, balance=True); wv, en = cache.audio_batch(anchors); target = cache.mel_batch(anchors).to(device)
            with torch.no_grad(): code = flow.code(aligner, wv.to(device), en.to(device))
        for group in optimizer.param_groups: group['lr'] = lr_at((time.time()-began)/max(deadline-began, 1), step, peak=peak)
        loss = flow.loss(code, target); optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.net.parameters(), 1.); optimizer.step(); step += 1
        if step % 50 == 0: append_jsonl(folder/'flow_training.jsonl', {'step': step, 'minutes': round((time.time()-began)/60, 1), 'loss': loss.item()})
        if time.time()-last >= validate_minutes*60: last = time.time(); validate()
    validate(final=True); status(folder, stage=f'flow ({stage}) done', step=step)


@torch.no_grad()
def seed_agreement(gen, cache, items, device, n=64):
    """Dynamic Mel r between two samples (seeds 0 and 1) of the same condition: 1 = the condition decides everything,
    low = the model invents much of the output. For EEG codes and for true codes."""
    import random
    picked = random.Random(0).sample(items, min(n, len(items))); out = {}
    for label in ('eeg', 'true_code'):
        values = []
        for dataset in sorted({d for d, _, _ in picked}):
            rows = cache.records[dataset]; part = [(i, a) for d, i, a in picked if d == dataset]
            if label == 'eeg': code = gen.code(*_eeg_batch(cache, dataset, part, device))
            else:
                wv, en = cache.audio_batch([(dataset, rows[i]['audio'], a) for i, a in part]); code = gen.audio_decoder.code(gen.aligner, wv.to(device), en.to(device))
            frames = mel_frames(code.shape[1]); a_, b_ = gen.audio_decoder.sample(code, frames, seed=0), gen.audio_decoder.sample(code, frames, seed=1)
            values += (descriptor(a_)*descriptor(b_)).sum(1).cpu().tolist()
        out[label] = float(np.mean(values))
    return out


@torch.no_grad()
def evaluate_flow(cfg, cache, folder, out, run, device='mps', steps=8, guidance=1., max_records=None, sentences=5, bootstrap=2000, permutations=1000):
    """The adjusted evaluation (long stretches, frames, words, sentences) with the flow decoder in place of the
    deterministic one, plus the seed agreement. Needs a stage-'eeg' run (or any flow run plus its bridge)."""
    folder, out = Path(folder), Path(out); out.mkdir(parents=True, exist_ok=True)
    saved = torch.load(folder/'flow.pt', map_location=device, weights_only=False)
    if not saved['bridge']: raise ValueError('this flow run has no EEG bridge; train stage eeg or pass a bridge')
    gen = build_generator(cache, saved['align'], saved['audio'], device, init=Path(saved['bridge'])/'best.pt'); gen.eval()
    gen.audio_decoder = build_flow(saved['audio'], device, folder/'flow.pt', steps, guidance).eval(); spans = word_spans(cfg, cache)
    limit = 30 if max_records else None
    summary = {'run': run, 'checkpoint': 'flow.pt', 'step': saved['step'], 'minutes': saved['minutes'], 'route': gen.audio_decoder.route, 'smoke': cache.smoke,
               'flow': {'stage': saved['stage'], 'steps': steps, 'guidance': guidance, 'sigma': saved['sigma'], 'bridge': saved['bridge']}}
    groups = (('seen_people_unseen_speech', set(cache.subjects)), ('unseen_people_unseen_speech', set(cfg.fold_subjects('P1', 'F0', 'test'))))
    for name, people in groups:
        status(folder, stage=f'flow evaluation: {name}'); items = fixed_set(cache, people, 'test')
        summary[name] = {'long': recon_eval.long_stretch_metrics(gen, cache, people, device, max_records, decode_device=device),
                         'frames': reconstruction_metrics(gen, cache, items, device, limit=limit, permutations=permutations, bootstrap=bootstrap)['pooled'],
                         'words': word_identification(gen, cache, spans, items, device, limit=limit, bootstrap=bootstrap)['pooled']}
    summary['seed_agreement'] = seed_agreement(gen, cache, fixed_set(cache, set(cache.subjects), 'test'), device)
    gen = gen.to('cpu'); summary['sentences'] = {}
    for name, people in groups:
        rows, heard = listen(cfg, cache, gen, gen.audio_decoder, gen.aligner, people, out/name.split('_')[0], sentences if name.startswith('seen') else 2)
        summary['sentences'][name] = {'summary': heard, 'clips': rows}
    atomic_json(out/'evaluation.json', summary); recon_eval.write(summary, out)
    f, s = summary['flow'], summary['seed_agreement']
    note = ['', '## 生成式解码器（流匹配）', '',
            f"- 阶段 {f['stage']}；采样 {f['steps']} 步，引导系数 {f['guidance']}，起点噪声 σ = {f['sigma']}；固定种子，所以结果可复现。桥接来自 `{f['bridge']}`。",
            f"- 两个种子之间的一致度（动态 Mel r，1 = 完全由条件决定）：EEG 代码 {s['eeg']:.2f}，真实代码 {s['true_code']:.2f}。"
            '一致度低说明输出里有不少细节是模型自己生成的，而不是来自条件；所以必须同时看错时与无 EEG 两个对照。',
            '- 对照的确定性解码器结果见 `outputs/reconstruct/v3/<路线>/eval2/evaluation.md`。']
    with open(out/'evaluation.md', 'a') as handle: handle.write('\n'.join(note)+'\n')
    status(folder, stage='flow evaluation done')
    return summary

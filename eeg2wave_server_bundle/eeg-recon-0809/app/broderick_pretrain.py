#!/usr/bin/env python3
"""Pretrain the EEG encoder trunk on Broderick 2018 envelope tracking.

The trunk (signed spatial filters -> strided temporal conv -> dilated residual
blocks -> LayerNorm) is identical to the DS004940 recovery encoder, so its
weights can initialise that model (``aligned_recovery.py --initialize-trunk``).
Here it predicts the speech envelope at the 64 Hz token rate from random
4.6 s windows of ~19 hours of natural-speech EEG (runs 1-18 of every
participant); runs 19-20 are held out for validation.  The loss is
1 - Pearson r per window plus a small MSE on the standardised envelope, i.e.
the same quantity the G1 linear gate measures, but learned by the network.
Nothing from DS004940 is used, so there is no leakage into its held-out
sentences.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('ALIGNED_TARGET_CACHE_NAME', 'targets_adapted.h5')
import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src'))
from aligned_recovery_model import TemporalResidual, augment_eeg
from eeg2speech.aligned import EEG_SAMPLES

SOURCE_RATE = 128; TARGET_RATE = 256; TOKEN_STRIDE = 4
TRUNK_KEYS = ('spatial', 'temporal', 'blocks', 'output_norm')
CONTRACT = 'broderick_trunk_v1'


class EnvelopeTrunk(nn.Module):
    """Same module names as RecoveryEEGModel so the state dict transfers directly."""
    def __init__(self, channels=128, width=128, subjects=0, dropout=.1):
        super().__init__()
        self.channels, self.subjects = channels, int(subjects)
        if self.subjects:
            self.subject_delta = nn.Parameter(torch.zeros(self.subjects, channels, channels))
        self.spatial = nn.Conv1d(channels, width, 1, bias=False)
        self.temporal = nn.Conv1d(width, width, 15, stride=TOKEN_STRIDE, padding=7)
        self.blocks = nn.Sequential(*[TemporalResidual(width, 2 ** i, dropout) for i in range(6)])
        self.output_norm = nn.LayerNorm(width)
        self.envelope = nn.Conv1d(width, 1, 1)

    def forward(self, eeg, subject=None):
        x = eeg
        if subject is not None and self.subjects:
            x = torch.bmm(torch.eye(self.channels, device=x.device, dtype=x.dtype) + self.subject_delta[subject], x)
        x = F.gelu(self.temporal(self.spatial(x)))
        x = self.output_norm(self.blocks(x).transpose(1, 2)).transpose(1, 2)
        return self.envelope(x).squeeze(1)                       # B, tokens

    def trunk_state(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k.split('.')[0] in TRUNK_KEYS}


class Windows:
    """Random (train) or strided (validation) 4.6 s windows from the Broderick shards."""
    def __init__(self, shards, runs, seed):
        self.rng = np.random.default_rng(seed)
        self.items = []
        for index, path in enumerate(shards):
            with h5py.File(path, 'r') as h5:
                table = {k: h5['runs'][k][:] for k in h5['runs']}
                spans = [(int(s), int(e)) for r, s, e in zip(table['run'], table['start'], table['end']) if r in runs]
                self.items.append(dict(path=path, index=index, spans=spans, center=np.asarray(h5.attrs['normalizer_center'], np.float32)[:, None],
                                       scale=np.asarray(h5.attrs['normalizer_scale'], np.float32)[:, None]))
        self.source_samples = EEG_SAMPLES // 2                    # 589 samples at 128 Hz -> 1178 at 256 Hz
        self.token_times = np.arange((EEG_SAMPLES + 2 * 7 - 15) // TOKEN_STRIDE + 1) * TOKEN_STRIDE / TARGET_RATE

    def strided(self, stride_s=2.):
        out = []
        for item in self.items:
            for start, end in item['spans']:
                for s in range(start, end - self.source_samples, int(stride_s * SOURCE_RATE)):
                    out.append((item['index'], s))
        return out

    def random(self, count):
        out = []
        for _ in range(count):
            item = self.items[self.rng.integers(len(self.items))]
            start, end = item['spans'][self.rng.integers(len(item['spans']))]
            out.append((item['index'], int(self.rng.integers(start, end - self.source_samples))))
        return out

    def load(self, keys):
        from scipy.signal import resample_poly
        eeg, envelope, subject = [], [], []
        handles = {}
        try:
            for index, start in keys:
                item = self.items[index]
                h5 = handles.setdefault(index, h5py.File(item['path'], 'r'))
                x = (h5['eeg'][:, start:start + self.source_samples] - item['center']) / item['scale']
                eeg.append(resample_poly(x.astype(np.float64), 2, 1, axis=1)[:, :EEG_SAMPLES].astype(np.float32))
                env = h5['envelope'][start:start + self.source_samples]
                envelope.append(env[np.minimum((self.token_times * SOURCE_RATE).round().astype(int), len(env) - 1)])
                subject.append(index)
        finally:
            for h5 in handles.values():
                h5.close()
        return torch.from_numpy(np.stack(eeg)), torch.from_numpy(np.stack(envelope).astype(np.float32)), torch.tensor(subject)


def pearson_loss(prediction, target):
    p = prediction - prediction.mean(1, keepdim=True); t = target - target.mean(1, keepdim=True)
    r = (p * t).sum(1) / (p.norm(dim=1) * t.norm(dim=1) + 1e-6)
    return (1 - r).mean(), r


@torch.no_grad()
def validate(model, windows, keys, device, batch=64):
    model.eval(); correlations = []; shuffled = []
    for start in range(0, len(keys), batch):
        eeg, env, subject = windows.load(keys[start:start + batch])
        eeg, env, subject = eeg.to(device), env.to(device), subject.to(device)
        env = (env - env.mean(1, keepdim=True)) / (env.std(1, keepdim=True) + 1e-6)
        prediction = model(eeg, subject)
        correlations.extend(pearson_loss(prediction, env)[1].cpu().tolist())
        shuffled.extend(pearson_loss(prediction, env.roll(1, 0))[1].cpu().tolist())     # wrong-window null
    return float(np.mean(correlations)), float(np.mean(shuffled))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shards', default=str(ROOT / 'artifacts/broderick2018/shards'))
    parser.add_argument('--output', default=str(ROOT / 'outputs/broderick2018/trunk'))
    parser.add_argument('--updates', type=int, default=6000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--seed', type=int, default=31)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    torch.manual_seed(args.seed); torch.set_num_threads(4)
    device = torch.device('mps' if args.device == 'auto' and torch.backends.mps.is_available() else ('cpu' if args.device == 'auto' else args.device))
    shards = sorted(Path(args.shards).glob('Subject*.h5'), key=lambda p: int(p.stem.replace('Subject', '')))
    train = Windows(shards, runs=set(range(1, 19)), seed=args.seed); held = Windows(shards, runs={19, 20}, seed=args.seed + 1)
    validation_keys = held.strided()
    model = EnvelopeTrunk(width=args.width, subjects=len(shards)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.05)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    best = -1.; history = []; started = time.monotonic(); collected = []
    for step in range(1, args.updates + 1):
        lr = args.lr * min(1., step / 200) * (.1 + .9 * .5 * (1 + math.cos(math.pi * min(1., step / args.updates))))
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train(); optimizer.zero_grad(set_to_none=True)
        eeg, env, subject = train.load(train.random(args.batch_size))
        eeg, env, subject = eeg.to(device), env.to(device), subject.to(device)
        env = (env - env.mean(1, keepdim=True)) / (env.std(1, keepdim=True) + 1e-6)
        eeg = augment_eeg(eeg, torch.ones(eeg.shape[:2], dtype=torch.bool, device=device))
        prediction = model(eeg, subject)
        loss_r, r = pearson_loss(prediction, env)
        loss = loss_r + .1 * F.mse_loss(prediction, env)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
        collected.append(float(r.mean()))
        if step % 50 == 0:
            print(json.dumps(dict(update=step, train_r=round(float(np.mean(collected)), 4), lr=lr, seconds=round(time.monotonic() - started, 1))), flush=True); collected = []
        if step % args.eval_every == 0 or step == args.updates:
            r_val, r_null = validate(model, held, validation_keys, device)
            history.append(dict(update=step, validation_r=r_val, wrong_window_r=r_null, windows=len(validation_keys)))
            print(json.dumps(history[-1]), flush=True)
            if r_val > best:
                best = r_val
                torch.save(dict(contract=CONTRACT, trunk=model.trunk_state(), width=args.width, update=step, validation_r=r_val,
                                wrong_window_r=r_null, shards=[p.name for p in shards], seed=args.seed), output / 'best.pt')
            (output / 'metrics.json').write_text(json.dumps(dict(contract=CONTRACT, history=history, best_validation_r=best), indent=2) + '\n')
    print(json.dumps(dict(best_validation_r=best, checkpoint=str(output / 'best.pt'))))


if __name__ == '__main__':
    main()

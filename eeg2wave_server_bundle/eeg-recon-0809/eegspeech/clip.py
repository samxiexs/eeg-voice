"""Personal CLIP encoders: EEG trial features -> the speech-embedding space of the vocabulary.

The CLIP space is anchored on speech: each item k has an anchor a_k, the (centred, PCA-reduced, unit)
HuBERT embedding of its synthesised renderings.  Each person gets a linear map from standardised
filter-bank log-power features to a vector z in that space, trained only on the person's own
labelled trials (personal calibration from data; no identity enters any model) with

    clip      InfoNCE in both directions between trials and item speech embeddings, logits z . a_k
              (EEG -> speech: which item's speech; speech -> EEG: which trials of the batch)
    supcon    optional supervised contrast between trials (same item = positive, whatever the
              modality: heard, spoken and imagined trials of one item are pulled together)

With a linear map into the K - 1 dimensional anchor space, the dot-product InfoNCE has the capacity
of multinomial logistic regression and is convex.  L-BFGS solves it to convergence and matches
logistic regression within a person (BCI2020 0.370 vs 0.367); the usual cosine form trained with
Adam lost 3-4 points.  The posterior p(k | trial) = softmax(s z . a_k), with the temperature s
calibrated on cross-fitted trials, conditions the generator through e = sum_k p_k a_k.  All persons
of a dataset train in parallel as one batch of independent models.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .losses import supcon


def anchors(item_embeddings):
    """Unit anchors (K, K - 1) from item speech embeddings (K, D): centre, PCA, normalise."""
    h = np.asarray(item_embeddings, np.float64)
    h = h - h.mean(0)
    _, _, vt = np.linalg.svd(h, full_matrices=False)
    a = h @ vt[:len(h) - 1].T
    return (a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float32)


@dataclass
class Person:
    x: np.ndarray            # (n, F) features
    item: np.ndarray         # (n,) item index into the anchors
    weight: np.ndarray       # (n,) loss weight (aux modalities < 1)


class PersonalClip:
    """A batch of per-person linear CLIP encoders (P persons)."""

    def __init__(self, mean, std, weight, bias):
        self.mean, self.std, self.weight, self.bias = mean, std, weight, bias

    @torch.no_grad()
    def embed(self, p, x):
        """CLIP embeddings (n, d) of person p's feature rows x (n, F); logits are ``z @ anchors.T``."""
        f = x.shape[1]
        x = torch.as_tensor((np.asarray(x, np.float32) - self.mean[p][:f]) / self.std[p][:f])
        return (x @ self.weight[p][:, :f].T + self.bias[p]).numpy()


def train(persons, anchor, *, weight_decay=1e-2, supcon_weight=0., temperature=.1, steps=300, lr=1e-2,
          batch=512, seed=0):
    """Fit one linear CLIP encoder per person (in parallel).  ``anchor``: (K, d) unit item anchors.

    Without supervised contrast the objective is convex and L-BFGS runs up to ``steps`` iterations on
    all trials; with it, Adam runs ``steps`` steps on up to ``batch`` trials per person.
    """
    generator = torch.Generator().manual_seed(seed)
    P, F_max = len(persons), max(p.x.shape[1] for p in persons)
    N_max = max(len(p.x) for p in persons)
    X = torch.zeros(P, N_max, F_max); Y = torch.zeros(P, N_max, dtype=torch.long)
    Wt = torch.zeros(P, N_max); V = torch.zeros(P, N_max, dtype=torch.bool)
    mean, std = np.zeros((P, F_max), np.float32), np.ones((P, F_max), np.float32)
    for i, p in enumerate(persons):
        n, f = p.x.shape
        mean[i, :f], std[i, :f] = p.x.mean(0), p.x.std(0) + 1e-3
        X[i, :n, :f] = torch.from_numpy((p.x - mean[i, :f]) / std[i, :f])
        Y[i, :n], Wt[i, :n], V[i, :n] = torch.from_numpy(p.item), torch.from_numpy(p.weight.astype(np.float32)), True
    A = torch.as_tensor(anchor)
    K, d = A.shape
    W = (torch.randn(P, d, F_max, generator=generator) / np.sqrt(F_max)).requires_grad_()
    b = torch.zeros(P, d, requires_grad=True)

    def objective(x, y, w, v):
        z = torch.einsum('pnf,pdf->pnd', x, W) + b[:, None]
        logits = z @ A.T                                                                  # P, N, K
        w = w * v
        e2a = (F.cross_entropy(logits.transpose(1, 2), y, reduction='none') * w).sum(1) / w.sum(1).clamp_min(1e-6)
        # speech -> EEG: for each item present, which trials of the batch are its own
        to_trials = logits.transpose(1, 2).masked_fill(~v[:, None, :], -1e4).log_softmax(-1)          # P, K, N
        own = F.one_hot(y, K).transpose(1, 2).bool() & v[:, None, :]
        present = own.any(-1)
        per_item = -to_trials.masked_fill(~own, 0).sum(-1) / own.sum(-1).clamp_min(1)
        a2e = (per_item * present).sum(1) / present.sum(1).clamp_min(1)
        loss = (e2a + .5 * a2e + weight_decay * W.square().sum((1, 2))).sum()
        if supcon_weight:
            loss = loss + supcon_weight * sum(supcon(F.normalize(z[i][v[i]], dim=-1), y[i][v[i]], temperature)
                                              for i in range(P))
        return loss

    if not supcon_weight:
        optimizer = torch.optim.LBFGS([W, b], max_iter=steps, history_size=20, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            loss = objective(X, Y, Wt, V)
            loss.backward()
            return loss
        optimizer.step(closure)
    else:
        optimizer = torch.optim.Adam([W, b], lr=lr)
        for step in range(steps):
            for group in optimizer.param_groups:
                group['lr'] = lr * .5 * (1 + np.cos(np.pi * step / steps))
            if N_max > batch:                       # random subset of each person's trials
                idx = torch.stack([torch.randperm(N_max, generator=generator)[:batch] for _ in range(P)])
                tensors = [torch.gather(t, 1, idx if t.dim() == 2 else idx[..., None].expand(-1, -1, F_max))
                           for t in (X, Y, Wt, V)]
            else:
                tensors = [X, Y, Wt, V]
            optimizer.zero_grad()
            objective(*tensors).backward()
            optimizer.step()
    return PersonalClip(mean, std, W.detach(), b.detach())


def posterior(z, anchor, scale):
    """p(k | z) = softmax(scale * z . a_k); ``scale`` inf gives the one-hot decision."""
    logits = torch.as_tensor(z) @ torch.as_tensor(anchor).T
    if np.isinf(scale):
        return F.one_hot(logits.argmax(-1), logits.shape[-1]).float().numpy()
    return torch.softmax(logits * scale, -1).numpy()


def calibrate(z, items, anchor):
    """Temperature minimising the log-loss of cross-fitted embeddings (one per dataset)."""
    logits = torch.as_tensor(z) @ torch.as_tensor(anchor).T
    y = torch.as_tensor(items)
    grid = np.exp(np.linspace(np.log(1e-3), np.log(100), 300))
    losses = [float(F.cross_entropy(logits * s, y)) for s in grid]
    return float(grid[int(np.argmin(losses))]), float(min(losses))

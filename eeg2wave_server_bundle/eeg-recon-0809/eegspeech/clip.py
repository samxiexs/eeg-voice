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
logistic regression within a person; the usual cosine form trained with Adam lost 3-4 points.  The
posterior p(k | trial) = softmax(s z . a_k) has its temperature s calibrated on cross-fitted trials.
All persons of a dataset train in parallel as one batch of independent models.

Open vocabulary (``train_sentences``): a sentence's anchor comes from its speech, so a person's map
ranks sentences never seen in training; trained with symmetric InfoNCE between the trials of a run and
the sentences of that run, or by ridge regression onto the anchors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


def supcon(z, labels, temperature=.1):
    """Supervised contrastive loss (Khosla et al. 2020) on unit embeddings z (B, D): trials of the same
    item (whatever their modality) are positives."""
    logits = z @ z.T / temperature
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positive = (labels[:, None] == labels[None, :]) & ~eye
    logits = logits.masked_fill(eye, float('-inf'))
    log_prob = logits - torch.logsumexp(logits, 1, keepdim=True)
    count = positive.sum(1)
    keep = count > 0
    if not keep.any():
        return z.sum() * 0.
    return (-(log_prob.masked_fill(~positive, 0).sum(1)[keep] / count[keep])).mean()


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


# ----------------------------------------------------------------------------- open vocabulary (sentences)
class SentenceEncoder:
    """One person's linear map from trial features to the sentence CLIP space; a candidate's score is z . a_s."""

    def __init__(self, mean, std, weight):
        self.mean, self.std, self.weight = mean, std, weight

    def embed(self, x):
        return ((np.asarray(x, np.float32) - self.mean) / self.std) @ self.weight


def run_sets(sentence, run):
    """Per run: (trial indices, distinct sentences = the run's candidates, each trial's candidate position)."""
    out = []
    for r in np.unique(run):
        trials = np.flatnonzero(run == r)
        candidates, position = np.unique(sentence[trials], return_inverse=True)
        out.append((trials, candidates, position))
    return out


class Reducer:
    """Standardisation of trial features by the training trials and, above ``components`` features, the
    projection on their top principal directions (unsupervised, randomised SVD)."""

    def __init__(self, mean, std, basis=None):
        self.mean, self.std, self.basis = mean, std, basis

    @classmethod
    def fit(cls, x, components=None, seed=0):
        x = np.asarray(x, np.float32)
        mean, std = x.mean(0), x.std(0) + 1e-3
        basis = None
        if components and x.shape[1] > components:
            torch.manual_seed(seed)
            _, _, v = torch.svd_lowrank(torch.from_numpy((x - mean) / std), q=components, niter=3)
            basis = v.numpy()
        return cls(mean, std, basis)

    def __call__(self, x):
        z = (np.asarray(x, np.float32) - self.mean) / self.std
        return z if self.basis is None else z @ self.basis


def train_sentences(x, sentence, run, anchor, *, method='clip', weight_decay=.1, steps=200, standardize=True):
    """Fit one person's sentence encoder.  ``x`` (n, F) trial features, ``sentence`` (n,) the row of
    ``anchor`` (K, d) each trial imagined, ``run`` (n,) its recording run.

    ridge  closed-form regression onto the trial's anchor: mean ||z - a||^2 + wd ||W||^2
    clip   symmetric InfoNCE between the trials of a run and the sentences imagined in that run (at test
           time the candidates are also a run's sentences), started from the ridge solution and solved
           by L-BFGS: the objective is convex (a linear map inside log-sum-exp).  No bias: a sentence's
           score depends on the trial's EEG only.
    """
    x = np.asarray(x, np.float64)
    if standardize:
        mean, std = x.mean(0), x.std(0) + 1e-3
    else:                                                   # already standardised or projected (``Reducer``)
        mean, std = np.zeros(x.shape[1]), np.ones(x.shape[1])
    X = (x - mean) / std
    n, f = X.shape
    A = np.asarray(anchor, np.float64)
    W = np.linalg.solve(X.T @ X / n + weight_decay * np.eye(f), X.T @ A[sentence] / n)
    if method == 'clip':
        sets = run_sets(np.asarray(sentence), np.asarray(run))
        R, n_max, m_max = len(sets), max(len(t) for t, _, _ in sets), max(len(c) for _, c, _ in sets)
        T = torch.zeros(R, n_max, dtype=torch.long); C = torch.zeros(R, m_max, dtype=torch.long)
        valid_t = torch.zeros(R, n_max, dtype=torch.bool); valid_c = torch.zeros(R, m_max, dtype=torch.bool)
        target = torch.zeros(R, n_max, dtype=torch.long)
        for i, (trials, candidates, position) in enumerate(sets):
            T[i, :len(trials)], valid_t[i, :len(trials)] = torch.from_numpy(trials), True
            C[i, :len(candidates)], valid_c[i, :len(candidates)] = torch.from_numpy(candidates), True
            target[i, :len(trials)] = torch.from_numpy(position)
        positive = F.one_hot(target, m_max).bool() & valid_t[..., None] & valid_c[:, None, :]       # R, n, m
        Xt, At = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(A, dtype=torch.float32)
        Wt = torch.as_tensor(W, dtype=torch.float32).requires_grad_()
        pair = valid_t[..., None] & valid_c[:, None, :]

        def objective():
            logits = torch.einsum('rnd,rmd->rnm', (Xt @ Wt)[T], At[C]).masked_fill(~pair, -1e4)
            to_sentences = logits.log_softmax(-1)                       # EEG -> which sentence of the run
            e2a = -(to_sentences * positive).sum(-1)[valid_t].mean()
            to_trials = logits.log_softmax(1)                           # sentence -> which trial of the run
            own = positive.sum(1)
            a2e = (-(to_trials * positive).sum(1) / own.clamp_min(1))[own > 0].mean()
            return .5 * (e2a + a2e) + weight_decay * Wt.square().sum()

        optimizer = torch.optim.LBFGS([Wt], max_iter=steps, history_size=20, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()
            loss = objective()
            loss.backward()
            return loss
        optimizer.step(closure)
        W = Wt.detach().double().numpy()
    return SentenceEncoder(mean.astype(np.float32), std.astype(np.float32), W.astype(np.float32))


def rank_percentile(scores, position):
    """(n,) percentile of each row's true candidate among the row's candidates: 1 best, 0 worst, 0.5 chance
    (ties count half).  ``scores`` (n, C), ``position`` (n,) the true column."""
    scores = np.asarray(scores)
    true = scores[np.arange(len(scores)), position][:, None]
    above = (scores > true).sum(1) + .5 * ((scores == true).sum(1) - 1)
    return 1 - above / max(scores.shape[1] - 1, 1)


def calibrate_sets(score_sets, position_sets):
    """Temperature minimising the log-loss of the true candidate over lists of (n_r, C_r) score matrices."""
    grid = np.exp(np.linspace(np.log(1e-3), np.log(1e3), 241))
    loss = []
    for s in grid:
        total = sum(-torch.log_softmax(torch.as_tensor(S * s), -1)[np.arange(len(p)), p].sum().item()
                    for S, p in zip(score_sets, position_sets))
        loss.append(total / sum(len(p) for p in position_sets))
    best = int(np.argmin(loss))
    return float(grid[best]), float(loss[best])

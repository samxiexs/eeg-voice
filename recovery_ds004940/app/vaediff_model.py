"""VAE + latent-conditioned mel diffusion for EEG -> speech (``vaediff``).

Two posteriors share one latent space (``latent`` channels at 25 Hz, 40 ms frames):

* audio ``q_a(z | WavLM layers of the presented sentence)``: trained in stage 1 together with the decoder, then frozen;
* EEG ``q_e(z | EEG, subject)``: stage 2.

The decoder ``p(mel | z)`` is the v-prediction diffusion model of ``generative_recovery.py`` (its adaLN-zero
``Block``), over the native Vocos log-mel, with ``z`` as the only conditioning; a learned null stream gives
classifier-free guidance.  At test time the EEG is the only input: no text, audio or duration.

Why a VAE: different people hear the same sentence with different EEG, so the EEG code cannot be one point per
sentence.  ``q_e`` is a distribution whose KL is taken against the audio posterior widened by a "person spread"
``s`` (``N(mu_a, sigma_a^2 + s^2)``): each person's code may sit anywhere in that neighbourhood of the sentence
code at little cost, instead of being forced onto it.

The stage-2 objective is ``w1 * L_voice + w2 * L_text + w3 * L_diff + beta * KL``:

* ``L_voice`` (timbre / content): chunk InfoNCE between EEG and audio codes (0.4 s chunks, same position, other
  sentences as negatives, divided by log #candidates so chance = 1) plus the timbre distance of the decoder's
  one-step estimate (long-term mel spectrum, mean removed, in dB, divided by its threshold);
* ``L_text``: InfoNCE between a text head on the EEG code and the frozen MiniLM sentence embedding (chance = 1);
* ``L_diff``: the diffusion loss over EEG-, audio- and null-conditioned samples;
* ``KL``: the VAE term above (fixed ``beta``, free bits).

``WeightController`` adjusts ``w1`` and ``w2`` every ``every`` updates from dev measures with hard thresholds,
and a ``control.json`` file in the run folder overrides it (the hook for a person, or Claude, to set the weights).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from aligned_recovery_model import TemporalResidual
from eeg2speech.aligned import sample_at
from generative_recovery import Block, sinusoidal

LATENT_RATE = 25.
DB = 20. / math.log(10.)                 # natural-log magnitude -> dB


def latent_times(seconds: float) -> torch.Tensor:
    return (torch.arange(int(round(seconds * LATENT_RATE))).float() + .5) / LATENT_RATE


def reparameterize(mu, logvar):
    return mu + torch.randn_like(mu) * (.5 * logvar).exp()


def masked_mean(value, mask):
    weight = mask.to(value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(1)
    weight = weight.expand_as(value)
    return (value * weight).sum() / weight.sum().clamp_min(1.)


# ----------------------------------------------------------------------------- posteriors
class AudioPosterior(nn.Module):
    """WavLM layers (learned mixture) at 50 Hz -> small conv stack -> (mu, logvar) at the latent frame times."""
    def __init__(self, layers=4, width=768, hidden=256, latent=32):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(layers))
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(nn.Conv1d(width, hidden, 3, padding=1), nn.GELU(),
                                 nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(), nn.Conv1d(hidden, 2 * latent, 1))

    def forward(self, wavlm, wavlm_times, query_times):
        """wavlm (B, layers, T, 768) -> mu, logvar (B, latent, T_latent)."""
        weights = torch.softmax(self.layer_logits, 0)
        h = self.norm((weights[None, :, None, None] * wavlm.float()).sum(1))            # B, T, 768
        out = self.net(h.transpose(1, 2)).transpose(1, 2)                                # B, T, 2d (50 Hz)
        out = sample_at(out, wavlm_times, query_times).transpose(1, 2)                  # B, 2d, T_latent
        mu, logvar = out.chunk(2, 1)
        return mu, logvar.clamp(-8., 4.)


class TextHead(nn.Module):
    """Attention pooling over speech frames of a latent sequence -> unit vector in the frozen text space."""
    def __init__(self, latent, text_dim=384, hidden=256):
        super().__init__()
        self.score = nn.Conv1d(latent, 1, 1)
        self.mlp = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, text_dim))

    def forward(self, z, mask):
        score = self.score(z).squeeze(1).masked_fill(~mask, -1e4)
        pooled = (z * torch.softmax(score, -1)[:, None]).sum(-1)
        return F.normalize(self.mlp(pooled), dim=-1)


class EEGPosterior(nn.Module):
    """EEG (B, 128, samples) -> (mu, logvar) at the latent frame times, plus a text prediction.

    The September encoder trunk (per-subject signed channel mixing, shared spatial filters, stride-2 temporal
    conv, six dilated residual blocks, time-since-onset code) with two changes: a per-subject FiLM after the
    trunk, and a Gaussian head.  Subject index -1 (a person never seen in training) takes the shared path.
    The field of view is about +-2 s, so a fixed playback / neural delay is absorbed without an explicit lag.
    """
    def __init__(self, channels=128, width=128, latent=32, subjects=0, rate=128, start=-.25, samples=493,
                 blocks=6, dropout=.1, text_dim=384, positional=True):
        super().__init__()
        self.channels, self.subjects = channels, int(subjects)
        if self.subjects:
            self.subject_delta = nn.Parameter(torch.zeros(self.subjects, channels, channels))
            self.film = nn.Parameter(torch.zeros(self.subjects, 2, width))
        self.spatial = nn.Conv1d(channels, width, 1, bias=False)
        self.temporal = nn.Conv1d(width, width, 9, stride=2, padding=4)
        self.blocks = nn.Sequential(*[TemporalResidual(width, 2 ** i, dropout) for i in range(blocks)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Conv1d(width, 2 * latent, 1)
        nn.init.normal_(self.head.weight, std=.01); nn.init.zeros_(self.head.bias)
        tokens = (samples + 2 * 4 - 9) // 2 + 1
        self.register_buffer('token_times', start + torch.arange(tokens).float() * 2 / rate)
        self.positional = bool(positional)
        if self.positional:
            self.position = nn.Parameter(torch.zeros(1, width, tokens))
        self.text = TextHead(latent, text_dim)

    def _known(self, subject):
        return subject.clamp_min(0), (subject >= 0).float()

    def forward(self, eeg, channel_mask, subject, query_times):
        x = eeg * channel_mask[:, :, None]
        if self.subjects:
            index, known = self._known(subject)
            delta = self.subject_delta[index] * known[:, None, None]
            x = x + torch.bmm(delta, x)
        h = F.gelu(self.temporal(self.spatial(x)))
        if self.positional:
            h = h + self.position
        h = self.norm(self.blocks(h).transpose(1, 2))                                  # B, T, W
        if self.subjects:
            film = self.film[index] * known[:, None, None]
            h = h * (1 + film[:, 0][:, None]) + film[:, 1][:, None]
        out = sample_at(self.head(h.transpose(1, 2)).transpose(1, 2), self.token_times, query_times).transpose(1, 2)
        mu, logvar = out.chunk(2, 1)
        return mu, logvar.clamp(-8., 4.)


# ----------------------------------------------------------------------------- decoder
class LatentMelDiffusion(nn.Module):
    """v-prediction diffusion over the standardised (bins, frames) Vocos log-mel, cosine schedule.

    The conditioning stream is the latent ``z`` interpolated to the mel frame times; a learned null stream
    replaces it for classifier-free guidance.  Same blocks, schedule and DDIM sampler as generative_recovery.py.
    """
    def __init__(self, latent=32, bins=100, frames=338, hidden=256, blocks=6, heads=4, dropout=.1, timesteps=1000):
        super().__init__()
        self.bins, self.frames, self.hidden, self.timesteps = bins, frames, hidden, timesteps
        self.input = nn.Conv1d(bins, hidden, 3, padding=1)
        self.cond_input = nn.Conv1d(latent, hidden, 3, padding=1)
        self.null = nn.Parameter(torch.zeros(1, frames, hidden))
        self.position = nn.Parameter(torch.randn(1, frames, hidden) * .02)
        self.time = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden))
        dilations = [1, 2, 4, 8]
        self.blocks = nn.ModuleList([Block(hidden, hidden, dilations[i % 4], heads, i % 2 == 1, dropout) for i in range(blocks)])
        self.output_norm = nn.LayerNorm(hidden)
        self.output = nn.Conv1d(hidden, bins, 3, padding=1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        steps = torch.arange(timesteps + 1).double() / timesteps
        alpha_bar = torch.cos((steps + .008) / 1.008 * math.pi / 2) ** 2
        self.register_buffer('alpha_bar', (alpha_bar / alpha_bar[0]).clamp(1e-5, 1.).float())

    def condition(self, z, latent_times, mel_times):
        """z (B, latent, T_latent) -> conditioning stream (B, frames, hidden)."""
        zi = sample_at(z.transpose(1, 2), latent_times, mel_times)                     # B, F, latent
        return self.cond_input(zi.transpose(1, 2)).transpose(1, 2)

    def null_stream(self, batch):
        return self.null.expand(batch, -1, -1)

    def forward(self, x, t, c):
        h = self.input(x).transpose(1, 2) + self.position
        te = self.time(sinusoidal(t, self.hidden))
        for block in self.blocks:
            h = block(h, c, te)
        return self.output(self.output_norm(h).transpose(1, 2))

    def noisy(self, x0, t, noise):
        ab = self.alpha_bar[t][:, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise, ab.sqrt() * noise - (1 - ab).sqrt() * x0, ab

    def loss(self, x0, c, t=None, noise=None):
        """Returns (per-sample v-MSE (B,), one-step x0 estimate, alpha_bar (B,))."""
        t = torch.randint(1, self.timesteps + 1, (len(x0),), device=x0.device) if t is None else t
        noise = torch.randn_like(x0) if noise is None else noise
        xt, v, ab = self.noisy(x0, t, noise)
        v_hat = self(xt, t, c)
        x0_hat = ab.sqrt() * xt - (1 - ab).sqrt() * v_hat
        return (v_hat - v).square().mean((1, 2)), x0_hat, ab.flatten()

    @torch.no_grad()
    def sample(self, c, null, *, steps=50, guidance=2., noise=None, clamp=(-4., 4.)):
        """DDIM (eta = 0) with classifier-free guidance on v."""
        batch = c.shape[0]
        x = torch.randn(batch, self.bins, self.frames, device=c.device) if noise is None else noise.to(c.device)
        schedule = torch.linspace(self.timesteps, 0, steps + 1).round().long().tolist()
        for i in range(steps):
            t, t_next = schedule[i], schedule[i + 1]
            tt = torch.full((batch,), t, device=c.device)
            if guidance == 1.:
                v = self(x, tt, c)
            else:
                v_c, v_u = self(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([c, null])).chunk(2)
                v = v_u + guidance * (v_c - v_u)
            ab, ab_next = self.alpha_bar[t], self.alpha_bar[t_next]
            x0 = (ab.sqrt() * x - (1 - ab).sqrt() * v).clamp(*clamp)
            eps = (1 - ab).sqrt() * x + ab.sqrt() * v
            x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
        return x


class MelScaler(nn.Module):
    """Per-bin standardisation of the floored Vocos log-mel (fitted on training sentences)."""
    def __init__(self, bins=100, floor=-11.5):
        super().__init__()
        self.floor = float(floor)
        self.register_buffer('mean', torch.zeros(bins)); self.register_buffer('scale', torch.ones(bins))

    def fit(self, mel):
        x = mel.clamp_min(self.floor)
        self.mean.copy_(x.mean((0, 2))); self.scale.copy_(x.std((0, 2)).clamp_min(1e-3))
        return self

    def encode(self, mel):
        return (mel.clamp_min(self.floor) - self.mean[:, None]) / self.scale[:, None]

    def decode(self, x):
        return x * self.scale[:, None] + self.mean[:, None]


# ----------------------------------------------------------------------------- losses and measures
def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p, mask, free_bits=0.):
    """KL(N_q || N_p) in nats per latent dimension, averaged over masked frames; (B, d, T) inputs, mask (B, T)."""
    kl = .5 * (logvar_p - logvar_q + (logvar_q.exp() + (mu_q - mu_p).square()) / logvar_p.exp() - 1)
    per_frame = kl.mean(1)
    if free_bits > 0:
        per_frame = per_frame.clamp_min(free_bits)
    return masked_mean(per_frame, mask)


def widened(logvar, spread):
    """log(sigma^2 + s^2): the audio posterior widened by the allowed person-to-person spread."""
    return torch.log(logvar.exp() + spread ** 2) if spread > 0 else logvar


def chunk_pool(z, mask, size):
    """(B, d, T), (B, T) -> chunk means (B, d, K) and chunk speech mask (B, K): a chunk is speech if most frames are."""
    k = z.shape[-1] // size
    pooled = z[..., :k * size].reshape(*z.shape[:2], k, size).mean(-1)
    speech = mask[:, :k * size].reshape(len(mask), k, size).float().mean(-1) > .5
    return pooled, speech


def chunk_infonce(a, b, mask, exclude=None, size=10, temperature=.1):
    """Symmetric InfoNCE between chunks of ``a`` (EEG) and ``b`` (audio) at the same position.

    Negatives are the other trials' chunks at that position that are speech; ``exclude`` (B, B) removes pairs
    that are not real negatives (the same sentence frame, NPC vs NPI).  Returns (loss / log(mean candidates),
    accuracy): chance is 1 for the loss.
    """
    pa, speech = chunk_pool(a, mask, size)
    pb, _ = chunk_pool(b, mask, size)
    pa, pb = F.normalize(pa, dim=1), F.normalize(pb, dim=1)
    logits = torch.einsum('idk,jdk->kij', pa, pb) / temperature                        # K, B, B
    B = a.shape[0]
    allowed = speech.T[:, None, :] & speech.T[:, :, None]                               # K, B(i), B(j)
    if exclude is not None:
        eye = torch.eye(B, dtype=torch.bool, device=a.device)
        allowed = allowed & ~(exclude & ~eye)[None]
    anchors = allowed.diagonal(dim1=1, dim2=2) & (allowed.sum(-1) >= 2)                 # K, B
    if not bool(anchors.any()):
        return a.sum() * 0, 0.
    labels = torch.arange(B, device=a.device)[None].expand_as(anchors)
    forward = logits.masked_fill(~allowed, -1e4)
    backward = logits.transpose(1, 2).masked_fill(~allowed.transpose(1, 2), -1e4)
    loss = .5 * (F.cross_entropy(forward[anchors], labels[anchors]) + F.cross_entropy(backward[anchors], labels[anchors]))
    chance = torch.log(allowed.sum(-1)[anchors].float()).mean().clamp_min(math.log(2.))
    accuracy = float((forward[anchors].argmax(-1) == labels[anchors]).float().mean())
    return loss / chance, accuracy


def text_infonce(pred, text, temperature=.07):
    """Symmetric InfoNCE between predicted and true sentence embeddings of distinct sentences; chance = 1."""
    logits = pred @ text.T / temperature
    labels = torch.arange(len(pred), device=pred.device)
    loss = .5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    return loss / math.log(max(len(pred), 2)), float((logits.argmax(1) == labels).float().mean())


def long_term_spectrum(logmel, mask):
    """Mean log-mel over speech frames in dB, overall level removed: (B, bins, F), (B, F) -> (B, bins)."""
    weight = mask[:, None].to(logmel.dtype)
    ltas = DB * (logmel * weight).sum(-1) / weight.sum(-1).clamp_min(1.)
    return ltas - ltas.mean(1, keepdim=True)


def timbre_distance(logmel_a, logmel_b, mask):
    """RMS difference (dB) of the level-free long-term spectra: a timbre / spectral-envelope distance, (B,)."""
    return (long_term_spectrum(logmel_a, mask) - long_term_spectrum(logmel_b, mask)).square().mean(1).add(1e-8).sqrt()


def rank_percentile(scores, own):
    """Tie-averaged rank percentile of ``own`` within each row of ``scores`` (higher score = better): 0 best, .5 chance."""
    better = (scores > own[:, None]).sum(1).float()
    ties = (scores == own[:, None]).sum(1).float() - 1
    n = scores.shape[1]
    return (better + .5 * ties) / max(n - 1, 1)


def product_of_experts(mu, logvar):
    """Pool trials of one sentence: precision-weighted mean, mean precision (single-trial noise level kept).

    The decoder only ever saw single-trial posteriors; keeping the mean precision avoids feeding it a code that is
    far more certain than anything in training, while the mean is still the precision-weighted consensus.
    """
    precision = (-logvar).exp()
    mu_pool = (mu * precision).sum(0) / precision.sum(0)
    return mu_pool, -torch.log(precision.mean(0))


# ----------------------------------------------------------------------------- weight controller
class WeightController:
    """Feedback on the loss weights from dev measures with hard thresholds.

    Each controlled term has constraints ``measure <= threshold``; its ratio is the largest ``measure / threshold``.
    Every ``every`` updates: ``w <- w * exp(eta * clip(ratio - 1, -.5, .5))``, bounded to ``[w0 / bound, w0 * bound]``
    (a violated constraint raises its weight, a comfortably met one lowers it).  ``w_diff`` is the anchor and only
    changes through the control file.

    ``control.json`` in the run folder is read whenever it changes: ``{"weights": {...}, "thresholds": {...},
    "mode": "auto" | "hold", "eta": .5, "note": "..."}``.  Weights given there become the new base; ``hold`` stops the
    automatic rule.  This is the hook for a person, or Claude, to guess the next setting from ``tune.jsonl``.
    """
    TERMS = ('voice', 'text', 'diff')

    def __init__(self, weights, constraints, *, eta=.5, bound=4., mode='auto', control: Path | None = None):
        self.weights = {k: float(weights[k]) for k in self.TERMS}
        self.base = dict(self.weights)
        self.constraints = {term: dict(c) for term, c in constraints.items()}
        self.eta, self.bound, self.mode, self.control = float(eta), float(bound), mode, control
        self.control_mtime = None

    def ratio(self, term, measures):
        values = [measures[m] / t for m, t in self.constraints.get(term, {}).items() if measures.get(m) is not None and t > 0]
        return max(values) if values else None

    def read_control(self):
        if self.control is None or not self.control.exists():
            return None
        mtime = self.control.stat().st_mtime
        if mtime == self.control_mtime:
            return None
        self.control_mtime = mtime
        try:
            value = json.loads(self.control.read_text())
        except json.JSONDecodeError as exc:
            return dict(error=f'control file unreadable: {exc}')
        for term, w in value.get('weights', {}).items():
            if term in self.weights:
                self.weights[term] = float(w); self.base[term] = float(w)
        for term, items in value.get('thresholds', {}).items():
            self.constraints.setdefault(term, {}).update({k: float(v) for k, v in items.items()})
        self.mode = value.get('mode', self.mode)
        self.eta = float(value.get('eta', self.eta))
        return dict(override=value)

    def step(self, update, measures):
        override = self.read_control()
        ratios = {term: self.ratio(term, measures) for term in self.TERMS}
        before = dict(self.weights)
        if self.mode == 'auto':
            for term, r in ratios.items():
                if r is None or term == 'diff':
                    continue
                w = self.weights[term] * math.exp(self.eta * max(-.5, min(.5, r - 1)))
                self.weights[term] = min(self.base[term] * self.bound, max(self.base[term] / self.bound, w))
        violated = [term for term, r in ratios.items() if r is not None and r > 1]
        return dict(update=update, mode=self.mode, measures=measures, ratios=ratios, violated=violated,
                    weights_before=before, weights=dict(self.weights), **(override or {}))

    def state(self):
        return dict(weights=self.weights, base=self.base, constraints=self.constraints, eta=self.eta, bound=self.bound, mode=self.mode)

    def load(self, state):
        self.weights, self.base = dict(state['weights']), dict(state['base'])
        self.constraints = {k: dict(v) for k, v in state['constraints'].items()}
        self.eta, self.bound, self.mode = state['eta'], state['bound'], state['mode']

"""Capacity-controlled TCN, physiology priors and group-safe temporal alignment."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class TemporalBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        for conv, norm in zip(self.convs, self.norms):
            x = self.dropout(F.gelu(norm(conv(x).transpose(1, 2)))).transpose(1, 2)
        return x + residual


class Adapter(nn.Module):
    def __init__(self, width=128, rank=16):
        super().__init__()
        self.down = nn.Linear(width, rank)
        self.up = nn.Linear(rank, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def residual(self, x): return self.up(F.gelu(self.down(x)))
    def forward(self, x): return x + self.residual(x)


class EEGEncoder(nn.Module):
    def __init__(self, cfg, groups, physiological=True):
        super().__init__()
        self.physiological = physiological
        e = cfg["encoder"]
        self.width, self.rate, self.left = e["width"], cfg["preprocessing"]["eeg_rate"], cfg["window"]["eeg_left"]
        self.register_buffer("groups", torch.tensor(groups, dtype=torch.long))
        self.register_buffer("lags", torch.tensor(e["lags_seconds"] if physiological else [0.0]))
        self.lag_logits = nn.Parameter(torch.zeros(len(self.lags)), requires_grad=physiological)
        if physiological:
            self.spatial = nn.ModuleList([nn.Linear(sum(int(g == i) for g in groups), e["roi_width"]) for i in range(5)])
            self.merge = nn.Linear(5 * e["roi_width"], e["width"])
        else:
            self.spatial = nn.Linear(len(groups), e["width"])
        self.blocks = nn.ModuleList([TemporalBlock(e["width"], d, e["dropout"]) for d in e["dilations"]])
        self.content = nn.Linear(e["width"], cfg["audio"]["teacher_dim"])
        self.acoustics = nn.Linear(e["width"], 2)

    def hidden(self, eeg, times, channel_mask=None):
        if channel_mask is not None: eeg = eeg * channel_mask[:, None].to(eeg.dtype)
        if self.physiological:
            x = self.merge(torch.cat([layer(eeg[..., self.groups == i]) for i, layer in enumerate(self.spatial)], dim=-1))
        else: x = self.spatial(eeg)
        x = x.transpose(1, 2)
        for block in self.blocks: x = block(x)
        x = x.transpose(1, 2)
        if times.ndim == 1: times = times.unsqueeze(0).expand(eeg.shape[0], -1)
        indices = (times[:, :, None] + self.left + self.lags[None, None]) * self.rate
        if indices.min() < 0 or indices.max() > x.shape[1] - 1:
            raise ValueError("Insufficient EEG context for physiology lag pooling")
        left = indices.floor().long()
        right = (left + 1).clamp(max=x.shape[1] - 1)
        b = torch.arange(len(x), device=x.device)[:, None, None]
        fraction = (indices - left).unsqueeze(-1)
        sampled = x[b, left] * (1 - fraction) + x[b, right] * fraction
        return (sampled * self.lag_logits.softmax(0)[None, None, :, None]).sum(2)

    def forward(self, eeg, times, channel_mask=None, adapter=None):
        h = self.hidden(eeg, times, channel_mask)
        if adapter is not None: h = adapter(h)
        return {"teacher": self.content(h), "phi": self.acoustics(h), "hidden": h}


def temporal_contrastive(predicted, target, audio_ids, starts, times, temperature=.07):
    """Symmetric multi-positive InfoNCE; duplicate/overlapping stimuli are safe."""
    if times.ndim == 1: times = times[None].expand(len(predicted), -1)
    pred_parts, true_parts, clip_ids, absolute, window_starts = [], [], [], [], []
    mapping = {s: i for i, s in enumerate(sorted(set(audio_ids)))}
    for b in range(len(predicted)):
        for second in range(4):
            valid = (times[b] >= second) & (times[b] < second + 1)
            if not valid.any(): continue
            pred_parts.append(predicted[b, valid].mean(0))
            true_parts.append(target[b, valid].mean(0))
            clip_ids.append(mapping[audio_ids[b]])
            absolute.append(float(starts[b]) + second)
            window_starts.append(float(starts[b]))
    if not pred_parts: return predicted.sum() * 0
    pred = F.normalize(torch.stack(pred_parts), dim=-1)
    truth = F.normalize(torch.stack(true_parts), dim=-1)
    logits = pred @ truth.T / temperature
    ids = torch.tensor(clip_ids, device=logits.device)
    position = torch.tensor(absolute, device=logits.device)
    start = torch.tensor(window_starts, device=logits.device)
    same_clip = ids[:, None] == ids[None]
    positive = same_clip & ((position[:, None] - position[None]).abs() < 1e-3)
    overlapping = same_clip & ((start[:, None] - start[None]).abs() < 4)
    allowed = ~overlapping | positive
    def loss(scores, pos, permitted):
        numerator = torch.logsumexp(scores.masked_fill(~pos, -torch.inf), dim=1)
        denominator = torch.logsumexp(scores.masked_fill(~permitted, -torch.inf), dim=1)
        return (denominator - numerator).mean()
    return .5 * (loss(logits, positive, allowed) + loss(logits.T, positive.T, allowed.T))


def encoder_loss(output, batch, cfg, physiological=True):
    reg = F.mse_loss(output["teacher"], batch["teacher"])
    ctr = temporal_contrastive(output["teacher"], batch["teacher"], batch["audio"], batch["start"], batch["times"], cfg["encoder"]["contrast_temperature"])
    env = F.mse_loss(output["phi"][..., 0], batch["phi"][..., 0])
    onset = F.mse_loss(output["phi"][..., 1], batch["phi"][..., 1])
    loss = reg + cfg["encoder"]["contrast_weight"] * ctr
    if physiological: loss = loss + cfg["encoder"]["envelope_weight"] * env + cfg["encoder"]["onset_weight"] * onset
    return loss, {"reg": reg.detach(), "ctr": ctr.detach(), "env": env.detach(), "onset": onset.detach()}

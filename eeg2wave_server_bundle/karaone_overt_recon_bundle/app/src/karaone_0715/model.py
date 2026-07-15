from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, strength: float) -> torch.Tensor:  # type: ignore[override]
        ctx.strength = float(strength)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        return -ctx.strength * gradient, None


def grad_reverse(value: torch.Tensor, strength: float) -> torch.Tensor:
    return _GradientReverse.apply(value, float(strength))


@dataclass(frozen=True)
class AudioCodeModelConfig:
    codebooks: int = 8
    code_steps: int = 150
    vocab_size: int = 1024
    num_labels: int = 11
    d_model: int = 192
    condition_steps: int = 50
    encoder_layers: int = 3
    decoder_layers: int = 4
    heads: int = 6
    dropout: float = 0.10


@dataclass(frozen=True)
class EEGModelConfig:
    channels: int = 62
    eeg_len: int = 768
    d_model: int = 192
    condition_steps: int = 50
    code_steps: int = 150
    num_labels: int = 11
    num_train_subjects: int = 12
    transformer_layers: int = 3
    heads: int = 6
    dropout: float = 0.15
    temporal_kernels: tuple[int, ...] = (15, 31, 63)
    stem_stride: int = 4


def _transformer_encoder(d_model: int, heads: int, layers: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=heads,
        dim_feedforward=d_model * 4,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers)


class AudioConditionEncoder(nn.Module):
    """Compress exact EnCodec codes into the condition space aligned with EEG."""

    def __init__(self, cfg: AudioCodeModelConfig):
        super().__init__()
        self.cfg = cfg
        self.code_embeddings = nn.ModuleList([nn.Embedding(cfg.vocab_size, cfg.d_model) for _ in range(cfg.codebooks)])
        self.codebook_embedding = nn.Parameter(torch.zeros(1, cfg.codebooks, 1, cfg.d_model))
        self.position = nn.Parameter(torch.zeros(1, cfg.code_steps, cfg.d_model))
        self.encoder = _transformer_encoder(cfg.d_model, cfg.heads, cfg.encoder_layers, cfg.dropout)
        self.condition_refiner = _transformer_encoder(cfg.d_model, cfg.heads, 2, cfg.dropout)
        self.condition_norm = nn.LayerNorm(cfg.d_model)
        self.label_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.num_labels))
        nn.init.normal_(self.codebook_embedding, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, codes: torch.Tensor) -> dict[str, torch.Tensor]:
        if codes.ndim != 3 or codes.shape[1:] != (self.cfg.codebooks, self.cfg.code_steps):
            raise ValueError(f"codes must be [B,{self.cfg.codebooks},{self.cfg.code_steps}], got {tuple(codes.shape)}")
        streams = []
        for codebook, embedding in enumerate(self.code_embeddings):
            stream = embedding(codes[:, codebook].long())
            stream = stream + self.codebook_embedding[:, codebook]
            streams.append(stream)
        tokens = torch.stack(streams, dim=0).mean(dim=0) * math.sqrt(float(self.cfg.codebooks))
        encoded = self.encoder(tokens + self.position)
        condition = F.adaptive_avg_pool1d(encoded.transpose(1, 2), self.cfg.condition_steps).transpose(1, 2)
        condition = self.condition_norm(self.condition_refiner(condition))
        pooled = condition.mean(dim=1)
        return {"condition": condition, "pooled": pooled, "label_logits": self.label_head(pooled), "code_tokens": encoded}


class MaskedCodeDecoder(nn.Module):
    """MaskGIT-style conditional decoder over all EnCodec codebooks."""

    def __init__(self, cfg: AudioCodeModelConfig):
        super().__init__()
        self.cfg = cfg
        self.mask_id = int(cfg.vocab_size)
        self.code_embeddings = nn.ModuleList([nn.Embedding(cfg.vocab_size + 1, cfg.d_model) for _ in range(cfg.codebooks)])
        self.codebook_embedding = nn.Parameter(torch.zeros(1, cfg.codebooks, 1, cfg.d_model))
        self.position = nn.Parameter(torch.zeros(1, cfg.code_steps, cfg.d_model))
        self.label_embedding = nn.Parameter(torch.zeros(cfg.num_labels, cfg.d_model))
        layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=cfg.decoder_layers)
        self.output_heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.vocab_size)) for _ in range(cfg.codebooks)])
        nn.init.normal_(self.codebook_embedding, std=0.02)
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.label_embedding, std=0.02)

    def forward(
        self,
        codes: torch.Tensor,
        mask: torch.Tensor,
        condition: torch.Tensor,
        label_probabilities: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if codes.ndim != 3 or mask.shape != codes.shape:
            raise ValueError("codes and mask must both be [B,Q,T]")
        if codes.shape[1:] != (self.cfg.codebooks, self.cfg.code_steps):
            raise ValueError(f"Unexpected code shape: {tuple(codes.shape)}")
        masked_codes = torch.where(mask.bool(), torch.full_like(codes, self.mask_id), codes).long()
        streams = []
        for codebook, embedding in enumerate(self.code_embeddings):
            stream = embedding(masked_codes[:, codebook]) + self.codebook_embedding[:, codebook]
            streams.append(stream)
        target = torch.stack(streams, dim=0).mean(dim=0) * math.sqrt(float(self.cfg.codebooks))
        target = target + self.position
        if label_probabilities is not None:
            if label_probabilities.shape != (codes.shape[0], self.cfg.num_labels):
                raise ValueError("label_probabilities must be [B,num_labels]")
            target = target + (label_probabilities.to(target.dtype) @ self.label_embedding).unsqueeze(1)
        hidden = self.decoder(target, condition)
        return torch.stack([head(hidden) for head in self.output_heads], dim=1)

    @torch.no_grad()
    def generate(
        self,
        condition: torch.Tensor,
        label_probabilities: torch.Tensor,
        *,
        steps: int = 12,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        batch = condition.shape[0]
        codes = torch.zeros(batch, self.cfg.codebooks, self.cfg.code_steps, device=condition.device, dtype=torch.long)
        mask = torch.ones_like(codes, dtype=torch.bool)
        steps = max(1, int(steps))
        for step in range(steps):
            logits = self(codes, mask, condition, label_probabilities)
            probabilities = torch.softmax(logits.float() / max(float(temperature), 1e-6), dim=-1) if temperature > 0 else torch.softmax(logits.float(), dim=-1)
            if temperature > 0:
                flat = probabilities.reshape(-1, self.cfg.vocab_size)
                proposed = torch.multinomial(flat, 1, generator=generator).reshape_as(codes)
                confidence = probabilities.gather(-1, proposed.unsqueeze(-1)).squeeze(-1)
            else:
                confidence, proposed = probabilities.max(dim=-1)
            remaining_steps = steps - step
            for item in range(batch):
                remaining = torch.nonzero(mask[item].reshape(-1), as_tuple=False).flatten()
                if remaining.numel() == 0:
                    continue
                fill_count = remaining.numel() if remaining_steps == 1 else max(1, math.ceil(remaining.numel() / remaining_steps))
                item_confidence = confidence[item].reshape(-1)[remaining]
                chosen = remaining[torch.topk(item_confidence, k=min(fill_count, remaining.numel())).indices]
                flat_codes = codes[item].reshape(-1)
                flat_mask = mask[item].reshape(-1)
                flat_proposed = proposed[item].reshape(-1)
                flat_codes[chosen] = flat_proposed[chosen]
                flat_mask[chosen] = False
        if mask.any():
            logits = self(codes, mask, condition, label_probabilities)
            codes = torch.where(mask, logits.argmax(dim=-1), codes)
        return codes


class AudioCodeAutoencoder(nn.Module):
    def __init__(self, cfg: AudioCodeModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = AudioConditionEncoder(cfg)
        self.decoder = MaskedCodeDecoder(cfg)

    def forward(
        self,
        codes: torch.Tensor,
        mask: torch.Tensor,
        label_probabilities: torch.Tensor | None = None,
        condition_dropout: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder(codes)
        condition = encoded["condition"]
        if condition_dropout is not None:
            condition = condition * (~condition_dropout.bool()).to(condition.dtype).view(-1, 1, 1)
        logits = self.decoder(codes, mask, condition, label_probabilities)
        return {**encoded, "decoder_condition": condition, "code_logits": logits}


class _MultiScaleStem(nn.Module):
    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        input_channels = cfg.channels * 2
        branch_channels = cfg.d_model // len(cfg.temporal_kernels)
        if branch_channels * len(cfg.temporal_kernels) != cfg.d_model:
            raise ValueError("d_model must be divisible by the number of temporal kernels")
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(input_channels, branch_channels, kernel_size=kernel, stride=cfg.stem_stride, padding=kernel // 2, bias=False),
                    nn.GroupNorm(max(1, branch_channels // 16), branch_channels),
                    nn.GELU(),
                )
                for kernel in cfg.temporal_kernels
            ]
        )
        self.mix = nn.Sequential(
            nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=1, bias=False),
            nn.GroupNorm(max(1, cfg.d_model // 16), cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        car = eeg - eeg.mean(dim=1, keepdim=True)
        dual = torch.cat([eeg, car], dim=1)
        return self.mix(torch.cat([branch(dual) for branch in self.branches], dim=1))


class EEGConditionEncoder(nn.Module):
    """Small-data EEG encoder aligned to the audio codec condition manifold."""

    def __init__(self, cfg: EEGModelConfig):
        super().__init__()
        self.cfg = cfg
        self.stem = _MultiScaleStem(cfg)
        stem_steps = math.floor((cfg.eeg_len + 2 * (cfg.temporal_kernels[0] // 2) - cfg.temporal_kernels[0]) / cfg.stem_stride + 1)
        self.position = nn.Parameter(torch.zeros(1, stem_steps, cfg.d_model))
        self.encoder = _transformer_encoder(cfg.d_model, cfg.heads, cfg.transformer_layers, cfg.dropout)
        self.queries = nn.Parameter(torch.zeros(1, cfg.condition_steps, cfg.d_model))
        self.query_attention = nn.MultiheadAttention(cfg.d_model, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.condition_refiner = _transformer_encoder(cfg.d_model, cfg.heads, 2, cfg.dropout)
        self.condition_norm = nn.LayerNorm(cfg.d_model)
        self.label_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.num_labels))
        self.envelope_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, 1))
        self.timing_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, 2))
        self.subject_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, cfg.num_train_subjects)
        )
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.queries, std=0.02)

    def augment(self, eeg: torch.Tensor, *, channel_dropout: float, time_mask_ratio: float, noise_std: float) -> torch.Tensor:
        output = eeg
        if channel_dropout > 0:
            keep = (torch.rand(eeg.shape[0], eeg.shape[1], 1, device=eeg.device) > float(channel_dropout)).to(eeg.dtype)
            output = output * keep / keep.mean(dim=1, keepdim=True).clamp_min(0.25)
        if time_mask_ratio > 0:
            width = max(1, int(round(eeg.shape[-1] * float(time_mask_ratio))))
            for item in range(eeg.shape[0]):
                start = int(torch.randint(0, max(1, eeg.shape[-1] - width + 1), (1,), device=eeg.device).item())
                output[item, :, start : start + width] = 0
        if noise_std > 0:
            output = output + float(noise_std) * torch.randn_like(output)
        return output

    def forward(self, eeg: torch.Tensor, valid_len: torch.Tensor, *, subject_adversary_strength: float = 0.0) -> dict[str, torch.Tensor]:
        if eeg.ndim != 3 or eeg.shape[1] != self.cfg.channels:
            raise ValueError(f"eeg must be [B,{self.cfg.channels},T], got {tuple(eeg.shape)}")
        stem = self.stem(eeg).transpose(1, 2)
        stem_steps = stem.shape[1]
        valid_steps = torch.ceil(valid_len.float() / float(self.cfg.stem_stride)).long().clamp(1, stem_steps)
        valid_mask = torch.arange(stem_steps, device=eeg.device).unsqueeze(0) < valid_steps.unsqueeze(1)
        encoded = self.encoder(stem + self.position[:, :stem_steps], src_key_padding_mask=~valid_mask)
        queries = self.queries.expand(eeg.shape[0], -1, -1)
        condition, _ = self.query_attention(queries, encoded, encoded, key_padding_mask=~valid_mask, need_weights=False)
        condition = self.condition_norm(self.condition_refiner(condition))
        pooled = condition.mean(dim=1)
        envelope_logits = self.envelope_head(condition).squeeze(-1)
        envelope_logits = F.interpolate(envelope_logits.unsqueeze(1), size=self.cfg.code_steps, mode="linear", align_corners=False).squeeze(1)
        timing = torch.sigmoid(self.timing_head(pooled))
        return {
            "condition": condition,
            "pooled": pooled,
            "label_logits": self.label_head(pooled),
            "envelope_logits": envelope_logits,
            "onset": timing[:, 0],
            "duration": timing[:, 1],
            "subject_logits": self.subject_head(grad_reverse(pooled, subject_adversary_strength)),
            "eeg_tokens": encoded,
            "eeg_valid_mask": valid_mask,
        }


def random_code_mask(
    codes: torch.Tensor,
    *,
    min_ratio: float,
    max_ratio: float,
    full_mask_probability: float,
) -> torch.Tensor:
    if not 0.0 <= min_ratio <= max_ratio <= 1.0:
        raise ValueError("mask ratios must satisfy 0 <= min <= max <= 1")
    batch = codes.shape[0]
    ratios = torch.empty(batch, 1, 1, device=codes.device).uniform_(float(min_ratio), float(max_ratio))
    if full_mask_probability > 0:
        full = torch.rand(batch, 1, 1, device=codes.device) < float(full_mask_probability)
        ratios = torch.where(full, torch.ones_like(ratios), ratios)
    mask = torch.rand(codes.shape, device=codes.device) < ratios
    mask[:, :, 0] = True
    return mask

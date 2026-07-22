from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, strength: float) -> torch.Tensor:  # type: ignore[override]
        ctx.strength = float(strength)
        return value.view_as(value)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        return -ctx.strength * gradient, None


def grad_reverse(value: torch.Tensor, strength: float) -> torch.Tensor:
    return _GradientReverse.apply(value, float(strength))


def _transformer(d_model: int, heads: int, layers: int, dropout: float, *, expansion: int = 4) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=heads,
        dim_feedforward=d_model * expansion,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers)


def sinusoidal_positions(length: int, dimension: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if dimension < 2:
        raise ValueError("position dimension must be at least 2")
    half = dimension // 2
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    scale = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=device, dtype=torch.float32)
        / max(1, half - 1)
    )
    encoded = torch.cat((torch.sin(positions * scale), torch.cos(positions * scale)), dim=1)
    if encoded.shape[1] < dimension:
        encoded = F.pad(encoded, (0, dimension - encoded.shape[1]))
    return encoded[:, :dimension].to(dtype=dtype)


@dataclass(frozen=True)
class LabelFreeAudioConfig:
    codebooks: int = 8
    code_steps: int = 150
    vocab_size: int = 1024
    d_model: int = 192
    condition_steps: int = 50
    encoder_layers: int = 3
    decoder_layers: int = 4
    heads: int = 6
    dropout: float = 0.10
    text_dimension: int = 768
    xlsr_dimension: int = 1024


class LabelFreeAudioConditionEncoder(nn.Module):
    """Map exact codec tokens to a label-independent acoustic condition."""

    def __init__(self, cfg: LabelFreeAudioConfig):
        super().__init__()
        self.cfg = cfg
        self.code_embeddings = nn.ModuleList(
            [nn.Embedding(cfg.vocab_size, cfg.d_model) for _ in range(cfg.codebooks)]
        )
        self.codebook_embedding = nn.Parameter(torch.zeros(1, cfg.codebooks, 1, cfg.d_model))
        self.position = nn.Parameter(torch.zeros(1, cfg.code_steps, cfg.d_model))
        self.encoder = _transformer(cfg.d_model, cfg.heads, cfg.encoder_layers, cfg.dropout)
        self.condition_refiner = _transformer(cfg.d_model, cfg.heads, 2, cfg.dropout)
        self.condition_norm = nn.LayerNorm(cfg.d_model)
        self.acoustic_projection = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model))
        self.semantic_projection = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model))
        nn.init.normal_(self.codebook_embedding, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, codes: torch.Tensor, code_valid_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        expected = (self.cfg.codebooks, self.cfg.code_steps)
        if codes.ndim != 3 or tuple(codes.shape[1:]) != expected:
            raise ValueError(f"codes must be [B,{expected[0]},{expected[1]}], got {tuple(codes.shape)}")
        streams = [
            embedding(codes[:, index].long()) + self.codebook_embedding[:, index]
            for index, embedding in enumerate(self.code_embeddings)
        ]
        tokens = torch.stack(streams, dim=0).mean(dim=0) * math.sqrt(float(self.cfg.codebooks))
        if code_valid_mask is None:
            valid = torch.ones(codes.shape[0], self.cfg.code_steps, device=codes.device, dtype=torch.bool)
        else:
            if code_valid_mask.shape == codes.shape:
                valid = code_valid_mask.bool().any(dim=1)
            elif code_valid_mask.shape == (codes.shape[0], self.cfg.code_steps):
                valid = code_valid_mask.bool()
            else:
                raise ValueError("code_valid_mask must be [B,T] or [B,Q,T]")
        encoded = self.encoder(tokens + self.position, src_key_padding_mask=~valid)
        condition = F.adaptive_avg_pool1d(
            encoded.transpose(1, 2), self.cfg.condition_steps
        ).transpose(1, 2)
        condition = self.condition_norm(self.condition_refiner(condition))
        pooled = condition.mean(dim=1)
        return {
            "condition": condition,
            "pooled": pooled,
            "acoustic_global": F.normalize(self.acoustic_projection(pooled), dim=-1),
            "semantic_global": F.normalize(self.semantic_projection(pooled), dim=-1),
            "acoustic_local": F.normalize(condition, dim=-1),
            "code_tokens": encoded,
            "code_valid_mask": valid,
        }


class LabelFreeMaskedCodeDecoder(nn.Module):
    """MaskGIT decoder whose public API has no label or dataset argument."""

    def __init__(self, cfg: LabelFreeAudioConfig):
        super().__init__()
        self.cfg = cfg
        self.mask_id = int(cfg.vocab_size)
        self.code_embeddings = nn.ModuleList(
            [nn.Embedding(cfg.vocab_size + 1, cfg.d_model) for _ in range(cfg.codebooks)]
        )
        self.codebook_embedding = nn.Parameter(torch.zeros(1, cfg.codebooks, 1, cfg.d_model))
        self.position = nn.Parameter(torch.zeros(1, cfg.code_steps, cfg.d_model))
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
        self.output_heads = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.vocab_size)) for _ in range(cfg.codebooks)]
        )
        nn.init.normal_(self.codebook_embedding, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, codes: torch.Tensor, mask: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if codes.ndim != 3 or mask.shape != codes.shape:
            raise ValueError("codes and mask must share [B,Q,T]")
        if tuple(codes.shape[1:]) != (self.cfg.codebooks, self.cfg.code_steps):
            raise ValueError(f"unexpected code shape: {tuple(codes.shape)}")
        if condition.shape != (codes.shape[0], self.cfg.condition_steps, self.cfg.d_model):
            raise ValueError("condition must be [B,condition_steps,d_model]")
        masked = torch.where(mask.bool(), torch.full_like(codes, self.mask_id), codes).long()
        streams = [
            embedding(masked[:, index]) + self.codebook_embedding[:, index]
            for index, embedding in enumerate(self.code_embeddings)
        ]
        target = torch.stack(streams, dim=0).mean(dim=0) * math.sqrt(float(self.cfg.codebooks))
        hidden = self.decoder(target + self.position, condition)
        return torch.stack([head(hidden) for head in self.output_heads], dim=1)

    @torch.no_grad()
    def generate(
        self,
        condition: torch.Tensor,
        *,
        steps: int = 12,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        batch = condition.shape[0]
        codes = torch.zeros(
            batch,
            self.cfg.codebooks,
            self.cfg.code_steps,
            device=condition.device,
            dtype=torch.long,
        )
        mask = torch.ones_like(codes, dtype=torch.bool)
        for step in range(max(1, int(steps))):
            logits = self(codes, mask, condition)
            if temperature > 0:
                probabilities = torch.softmax(logits.float() / float(temperature), dim=-1)
                proposed = torch.multinomial(
                    probabilities.reshape(-1, self.cfg.vocab_size), 1, generator=generator
                ).reshape_as(codes)
                confidence = probabilities.gather(-1, proposed.unsqueeze(-1)).squeeze(-1)
            else:
                confidence, proposed = torch.softmax(logits.float(), dim=-1).max(dim=-1)
            remaining_steps = max(1, int(steps) - step)
            for item in range(batch):
                remaining = torch.nonzero(mask[item].reshape(-1), as_tuple=False).flatten()
                if remaining.numel() == 0:
                    continue
                count = remaining.numel() if remaining_steps == 1 else max(1, math.ceil(remaining.numel() / remaining_steps))
                selected = remaining[
                    torch.topk(confidence[item].reshape(-1)[remaining], k=min(count, remaining.numel())).indices
                ]
                flat_codes = codes[item].reshape(-1)
                flat_mask = mask[item].reshape(-1)
                flat_proposed = proposed[item].reshape(-1)
                flat_codes[selected] = flat_proposed[selected]
                flat_mask[selected] = False
        if mask.any():
            codes = torch.where(mask, self(codes, mask, condition).argmax(dim=-1), codes)
        return codes


class TextConditionProjector(nn.Module):
    """Evaluation/auxiliary text branch; never called by the main generator."""

    def __init__(self, cfg: LabelFreeAudioConfig):
        super().__init__()
        self.cfg = cfg
        self.semantic = nn.Sequential(
            nn.LayerNorm(cfg.text_dimension),
            nn.Linear(cfg.text_dimension, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.condition = nn.Sequential(
            nn.LayerNorm(cfg.text_dimension),
            nn.Linear(cfg.text_dimension, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.condition_steps * cfg.d_model),
        )

    def forward(self, text_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        if text_embedding.ndim != 2 or text_embedding.shape[1] != self.cfg.text_dimension:
            raise ValueError(f"text_embedding must be [B,{self.cfg.text_dimension}]")
        condition = self.condition(text_embedding).reshape(
            len(text_embedding), self.cfg.condition_steps, self.cfg.d_model
        )
        return {
            "semantic_global": F.normalize(self.semantic(text_embedding), dim=-1),
            "condition": condition,
        }


class XLSRConditionEncoder(nn.Module):
    """Project frozen XLS-R content tokens into the shared acoustic space."""

    def __init__(self, cfg: LabelFreeAudioConfig):
        super().__init__()
        self.cfg = cfg
        self.input = nn.Sequential(
            nn.LayerNorm(cfg.xlsr_dimension),
            nn.Linear(cfg.xlsr_dimension, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.encoder = _transformer(cfg.d_model, cfg.heads, 2, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.acoustic_projection = nn.Linear(cfg.d_model, cfg.d_model)
        self.semantic_projection = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if tokens.ndim != 3 or tokens.shape[-1] != self.cfg.xlsr_dimension:
            raise ValueError(f"XLS-R tokens must be [B,T,{self.cfg.xlsr_dimension}]")
        value = self.input(tokens)
        if value.shape[1] != self.cfg.condition_steps:
            value = F.adaptive_avg_pool1d(
                value.transpose(1, 2), self.cfg.condition_steps
            ).transpose(1, 2)
        condition = self.norm(self.encoder(value))
        pooled = condition.mean(dim=1)
        return {
            "condition": condition,
            "pooled": pooled,
            "acoustic_global": F.normalize(self.acoustic_projection(pooled), dim=-1),
            "semantic_global": F.normalize(self.semantic_projection(pooled), dim=-1),
            "acoustic_local": F.normalize(condition, dim=-1),
        }


class LabelFreeAudioModel(nn.Module):
    def __init__(self, cfg: LabelFreeAudioConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = LabelFreeAudioConditionEncoder(cfg)
        self.xlsr_encoder = XLSRConditionEncoder(cfg)
        self.decoder = LabelFreeMaskedCodeDecoder(cfg)
        self.text_projector = TextConditionProjector(cfg)

    def forward(
        self,
        codes: torch.Tensor,
        mask: torch.Tensor,
        *,
        code_valid_mask: torch.Tensor | None = None,
        xlsr_tokens: torch.Tensor | None = None,
        condition_dropout: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        codec_encoded = self.encoder(codes, code_valid_mask)
        encoded = self.xlsr_encoder(xlsr_tokens) if xlsr_tokens is not None else codec_encoded
        condition = encoded["condition"]
        if condition_dropout is not None:
            condition = condition * (~condition_dropout.bool()).to(condition.dtype).view(-1, 1, 1)
        return {
            **encoded,
            "codec_oracle_condition": codec_encoded["condition"],
            "decoder_condition": condition,
            "code_logits": self.decoder(codes, mask, condition),
        }


@dataclass(frozen=True)
class OpenVoiceEEGConfig:
    eeg_samples: int = 1280
    patch_size: int = 64
    patch_hop: int = 32
    d_model: int = 192
    condition_steps: int = 50
    code_steps: int = 150
    heads: int = 6
    latent_layers: int = 3
    dropout: float = 0.15
    specialists: int = 4
    specialist_bottleneck: int = 48
    soft_routing_epochs: int = 5
    top_k_specialists: int = 2
    expert_dropout: float = 0.10
    num_datasets: int = 3
    num_train_subjects: int = 38
    adapter_moe_enabled: bool = True
    text_dimension: int = 768


class _Adapter(nn.Module):
    def __init__(self, dimension: int, bottleneck: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, bottleneck, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, dimension, bias=False),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class AntiCollapseAdapterMoE(nn.Module):
    """Universal FFN plus soft/top-2 low-rank specialists."""

    def __init__(self, cfg: OpenVoiceEEGConfig):
        super().__init__()
        if cfg.top_k_specialists < 1 or cfg.top_k_specialists > cfg.specialists:
            raise ValueError("top_k_specialists must be within the specialist count")
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.d_model)
        self.universal = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
        )
        self.specialists = nn.ModuleList(
            [_Adapter(cfg.d_model, cfg.specialist_bottleneck, cfg.dropout) for _ in range(cfg.specialists)]
        )
        self.router = nn.Linear(cfg.d_model, cfg.specialists)

    def forward(
        self,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        epoch: int = 0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if valid_mask.shape != tokens.shape[:2]:
            raise ValueError("valid_mask must match the first two token dimensions")
        normalized = self.norm(tokens)
        logits = self.router(normalized)
        weights = torch.sigmoid(logits)
        if not self.cfg.adapter_moe_enabled:
            # Parameter/compute-matched dense control: all adapter parameters
            # are used with fixed uniform weights and there is no routing.
            weights = torch.full_like(weights, 1.0 / float(self.cfg.specialists))
        elif self.training and self.cfg.expert_dropout > 0:
            keep = torch.rand_like(weights) >= float(self.cfg.expert_dropout)
            all_dropped = ~keep.any(dim=-1, keepdim=True)
            keep = keep | all_dropped.expand_as(keep)
            weights = weights * keep.to(weights.dtype)
        if self.cfg.adapter_moe_enabled and int(epoch) >= int(self.cfg.soft_routing_epochs):
            top = torch.topk(weights, k=self.cfg.top_k_specialists, dim=-1).indices
            active = torch.zeros_like(weights, dtype=torch.bool).scatter_(-1, top, True)
            weights = weights * active.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        specialist_outputs = torch.stack([expert(normalized) for expert in self.specialists], dim=-2)
        specialist = (specialist_outputs * weights.unsqueeze(-1)).sum(dim=-2)
        output = tokens + self.universal(normalized) + specialist
        output = torch.where(valid_mask.unsqueeze(-1), output, torch.zeros_like(output))

        valid_weight = valid_mask.to(weights.dtype).unsqueeze(-1)
        denominator = valid_weight.sum().clamp_min(1.0)
        mass = (weights * valid_weight).sum(dim=(0, 1)) / denominator
        sample_denominator = valid_weight.sum(dim=1).clamp_min(1.0)
        sample_mass = (weights * valid_weight).sum(dim=1) / sample_denominator
        target = torch.full_like(mass, 1.0 / float(self.cfg.specialists))
        balance = ((mass - target).square().mean() * self.cfg.specialists)
        z_loss = (torch.logsumexp(logits.float(), dim=-1).square() * valid_mask).sum() / valid_mask.sum().clamp_min(1)
        entropy_per_token = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = (entropy_per_token * valid_mask).sum() / valid_mask.sum().clamp_min(1)
        load = ((weights > 0).to(weights.dtype) * valid_weight).sum(dim=(0, 1)) / denominator
        return output, {
            "specialist_mass": mass,
            "sample_specialist_mass": sample_mass,
            "specialist_load": load,
            "balance_loss": balance,
            "z_loss": z_loss.to(tokens.dtype),
            "entropy": entropy,
            "router_logits": logits,
            "router_weights": weights,
        }


class OpenVoiceEEGEncoder(nn.Module):
    """Permutation-aware, variable-channel EEG encoder with no label input."""

    def __init__(self, cfg: OpenVoiceEEGConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embedding = nn.Sequential(
            nn.LayerNorm(cfg.patch_size),
            nn.Linear(cfg.patch_size, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.coordinate_embedding = nn.Sequential(
            nn.Linear(3, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model)
        )
        self.quality_embedding = nn.Sequential(nn.Linear(1, cfg.d_model), nn.Tanh())
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, cfg.d_model))
        self.input_norm = nn.LayerNorm(cfg.d_model)
        self.moe = AntiCollapseAdapterMoE(cfg)
        self.queries = nn.Parameter(torch.zeros(1, cfg.condition_steps, cfg.d_model))
        self.query_attention = nn.MultiheadAttention(
            cfg.d_model, cfg.heads, dropout=cfg.dropout, batch_first=True
        )
        self.latent_transformer = _transformer(
            cfg.d_model, cfg.heads, cfg.latent_layers, cfg.dropout
        )
        self.condition_norm = nn.LayerNorm(cfg.d_model)
        self.acoustic_projection = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model))
        self.semantic_projection = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model))
        self.text_semantic_projection = nn.Sequential(
            nn.LayerNorm(cfg.text_dimension), nn.Linear(cfg.text_dimension, cfg.d_model),
            nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.local_projection = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model))
        self.envelope_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, 1))
        self.timing_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, 2)
        )
        self.dataset_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.num_datasets)
        )
        self.subject_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.num_train_subjects)
        )
        self.router_dataset_head = nn.Linear(cfg.specialists, cfg.num_datasets)
        self.router_subject_head = nn.Linear(cfg.specialists, cfg.num_train_subjects)
        self.patch_reconstruction = nn.Sequential(
            nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.patch_size)
        )
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.queries, std=0.02)

    def project_text(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """Training/evaluation auxiliary; deliberately outside ``forward``."""
        if text_embedding.ndim != 2 or text_embedding.shape[-1] != self.cfg.text_dimension:
            raise ValueError(f"text_embedding must be [B,{self.cfg.text_dimension}]")
        return F.normalize(self.text_semantic_projection(text_embedding), dim=-1)

    def _patches(
        self,
        eeg: torch.Tensor,
        channel_mask: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if eeg.ndim != 3:
            raise ValueError("eeg must be [B,C,T]")
        if channel_mask.shape != eeg.shape[:2]:
            raise ValueError("channel_mask must be [B,C]")
        if time_mask.shape != (eeg.shape[0], eeg.shape[2]):
            raise ValueError("time_mask must be [B,T]")
        if eeg.shape[-1] < self.cfg.patch_size:
            raise ValueError("EEG sequence is shorter than one patch")
        # The time mask is semantic, not merely advisory: values in padded
        # samples must be unable to affect even a boundary patch.
        masked_eeg = eeg * time_mask.to(eeg.dtype).unsqueeze(1)
        patches = masked_eeg.unfold(-1, self.cfg.patch_size, self.cfg.patch_hop)
        time_windows = time_mask.to(eeg.dtype).unfold(-1, self.cfg.patch_size, self.cfg.patch_hop)
        patch_time_valid = time_windows.mean(dim=-1) >= 0.5
        valid = channel_mask.bool().unsqueeze(-1) & patch_time_valid.bool().unsqueeze(1)
        return patches, valid

    def forward(
        self,
        eeg: torch.Tensor,
        channel_xyz: torch.Tensor,
        channel_mask: torch.Tensor,
        time_mask: torch.Tensor,
        *,
        epoch: int = 0,
        patch_mask: torch.Tensor | None = None,
        adversary_strength: float = 0.0,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if channel_xyz.shape != (*eeg.shape[:2], 3):
            raise ValueError("channel_xyz must be [B,C,3]")
        if not torch.isfinite(eeg).all() or not torch.isfinite(channel_xyz).all():
            raise ValueError("EEG and channel coordinates must be finite")
        patches, patch_valid = self._patches(eeg, channel_mask, time_mask)
        tokens = self.patch_embedding(patches)
        coordinate = self.coordinate_embedding(channel_xyz).unsqueeze(2)
        time = sinusoidal_positions(tokens.shape[2], self.cfg.d_model, eeg.device, tokens.dtype).view(
            1, 1, tokens.shape[2], self.cfg.d_model
        )
        quality = torch.log1p(torch.sqrt(patches.square().mean(dim=-1, keepdim=True) + 1e-8))
        tokens = tokens + coordinate + time + self.quality_embedding(quality)
        if patch_mask is not None:
            if patch_mask.shape != patch_valid.shape:
                raise ValueError("patch_mask must be [B,C,P]")
            active_mask = patch_mask.bool() & patch_valid
            tokens = torch.where(active_mask.unsqueeze(-1), self.mask_token.expand_as(tokens), tokens)
        else:
            active_mask = torch.zeros_like(patch_valid)
        tokens = self.input_norm(tokens)
        batch, channels, steps, dimension = tokens.shape
        flat = tokens.reshape(batch, channels * steps, dimension)
        flat_valid = patch_valid.reshape(batch, channels * steps)
        routed, router = self.moe(flat, flat_valid, epoch=epoch)
        queries = self.queries.expand(batch, -1, -1)
        condition, _ = self.query_attention(
            queries, routed, routed, key_padding_mask=~flat_valid, need_weights=False
        )
        condition = self.condition_norm(self.latent_transformer(condition))
        pooled = condition.mean(dim=1)
        local = F.normalize(self.local_projection(condition), dim=-1)
        envelope = F.softplus(self.envelope_head(condition).squeeze(-1))
        envelope = F.interpolate(
            envelope.unsqueeze(1), size=self.cfg.code_steps, mode="linear", align_corners=False
        ).squeeze(1)
        timing = torch.sigmoid(self.timing_head(pooled))
        # Keep the adversary sample-specific.  Expanding the batch aggregate
        # would give every example the same router feature and make the
        # shortcut audit mathematically incapable of detecting a leak.
        router_summary = router["sample_specialist_mass"]
        reconstruction = self.patch_reconstruction(routed).reshape(
            batch, channels, steps, self.cfg.patch_size
        )
        return {
            "condition": condition,
            "pooled": pooled,
            "acoustic_global": F.normalize(self.acoustic_projection(pooled), dim=-1),
            "semantic_global": F.normalize(self.semantic_projection(pooled), dim=-1),
            "acoustic_local": local,
            "envelope": envelope,
            "onset": timing[:, 0],
            "duration": timing[:, 1],
            "dataset_logits": self.dataset_head(grad_reverse(pooled, adversary_strength)),
            "subject_logits": self.subject_head(grad_reverse(pooled, adversary_strength)),
            "router_dataset_logits": self.router_dataset_head(grad_reverse(router_summary, adversary_strength)),
            "router_subject_logits": self.router_subject_head(grad_reverse(router_summary, adversary_strength)),
            "patch_reconstruction": reconstruction,
            "patch_target": patches,
            "patch_valid_mask": patch_valid,
            "patch_mask": active_mask,
            "router": router,
        }


class OpenVoiceGenerator(nn.Module):
    """Inference facade: EEG and validity metadata are the only inputs."""

    def __init__(self, eeg: OpenVoiceEEGEncoder, audio: LabelFreeAudioModel):
        super().__init__()
        self.eeg = eeg
        self.audio = audio

    @torch.no_grad()
    def generate(
        self,
        eeg: torch.Tensor,
        channel_xyz: torch.Tensor,
        channel_mask: torch.Tensor,
        time_mask: torch.Tensor,
        *,
        steps: int = 12,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        output = self.eeg(eeg, channel_xyz, channel_mask, time_mask)
        return self.audio.decoder.generate(
            output["condition"], steps=steps, temperature=temperature  # type: ignore[arg-type]
        )


def random_code_mask(
    codes: torch.Tensor,
    *,
    min_ratio: float,
    max_ratio: float,
    full_mask_probability: float,
) -> torch.Tensor:
    if not 0.0 <= min_ratio <= max_ratio <= 1.0:
        raise ValueError("mask ratios must satisfy 0 <= min <= max <= 1")
    ratios = torch.empty(len(codes), 1, 1, device=codes.device).uniform_(min_ratio, max_ratio)
    if full_mask_probability > 0:
        full = torch.rand(len(codes), 1, 1, device=codes.device) < float(full_mask_probability)
        ratios = torch.where(full, torch.ones_like(ratios), ratios)
    mask = torch.rand(codes.shape, device=codes.device) < ratios
    mask[:, :, 0] = True
    return mask


def random_patch_mask(valid_mask: torch.Tensor, ratio: float) -> torch.Tensor:
    if not 0.0 <= float(ratio) <= 1.0:
        raise ValueError("patch mask ratio must be in [0,1]")
    mask = (torch.rand(valid_mask.shape, device=valid_mask.device) < float(ratio)) & valid_mask.bool()
    for item in range(len(mask)):
        available = torch.nonzero(valid_mask[item], as_tuple=False)
        if available.numel() and not mask[item].any():
            first = available[0]
            mask[item, first[0], first[1]] = True
    return mask


__all__ = [
    "AntiCollapseAdapterMoE",
    "LabelFreeAudioConfig",
    "LabelFreeAudioModel",
    "OpenVoiceEEGConfig",
    "OpenVoiceEEGEncoder",
    "OpenVoiceGenerator",
    "TextConditionProjector",
    "XLSRConditionEncoder",
    "random_code_mask",
    "random_patch_mask",
]

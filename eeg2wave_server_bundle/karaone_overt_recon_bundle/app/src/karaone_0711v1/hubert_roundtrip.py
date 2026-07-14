from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class HubertRoundTripConfig:
    """Architecture for a frozen-HuBERT-to-EnCodec audit decoder.

    This model is deliberately separate from the EEG flow decoder.  Its input
    is an adapted HuBERT sequence, and it predicts only the cached continuous
    EnCodec latent.  It never accepts EEG, subject IDs, labels, or waveform
    samples as inputs.
    """

    source_dim: int = 768
    source_steps: int = 50
    latent_dim: int = 128
    latent_steps: int = 150
    d_model: int = 256
    heads: int = 4
    encoder_layers: int = 4
    refiner_layers: int = 2
    dropout: float = 0.1


class HubertToEncodecDecoder(nn.Module):
    """Predict EnCodec latents from a frozen, continuous HuBERT sequence."""

    def __init__(self, cfg: HubertRoundTripConfig = HubertRoundTripConfig()):
        super().__init__()
        self.cfg = cfg
        self.input_norm = nn.LayerNorm(cfg.source_dim)
        self.input_projection = nn.Linear(cfg.source_dim, cfg.d_model)
        self.source_position = nn.Parameter(torch.zeros(1, cfg.source_steps, cfg.d_model))
        self.latent_position = nn.Parameter(torch.zeros(1, cfg.latent_steps, cfg.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            cfg.d_model,
            cfg.heads,
            cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.source_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.encoder_layers)
        refiner_layer = nn.TransformerEncoderLayer(
            cfg.d_model,
            cfg.heads,
            cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.latent_refiner = nn.TransformerEncoder(refiner_layer, num_layers=cfg.refiner_layers)
        self.output_projection = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.latent_dim))
        nn.init.normal_(self.source_position, std=0.02)
        nn.init.normal_(self.latent_position, std=0.02)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError(f"Expected HuBERT sequence [B,T,D], got {tuple(sequence.shape)}")
        if sequence.shape[1:] != (self.cfg.source_steps, self.cfg.source_dim):
            raise ValueError(
                "HuBERT sequence shape mismatch: expected "
                f"[B,{self.cfg.source_steps},{self.cfg.source_dim}], got {tuple(sequence.shape)}"
            )
        encoded = self.input_projection(self.input_norm(sequence)) + self.source_position
        encoded = self.source_encoder(encoded)
        upsampled = F.interpolate(
            encoded.transpose(1, 2),
            size=self.cfg.latent_steps,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        refined = self.latent_refiner(upsampled + self.latent_position)
        return self.output_projection(refined)


def per_example_latent_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return trial-level latent MSE and cosine similarity for bootstrap summaries."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(f"Expected matching latent tensors [B,T,D], got {tuple(prediction.shape)} and {tuple(target.shape)}")
    mse = (prediction - target).pow(2).mean(dim=(1, 2))
    cosine = F.cosine_similarity(prediction.flatten(start_dim=1), target.flatten(start_dim=1), dim=1, eps=1e-8)
    return {"latent_mse": mse, "latent_cosine": cosine}

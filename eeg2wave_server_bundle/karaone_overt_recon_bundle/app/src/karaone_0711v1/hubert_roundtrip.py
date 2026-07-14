from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class HubertRoundTripConfig:
    """Architecture for a discrete HuBERT-token-to-EnCodec audit decoder.

    This model is deliberately separate from the EEG flow decoder. Its only
    input is the 50-step, 64-unit semantic_token_ids generated from adapted
    HuBERT. It never accepts EEG, continuous HuBERT hidden states, subject IDs,
    labels, waveforms, or true EnCodec latents as inputs.
    """

    vocab_size: int = 64
    token_steps: int = 50
    latent_dim: int = 128
    latent_steps: int = 150
    d_model: int = 128
    heads: int = 4
    encoder_layers: int = 2
    refiner_layers: int = 1
    dropout: float = 0.1


class HubertTokenToEncodecDecoder(nn.Module):
    """Predict EnCodec latents from discrete adapted-HuBERT token IDs only."""

    def __init__(self, cfg: HubertRoundTripConfig = HubertRoundTripConfig()):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.token_position = nn.Parameter(torch.zeros(1, cfg.token_steps, cfg.d_model))
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
        nn.init.normal_(self.token_position, std=0.02)
        nn.init.normal_(self.latent_position, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] != self.cfg.token_steps:
            raise ValueError(f"Expected token IDs [B,{self.cfg.token_steps}], got {tuple(token_ids.shape)}")
        if token_ids.numel() and (token_ids.min() < 0 or token_ids.max() >= self.cfg.vocab_size):
            raise ValueError(f"Token IDs must be in [0,{self.cfg.vocab_size - 1}]")
        encoded = self.token_embedding(token_ids.long()) + self.token_position
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

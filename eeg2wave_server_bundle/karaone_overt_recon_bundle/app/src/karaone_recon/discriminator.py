from __future__ import annotations

"""Adversarial discriminator over the acoustic target sequence (mel or latent).

A small PatchGAN-style 2D conv discriminator that treats the predicted/target
sequence [B, T, D] as an image [B, 1, D, T]. Used (behind the `lambda_gan`
switch) to fight the mean-collapse / over-smoothing of pure regression — the
generator must produce sharp, realistic acoustic frames, not the blurry mean
(NeuroTalk / HiFi-GAN style adversarial + feature-matching objective)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AcousticDiscriminator(nn.Module):
    def __init__(self, hidden: int = 64, n_layers: int = 3):
        super().__init__()
        layers = []
        in_ch = 1
        ch = hidden
        for i in range(n_layers):
            layers.append(
                nn.Conv2d(in_ch, ch, kernel_size=(3, 5), stride=(2, 2), padding=(1, 2))
            )
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = ch
            ch = min(ch * 2, 256)
        self.blocks = nn.ModuleList(layers)
        self.head = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # seq: [B, T, D] -> image [B, 1, D, T]
        x = seq.transpose(1, 2).unsqueeze(1)
        feats: list[torch.Tensor] = []
        for layer in self.blocks:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                feats.append(x)
        return self.head(x), feats


# -- LSGAN losses (stable, simple) -----------------------------------------
def discriminator_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    return 0.5 * (F.mse_loss(d_real, torch.ones_like(d_real)) + F.mse_loss(d_fake, torch.zeros_like(d_fake)))


def generator_adv_loss(d_fake: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(d_fake, torch.ones_like(d_fake))


def feature_matching_loss(feats_real: list[torch.Tensor], feats_fake: list[torch.Tensor]) -> torch.Tensor:
    loss = feats_fake[0].new_tensor(0.0) if feats_fake else torch.tensor(0.0)
    for fr, ff in zip(feats_real, feats_fake):
        loss = loss + F.l1_loss(ff, fr.detach())
    return loss / max(len(feats_fake), 1)

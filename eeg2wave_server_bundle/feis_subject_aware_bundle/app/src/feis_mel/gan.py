from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AcousticPatchDiscriminator(nn.Module):
    def __init__(self, target_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.ModuleList(
            [
                nn.Sequential(nn.Conv1d(target_dim, hidden, kernel_size=5, padding=2), nn.LeakyReLU(0.2), nn.Dropout(dropout)),
                nn.Sequential(nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(0.2), nn.Dropout(dropout)),
                nn.Sequential(nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2), nn.LeakyReLU(0.2), nn.Dropout(dropout)),
            ]
        )
        self.head = nn.Conv1d(hidden, 1, kernel_size=3, padding=1)

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = seq.transpose(1, 2)
        feats: list[torch.Tensor] = []
        for block in self.net:
            x = block(x)
            feats.append(x)
        return self.head(x).mean(dim=(1, 2)), feats


def discriminator_loss(discriminator: AcousticPatchDiscriminator, real_seq: torch.Tensor, fake_seq: torch.Tensor) -> torch.Tensor:
    real_logits, _ = discriminator(real_seq)
    fake_logits, _ = discriminator(fake_seq.detach())
    real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    return 0.5 * (real_loss + fake_loss)


def generator_loss(
    discriminator: AcousticPatchDiscriminator,
    real_seq: torch.Tensor,
    fake_seq: torch.Tensor,
    *,
    feat_match_weight: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fake_logits, fake_feats = discriminator(fake_seq)
    with torch.no_grad():
        _, real_feats = discriminator(real_seq)
    adv = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    feat = fake_seq.new_tensor(0.0)
    for fake, real in zip(fake_feats, real_feats):
        feat = feat + F.l1_loss(fake, real)
    total = adv + float(feat_match_weight) * feat
    return total, adv.detach(), feat.detach()

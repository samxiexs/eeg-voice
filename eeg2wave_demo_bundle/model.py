from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGEncoder(nn.Module):
    def __init__(self, in_channels: int = 14, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.net(eeg)


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 128,
        beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.beta = float(beta)
        self.decay = float(decay)
        self.eps = float(eps)

        embed = torch.randn(self.num_embeddings, self.embedding_dim)
        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, channels, steps = z_e.shape
        flat = z_e.permute(0, 2, 1).reshape(-1, channels)
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ self.embedding.t()
            + self.embedding.pow(2).sum(dim=1, keepdim=True).t()
        )
        codes = torch.argmin(distances, dim=1)
        encodings = F.one_hot(codes, self.num_embeddings).type(flat.dtype)
        quantized = F.embedding(codes, self.embedding).view(batch, steps, channels)

        if self.training:
            self._ema_update(flat, encodings)

        commitment = self.beta * F.mse_loss(z_e.permute(0, 2, 1), quantized.detach())
        quantized = z_e.permute(0, 2, 1) + (quantized - z_e.permute(0, 2, 1)).detach()
        quantized = quantized.permute(0, 2, 1).contiguous()

        avg_probs = encodings.float().mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return quantized, commitment, codes.view(batch, steps), perplexity

    def _ema_update(self, flat: torch.Tensor, encodings: torch.Tensor) -> None:
        with torch.no_grad():
            self.cluster_size.mul_(self.decay).add_(encodings.sum(dim=0), alpha=1.0 - self.decay)
            self.embed_avg.mul_(self.decay).add_(encodings.t() @ flat, alpha=1.0 - self.decay)

            total_count = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (
                total_count + self.num_embeddings * self.eps
            ) * total_count
            self.embedding.copy_(self.embed_avg / cluster_size.unsqueeze(1))


class WaveformDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, output_samples: int = 24000):
        super().__init__()
        self.output_samples = int(output_samples)
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, 64, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=9, stride=5, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv1d(16, 8, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(8, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        x = self.up1(z_q)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = F.interpolate(x, size=self.output_samples, mode="linear", align_corners=False)
        return self.refine(x)


class EEG2WaveVQModel(nn.Module):
    def __init__(
        self,
        n_channels_eeg: int = 14,
        hidden_dim: int = 128,
        codebook_size: int = 512,
        vq_beta: float = 0.25,
        vq_decay: float = 0.99,
        output_samples: int = 24000,
    ):
        super().__init__()
        self.encoder = EEGEncoder(in_channels=n_channels_eeg, hidden_dim=hidden_dim)
        self.quantizer = VectorQuantizerEMA(
            num_embeddings=codebook_size,
            embedding_dim=hidden_dim,
            beta=vq_beta,
            decay=vq_decay,
        )
        self.decoder = WaveformDecoder(hidden_dim=hidden_dim, output_samples=output_samples)

    def forward(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encoder(eeg)
        z_q, vq_loss, codes, perplexity = self.quantizer(z_e)
        recon = self.decoder(z_q)
        return recon, vq_loss, codes, perplexity

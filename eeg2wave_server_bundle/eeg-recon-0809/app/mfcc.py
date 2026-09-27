"""MFCC-80: the orthonormal DCT-II of the native SpeechT5 log-mel, frame by frame.

With all 80 coefficients the transform is a rotation, so it is exactly
invertible (DCT-III): a model that predicts or generates MFCC-80 keeps the
log-mel audio ceiling (HiFi-GAN receives ``mfcc_to_mel(c)``; oracle STOI
0.976, PESQ 3.80, the same as log-mel).  Truncation is what costs quality:
13 coefficients drop the pitch harmonics (STOI 0.715, PESQ 1.40; 40 keep
0.966 / 3.58), so nothing here truncates.

What does change against log-mel is the geometry of every loss: MFCC
coefficients are decorrelated and standardised one by one, so the envelope
(c0), the spectral tilt (c1, c2) and the harmonic fine structure (high
quefrencies) are weighted explicitly instead of through 80 correlated bins.

    mel_to_mfcc / mfcc_to_mel   (..., 80, T) <-> (..., 80, T), torch or numpy
    MFCCScaler                  per-coefficient standardisation fitted on train-fold audio
    mcd                         mel-cepstral distortion (dB) on speech frames
"""
from __future__ import annotations

import math

import numpy as np
import torch

COEFFICIENTS = 80
LOW = 13                      # the classic MFCC set: envelope, tilt and formant shape (c0 .. c12)
_DCT = {}


def dct_matrix(n: int = COEFFICIENTS) -> torch.Tensor:
    """Orthonormal DCT-II matrix D (n, n): c = D @ x, x = D.T @ c."""
    if n not in _DCT:
        k = torch.arange(n, dtype=torch.float64)[:, None]
        i = torch.arange(n, dtype=torch.float64)[None, :]
        d = torch.cos(math.pi / n * (i + .5) * k) * math.sqrt(2. / n)
        d[0] /= math.sqrt(2.)
        _DCT[n] = d.float()
    return _DCT[n]


def _apply(matrix: torch.Tensor, x):
    if isinstance(x, np.ndarray):
        return np.einsum('ij,...jt->...it', matrix.numpy(), x.astype(np.float32))
    return torch.einsum('ij,...jt->...it', matrix.to(x.device, x.dtype), x)


def mel_to_mfcc(mel):
    """(..., 80, T) log-mel -> (..., 80, T) MFCC-80 (coefficient axis replaces the bin axis)."""
    return _apply(dct_matrix(mel.shape[-2]), mel)


def mfcc_to_mel(mfcc):
    return _apply(dct_matrix(mfcc.shape[-2]).T, mfcc)


class MFCCScaler:
    """Clamp the log-mel at ``floor``, rotate to MFCC-80, standardise each coefficient.

    ``low`` / ``high`` are per-coefficient sampling bounds in standardised units
    (the 0.1 / 99.9 % quantiles of the training frames, widened by 1.5), used in
    place of the global mel clamp by the diffusion sampler.
    """
    kind = 'mfcc'

    def __init__(self, mean, scale, low, high, floor: float):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.scale = torch.as_tensor(scale, dtype=torch.float32)
        self.low = torch.as_tensor(low, dtype=torch.float32)
        self.high = torch.as_tensor(high, dtype=torch.float32)
        self.floor = float(floor)

    @classmethod
    def fit(cls, mel: torch.Tensor, floor: float):
        """``mel`` (N, 80, T) of train-fold audio."""
        c = mel_to_mfcc(mel.float().clamp_min(floor)).transpose(1, 2).reshape(-1, mel.shape[1])
        mean, scale = c.mean(0), c.std(0).clamp_min(1e-4)
        z = (c - mean) / scale
        sample = z[torch.randperm(len(z), generator=torch.Generator().manual_seed(0))[:200000]]
        low, high = torch.quantile(sample, .001, dim=0) - 1.5, torch.quantile(sample, .999, dim=0) + 1.5
        return cls(mean, scale, low, high, floor)

    def _shape(self, v, like):
        return v.to(like.device, like.dtype)[:, None]

    def encode(self, mel):
        c = mel_to_mfcc(mel.clamp_min(self.floor))
        return (c - self._shape(self.mean, c)) / self._shape(self.scale, c)

    def decode(self, z):
        return mfcc_to_mel(z * self._shape(self.scale, z) + self._shape(self.mean, z))

    def bounds(self, device):
        return self.low.to(device)[None, :, None], self.high.to(device)[None, :, None]

    def state(self):
        return dict(kind=self.kind, mean=self.mean.tolist(), scale=self.scale.tolist(), low=self.low.tolist(),
                    high=self.high.tolist(), floor=self.floor)


def standardised_mae(scaler: MFCCScaler, a, b, frames: int, coefficients=slice(None)) -> float:
    """Mean absolute difference of standardised MFCC-80 over the first ``frames`` frames."""
    za, zb = scaler.encode(a[..., :frames]), scaler.encode(b[..., :frames])
    return float((za[..., coefficients, :] - zb[..., coefficients, :]).abs().mean())


def mcd(reference: np.ndarray, degraded: np.ndarray, frames: int, coefficients=slice(1, None)) -> float:
    """Mel-cepstral distortion in dB between two log10-mels (80, T), c1 .. c79 by default.

    The cepstra are the orthonormal DCT of the natural-log mel, and the usual
    (10 / ln 10) * sqrt(2 * sum dc^2) is averaged over speech frames.  It is an
    internal scale (not SPTK mel-cepstra), consistent across every condition.
    """
    a = mel_to_mfcc(np.asarray(reference, np.float32)[:, :frames] * math.log(10.))
    b = mel_to_mfcc(np.asarray(degraded, np.float32)[:, :frames] * math.log(10.))
    diff = (a - b)[coefficients]
    return float(np.mean(10. / math.log(10.) * np.sqrt(2. * (diff ** 2).sum(0))))

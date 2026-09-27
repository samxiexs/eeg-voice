"""Unit checks for app/mfcc.py and the MFCC data space of the diffusion decoder (no data needed).

* the 80-point DCT is orthonormal, so MFCC-80 -> log-mel is exact (no ceiling loss);
* the scaler round-trips above the floor and standardises every coefficient;
* checkpoints written before the MFCC space still load as the mel scaler;
* MCD is zero for identical inputs and grows with a spectral change;
* DDIM with the exact v still returns x0 under per-coefficient clamp bounds.
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src'))
_cache_name = os.environ.get('ALIGNED_TARGET_CACHE_NAME')
import generative_recovery as gr
import mfcc
if _cache_name is None:
    os.environ.pop('ALIGNED_TARGET_CACHE_NAME', None)
else:
    os.environ['ALIGNED_TARGET_CACHE_NAME'] = _cache_name


def fake_mel(n=6, frames=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.linspace(-2., -6., 80)[None, :, None]
    return base + .8 * torch.randn(n, 80, frames, generator=g)


class TransformTest(unittest.TestCase):
    def test_orthonormal_and_exact_inverse(self):
        d = mfcc.dct_matrix()
        self.assertLess(float((d @ d.T - torch.eye(80)).abs().max()), 1e-5)
        mel = fake_mel()
        self.assertLess(float((mfcc.mfcc_to_mel(mfcc.mel_to_mfcc(mel)) - mel).abs().max()), 1e-4)
        numpy_mel = mel[0].numpy()
        self.assertTrue(np.allclose(mfcc.mfcc_to_mel(mfcc.mel_to_mfcc(numpy_mel)), numpy_mel, atol=1e-4))

    def test_c0_is_scaled_mean(self):
        mel = fake_mel()
        c = mfcc.mel_to_mfcc(mel)
        self.assertTrue(torch.allclose(c[:, 0], mel.mean(1) * np.sqrt(80), atol=1e-4))


class ScalerTest(unittest.TestCase):
    def test_round_trip_and_standardisation(self):
        mel = fake_mel(n=20)
        scaler = mfcc.MFCCScaler.fit(mel, floor=-7.)
        z = scaler.encode(mel)
        flat = z.transpose(1, 2).reshape(-1, 80)
        self.assertLess(float(flat.mean(0).abs().max()), 1e-3)
        self.assertLess(float((flat.std(0) - 1).abs().max()), 1e-2)
        self.assertLess(float((scaler.decode(z) - mel.clamp_min(-7.)).abs().max()), 1e-4)
        low, high = scaler.bounds('cpu')
        inside = ((z >= low) & (z <= high)).float().mean()
        self.assertGreater(float(inside), .9999)             # the sampler clamp almost never touches real frames

    def test_state_round_trip_and_legacy_checkpoints(self):
        scaler = gr.make_scaler('mfcc', fake_mel(n=8))
        again = gr.scaler_from_state(scaler.state())
        mel = fake_mel(n=2, seed=3)
        self.assertTrue(torch.allclose(again.encode(mel), scaler.encode(mel)))
        legacy_state = dict(mean=-3.7, scale=2.4)            # a pre-MFCC checkpoint signature
        self.assertIsInstance(gr.scaler_from_state(legacy_state), gr.MelScaler)
        self.assertEqual(gr.scaler_from_state(legacy_state).bounds('cpu'), (-2., 3.))


class MetricTest(unittest.TestCase):
    def test_mcd(self):
        mel = fake_mel(n=1)[0].numpy()
        self.assertAlmostEqual(mfcc.mcd(mel, mel, 40), 0., places=5)
        tilted = mel + np.linspace(0, 1, 80)[:, None]
        louder = mel + 1.                                   # only c0 changes: invisible to c1..c79
        self.assertGreater(mfcc.mcd(mel, tilted, 40), 1.)
        self.assertAlmostEqual(mfcc.mcd(mel, louder, 40), 0., places=3)


class SamplerTest(unittest.TestCase):
    def test_exact_v_returns_x0_with_coefficient_bounds(self):
        torch.manual_seed(0)
        x0 = torch.randn(2, 80, 16) * .5
        model = ExactV(x0)
        low, high = torch.full((1, 80, 1), -4.), torch.full((1, 80, 1), 4.)
        c = torch.zeros(2, 16, 16)
        out = model.sample(c, null=c, steps=20, guidance=1., clamp=(low, high))
        self.assertLess(float((out - x0).abs().max()), 1e-3)


class ExactV(gr.MelDiffusion):
    def __init__(self, x0):
        super().__init__(eeg_dimension=4, hidden=16, blocks=2, heads=2, timesteps=100, frames=x0.shape[-1])
        self.x0 = x0

    def forward(self, x, t, c):
        ab = self.alpha_bar[t][:, None, None]
        noise = (x - ab.sqrt() * self.x0[:len(x)]) / (1 - ab).sqrt().clamp_min(1e-6)
        return ab.sqrt() * noise - (1 - ab).sqrt() * self.x0[:len(x)]


if __name__ == '__main__':
    unittest.main()

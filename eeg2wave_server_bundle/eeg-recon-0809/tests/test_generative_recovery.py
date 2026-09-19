"""Unit checks for the diffusion decoder in app/generative_recovery.py (no data needed).

* the v-parameterisation identities: DDIM with the exact v returns x0, whatever the noise;
* adaLN-zero initialisation: the untrained network predicts v = 0, so the loss at
  initialisation is E[v^2] = 1 (a sanity anchor for the training log);
* classifier-free guidance at scale 1 must equal the conditional prediction exactly;
* mel normalisation round-trips above the floor, and the silence rule only fires on
  frames the generator left at the floor.
"""
import os
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src'))
_cache_name = os.environ.get('ALIGNED_TARGET_CACHE_NAME')
import generative_recovery as gr
if _cache_name is None:
    os.environ.pop('ALIGNED_TARGET_CACHE_NAME', None)
else:
    os.environ['ALIGNED_TARGET_CACHE_NAME'] = _cache_name


class ExactV(gr.MelDiffusion):
    """A denoiser that knows x0: returns the exact v for the given noisy input."""
    def __init__(self, x0):
        super().__init__(eeg_dimension=4, hidden=16, blocks=2, heads=2, timesteps=100, frames=x0.shape[-1])
        self.x0 = x0

    def forward(self, x, t, c):
        ab = self.alpha_bar[t][:, None, None]
        eps = (x - ab.sqrt() * self.x0) / (1 - ab).sqrt().clamp_min(1e-6)
        return ab.sqrt() * eps - (1 - ab).sqrt() * self.x0


class DiffusionTests(unittest.TestCase):
    def test_ddim_recovers_x0_with_exact_v(self):
        torch.manual_seed(0)
        x0 = torch.randn(2, gr.MEL_BINS, 20).clamp(-1.5, 1.5)
        model = ExactV(x0)
        c = torch.zeros(2, 20, 16)
        out = model.sample(c, null=c, steps=10, guidance=1., noise=torch.randn(2, gr.MEL_BINS, 20))
        self.assertLess(float((out - x0).abs().max()), 1e-3)

    def test_initial_loss_is_unit_variance_of_v(self):
        torch.manual_seed(0)
        model = gr.MelDiffusion(eeg_dimension=4, hidden=16, blocks=2, heads=2, timesteps=50, frames=12)
        x0 = torch.randn(64, gr.MEL_BINS, 12)
        c = model.condition('eeg', torch.randn(64, 4, 12))
        with torch.no_grad():
            self.assertEqual(float(model(x0, torch.full((64,), 10), c).abs().max()), 0.)
            losses = [float(model.loss(x0, c)) for _ in range(5)]
        self.assertAlmostEqual(sum(losses) / len(losses), 1., delta=.1)

    def test_guidance_one_equals_conditional(self):
        torch.manual_seed(0)
        model = gr.MelDiffusion(eeg_dimension=4, hidden=16, blocks=2, heads=2, timesteps=50, frames=12).eval()
        for p in model.parameters():
            p.data.normal_(0, .1)
        c = model.condition('eeg', torch.randn(3, 4, 12)); null = model.condition('null', torch.zeros(3, 1))
        noise = torch.randn(3, gr.MEL_BINS, 12)
        with torch.no_grad():
            a = model.sample(c, null=null, steps=5, guidance=1., noise=noise)
            b = model.sample(c, null=null, steps=5, guidance=1. + 1e-9, noise=noise)  # goes through the CFG branch
        self.assertLess(float((a - b).abs().max()), 1e-4)

    def test_scaler_and_silence_rule(self):
        mel = torch.tensor([[[-10., -6.5, -2., 0.]] * gr.MEL_BINS])
        scaler = gr.MelScaler.fit(mel)
        back = scaler.decode(scaler.encode(mel))
        self.assertTrue(torch.allclose(back[0, :, 2:], mel[0, :, 2:], atol=1e-5))
        self.assertTrue(torch.allclose(back[0, :, 0], torch.full((gr.MEL_BINS,), gr.MEL_FLOOR), atol=1e-5))
        silenced = gr.silence_tail(back)
        self.assertTrue(bool((silenced[0, :, :2] == gr.SILENCE_MEL).all()))
        self.assertTrue(torch.allclose(silenced[0, :, 2:], mel[0, :, 2:], atol=1e-5))

    def test_sharpness_prefers_structure_over_blur(self):
        torch.manual_seed(0)
        sharp = (torch.randn(gr.MEL_BINS, 40) * 2).numpy()
        blur = torch.nn.functional.avg_pool2d(torch.from_numpy(sharp)[None, None], 5, stride=1, padding=2)[0, 0].numpy()
        a, b = gr.sharpness(sharp, 40), gr.sharpness(blur, 40)
        self.assertGreater(a['spectral_contrast'], b['spectral_contrast'])
        self.assertGreater(a['frame_flux'], b['frame_flux'])


if __name__ == '__main__':
    unittest.main()

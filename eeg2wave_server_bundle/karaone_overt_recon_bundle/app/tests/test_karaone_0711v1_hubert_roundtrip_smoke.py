from __future__ import annotations

import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as error:  # pragma: no cover
    raise unittest.SkipTest(f"round-trip dependencies unavailable: {error}")

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.karaone_0711v1.hubert_roundtrip import HubertRoundTripConfig, HubertToEncodecDecoder, per_example_latent_metrics


class HubertRoundTripSmokeTest(unittest.TestCase):
    def test_roundtrip_decoder_shape_and_metrics(self) -> None:
        cfg = HubertRoundTripConfig(source_dim=12, source_steps=5, latent_dim=7, latent_steps=9, d_model=16, heads=4, encoder_layers=1, refiner_layers=1)
        model = HubertToEncodecDecoder(cfg)
        output = model(torch.randn(3, 5, 12))
        self.assertEqual(output.shape, (3, 9, 7))
        metrics = per_example_latent_metrics(output, torch.randn(3, 9, 7))
        self.assertEqual(metrics["latent_mse"].shape, (3,))
        self.assertEqual(metrics["latent_cosine"].shape, (3,))
        self.assertTrue(torch.isfinite(metrics["latent_mse"]).all())

    def test_roundtrip_decoder_rejects_mismatched_sequence_shape(self) -> None:
        cfg = HubertRoundTripConfig(source_dim=4, source_steps=3, latent_dim=2, latent_steps=5, d_model=8, heads=2, encoder_layers=1, refiner_layers=1)
        model = HubertToEncodecDecoder(cfg)
        with self.assertRaises(ValueError):
            model(torch.randn(1, 4, 4))


if __name__ == "__main__":
    unittest.main()

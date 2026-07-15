from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.karaone_0715.data import robust_baseline_normalise
from src.karaone_0715.eval import make_eeg_gate
from src.karaone_0715.losses import code_cross_entropy, condition_alignment_loss
from src.karaone_0715.model import (
    AudioCodeAutoencoder,
    AudioCodeModelConfig,
    EEGConditionEncoder,
    EEGModelConfig,
    random_code_mask,
)


class KaraOne0715SmokeTest(unittest.TestCase):
    def test_audio_code_autoencoder_and_maskgit_shapes(self) -> None:
        cfg = AudioCodeModelConfig(
            codebooks=2,
            code_steps=9,
            vocab_size=16,
            num_labels=3,
            d_model=24,
            condition_steps=3,
            encoder_layers=1,
            decoder_layers=1,
            heads=3,
            dropout=0.0,
        )
        model = AudioCodeAutoencoder(cfg)
        codes = torch.randint(0, cfg.vocab_size, (2, cfg.codebooks, cfg.code_steps))
        mask = random_code_mask(codes, min_ratio=0.5, max_ratio=0.8, full_mask_probability=0.0)
        labels = torch.nn.functional.one_hot(torch.tensor([0, 2]), num_classes=cfg.num_labels).float()
        output = model(codes, mask, labels)
        self.assertEqual(output["condition"].shape, (2, cfg.condition_steps, cfg.d_model))
        self.assertEqual(output["code_logits"].shape, (2, cfg.codebooks, cfg.code_steps, cfg.vocab_size))
        losses = code_cross_entropy(output["code_logits"], codes, mask)
        self.assertTrue(torch.isfinite(losses["total"]))
        generated = model.decoder.generate(output["condition"], labels, steps=3)
        self.assertEqual(generated.shape, codes.shape)
        self.assertGreaterEqual(int(generated.min()), 0)
        self.assertLess(int(generated.max()), cfg.vocab_size)

    def test_eeg_encoder_alignment_shapes(self) -> None:
        cfg = EEGModelConfig(
            channels=4,
            eeg_len=64,
            d_model=24,
            condition_steps=3,
            code_steps=9,
            num_labels=3,
            num_train_subjects=2,
            transformer_layers=1,
            heads=3,
            dropout=0.0,
            temporal_kernels=(3, 5, 7),
            stem_stride=2,
        )
        model = EEGConditionEncoder(cfg)
        output = model(torch.randn(2, cfg.channels, cfg.eeg_len), torch.tensor([64, 51]), subject_adversary_strength=0.1)
        self.assertEqual(output["condition"].shape, (2, cfg.condition_steps, cfg.d_model))
        self.assertEqual(output["label_logits"].shape, (2, cfg.num_labels))
        self.assertEqual(output["envelope_logits"].shape, (2, cfg.code_steps))
        self.assertEqual(output["subject_logits"].shape, (2, cfg.num_train_subjects))
        alignment = condition_alignment_loss(output["condition"], torch.randn_like(output["condition"]))
        self.assertTrue(torch.isfinite(alignment["total"]))

    def test_baseline_normalization_and_gate(self) -> None:
        rng = np.random.default_rng(15)
        clearing = rng.normal(3.0, 2.0, size=(4, 100)).astype(np.float32)
        overt = rng.normal(3.0, 2.0, size=(4, 50)).astype(np.float32)
        normalized = robust_baseline_normalise(overt, clearing)
        self.assertEqual(normalized.shape, overt.shape)
        self.assertTrue(np.isfinite(normalized).all())
        labels = np.repeat(np.arange(3), 5)
        prediction = labels.copy()
        condition = np.eye(15, dtype=np.float32)
        gate = make_eeg_gate(
            labels=labels,
            predictions=prediction,
            eeg_condition=condition,
            audio_condition=condition,
            coarse_code_accuracy=0.2,
            label_only_coarse_code_accuracy=0.1,
            min_balanced_accuracy=0.2,
            chance_accuracy=1.0 / 3.0,
            bootstrap_samples=50,
        )
        self.assertTrue(gate.passed)


if __name__ == "__main__":
    unittest.main()

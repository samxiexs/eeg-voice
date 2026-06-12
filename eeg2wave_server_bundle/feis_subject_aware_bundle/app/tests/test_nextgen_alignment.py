from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.alignment_losses import compute_alignment_losses  # noqa: E402
from src.alignment_model import EEGSpeechAlignmentModel  # noqa: E402
from src.dataset import FEISProtocolDataset  # noqa: E402
from src.phonemes import build_phoneme_vocab, encode_label_phonemes  # noqa: E402


class NextGenAlignmentTests(unittest.TestCase):
    def test_phoneme_encoding_for_feis_labels(self):
        vocab = build_phoneme_vocab(["fleece", "ng", "thought", "trap"])
        ids, mask = encode_label_phonemes("fleece", vocab)
        self.assertEqual(int(mask.sum()), 4)
        self.assertEqual(ids.shape[0], vocab.max_steps)
        self.assertIn("IY", vocab.token_to_id)
        thought_ids, thought_mask = encode_label_phonemes("thought", vocab)
        self.assertEqual(int(thought_mask.sum()), 3)
        self.assertEqual(thought_ids[0], vocab.token_to_id["TH"])

    def test_alignment_model_codec_and_phoneme_heads_shape(self):
        model = EEGSpeechAlignmentModel(
            n_channels_eeg=14,
            hidden_dim=16,
            speech_embedding_dim=128,
            prosody_dim=0,
            num_labels=16,
            latent_dim=32,
            target_steps=16,
            use_label_head=False,
            use_codec_scale_head=True,
            use_phoneme_head=True,
            num_phoneme_tokens=20,
            phoneme_steps=4,
        )
        outputs = model(torch.randn(2, 14, 512))
        self.assertEqual(tuple(outputs["speech_sequence"].shape), (2, 16, 128))
        self.assertEqual(tuple(outputs["codec_log_rms"].shape), (2,))
        self.assertEqual(tuple(outputs["phoneme_logits"].shape), (2, 4, 20))

    def test_auxiliary_losses_are_reported(self):
        pred = torch.randn(3, 5, 8)
        target = torch.randn(3, 5, 8)
        losses = compute_alignment_losses(
            pred_sequence=pred,
            target_sequence=target,
            pred_codec_log_rms=torch.zeros(3),
            target_log_rms=torch.ones(3) * -3.0,
            lambda_codec_scale=0.2,
            phoneme_logits=torch.randn(3, 2, 6),
            phoneme_ids=torch.tensor([[1, 2], [2, 3], [3, 4]]),
            phoneme_mask=torch.ones(3, 2),
            lambda_phoneme=0.2,
        )
        self.assertIn("codec_scale_loss", losses)
        self.assertIn("phoneme_loss", losses)
        self.assertGreater(float(losses["codec_scale_loss"]), 0.0)
        self.assertGreater(float(losses["phoneme_loss"]), 0.0)

    def test_encodec_cache_targets_are_normalized_in_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "feis"
            (root / "subjects").mkdir(parents=True)
            (root / "audio").mkdir()
            wavfile.write(str(root / "audio" / "01_f.wav"), 16000, np.zeros(16000, dtype=np.int16))
            with (root / "segments.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "subject_id",
                        "segment_stage",
                        "trial_index",
                        "label",
                        "audio_path",
                        "is_clean_subject",
                        "eeg_valid_num_samples",
                    ],
                )
                writer.writeheader()
                for trial_idx in range(3):
                    writer.writerow(
                        {
                            "subject_id": "01",
                            "segment_stage": "thinking",
                            "trial_index": trial_idx,
                            "label": "f",
                            "audio_path": "audio/01_f.wav",
                            "is_clean_subject": "true",
                            "eeg_valid_num_samples": "512",
                        }
                    )
            np.savez(
                root / "subjects" / "01.npz",
                stage__thinking=np.zeros((3, 14, 512), dtype=np.float32),
                trial_indices=np.asarray([0, 1, 2], dtype=np.int32),
                labels=np.asarray(["f", "f", "f"]),
                audio_relpaths=np.asarray(["audio/01_f.wav"] * 3),
                channel_names=np.asarray([f"ch{i}" for i in range(14)]),
            )
            raw = np.asarray([[[2.0, 4.0], [4.0, 8.0]]], dtype=np.float32)
            cache_path = Path(tmp) / "targets.npz"
            np.savez(
                cache_path,
                template_ids=np.asarray(["01:f"]),
                subject_ids=np.asarray(["01"]),
                labels=np.asarray(["f"]),
                audio_paths=np.asarray(["audio/01_f.wav"]),
                speech_embeddings=raw.mean(axis=1),
                target_sequences=raw,
                target_masks=np.ones((1, 2), dtype=np.float32),
                target_summaries=raw.mean(axis=1),
                prosody_targets=np.zeros((1, 4), dtype=np.float32),
                feature_backend=np.asarray(["encodec_latent"]),
                target_kind=np.asarray("encodec_latent"),
                target_mean=np.asarray([3.0, 6.0], dtype=np.float32),
                target_std=np.asarray([1.0, 2.0], dtype=np.float32),
                target_rms=np.asarray([0.08], dtype=np.float32),
                target_log_rms=np.asarray([np.log(0.08)], dtype=np.float32),
            )
            dataset = FEISProtocolDataset(
                data_root=root,
                protocol="S",
                split="train",
                subject_id="01",
                target_cache_path=cache_path,
                require_targets=True,
            )
            target = dataset.get_template_target("01:f")
            np.testing.assert_allclose(target["target_sequence"], [[-1.0, -1.0], [1.0, 1.0]], atol=1e-6)
            self.assertTrue(dataset.target_cache["targets_are_normalized"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
except ModuleNotFoundError as error:  # pragma: no cover - dependency bootstrap case
    raise unittest.SkipTest(f"0711v1 dependencies unavailable: {error}")

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.karaone_0711v1.data import FitAudit, SplitManifest, TopographicProjector, TrialRecord
from src.karaone_0711v1.losses import multi_positive_clip_loss, symmetric_view_contrastive
from src.karaone_0711v1.model import ConditionalFlowDecoder, EEG0711Config, EEG0711Encoder


class KaraOne0711SmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = SplitManifest("0711v1", ("S01", "S02"), "VAL", "TEST", "csv", 4)

    def test_train_only_fit_audit_rejects_heldout(self) -> None:
        train = TrialRecord("S01", 1, "/m/", "a.wav", "s01.npz")
        audit = FitAudit.from_records("unit", self.manifest, [train])
        self.assertEqual(audit.fit_split, "subject_train")
        heldout = TrialRecord("VAL", 2, "/m/", "b.wav", "val.npz")
        with self.assertRaises(ValueError):
            FitAudit.from_records("unit", self.manifest, [heldout])

    def test_topography_shape_and_padding(self) -> None:
        n = 62
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        positions = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
        projector = TopographicProjector([f"C{idx}" for idx in range(n)], positions)
        eeg = np.random.default_rng(1).normal(size=(n, 128)).astype(np.float32)
        topo = projector.transform(eeg, valid_len=96, time_bins=8)
        self.assertEqual(topo.shape, (5, 8, 9, 9))
        self.assertTrue(np.isfinite(topo).all())

    def test_multi_positive_clip_excludes_same_subject_false_negatives(self) -> None:
        eeg = torch.eye(4)
        audio = torch.eye(4)
        labels = torch.tensor([0, 0, 1, 1])
        loss = multi_positive_clip_loss(eeg, audio, labels, ["A", "A", "B", "C"])
        self.assertTrue(torch.isfinite(loss["total"]))
        self.assertEqual(loss["per_example"].shape, (4,))

    def test_encoder_and_flow_have_no_subject_input(self) -> None:
        cfg = EEG0711Config(channels=8, d_model=32, layers=1, heads=4, embed_dim=16, semantic_steps=6, semantic_vocab=7)
        encoder = EEG0711Encoder(cfg)
        eeg = torch.randn(2, 8, 128)
        valid = torch.tensor([128, 96])
        topo = torch.randn(2, 5, 6, 9, 9)
        output = encoder(eeg, valid, topo)
        self.assertEqual(output["eeg_embed"].shape, (2, 16))
        self.assertEqual(output["token_logits"].shape, (2, 6, 7))
        self.assertTrue(torch.isfinite(symmetric_view_contrastive(output["raw_embed"], output["topo_embed"])["total"]))
        flow = ConditionalFlowDecoder(latent_dim=10, eeg_dim=32, d_model=32, heads=4, layers=1)
        velocity = flow(torch.randn(2, 9, 10), torch.rand(2), output["tokens"], output["pred_onset_sec"], output["pred_duration_sec"], output["pred_active_logit"])
        self.assertEqual(velocity.shape, (2, 9, 10))

    def test_no_retired_module_import(self) -> None:
        paths = list((ROOT / "src" / "karaone_0711v1").glob("*.py")) + [ROOT / "scripts" / "train_karaone_0711v1.py"]
        text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        self.assertNotIn("karaone_v10", text)
        self.assertNotIn("karaone_v11", text)
        self.assertNotIn("karaone_v12", text)


if __name__ == "__main__":
    unittest.main()

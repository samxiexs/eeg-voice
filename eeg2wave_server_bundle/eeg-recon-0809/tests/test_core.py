"""Fast synthetic tests of the pieces every result depends on (no dataset needed).

    python -m unittest discover -s tests
"""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegspeech.data import ListenSource, TrialSource, collate, subject_folds      # noqa: E402
from eegspeech.losses import match_mismatch, pair_infonce, supcon                  # noqa: E402
from eegspeech.evaluation import few_shot                                          # noqa: E402
from eegspeech.model import Model                                                  # noqa: E402
from eegspeech.signal import BIOSEMI64, positions, speech_features, whitening       # noqa: E402
from eegspeech.store import Store, StoreWriter                                     # noqa: E402


def toy_store(path, rate=128., subjects=3):
    """Trials of 2 items in 2 modalities plus one 60 s listening segment per subject."""
    rng = np.random.default_rng(0)
    xyz, _ = positions(BIOSEMI64[:16], 'biosemi64')
    writer = StoreWriter(path, name='toy', rate=rate, band=[.5, 45], reference='average', unit='uV')
    writer.add_stimulus('story', rng.standard_normal((18, 64 * 60)).astype(np.float32), [f'r{i}' for i in range(18)])
    for s in range(subjects):
        segments = [dict(eeg=rng.standard_normal((16, 256)), valid=np.ones(16, bool), modality=m, item=f'w{k}')
                    for m in ('overt', 'imagine') for k in (0, 1) for _ in range(6)]
        segments.append(dict(eeg=rng.standard_normal((16, int(60 * rate))), valid=np.ones(16, bool), modality='listen',
                             stimulus='story', offset=-.5))
        writer.add_subject(f'sub-{s}', BIOSEMI64[:16], xyz, segments)
    writer.close()
    return Store(path)


class StoreTest(unittest.TestCase):
    def test_round_trip_and_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = toy_store(Path(tmp) / 'toy.h5')
            self.assertEqual(len(store.subjects), 3)
            self.assertEqual(len(store.table), 3 * 25)
            self.assertEqual(store.items, ['w0', 'w1'])
            row = store.table.iloc[0]
            self.assertEqual(store.segment(row).shape, (16, 256))
            trials = TrialSource(store, store.subjects, window_s=1.5, modalities=['imagine'])
            self.assertTrue((trials.rows.modality_name == 'imagine').all())
            batch = collate(trials.sample(8), ('item',))
            self.assertEqual(tuple(batch.eeg.shape), (8, 16, 192))
            listen = ListenSource(store, store.subjects, window_s=5., mismatches=3)
            windows, pairs = listen.sample(6, partner_fraction=1.)
            self.assertEqual(windows[0]['features'].shape, (4, 18, 320))
            self.assertTrue(pairs and all(windows[a]['time'] == windows[p]['time'] for a, p in pairs))
            self.assertTrue(all(windows[a]['subject'] != windows[p]['subject'] for a, p in pairs))

    def test_alignment_whitens(self):
        rng = np.random.default_rng(1)
        mixing = rng.standard_normal((8, 8))
        x = mixing @ rng.standard_normal((8, 20000))
        w = whitening(x @ x.T / x.shape[1], shrink=0.)
        cov = (w @ x) @ (w @ x).T / x.shape[1]
        self.assertTrue(np.allclose(cov / cov[0, 0], np.eye(8), atol=.05))

    def test_folds_are_stable(self):
        a = subject_folds(['s1', 's2', 's3', 's4', 's5'], 5)
        b = subject_folds(['s1', 's2', 's3', 's4', 's5', 's6'], 5)
        self.assertEqual(len(set(a.values())), 5)
        self.assertEqual(a, subject_folds(list(reversed(list(a))), 5))
        self.assertTrue(all(isinstance(b[s], int) for s in b))

    def test_speech_features_rate(self):
        wave = np.sin(np.linspace(0, 2000, 3 * 16000)) * (np.arange(3 * 16000) % 8000 < 4000)
        features = speech_features(wave, 16000)
        self.assertEqual(features.shape[0], 18)
        self.assertLessEqual(abs(features.shape[1] - 3 * 64), 1)


class LossTest(unittest.TestCase):
    def test_match_mismatch_prefers_candidate_zero(self):
        eeg = torch.randn(4, 20, 8)
        speech = torch.randn(4, 5, 20, 8)
        speech[:, 0] = eeg
        loss, accuracy = match_mismatch(eeg, speech, torch.ones(4, 20, dtype=torch.bool))
        self.assertEqual(float(accuracy), 1.)

    def test_pair_infonce_identity(self):
        a = torch.randn(6, 10, 8)
        loss, accuracy = pair_infonce(a, a.clone(), torch.ones(6, 10, dtype=torch.bool))
        self.assertEqual(float(accuracy), 1.)

    def test_supcon_orders_clusters(self):
        z = torch.nn.functional.normalize(torch.tensor([[1., 0], [1., .1], [0, 1.], [.1, 1.]]), dim=-1)
        good = supcon(z, torch.tensor([0, 0, 1, 1]))
        bad = supcon(z, torch.tensor([0, 1, 0, 1]))
        self.assertLess(float(good), float(bad))
        cross = supcon(z, torch.tensor([0, 0, 1, 1]), cross_only=torch.tensor([0, 1, 0, 1]))
        self.assertTrue(torch.isfinite(cross))

    def test_few_shot_separable(self):
        labels = np.repeat([0, 1, 2], 8)
        emb = torch.nn.functional.normalize(torch.eye(3)[labels] + .05 * torch.randn(24, 3), dim=-1)
        self.assertGreater(few_shot(emb, labels, 2), .9)


class ModelTest(unittest.TestCase):
    def test_montage_agnostic_forward_and_reload(self):
        model = Model({'toy': 3}, person_dim=8, virtual=16, width=32, dim=16)
        for channels in (16, 40):
            xyz, _ = positions(BIOSEMI64[:channels], 'biosemi64')
            eeg = torch.randn(2, channels, 256)
            frames, mask, pooled, z = model.encoder(eeg, torch.as_tensor(xyz).expand(2, -1, -1),
                                                    torch.ones(2, channels, dtype=torch.bool), torch.tensor([256, 200]),
                                                    person=torch.randn(2, 8))
            self.assertEqual(tuple(frames.shape), (2, 64, 32))
            self.assertEqual(int(mask[1].sum()), 50)
            self.assertTrue(torch.allclose(z.norm(dim=-1), torch.ones(2), atol=1e-5))
        speech = model.speech(torch.randn(2, 18, 128))
        self.assertEqual(tuple(speech.shape), (2, 64, 32))
        with tempfile.TemporaryDirectory() as tmp:
            model.save(Path(tmp) / 'm.pt')
            loaded, _ = Model.load(Path(tmp) / 'm.pt', heads={'other': 5})
            self.assertIn('other', loaded.heads)
            self.assertTrue(torch.equal(loaded.encoder.spatial.logits.weight, model.encoder.spatial.logits.weight))


if __name__ == '__main__':
    unittest.main()

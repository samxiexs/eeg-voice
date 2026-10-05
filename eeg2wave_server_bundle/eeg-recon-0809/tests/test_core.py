"""Fast synthetic tests of the pieces every result depends on (no dataset, no pretrained model).

    python -m unittest discover -s tests
"""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from eegspeech import audio, clip                                                   # noqa: E402
from eegspeech.data import ListenSource, TrialSource, collate, within_fold          # noqa: E402
from eegspeech.diffusion import EMA, MelDiffusion, MelScaler                        # noqa: E402
from eegspeech.losses import match_mismatch, supcon                                 # noqa: E402
from eegspeech.model import Model                                                   # noqa: E402
from eegspeech.signal import BIOSEMI64, positions, whitening                        # noqa: E402
from eegspeech.store import Store, StoreWriter                                      # noqa: E402
import reconstruct                                                                  # noqa: E402


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
            self.assertEqual(len(store.table), 3 * 25)
            self.assertEqual(store.items, ['w0', 'w1'])
            self.assertEqual(store.segment(store.table.iloc[0]).shape, (16, 256))
            trials = TrialSource(store, store.subjects, window_s=1.5, modalities=['imagine'])
            self.assertTrue((trials.rows.modality_name == 'imagine').all())
            self.assertEqual(tuple(collate(trials.sample(8), ('item',)).eeg.shape), (8, 16, 192))
            listen = ListenSource(store, store.subjects, window_s=5., mismatches=3)
            windows, pairs = listen.sample(6, partner_fraction=1.)
            self.assertEqual(windows[0]['features'].shape, (4, 18, 320))
            self.assertTrue(pairs and all(windows[a]['subject'] != windows[p]['subject'] for a, p in pairs))

    def test_alignment_whitens(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((8, 8)) @ rng.standard_normal((8, 20000))
        w = whitening(x @ x.T / x.shape[1], shrink=0.)
        cov = (w @ x) @ (w @ x).T / x.shape[1]
        self.assertTrue(np.allclose(cov / cov[0, 0], np.eye(8), atol=.05))

    def test_missing_positions_are_masked(self):
        xyz, _ = positions(BIOSEMI64[:8], 'biosemi64')
        xyz[2] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            writer = StoreWriter(Path(tmp) / 'p.h5', name='p', rate=128., band=[.5, 45], reference='average', unit='uV')
            writer.add_subject('s', BIOSEMI64[:8], xyz, [dict(eeg=np.ones((8, 64)), item='a')])
            writer.close()
            store = Store(Path(tmp) / 'p.h5')
            self.assertTrue(np.isfinite(store.xyz('s')).all())
            self.assertFalse(store.file['subjects']['s']['valid'][0][2])
        frames, _, pooled, _ = Model({'p': 2}, virtual=8, width=16, dim=8).encoder(
            torch.randn(1, 8, 128), torch.as_tensor(xyz)[None], torch.ones(1, 8, dtype=torch.bool))
        self.assertTrue(torch.isfinite(frames).all() and torch.isfinite(pooled).all())


class DeepModelTest(unittest.TestCase):
    def test_losses(self):
        eeg = torch.randn(4, 20, 8)
        speech = torch.randn(4, 5, 20, 8)
        speech[:, 0] = eeg
        self.assertEqual(float(match_mismatch(eeg, speech, torch.ones(4, 20, dtype=torch.bool))[1]), 1.)
        z = torch.nn.functional.normalize(torch.tensor([[1., 0], [1., .1], [0, 1.], [.1, 1.]]), dim=-1)
        self.assertLess(float(supcon(z, torch.tensor([0, 0, 1, 1]))), float(supcon(z, torch.tensor([0, 1, 0, 1]))))

    def test_montage_agnostic_forward_and_reload(self):
        model = Model({'toy': 3}, person_dim=8, virtual=16, width=32, dim=16)
        for channels in (16, 40):
            xyz, _ = positions(BIOSEMI64[:channels], 'biosemi64')
            frames, mask, _, z = model.encoder(torch.randn(2, channels, 256), torch.as_tensor(xyz).expand(2, -1, -1),
                                               torch.ones(2, channels, dtype=torch.bool), torch.tensor([256, 200]),
                                               person=torch.randn(2, 8))
            self.assertEqual(tuple(frames.shape), (2, 64, 32))
            self.assertEqual(int(mask[1].sum()), 50)
            self.assertTrue(torch.allclose(z.norm(dim=-1), torch.ones(2), atol=1e-5))
        with tempfile.TemporaryDirectory() as tmp:
            model.save(Path(tmp) / 'm.pt')
            loaded, _ = Model.load(Path(tmp) / 'm.pt', heads={'other': 5})
            self.assertIn('other', loaded.heads)
            self.assertTrue(torch.equal(loaded.encoder.spatial.logits.weight, model.encoder.spatial.logits.weight))


class ReconstructionTest(unittest.TestCase):
    def test_personal_clip_learns_only_where_there_is_signal(self):
        rng = np.random.default_rng(0)
        anchor = clip.anchors(rng.standard_normal((4, 16)))
        self.assertEqual(anchor.shape, (4, 3))
        train, test = [], []
        for strength in (0., 1.):                                   # one person without signal, one with
            pattern, y = rng.standard_normal((4, 60)), rng.integers(0, 4, 300)
            x = (strength * pattern[y] + rng.standard_normal((300, 60))).astype(np.float32)
            train.append(clip.Person(x[:200], y[:200], np.ones(200)))
            test.append((x[200:], y[200:]))
        for settings in (dict(), dict(supcon_weight=.1, steps=150)):          # L-BFGS and Adam paths
            model = clip.train(train, anchor, weight_decay=1e-2, **settings)
            accuracy = [((model.embed(p, x) @ anchor.T).argmax(1) == y).mean() for p, (x, y) in enumerate(test)]
            self.assertLess(accuracy[0], .45)                                  # chance .25
            self.assertGreater(accuracy[1], .9)
        z = model.embed(1, test[1][0])
        np.testing.assert_array_equal(clip.posterior(z, anchor, np.inf).argmax(1), (z @ anchor.T).argmax(1))

    def test_diffusion_is_conditional_and_seeded(self):
        torch.manual_seed(0)
        model = MelDiffusion(3, bins=8, frames=16, hidden=32, blocks=2, heads=2, timesteps=50)
        x0, c, null = torch.randn(4, 8, 16), torch.randn(4, 3), torch.tensor([True, False, False, False])
        optimizer = torch.optim.Adam(model.parameters(), 1e-2)
        for _ in range(5):
            optimizer.zero_grad()
            model.loss(x0, c, null).backward()
            optimizer.step()
        x, t = torch.randn(1, 8, 16), torch.tensor([10])
        e = [model.embed(torch.full((1, 3), v), torch.tensor([False])) for v in (0., 1.)]
        self.assertFalse(torch.allclose(model(x, t, e[0]), model(x, t, e[1])))    # the condition reaches the output
        ema = EMA(model, .9)
        ema.update(model)
        a, b = (ema.model.sample(c, steps=5, generator=torch.Generator().manual_seed(0)) for _ in range(2))
        torch.testing.assert_close(a, b)
        mel = torch.full((1, 80, 10), -10.)
        mel[:, :, 3:7] = -2.
        scaler = MelScaler.fit(mel)
        torch.testing.assert_close(scaler.decode(scaler.encode(mel)), mel)        # floor frames -> digital silence

    def test_dtw_mcd(self):
        rng = np.random.default_rng(0)
        a, other = rng.standard_normal((30, 24)), rng.standard_normal((25, 24))
        d = audio.dtw_mcd([(a, a), (a, np.repeat(a, 2, axis=0)), (a, other)])
        np.testing.assert_allclose(d[:2], 0, atol=1e-6)                            # same sequence, any speed
        self.assertGreater(d[2], 1)

    def test_within_person_folds_and_wrong_trials(self):
        table = pd.DataFrame(dict(subject=np.repeat(['a', 'b'], 20),
                                  modality_name=np.tile(np.repeat(['imagine', 'overt'], 10), 2)))
        folds = [set(within_fold(table, 5, k).tolist()) for k in range(5)]
        self.assertEqual(sum(len(f) for f in folds), len(set().union(*folds)))     # disjoint
        self.assertEqual(set().union(*folds), set(table.index))
        self.assertEqual(sorted(folds[0]), [0, 1, 10, 11, 20, 21, 30, 31])          # contiguous per person and modality
        who = np.array(['a'] * 5 + ['b'] * 4)
        other = reconstruct.derangement(who, np.random.default_rng(0))
        self.assertTrue((other != np.arange(len(who))).all() and (who[other] == who).all())


if __name__ == '__main__':
    unittest.main()

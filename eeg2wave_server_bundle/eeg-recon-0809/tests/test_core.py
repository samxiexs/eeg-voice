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
from eegspeech import audio, clip, deep, features                                   # noqa: E402
from eegspeech.diffusion import EMA, MelDiffusion, MelScaler                        # noqa: E402
from eegspeech.features import blocks, within_fold                                  # noqa: E402
from eegspeech.signal import BIOSEMI64, positions, whitening                        # noqa: E402
from eegspeech.store import Store, StoreWriter                                      # noqa: E402
import reconstruct                                                                  # noqa: E402


def toy_store(path, rate=128., subjects=3):
    """Trials of 2 items in 2 modalities plus one 60 s rest segment per subject."""
    rng = np.random.default_rng(0)
    xyz, _ = positions(BIOSEMI64[:16], 'biosemi64')
    writer = StoreWriter(path, name='toy', rate=rate, band=[.5, 45], reference='average', unit='uV')
    for s in range(subjects):
        segments = [dict(eeg=rng.standard_normal((16, 256)), valid=np.ones(16, bool), modality=m, item=f'w{k}',
                         text=f'w{k}') for m in ('overt', 'imagine') for k in (0, 1) for _ in range(6)]
        segments.append(dict(eeg=rng.standard_normal((16, int(60 * rate))), valid=np.ones(16, bool), modality='rest'))
        writer.add_subject(f'sub-{s}', BIOSEMI64[:16], xyz, segments)
    writer.close()
    return Store(path)


class StoreTest(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = toy_store(Path(tmp) / 'toy.h5')
            self.assertEqual(len(store.table), 3 * 25)
            self.assertEqual(store.items, ['w0', 'w1'])
            self.assertEqual(store.modalities, ['overt', 'imagine', 'rest'])
            self.assertEqual(store.segment(store.table.iloc[0]).shape, (16, 256))
            self.assertEqual(store.table.text.iloc[0], 'w0')
            self.assertEqual(int(store.table.item.iloc[-1]), -1)

    def test_alignment_can_leave_out_held_trials(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = toy_store(Path(tmp) / 'toy.h5')
            rows = store.table[store.table.subject == 'sub-0'].index[:12]
            full, without = store.alignment('sub-0'), store.alignment('sub-0', exclude=rows)
            self.assertFalse(np.allclose(full, without))
            again = Store(Path(tmp) / 'toy.h5')
            again.table = again.table.drop(rows)                                    # as if never recorded
            np.testing.assert_allclose(again.alignment('sub-0'), without, rtol=1e-5)

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


class DeepEncoderTest(unittest.TestCase):
    def test_supcon_prefers_same_item_neighbours(self):
        z = torch.nn.functional.normalize(torch.tensor([[1., 0], [1., .1], [0, 1.], [.1, 1.]]), dim=-1)
        self.assertLess(float(clip.supcon(z, torch.tensor([0, 0, 1, 1]))), float(clip.supcon(z, torch.tensor([0, 1, 0, 1]))))

    def test_filter_bank_passes_its_band(self):
        bank = deep.sinc_bank(8, 33, 128)
        t = np.arange(512) / 128
        edges = np.geomspace(1, .45 * 128, 9)
        centre = np.sqrt(edges[5] * edges[6])                                  # inside filter 5
        y = np.stack([np.convolve(np.sin(2 * np.pi * centre * t), h, 'valid') for h in bank])
        self.assertEqual(int(np.argmax(y.std(1))), 5)

    def test_sentence_net_is_personal_and_learns(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        anchor = torch.nn.functional.normalize(torch.randn(6, 4), dim=-1)
        n, channels, samples = 96, 6, 160
        sentence = rng.integers(0, 6, n)
        x = rng.standard_normal((n, channels, samples)).astype(np.float32)
        t = np.arange(samples) / 64
        for i in range(n):                                     # 8 Hz power in channel 0 for the first half of items
            x[i, 0] += (sentence[i] < 3) * 3 * np.sin(2 * np.pi * 8 * t)
        data = x.astype(np.float16)
        person = np.repeat([0, 1], n // 2)
        model = deep.SentenceNet(channels, 2, samples, 4, 64, filters=8, depth=1, kernel=.25, pool=.5, stride=.25,
                                 dropout=0.)
        z = model(torch.from_numpy(x[:4]), torch.tensor([0, 0, 1, 1]))
        self.assertEqual(tuple(z.shape), (4, 4))
        torch.testing.assert_close(z.norm(dim=-1), torch.ones(4))
        with torch.no_grad():                                   # the persons' spatial filters are separate
            model.spatial[1].add_(1.)
        self.assertFalse(torch.allclose(model(torch.from_numpy(x[:1]), torch.tensor([0])),
                                        model(torch.from_numpy(x[:1]), torch.tensor([1]))))
        examples = deep.Examples(data, [np.eye(channels)] * 2, person, samples, torch.device('cpu'))
        a, p = examples(np.array([5, 1, 3]))
        torch.testing.assert_close(a[1], torch.from_numpy(data[1].astype(np.float32)))   # order kept, identity alignment
        self.assertEqual(p.tolist(), [0, 0, 0])
        cfg = dict(rate=64, crop=2.25, filters=8, depth=1, kernel=.25, pool=.5, stride=.25, dropout=0., batch=32,
                   lr=.01, weight_decay=0., epochs=15, patience=15)
        run = np.repeat(np.arange(8), n // 8)
        model, examples, scale, best = deep.fit(data, [np.eye(channels)] * 2, person, sentence, run,
                                                np.asarray(anchor), np.arange(64), np.arange(64, n), cfg,
                                                torch.device('cpu'), lambda m: None)
        self.assertGreater(best, .6)                                              # chance 0.5


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
        model = MelDiffusion(3, bins=8, frames=16, hidden=32, blocks=2, heads=2, timesteps=50, trait_dim=5)
        x0, c, null = torch.randn(4, 8, 16), torch.randn(4, 3), torch.tensor([True, False, False, False])
        traits = torch.randn(4, 5)
        optimizer = torch.optim.Adam(model.parameters(), 1e-2)
        for _ in range(5):
            optimizer.zero_grad()
            model.loss(x0, c, null, traits).backward()
            optimizer.step()
        x, t, no = torch.randn(1, 8, 16), torch.tensor([10]), torch.tensor([False])
        e = [model.embed(torch.full((1, 3), v), no, traits[:1]) for v in (0., 1.)]
        self.assertFalse(torch.allclose(model(x, t, e[0]), model(x, t, e[1])))    # the content reaches the output
        e = [model.embed(c[:1], no, traits[i:i + 1]) for i in (0, 1)]
        self.assertFalse(torch.allclose(model(x, t, e[0]), model(x, t, e[1])))    # and so does the voice
        ema = EMA(model, .9)
        ema.update(model)
        a, b = (ema.model.sample(c, traits=traits, steps=5, generator=torch.Generator().manual_seed(0)) for _ in range(2))
        torch.testing.assert_close(a, b)
        mel = torch.full((1, 80, 10), -10.)
        mel[:, :, 3:7] = -2.
        scaler = MelScaler.fit(mel)
        torch.testing.assert_close(scaler.decode(scaler.encode(mel)), mel)        # floor frames -> digital silence

    def test_band_power_and_its_time_course(self):
        rng = np.random.default_rng(3)
        bands = features.bands_for(256)
        x = rng.standard_normal((4, 768))
        x[0, 512:] *= 4                                                          # channel 0 louder at the end
        whole, windows = features.log_power(x, 256, bands, windows=6)
        self.assertEqual(whole.shape, (4 * len(bands),))
        self.assertEqual(windows.shape, (6, 4 * len(bands)))
        course = features.time_course(whole[None], windows[None], 3).reshape(3, len(bands), 4)
        self.assertTrue((course[2, 1:, 0] > course[0, 1:, 0] + 1).all())         # channel 0 rises in every band
                                                                                 # above 4 Hz (1-4 Hz: too slow for 1 s)
        high = features.band_columns(bands, whole.size, 55)
        self.assertEqual(len(high), 4 * sum(lo >= 55 for lo, _ in bands))
        self.assertEqual(high[0], 4 * [lo >= 55 for lo, _ in bands].index(True))

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
        who, episode = np.repeat(['a', 'b'], 24), np.repeat(np.arange(12), 4)       # 4 repetitions per cue
        other = reconstruct.derangement(who, np.random.default_rng(0), episode)
        self.assertTrue((episode[other] != episode).all() and (who[other] == who).all())

    def test_episode_folds_hold_out_whole_episodes(self):
        rng = np.random.default_rng(0)
        item = np.repeat(np.arange(5), 80)                                          # 20 cues x 4 repetitions per item
        episode = np.repeat(np.arange(100), 4)
        order = rng.permutation(400)                                                # released order != recording order
        table = pd.DataFrame(dict(subject='a', modality_name='imagine', item=item[order], episode=episode[order]))
        fold = blocks(table, 5)
        self.assertEqual(set(fold), set(range(5)))
        self.assertTrue((table.assign(fold=fold).groupby('episode').fold.nunique() == 1).all())   # never split
        counts = pd.crosstab(fold, table.item)
        self.assertTrue((counts.to_numpy() == 16).all())                            # class balance kept
        held = set(within_fold(table, 5, 2).tolist())
        self.assertEqual(held, set(np.flatnonzero(fold == 2).tolist()))

    def test_episodes_are_recovered_from_near_duplicates(self):
        rng = np.random.default_rng(1)
        state = rng.standard_normal((10, 40))                                       # one state per cue episode
        x = np.repeat(state, 4, axis=0) + .5 * rng.standard_normal((40, 40))
        order = rng.permutation(40)
        x = x[order] / np.linalg.norm(x[order], axis=1, keepdims=True)
        groups = features.capped_groups(x @ x.T, 4)
        truth = np.repeat(np.arange(10), 4)[order]
        self.assertTrue(all(len(set(truth[g])) == 1 and len(g) == 4 for g in groups))

    def test_run_folds_hold_out_whole_runs_in_run_order(self):
        table = pd.DataFrame(dict(subject='a', modality_name='imagine', item=0,
                                  block=np.repeat([10, 1, 2, 3, 4, 5, 6, 7, 8, 9], 6)))   # stored in name order
        fold = blocks(table, 5)
        self.assertTrue((table.assign(fold=fold).groupby('block').fold.nunique() == 1).all())
        self.assertEqual(sorted(table.block[fold == 0].unique()), [1, 2])                  # contiguous in run order
        self.assertEqual(sorted(table.block[fold == 4].unique()), [9, 10])

    def test_sentence_retrieval_learns_only_where_there_is_signal(self):
        rng = np.random.default_rng(4)
        anchor = rng.standard_normal((400, 8))
        anchor /= np.linalg.norm(anchor, axis=1, keepdims=True)
        sentence, run = rng.permutation(400), np.repeat(np.arange(10), 40)
        mixing = rng.standard_normal((8, 30))
        for strength, low, high in ((0., .4, .6), (.5, .9, 1.)):
            x = strength * anchor[sentence] @ mixing + rng.standard_normal((400, 30))
            train, test = run < 8, run >= 8
            for method in ('ridge', 'clip'):
                model = clip.train_sentences(x[train], sentence[train], run[train], anchor, method=method, steps=50)
                z = model.embed(x[test])
                score = np.mean(np.concatenate([clip.rank_percentile(z[t] @ anchor[c].T, p)
                                                for t, c, p in clip.run_sets(sentence[test], run[test])]))
                self.assertTrue(low < score < high, (strength, method, score))
        np.testing.assert_allclose(clip.rank_percentile(np.array([[3., 2., 1.], [1., 1., 1.]]), np.array([0, 2])), [1, .5])

    def test_drift_baseline_is_causal(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((30, 6)).astype(np.float32)
        run = np.repeat(['s1/imagine', 's1/overt'], 15)
        held = np.zeros(30, bool)
        held[5:10] = True
        prior = np.zeros((30, 6))
        base = features.local_baseline(x, run, held, prior, window=4)
        changed = x.copy()
        changed[12] += 100                                                          # a later trial
        changed[7] += 100                                                           # a held-out trial
        again = features.local_baseline(changed, run, held, prior, window=4)
        np.testing.assert_allclose(again[:8], base[:8])                             # nothing from the future
        np.testing.assert_allclose(again[10:12], base[10:12])                       # training rows skip held-out ones
        self.assertFalse(np.allclose(again[8], base[8]))                            # held-out rows follow the stream
        np.testing.assert_allclose(again[15:], base[15:])                           # runs are separate
        np.testing.assert_allclose(base[0], 0)                                      # first trial: the prior


if __name__ == '__main__':
    unittest.main()

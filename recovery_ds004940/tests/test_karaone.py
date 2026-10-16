import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'app'))
import prepare_karaone as prepare
import karaone as baselines


def write_epoch_inds(path, clearing, thinking, mixed):
    import scipy.io as sio
    def cell(pairs):
        array = np.empty((1, len(pairs)), dtype=object)
        for i, p in enumerate(pairs):
            array[0, i] = np.array(p, dtype=float)[None, :]
        return array
    sio.savemat(path, {'clearing_inds': cell(clearing), 'thinking_inds': cell(thinking), 'speaking_inds': cell(mixed)})


class KaraOneTests(unittest.TestCase):
    def test_stage_indices_are_located_by_position_and_corrupt_entries_are_flagged(self):
        clearing = [(100, 5100), (10100, 15100), (20100, 25100)]
        thinking = [(7200, 9900), (17200, 19900), (27200, 29900)]
        # trial 1's stimulus entry duplicates its clearing entry; trial 2 has a duplicated (identical) speaking entry
        mixed = [(5100, 7100), (9950, 10050), (10100, 15100), (19950, 20050), (25100, 27100), (29950, 30050), (29950, 30050)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'epoch_inds.mat'; write_epoch_inds(path, clearing, thinking, mixed)
            frame = prepare.read_stage_indices(path)
        self.assertEqual(frame.thinking_valid.tolist(), [True, True, True])
        self.assertEqual(frame.stimulus_valid.tolist(), [True, False, True])
        self.assertEqual(frame.speaking_valid.tolist(), [True, True, True])
        self.assertEqual(int(frame.stimulus_start[0]), 5100); self.assertEqual(int(frame.stimulus_start[1]), -1)
        self.assertEqual(int(frame.speaking_start[2]), 29950)
        self.assertEqual(prepare.to_target_rate(1000), 256)

    def test_biosemi_map_rows_are_partition_of_unity_and_use_located_channels_only(self):
        import mne
        template = mne.channels.make_standard_montage('standard_1005').get_positions()['ch_pos']
        names = ['CZ', 'PZ', 'FZ', 'C3', 'C4', 'O1', 'FP1', 'FP2', 'T7', 'T8', 'CB1']
        positions = {n: template[{k.upper(): k for k in template}[n]] for n in names if n != 'CB1'}
        matrix, targets, xyz = prepare.biosemi128_map(names, positions)
        self.assertEqual(matrix.shape, (128, len(names))); self.assertEqual(len(targets), 128)
        self.assertTrue(np.allclose(matrix.sum(1), 1., atol=1e-4))
        self.assertEqual(float(np.abs(matrix[:, names.index('CB1')]).sum()), 0.)
        self.assertEqual(names[int(np.argmax(matrix[targets.index('A1')]))], 'CZ')

    def test_folds_cover_every_trial_once(self):
        for maker in (baselines.block_folds, baselines.interleaved_folds):
            folds = maker(133, 6)
            self.assertEqual(sorted(np.concatenate(folds).tolist()), list(range(133)))
        block = baselines.block_folds(133, 6)
        self.assertTrue(all(np.all(np.diff(f) == 1) for f in block))

    def test_tangent_features_and_dual_lda_match_primal(self):
        rng = np.random.default_rng(0)
        windows = rng.standard_normal((6, 5, 200)).astype(np.float32)
        features = baselines.tangent_features(windows)
        self.assertEqual(features.shape, (6, 15))
        n, d, k = 60, 300, 4
        y = rng.integers(0, k, n); x = rng.standard_normal((n, d)) + np.eye(k)[y] @ rng.standard_normal((k, d))
        means = np.stack([x[y == c].mean(0) for c in range(k)]); centered = np.concatenate([x[y == c] - means[c] for c in range(k)])
        cov = centered.T @ centered / (n - k); cov = .8 * cov + .2 * np.trace(cov) / d * np.eye(d)
        primal = x @ np.linalg.solve(cov, means.T) - .5 * np.einsum('kd,dk->k', means, np.linalg.solve(cov, means.T)) + np.log(np.bincount(y) / n)
        dual = baselines.ShrinkageLDA(.2).fit(x, y).decision(x)
        self.assertLess(float(np.abs(primal - dual).max()), 1e-8)

    def test_block_cv_and_permutation_detect_signal_and_reject_noise(self):
        rng = np.random.default_rng(1)
        labels = np.array([f'p{i % 5}' for i in range(100)])
        signal = rng.standard_normal((100, 20)) + 2 * np.eye(5)[[i % 5 for i in range(100)]] @ rng.standard_normal((5, 20))
        noise = rng.standard_normal((100, 20))
        folds = baselines.block_folds(100, 5); lda = lambda: baselines.ShrinkageLDA(.2)
        strong, _, _ = baselines.cross_validate(signal, labels, folds, lda)
        weak, _, _ = baselines.cross_validate(noise, labels, folds, lda)
        self.assertGreater(strong, .8); self.assertLess(weak, .4)
        p_signal, _ = baselines.permutation_p_value(signal, labels, folds, lda, strong, 30, rng)
        p_noise, _ = baselines.permutation_p_value(noise, labels, folds, lda, weak, 30, rng)
        self.assertLess(p_signal, .05); self.assertGreater(p_noise, .05)

    def test_bandpass_and_bandpower_shapes(self):
        rng = np.random.default_rng(2)
        windows = rng.standard_normal((3, 4, 512)).astype(np.float32)
        low = baselines.bandpass_fft(windows, .5, 20.); self.assertEqual(low.shape, windows.shape)
        power = np.abs(np.fft.rfft(low, axis=-1)) ** 2; freqs = np.fft.rfftfreq(512, 1 / 256)
        self.assertLess(power[..., freqs > 25].sum(), 1e-6 * power.sum())
        self.assertEqual(baselines.bandpower_features(windows).shape, (3, 16))
        self.assertEqual(baselines.aux_features(windows, ['M1', 'VEO', 'EKG', 'EMG']).shape, (3, 5))


if __name__ == '__main__':
    unittest.main()

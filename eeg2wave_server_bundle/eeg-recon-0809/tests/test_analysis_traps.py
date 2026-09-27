"""Regression tests for the methodological traps found while running the analyses.

Each test encodes a mistake that produced a wrong number at least once:

* retrieval scored with the trial's own duration mask (a duration shortcut);
* forced-choice foils drawn per trial rather than per sentence (the foil is
  then usually the same sentence, so even an oracle looks bad);
* an evaluation whose 'no information' reference is the mean rather than the
  L1-optimal median template;
* pooling repeated presentations *after* the encoder instead of at its input,
  which measures the weaker of the two poolings and hides the signal;
* scoring against a prior-only prediction, or against the model's own zero-EEG
  output (which the model can let decay).
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src'))
_cache_name = os.environ.get('ALIGNED_TARGET_CACHE_NAME')
import audio_comparison as audio
import aligned_recovery_eval as evaluation
import envelope_decoder
# Some app modules default the cache name at import time; restore
# the caller's environment so other test modules still see the default
# targets.h5 (importing them later, inside a test, leaks it again).
if _cache_name is None:
    os.environ.pop('ALIGNED_TARGET_CACHE_NAME', None)
else:
    os.environ['ALIGNED_TARGET_CACHE_NAME'] = _cache_name


class FoilSelection(unittest.TestCase):
    def setUp(self):
        # 4 sentences x 3 trials; duration is a property of the sentence.
        self.keys = [f's{s}t{t}' for s in range(4) for t in range(3)]
        self.contents = {k: k.split('t')[0] for k in self.keys}
        self.lengths = {k: 16000 + 4000 * int(k[1]) for k in self.keys}

    def test_foil_is_never_the_same_sentence(self):
        for key in self.keys:
            for matched in (True, False):
                pool = audio.foil_pool(self.keys, self.lengths, self.contents, key, duration_matched=matched)
                self.assertTrue(pool, 'pool must not be empty')
                self.assertTrue(all(self.contents[k] != self.contents[key] for k in pool))

    def test_duration_matched_pool_keeps_one_trial_per_sentence_and_prefers_close_durations(self):
        pool = audio.foil_pool(self.keys, self.lengths, self.contents, 's0t0', duration_matched=True)
        self.assertEqual(len(pool), len({self.contents[k] for k in pool}), 'one trial per sentence')
        distances = [abs(self.lengths[k] - self.lengths['s0t0']) for k in pool]
        self.assertEqual(distances, sorted(distances), 'closest durations first')
        self.assertEqual(self.contents[pool[0]], 's1', 'the nearest-duration sentence comes first')

    def test_oracle_wins_its_own_sentence_under_the_forced_choice(self):
        """A near-copy of the original must be matched to it, or the harness is broken."""
        rate = audio.RATE; rng = np.random.default_rng(0)
        import soundfile as sf, tempfile
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); index = []; folders = {}; contents = {}
            for s in range(4):
                time = np.arange(int(1.5 * rate)) / rate
                base = (np.sin(2 * np.pi * (110 + 40 * s) * time) * (.5 + .5 * np.sin(2 * np.pi * (2 + s) * time))).astype(np.float32)
                for t in range(2):
                    key = f's{s}t{t}'; item = root / key; item.mkdir()
                    sf.write(item / 'original.wav', base, rate, subtype='FLOAT')
                    sf.write(item / 'oracle.wav', (base + .01 * rng.standard_normal(len(base))).astype(np.float32), rate, subtype='FLOAT')
                    sf.write(item / 'unrelated.wav', rng.standard_normal(len(base)).astype(np.float32) * .1, rate, subtype='FLOAT')
                    index.append(dict(folder=key, oracle_duration_frames=int(1.5 * rate) // 256))
                    folders[key] = item; contents[key] = f's{s}'
            result = audio.two_alternative(folders, index, ('oracle', 'unrelated'), contents, rng, repeats=2)
        self.assertGreater(result['oracle']['stoi']['accuracy'], .9)
        self.assertLess(result['unrelated']['stoi']['accuracy'], .8)


class ResidualMetrics(unittest.TestCase):
    """Removing a shared prior from both sides inflates the correlation.

    The first envelope run reported a "residual correlation" of 0.37 for real
    EEG and 0.46 for zero EEG, both far above the linear gate's 0.14, because
    both sides were z-scored and had the same prior subtracted: the shared
    -z(prior) term dominated. Partial correlation removes it properly.
    """
    def test_subtracting_a_shared_prior_inflates_but_partial_correlation_does_not(self):
        import torch
        from envelope_decoder import masked_correlation, partial_correlation, standardize
        torch.manual_seed(0)
        prior = torch.randn(8, 295)
        target = .5 * prior + .5 * torch.randn(8, 295)
        useless = torch.randn(8, 295)
        mask = torch.ones(8, 295, dtype=torch.bool)
        naive = masked_correlation(standardize(useless, mask) - standardize(prior, mask),
                                   standardize(target, mask) - standardize(prior, mask), mask).mean()
        self.assertGreater(float(naive), .15)                       # the trap
        self.assertLess(abs(float(partial_correlation(useless, target, prior, mask).mean())), .12)
        self.assertLess(abs(float(partial_correlation(prior, target, prior, mask).mean())), .05)

    def test_partial_correlation_keeps_information_beyond_the_prior(self):
        import torch
        from envelope_decoder import partial_correlation
        torch.manual_seed(1)
        prior = torch.randn(8, 295)
        extra = torch.randn(8, 295)
        target = .5 * prior + .5 * extra
        prediction = .5 * prior + .45 * extra + .1 * torch.randn(8, 295)
        mask = torch.ones(8, 295, dtype=torch.bool)
        self.assertGreater(float(partial_correlation(prediction, target, prior, mask).mean()), .8)


class TemplateAndRetrieval(unittest.TestCase):
    def test_template_uses_speaking_sentences_only_and_the_median(self):
        import h5py, tempfile
        rng = np.random.default_rng(2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'targets.h5'
            durations = [40, 80, 120]
            with h5py.File(path, 'w') as h5:
                group = h5.create_group('targets')
                for i, frames in enumerate(durations):
                    mel = np.full((80, 251), -10., dtype=np.float32)
                    mel[:, :frames] = 1. + i                       # distinct speech levels: median is the middle one
                    item = group.create_group(f'a{i}')
                    item.create_dataset('mel', data=mel)
                    item.attrs['source_samples_16k'] = (frames - 1) * 256
            dataset = type('D', (), dict(cache=path, frame=pd.DataFrame(dict(audio_key=[f'a{i}' for i in range(3)]))))()
            speech_mean, speech_median, full_mean, full_median = evaluation.train_templates(dataset)
        # frame 0: all three speak -> median 2.0 ; frame 100: only the longest speaks -> its own level 3.0
        self.assertAlmostEqual(float(speech_median[0, 0]), 2., places=5)
        self.assertAlmostEqual(float(speech_median[0, 100]), 3., places=5)
        self.assertLess(float(full_median[0, 100]), -5.)           # the full-window template predicts padding silence
        self.assertGreater(float(speech_mean[0, 0]), 1.9)

    def test_fixed_retrieval_window_is_shorter_than_the_shortest_sentence(self):
        manifest = ROOT / 'artifacts/aligned_speech_local_v1/manifest.csv'
        if not manifest.exists():
            self.skipTest('manifest not installed')
        shortest = pd.read_csv(manifest, keep_default_na=False).stimulus_duration_seconds.astype(float).min()
        self.assertLess(evaluation.FIXED_WINDOW_S, shortest,
                        'the oracle-free retrieval window must fit inside every sentence')


class PoolingAndPriorTraps(unittest.TestCase):
    """Pooling presentations and scoring against priors."""

    def test_a_prior_only_envelope_prediction_discriminates_no_candidate(self):
        # The trap the partial correlation exists to close: scoring candidates by
        # raw correlation with a prediction that is only the prior ranks them by
        # how prior-like they are, identically for every trial.
        import torch
        times = np.linspace(0, 1, 64)
        prior = torch.from_numpy(np.sin(2 * np.pi * times)).float()[None]
        rng = np.random.default_rng(3)
        # Candidates differ in how prior-like they are, which is exactly what a
        # raw correlation would rank them by.
        candidates = torch.from_numpy(np.stack([
            weight * np.sin(2 * np.pi * times) + rng.normal(size=64)
            for weight in (.2, .6, 1., 1.6, 2.4)])).float()
        mask = torch.ones(5, 64, dtype=torch.bool)
        expanded = prior.expand(5, -1)
        partial = envelope_decoder.partial_correlation(expanded, candidates, expanded, mask)
        self.assertLess(float(partial.abs().max()), 1e-2)
        raw = envelope_decoder.masked_correlation(expanded, candidates, mask)
        self.assertGreater(float(raw.max() - raw.min()), .2)

    def test_a_gain_measured_against_your_own_no_eeg_output_is_gameable(self):
        """The band-split run scored +0.101 by letting its zero-EEG output rot."""
        rot = dict(correct=dict(raw=.4682), zero=dict(raw=.3671), prior_raw=.4374)
        honest = dict(correct=dict(raw=.4638), zero=dict(raw=.4345), prior_raw=.4374)
        gain = lambda r: r['correct']['raw'] - r['zero']['raw']
        over_prior = lambda r: r['correct']['raw'] - r['prior_raw']
        self.assertGreater(gain(rot), 3 * gain(honest))          # looks 3x better ...
        self.assertLess(over_prior(rot) - over_prior(honest), .005)   # ... but is not
        self.assertGreater(rot['prior_raw'] - rot['zero']['raw'], .05)
        self.assertLess(honest['prior_raw'] - honest['zero']['raw'], .01)

    def test_input_pooling_and_output_pooling_are_not_the_same_measurement(self):
        # A nonlinear encoder: averaging its inputs raises SNR, averaging its
        # outputs does not.  Output pooling is why pooled sentence identification
        # once looked dead at 4.9% where input pooling gave 14.6%.
        rng = np.random.default_rng(11)
        signal = rng.normal(size=32)
        trials = np.stack([signal + 2.5 * rng.normal(size=32) for _ in range(16)])
        encode = np.tanh
        self.assertGreater(np.corrcoef(encode(trials.mean(0)), encode(signal))[0, 1],
                           np.corrcoef(encode(trials).mean(0), encode(signal))[0, 1])


if __name__ == '__main__':
    unittest.main()

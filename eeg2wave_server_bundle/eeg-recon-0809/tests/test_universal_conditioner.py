"""The subject-free conditioner of the diffusion decoder (generative_recovery.UniversalConditioner), synthetic.

* output is (B, components + 1 duration + 2 acoustic, mel frames) and ignores the participant argument;
* each trial is read by its own fold's encoder (cross-fitting routing);
* the identity ``mix`` makes pooled conditioning the encoder applied to the averaged EEG;
* the deterministic route has the mel shape and the silence value past its own duration.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src')); sys.path.insert(0, str(ROOT / 'tests'))
_cache_name = os.environ.get('ALIGNED_TARGET_CACHE_NAME')
import generative_recovery as gr
import universal_model as um
import universal_train as ut
from test_universal import decoder_payload, montage_xyz, tiny_decoder
from eeg2speech.aligned import EEG_SAMPLES
if _cache_name is None:
    os.environ.pop('ALIGNED_TARGET_CACHE_NAME', None)
else:
    os.environ['ALIGNED_TARGET_CACHE_NAME'] = _cache_name


def checkpoint(folder, seed):
    torch.manual_seed(seed)
    decoder = tiny_decoder()
    spec = dict(virtual=8, width=16, dropout=.1, positional=True, lag_ms=0, harmonics=4, input_norm='trial_rms')
    model = um.UniversalEEGModel({'speech': decoder}, **spec)
    for p in model.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, std=.2)
    path = Path(folder) / f'fold{seed}.pt'
    torch.save(dict(contract=ut.CONTRACT, decoder_specs={'speech': decoder_payload(decoder)['decoder_spec']}, model_spec=spec,
                    model=model.state_dict(), selection_cohort='unseen'), path)
    return path


class UniversalConditionerTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        paths = [checkpoint(self.folder.name, s) for s in (0, 1)]
        xyz = montage_xyz('biosemi128')[:16]
        teacher = np.random.default_rng(0).normal(size=(10, 199, 12)).astype(np.float32)
        self.conditioner = gr.UniversalConditioner(paths, torch.linspace(0, 4, 251), xyz, components=4, teacher_train=teacher)
        g = torch.Generator().manual_seed(3)
        self.eeg = torch.randn(4, 16, EEG_SAMPLES, generator=g) * 1e-5
        self.mask = torch.ones(4, 16, dtype=torch.bool)

    def tearDown(self):
        self.folder.cleanup()

    def test_shape_and_no_participant_dependence(self):
        fold = torch.tensor([0, 1, 0, 1])
        a = self.conditioner.raw(self.eeg, self.mask, torch.tensor([0, 1, 2, 3]), fold)
        b = self.conditioner.raw(self.eeg, self.mask, None, fold)
        self.assertEqual(tuple(a.shape), (4, 4 + 1 + 2, 251))
        self.assertEqual(self.conditioner.dimension, 7)
        torch.testing.assert_close(a, b)

    def test_fold_routing(self):
        both = self.conditioner.raw(self.eeg, self.mask, None, torch.tensor([0, 1, 0, 1]))
        zero = self.conditioner.raw(self.eeg, self.mask, None, torch.zeros(4, dtype=torch.long))
        one = self.conditioner.raw(self.eeg, self.mask, None, torch.ones(4, dtype=torch.long))
        torch.testing.assert_close(both[[0, 2]], zero[[0, 2]]); torch.testing.assert_close(both[[1, 3]], one[[1, 3]])
        self.assertGreater(float((zero - one).abs().max()), 1e-4)

    def test_pooled_premix_is_the_average(self):
        fold = torch.zeros(1, dtype=torch.long)
        a, b = self.conditioner.mix(self.eeg, self.mask, None, fold.expand(4))
        torch.testing.assert_close(a, b)
        pooled = self.conditioner.raw(self.eeg[:1], self.mask[:1], None, fold, premixed=(a.mean(0, keepdim=True), b.mean(0, keepdim=True)))
        direct = self.conditioner.raw(self.eeg.mean(0, keepdim=True), self.mask[:1], None, fold)
        torch.testing.assert_close(pooled, direct)

    def test_regression_route(self):
        mel = self.conditioner.regression(self.eeg, self.mask, torch.tensor([0, 1, 0, 1]))
        self.assertEqual(tuple(mel.shape), (4, 80, 251))
        self.assertTrue(bool(torch.isfinite(mel).all()))


if __name__ == '__main__':
    unittest.main()

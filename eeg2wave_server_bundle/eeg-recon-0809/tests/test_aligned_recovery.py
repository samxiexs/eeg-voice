import sys
import os
from pathlib import Path
import unittest
import argparse
import copy
import signal
import tempfile
from unittest.mock import patch

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
_cache_name = os.environ.get('ALIGNED_TARGET_CACHE_NAME')
import aligned_recovery as runner
if _cache_name is None:
    os.environ.pop('ALIGNED_TARGET_CACHE_NAME', None)
else:
    os.environ['ALIGNED_TARGET_CACHE_NAME'] = _cache_name
from aligned_recovery_model import RecoveryEEGModel, diverse_batches, recovery_loss
from eeg2speech.aligned import AcousticDecoder


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(322)

    def test_batches_keep_contents_distinct_and_cover_unequal_groups(self):
        frame = pd.DataFrame({'content_group': ['a'] * 11 + ['b'] * 3 + ['c'] * 5})
        batches = diverse_batches(frame, 8, 322)
        self.assertEqual(batches, diverse_batches(frame, 8, 322))
        self.assertEqual(set(i for batch in batches for i in batch), set(frame.index))
        for batch in batches:
            self.assertGreaterEqual(len(batch), 2)
            self.assertEqual(len(frame.iloc[batch].content_group.unique()), len(batch))
        with self.assertRaises(ValueError):
            diverse_batches(frame, 1, 322)

    def test_spatial_filters_sequence_and_frozen_decoder_gradient_path(self):
        decoder = AcousticDecoder(torch.linspace(.02, 3.98, 40), torch.linspace(0,4,64),
                                   speech_dimension=12, hidden=16, layers=2)
        model = RecoveryEEGModel(decoder, channels=4, width=16)
        eeg = torch.randn(4,4,1178)
        xyz = torch.zeros(4,4,3); mask=torch.ones(4,4,dtype=torch.bool)
        times = torch.ones(4,1178,dtype=torch.bool)
        state = model(eeg,xyz,mask,times)
        reversed_state = model(-eeg,xyz,mask,times)
        self.assertGreater(float((state.native_mel-reversed_state.native_mel).abs().mean().detach()),1e-5)
        teacher = torch.randn_like(state.aligned_sequence); target = torch.randn_like(state.native_mel)
        bank = torch.randn(8,12); indices=torch.arange(4)
        loss, metrics = recovery_loss(state,teacher,target,decoder.normalizer,bank,indices)
        loss.backward()
        self.assertGreater(float(model.spatial.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in decoder.parameters()))
        self.assertTrue(torch.isfinite(loss))
        # Acoustic supervision alone must reach the signed EEG spatial filters.
        model.zero_grad(); model(eeg,xyz,mask,times).native_mel.square().mean().backward()
        self.assertGreater(float(model.spatial.weight.grad.abs().sum()),0.)
        with self.assertRaises(ValueError):
            recovery_loss(state,teacher,target,decoder.normalizer,bank,torch.zeros(4,dtype=torch.long))

    def test_real_time_lag_and_no_audio_inference_inputs(self):
        decoder = AcousticDecoder(torch.linspace(.02,3.98,40),torch.linspace(0,4,64),
                                   speech_dimension=12,hidden=16,layers=2)
        model = RecoveryEEGModel(decoder,channels=4,width=16,lag_ms=400)
        self.assertAlmostEqual(float(model.token_times[0]),-.25)
        self.assertAlmostEqual(float(model.token_times[1]-model.token_times[0]),4/256)
        self.assertEqual(len(model.token_times),295)
        import inspect
        self.assertEqual(list(inspect.signature(model.forward).parameters),
                         ['eeg','channel_xyz','channel_mask','time_mask'])

    def test_pilot_cannot_authorize_test_evaluation(self):
        from evaluate_aligned_recovery import check_role
        report = dict(beats_template=True, beats_wrong_trial=True, zero_gain=1, time_block_shuffle_gain=1)
        with self.assertRaises(ValueError):
            check_role(dict(signature={'mode':'pilot'}, evaluation=report), 'test')
        with self.assertRaises(ValueError):
            check_role(dict(signature={'mode':'m0'}, evaluation=report), 'validation')
        check_role(dict(signature={'mode':'full'}, evaluation=report), 'test')

    def test_resume_matches_uninterrupted_updates(self):
        class Dataset:
            def __init__(self):
                self.frame=pd.DataFrame(dict(trial_id=['a','b','c','d'], content_group=['a','b','c','d'], subject=['s']*4))
                self.teacher_sha256='teacher'
                self.records=[dict(eeg=torch.randn(4,1178), channel_xyz=torch.zeros(4,3),
                    channel_mask=torch.ones(4,dtype=torch.bool),time_mask=torch.ones(1178,dtype=torch.bool),
                    teacher=torch.randn(40,12),mel=torch.randn(80,64),content=c) for c in 'abcd']
            def __getitem__(self,i): return self.records[i]
            def __len__(self): return 4
        data=Dataset();decoder=AcousticDecoder(torch.linspace(.02,3.98,40),torch.linspace(0,4,64),
                                               speech_dimension=12,hidden=16,layers=2)
        spec=dict(speech_times=decoder.speech_times.tolist(),mel_times=decoder.mel_times.tolist(),
                  speech_dimension=12,hidden=16,layers=2)
        source=dict(stage='adapt',teacher_sha256='teacher',decoder_spec=spec,decoder=decoder.state_dict())
        report=dict(native_mel_mae=1.,m0_passed=True, prediction_variance_ratio=.8,retrieval_r1=.9,chance_r1=.25)
        bank=torch.randn(4,12)
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);initial=base/'decoder.pt';initial.touch()
            args=argparse.Namespace(seed=322,device='cpu',mode='m0',width=16,lag_ms=0,
                decoder=str(initial),batch_size=2,updates=4,lr=.0001,eval_every=2,
                initialize=None,m0_checkpoint=None,output=str(base/'continuous'))
            with patch.object(runner.legacy,'dataset_for',return_value=data), \
                 patch.object(runner.legacy,'load_payload',return_value=source), \
                 patch.object(runner.legacy,'check_eeg_artifacts'), \
                 patch.object(runner,'dataset_signature',return_value=dict(manifest='manifest',cache='cache')), \
                 patch.object(runner,'fixed_bank',return_value=(list('abcd'),bank)), \
                 patch.object(runner,'metrics',return_value=report):
                runner.train(args,{})
                args.output=str(base/'resumed')
                original=torch.optim.AdamW.step
                def interrupt_after_step(optimizer,*a,**kw):
                    result=original(optimizer,*a,**kw)
                    os.kill(os.getpid(),signal.SIGINT)
                    return result
                with patch.object(torch.optim.AdamW,'step',interrupt_after_step):
                    with self.assertRaises(SystemExit) as cm: runner.train(args,{})
                    self.assertEqual(cm.exception.code,130)
                runner.train(args,{})
            a=runner.load_checkpoint(base/'continuous/training_state.pt')
            b=runner.load_checkpoint(base/'resumed/training_state.pt')
            self.assertEqual(a['step'],b['step'])
            for key in a['model']:
                torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()

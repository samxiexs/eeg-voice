import sys
import os
from pathlib import Path
import unittest
import argparse
import copy
import signal
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
_cache_name = os.environ.get('ALIGNED_TARGET_CACHE_NAME')
import aligned_recovery as runner
from aligned_recovery_model import (RecoveryEEGModel, apply_tail, augment_eeg, diverse_batches,
                                    duration_fraction, recovery_loss, speech_frame_masks)
import aligned_recovery_eval as evaluation
import linear_envelope_check as linear
from eeg2speech.aligned import AcousticDecoder
# aligned_recovery and linear_envelope_check default the cache name at import;
# restore the caller's environment so other test modules keep targets.h5.
if _cache_name is None:
    os.environ.pop('ALIGNED_TARGET_CACHE_NAME', None)
else:
    os.environ['ALIGNED_TARGET_CACHE_NAME'] = _cache_name


def tiny_decoder():
    return AcousticDecoder(torch.linspace(.0125, 3.9725, 199), torch.linspace(0, 4, 251),
                           speech_dimension=12, hidden=16, layers=2)


def inputs(batch=4, channels=4):
    return (torch.randn(batch, channels, 1178), torch.zeros(batch, channels, 3),
            torch.ones(batch, channels, dtype=torch.bool), torch.ones(batch, 1178, dtype=torch.bool))


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

    def test_speech_masks_exclude_padding_from_every_loss_term(self):
        decoder = tiny_decoder()
        frames = torch.tensor([100, 251, 2, 180])
        mel_mask, speech_mask = speech_frame_masks(frames, decoder.mel_times, decoder.speech_times)
        self.assertEqual(mel_mask.sum(1).tolist(), [100, 251, 2, 180])
        # Speech-teacher frames end within one hop of the last speech mel frame.
        self.assertEqual(speech_mask.sum(1).tolist(), [79, 199, 1, 143])
        with self.assertRaises(ValueError):
            speech_frame_masks(torch.tensor([1]), decoder.mel_times, decoder.speech_times)
        with self.assertRaises(ValueError):
            speech_frame_masks(torch.tensor([252]), decoder.mel_times, decoder.speech_times)
        model = RecoveryEEGModel(decoder, channels=4, width=16, subjects=2, dropout=0.)
        model.eval()
        eeg, xyz, mask, times = inputs()
        state = model(eeg, xyz, mask, times, torch.tensor([0, 1, 0, 1]))
        teacher = torch.randn(4, 199, 12); mel = torch.randn(4, 80, 251)
        fraction = duration_fraction(frames, 251); indices = torch.arange(4)
        base, metrics = recovery_loss(state, teacher, mel, decoder.normalizer, speech_mask, mel_mask, fraction, indices)
        corrupted_teacher = torch.where(speech_mask[:, :, None], teacher, 7 + 3 * torch.randn_like(teacher))
        corrupted_mel = torch.where(mel_mask[:, None, :], mel, 7 + 3 * torch.randn_like(mel))
        changed, _ = recovery_loss(state, corrupted_teacher, corrupted_mel, decoder.normalizer, speech_mask, mel_mask, fraction, indices)
        self.assertAlmostEqual(float(base), float(changed), places=4)
        self.assertNotIn('variance_penalty', metrics)
        for key in ('contrastive', 'sequence_mse', 'delta', 'mel', 'duration', 'batch_accuracy'):
            self.assertIn(key, metrics)

    def test_spatial_subject_and_frozen_decoder_gradient_path(self):
        decoder = tiny_decoder()
        model = RecoveryEEGModel(decoder, channels=4, width=16, subjects=3)
        eeg, xyz, mask, times = inputs()
        subject = torch.tensor([0, 1, 2, 0])
        state = model(eeg, xyz, mask, times, subject)
        reversed_state = model(-eeg, xyz, mask, times, subject)
        self.assertGreater(float((state.native_mel - reversed_state.native_mel).abs().mean().detach()), 1e-5)
        frames = torch.tensor([120, 200, 90, 251])
        mel_mask, speech_mask = speech_frame_masks(frames, decoder.mel_times, decoder.speech_times)
        teacher = torch.randn_like(state.aligned_sequence); target = torch.randn_like(state.native_mel)
        loss, metrics = recovery_loss(state, teacher, target, decoder.normalizer, speech_mask, mel_mask,
                                      duration_fraction(frames, 251), torch.arange(4))
        loss.backward()
        self.assertGreater(float(model.spatial.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.subject_delta.grad.abs().sum()), 0.)
        self.assertGreater(float(model.duration.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in decoder.parameters()))
        self.assertTrue(torch.isfinite(loss))
        # Acoustic supervision alone must reach the signed EEG spatial filters.
        model.zero_grad(); model(eeg, xyz, mask, times, subject).native_mel.square().mean().backward()
        self.assertGreater(float(model.spatial.weight.grad.abs().sum()), 0.)
        with self.assertRaises(ValueError):
            recovery_loss(state, teacher, target, decoder.normalizer, speech_mask, mel_mask,
                          duration_fraction(frames, 251), torch.zeros(4, dtype=torch.long))
        with self.assertRaises(ValueError):
            model(eeg, xyz, mask, times, torch.tensor([0, 1, 3, 0]))
        # Unknown subject falls back to the shared identity path; zero EEG stays zero.
        plain = RecoveryEEGModel(decoder, channels=4, width=16)
        self.assertEqual(plain(eeg, xyz, mask, times).native_mel.shape, (4, 80, 251))
        model.eval(); zero = model(torch.zeros_like(eeg), xyz, mask, times, subject).native_mel
        torch.testing.assert_close(zero[0], zero[1])

    def test_positional_code_is_trial_independent_and_optional(self):
        decoder = tiny_decoder()
        eeg, xyz, mask, times = inputs()
        with_code = RecoveryEEGModel(decoder, channels=4, width=16, positional=True).eval()
        without = RecoveryEEGModel(decoder, channels=4, width=16).eval()
        self.assertEqual(tuple(with_code.position.shape), (1, 16, 295))
        self.assertFalse(hasattr(without, 'position'))
        # Zero-initialized code changes nothing until trained; once set it moves every trial identically.
        without.load_state_dict({k: v for k, v in with_code.state_dict().items() if k != 'position'})
        torch.testing.assert_close(with_code(eeg, xyz, mask, times).native_mel, without(eeg, xyz, mask, times).native_mel)
        with torch.no_grad():
            with_code.position.normal_()
        shifted = with_code(torch.zeros_like(eeg), xyz, mask, times).native_mel
        torch.testing.assert_close(shifted[0], shifted[3])
        self.assertGreater(float((shifted - without(torch.zeros_like(eeg), xyz, mask, times).native_mel).abs().mean()), 1e-4)

    def test_real_time_lag_and_no_audio_inference_inputs(self):
        decoder = tiny_decoder()
        model = RecoveryEEGModel(decoder, channels=4, width=16, lag_ms=400)
        self.assertAlmostEqual(float(model.token_times[0]), -.25)
        self.assertAlmostEqual(float(model.token_times[1] - model.token_times[0]), 4 / 256)
        self.assertEqual(len(model.token_times), 295)
        import inspect
        self.assertEqual(list(inspect.signature(model.forward).parameters),
                         ['eeg', 'channel_xyz', 'channel_mask', 'time_mask', 'subject'])

    def test_augmentation_and_tail_policy(self):
        eeg, _, mask, _ = inputs(batch=6, channels=8)
        mask[:, 5] = False; eeg = eeg * mask[:, :, None]
        out = augment_eeg(eeg, mask)
        self.assertEqual(out.shape, eeg.shape)
        self.assertEqual(float(out[:, 5].abs().sum()), 0.)
        self.assertGreater(float((out - eeg).abs().mean()), 0.)
        torch.testing.assert_close(augment_eeg(eeg, mask, channel_drop=0, spans=0, noise=0, gain=0), eeg)
        mel = torch.zeros(2, 80, 251)
        cut = apply_tail(mel, torch.tensor([10, 251]))
        self.assertEqual(float(cut[0, :, 10:].mean()), -10.); self.assertEqual(float(cut[0, :, :10].mean()), 0.)
        self.assertEqual(float(cut[1].mean()), 0.)

    def test_learning_rate_schedule(self):
        self.assertAlmostEqual(runner.learning_rate(0, 1., 1000, 100), .01)
        self.assertAlmostEqual(runner.learning_rate(99, 1., 1000, 100), 1.)
        self.assertAlmostEqual(runner.learning_rate(1000, 1., 1000, 100), .1)
        self.assertGreater(runner.learning_rate(300, 1., 1000, 100), runner.learning_rate(600, 1., 1000, 100))

    def test_masked_evaluation_reports_speech_region_gates(self):
        decoder = tiny_decoder()
        model = RecoveryEEGModel(decoder, channels=4, width=16, subjects=2, dropout=0.)

        class Dataset:
            def __init__(self):
                self.frame = pd.DataFrame(dict(trial_id=list('abcdef'), content_group=list('xyzxyz'),
                                               subject=['s1'] * 3 + ['s2'] * 3, audio_key=list('xyzxyz'),
                                               task=['A', 'A', 'P', 'A', 'P', 'P'],
                                               stimulus_duration_seconds=[1.9, 2.2, 2.5, 1.9, 2.2, 2.5]))
                self.mel_times = decoder.mel_times; self.speech_times = decoder.speech_times
                self.records = [dict(eeg=torch.randn(4, 1178), channel_xyz=torch.zeros(4, 3),
                                     channel_mask=torch.ones(4, dtype=torch.bool), time_mask=torch.ones(1178, dtype=torch.bool),
                                     teacher=torch.randn(199, 12), mel=torch.randn(80, 251),
                                     oracle_duration_frames=torch.tensor(120 + 20 * (i % 3)), trial_id=t, content=c, subject=s)
                                for i, (t, c, s) in enumerate(zip(self.frame.trial_id, self.frame.content_group, self.frame.subject))]
                for r in self.records:
                    r['mel'][:, int(r['oracle_duration_frames']):] = -10.
            def __getitem__(self, i): return self.records[i]
            def __len__(self): return len(self.records)
        data = Dataset()
        templates = tuple(torch.zeros(80, 251) for _ in range(4))
        report = evaluation.evaluate_recovery(model, data, data, torch.device('cpu'), 4, {'s1': 0, 's2': 1},
                                              bootstrap=True, include_records=True, templates=templates)
        self.assertEqual(report['pairs'], 6); self.assertEqual(report['unique_contents'], 3)
        self.assertEqual(report['common_speech_frames'], 120)
        for key in ('native_mel_mae', 'template_mae', 'template_mean_mae', 'zero_gain', 'wrong_trial_gain',
                    'time_block_shuffle_gain', 'channel_shuffle_gain', 'envelope_corr', 'envelope_zero_gain',
                    'duration_corr', 'full_native_mel_mae', 'full_template_mae', 'retrieval_r1', 'm0_passed'):
            self.assertIn(key, report)
        # Speech-region error ignores the -10 padding; the full-window error does not.
        self.assertLess(abs(report['native_mel_mae'] - report['records'][0]['native_mel_mae']) + 1, 100)
        self.assertNotAlmostEqual(report['native_mel_mae'], report['full_native_mel_mae'])
        self.assertIn('zero_gain', report['subject_content_bootstrap'])
        self.assertEqual(set(report['per_task']), {'A', 'P'}); self.assertEqual(report['per_task']['A']['pairs'], 3)
        matched = evaluation.matched_wrong_trial_indices(data.frame)
        self.assertEqual(matched[0], 1)                             # same participant, same task, covering duration
        self.assertFalse(evaluation.passes_validation_controls(dict(beats_wrong_trial=True, beats_chance=True, zero_gain=1,
                                                                     time_block_shuffle_gain=1, envelope_zero_gain=-1e-3)))
        self.assertTrue(evaluation.passes_validation_controls(dict(beats_wrong_trial=True, beats_chance=True, zero_gain=1e-3,
                                                                    time_block_shuffle_gain=1e-3, envelope_zero_gain=1e-3)))
        self.assertAlmostEqual(evaluation.chance_mrr(1), 1.); self.assertAlmostEqual(evaluation.chance_mrr(2), .75)
        frame = pd.DataFrame(dict(trial_id=list('abcdef'), subject=['s'] * 6, content_group=list('xyzxyz'),
                                  stimulus_duration_seconds=[2.0, 3.0, 2.5, 2.0, 3.5, 1.0]))
        matched = evaluation.matched_wrong_trial_indices(frame)
        for i, j in enumerate(matched):
            self.assertNotEqual(frame.content_group[i], frame.content_group[j])
        self.assertEqual(matched[0], 2)     # 2.0 s target x: closest covering other content is z at 2.5 s
        self.assertIn(matched[5], (0, 3))   # 1.0 s target z: closest covering is x at 2.0 s (hash tie-break)
        self.assertEqual(matched[4], 2)     # 3.5 s target: no other content covers it, so the closest (z, 2.5 s) is used
        with self.assertRaises(ValueError):
            evaluation.evaluate_recovery(model, data, data, torch.device('cpu'), 4, {'s1': 0}, bootstrap=False, templates=templates)

    def test_linear_check_recovers_planted_envelope_and_rejects_noise(self):
        rng = np.random.default_rng(3)
        mel_times = np.linspace(0, 4, 251, dtype=np.float32)

        def rows(count, subject, planted):
            out = []
            for i in range(count):
                frames = int(rng.integers(110, 240)); grid = np.arange(0, float(mel_times[frames - 1]), 1 / linear.RATE)
                envelope = np.convolve(rng.standard_normal(len(grid)), np.ones(5) / 5, 'same').astype(np.float32)
                low = rng.standard_normal((6, 1178 // 8)).astype(np.float32)
                if planted:
                    # Channel 0 carries the envelope 125 ms after the acoustic frame.
                    index = np.clip(np.round((grid + .125 - linear.EEG_START) * linear.RATE).astype(int), 0, low.shape[1] - 1)
                    low[0, index] += 3 * envelope
                out.append(dict(subject=subject, content=f'{subject}-{i}', trial_id=f'{subject}-{i}',
                                low=low, y=envelope, grid=grid.astype(np.float32)))
            return out
        for planted in (True, False):
            train = rows(60, 'a', planted) + rows(60, 'b', planted)
            held = rows(20, 'a', planted) + rows(20, 'b', planted)
            result = linear.run_check(train, held, linear.wrong_indices(held), alphas=(1e1, 1e3), seed=1)
            pooled = result['pooled']
            if planted:
                self.assertGreater(pooled['r_model'], .5)
                self.assertGreater(pooled['gain_over_prior_subject_mean'], .4)
                self.assertGreater(pooled['r_residual'], pooled['r_residual_shift_null'] + .3)
                self.assertLess(pooled['r_wrong_trial'], pooled['r_model'] - .3)
                self.assertEqual(result['best_single_lag_ms'], 125)
            else:
                self.assertLess(abs(pooled['r_model'] - pooled['r_prior']), .15)
                self.assertLess(abs(pooled['r_residual'] - pooled['r_residual_shift_null']), .15)

    def test_same_content_partners_and_averaging(self):
        from aligned_recovery_model import average_eeg, same_content_partners
        frame = pd.DataFrame(dict(trial_id=list('abcdefg'), content_group=list('xxyyzzw'),
                                  subject=['s1', 's2', 's1', 's1', 's2', 's3', 's1'], task=['A', 'A', 'A', 'P', 'A', 'A', 'A']))
        self.assertEqual(same_content_partners(frame, 0., np.random.default_rng(0)), {})
        plan = same_content_partners(frame, 1., np.random.default_rng(0))
        self.assertNotIn(6, plan)                                   # singleton content has no partner
        for i, j in plan.items():
            self.assertNotEqual(i, j); self.assertEqual(frame.content_group[i], frame.content_group[j])
        self.assertEqual(plan[2], 3); self.assertEqual(plan[3], 2)  # same participant, other task preferred
        self.assertEqual(same_content_partners(frame, 1., np.random.default_rng(1)), plan)
        eeg = torch.ones(1, 3, 4); partner = torch.full((1, 3, 4), 3.)
        mask = torch.tensor([[True, True, False]]); partner_mask = torch.tensor([[True, False, True]])
        mixed = average_eeg(eeg, partner, mask, partner_mask)
        self.assertEqual(mixed[0, :, 0].tolist(), [2., 1., 3.])

    def test_new_augmentation_terms_are_off_by_default_and_bounded(self):
        from aligned_recovery_model import shift_eeg
        eeg, _, mask, _ = inputs(batch=6, channels=8)
        mask[:, 5] = False; eeg = eeg * mask[:, :, None]

        def reference(eeg, channel_mask, channel_drop=.1, spans=2, span_max=64, noise=.1, gain=.2):
            # The augmentation as released with v3: the default call must keep drawing exactly this.
            batch, channels, samples = eeg.shape
            keep = (torch.rand(batch, channels) >= channel_drop) & channel_mask
            keep = torch.where(keep.any(1, keepdim=True), keep, channel_mask)
            out = eeg * keep[:, :, None]
            positions = torch.arange(samples)[None, None, :]
            starts = torch.randint(0, samples, (batch, spans, 1)); lengths = torch.randint(0, span_max + 1, (batch, spans, 1))
            out = out * (~((positions >= starts) & (positions < starts + lengths)).any(1))[:, None, :]
            out = out * (1 + (torch.rand(batch, 1, 1) * 2 - 1) * gain)
            return out + noise * torch.randn_like(out) * channel_mask[:, :, None]
        torch.manual_seed(5); expected = reference(eeg, mask)
        torch.manual_seed(5); torch.testing.assert_close(augment_eeg(eeg, mask), expected)
        # Latency jitter: zero filled, never circular, direction = delay for positive shifts.
        shifted = shift_eeg(eeg, torch.tensor([3, -2, 0, 0, 0, 0]))
        torch.testing.assert_close(shifted[0, :, 3:], eeg[0, :, :-3]); self.assertEqual(float(shifted[0, :, :3].abs().sum()), 0.)
        torch.testing.assert_close(shifted[1, :, :-2], eeg[1, :, 2:]); self.assertEqual(float(shifted[1, :, -2:].abs().sum()), 0.)
        torch.testing.assert_close(shifted[2:], eeg[2:])
        out = augment_eeg(eeg, mask, channel_drop=0, spans=0, noise=0, gain=0, shift_max=10)
        self.assertEqual(float(out[:, 5].abs().sum()), 0.)
        self.assertTrue(all(any(torch.allclose(out[b], shift_eeg(eeg[b:b + 1], torch.tensor([k]))[0]) for k in range(-10, 11))
                            for b in range(6)))
        # Per-channel gain stays inside 1 +/- g and leaves invalid channels at zero.
        out = augment_eeg(eeg, mask, channel_drop=0, spans=0, noise=0, gain=0, channel_gain=.15)
        ratio = out[:, :5] / eeg[:, :5]
        self.assertTrue(bool(((ratio >= .85 - 1e-5) & (ratio <= 1.15 + 1e-5)).all()))
        self.assertGreater(float((ratio[:, 0, 0] - ratio[:, 1, 0]).abs().max()), 0.)   # differs across channels
        self.assertEqual(float(out[:, 5].abs().sum()), 0.)

    def test_background_mixing_uses_same_participant_other_sentence(self):
        from aligned_recovery_model import background_partners, mix_background
        frame = pd.DataFrame(dict(trial_id=list('abcdefg'), content_group=list('xxyyzzw'),
                                  subject=['s1', 's2', 's1', 's1', 's2', 's3', 's1'], task=['A'] * 7))
        self.assertEqual(background_partners(frame, 0., np.random.default_rng(0)), {})
        plan = background_partners(frame, 1., np.random.default_rng(0))
        self.assertNotIn(5, plan)                                   # s3 has no other sentence
        self.assertEqual(sorted(plan), [0, 1, 2, 3, 4, 6])
        for i, j in plan.items():
            self.assertEqual(frame.subject[i], frame.subject[j]); self.assertNotEqual(frame.content_group[i], frame.content_group[j])
        self.assertEqual(background_partners(frame, 1., np.random.default_rng(0)), plan)
        eeg = torch.ones(2, 3, 4); noise = torch.full((2, 3, 4), 2.)
        mask = torch.tensor([[True, True, False]] * 2); noise_mask = torch.tensor([[True, False, True]] * 2)
        mixed = mix_background(eeg, noise, mask, noise_mask, torch.tensor([.5, 0.]))
        self.assertEqual(mixed[0, :, 0].tolist(), [2., 1., 1.]); torch.testing.assert_close(mixed[1], eeg[1])

    def test_multi_presentation_pool_and_curriculum(self):
        from aligned_recovery_model import average_eeg, average_eeg_many, mix_probability, same_content_pool
        frame = pd.DataFrame(dict(trial_id=list('abcdefghi'), content_group=list('xxxxyyyzw'),
                                  subject=['s1', 's2', 's3', 's4', 's1', 's1', 's2', 's1', 's1'],
                                  task=['A', 'A', 'A', 'A', 'A', 'P', 'A', 'A', 'A']))
        self.assertEqual(same_content_pool(frame, 0., np.random.default_rng(0), 3), {})
        self.assertEqual(same_content_pool(frame, 1., np.random.default_rng(0), 0), {})
        plan = same_content_pool(frame, 1., np.random.default_rng(0), 3)
        self.assertNotIn(7, plan); self.assertNotIn(8, plan)          # singleton contents
        for i, js in plan.items():
            self.assertTrue(1 <= len(js) <= 3); self.assertEqual(len(set(js)), len(js)); self.assertNotIn(i, js)
            for j in js:
                self.assertEqual(frame.content_group[i], frame.content_group[j])
        self.assertEqual(plan[4][0], 5); self.assertEqual(plan[5][0], 4)   # own other-task presentation first
        self.assertEqual(same_content_pool(frame, 1., np.random.default_rng(0), 3), plan)
        sizes = {len(js) for _ in range(20) for js in same_content_pool(frame, 1., np.random.default_rng(_), 3).values()}
        self.assertEqual(sizes, {1, 2, 3})
        # One partner reduces to the pairwise average; several average the partners first.
        eeg = torch.randn(2, 3, 4); partner = torch.randn(2, 3, 4)
        mask = torch.tensor([[True, True, False]] * 2); partner_mask = torch.tensor([[True, False, True]] * 2)
        torch.testing.assert_close(average_eeg_many(eeg, partner[:, None], mask, partner_mask[:, None]),
                                   average_eeg(eeg, partner, mask, partner_mask))
        partners = torch.stack([torch.ones(2, 3, 4), torch.full((2, 3, 4), 3.)], 1)
        masks = torch.tensor([[[True, True, False], [True, False, False]]] * 2)
        many = average_eeg_many(torch.zeros(2, 3, 4), partners, torch.ones(2, 3, dtype=torch.bool), masks)
        self.assertEqual(many[0, :, 0].tolist(), [1., .5, 0.])       # mean(1, 3) / 2, 1 / 2, anchor only
        self.assertEqual(mix_probability(3, .2), .2)
        self.assertAlmostEqual(mix_probability(0, .2, .8, 10), .8); self.assertAlmostEqual(mix_probability(5, .2, .8, 10), .5)
        self.assertAlmostEqual(mix_probability(30, .2, .8, 10), .2)

    def test_pilot_cannot_authorize_test_evaluation(self):
        from recovery_reports import check_role
        report = dict(beats_template=True, beats_wrong_trial=True, beats_chance=True, zero_gain=1,
                      time_block_shuffle_gain=1, envelope_zero_gain=1)
        with self.assertRaises(ValueError):
            check_role(dict(signature={'mode': 'pilot'}, evaluation=report), 'test')
        with self.assertRaises(ValueError):
            check_role(dict(signature={'mode': 'full'}, evaluation=dict(report, envelope_zero_gain=-1)), 'test')
        with self.assertRaises(ValueError):
            check_role(dict(signature={'mode': 'm0'}, evaluation=report), 'validation')
        check_role(dict(signature={'mode': 'full'}, evaluation=report), 'test')

    def test_resume_matches_uninterrupted_updates(self):
        decoder = tiny_decoder()

        class Dataset:
            def __init__(self):
                # Contents a and b are observed twice so the same-content averaging path runs.
                self.frame = pd.DataFrame(dict(trial_id=list('abcdef'), content_group=list('abcdab'),
                                               subject=list('ssttts'), audio_key=list('abcdab')))
                self.teacher_sha256 = 'teacher'
                self.mel_times = decoder.mel_times; self.speech_times = decoder.speech_times
                self.records = [dict(eeg=torch.randn(4, 1178), channel_xyz=torch.zeros(4, 3),
                                     channel_mask=torch.ones(4, dtype=torch.bool), time_mask=torch.ones(1178, dtype=torch.bool),
                                     teacher=torch.randn(199, 12), mel=torch.randn(80, 251), content=c, subject=s,
                                     oracle_duration_frames=torch.tensor(150), trial_id=t)
                                for t, c, s in zip('abcdef', 'abcdab', 'ssttts')]
            def __getitem__(self, i): return self.records[i]
            def __len__(self): return 6
        data = Dataset()
        spec = dict(speech_times=decoder.speech_times.tolist(), mel_times=decoder.mel_times.tolist(),
                    speech_dimension=12, hidden=16, layers=2)
        source = dict(stage='adapt', teacher_sha256='teacher', decoder_spec=spec, decoder=decoder.state_dict())
        report = dict(native_mel_mae=1., m0_passed=True, prediction_variance_ratio=.8, retrieval_r1=.9, chance_r1=.25,
                      template_mae=1.2, template_improvement=.2, zero_gain=.1, wrong_trial_gain=.1, time_block_shuffle_gain=.1,
                      envelope_corr=.1, envelope_zero_gain=.0, duration_corr=.0, beats_template=True, beats_wrong_trial=True,
                      beats_chance=True)
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder); initial = base / 'decoder.pt'; initial.touch()
            args = argparse.Namespace(seed=322, device='cpu', mode='m0', width=16, lag_ms=0,
                                      decoder=str(initial), batch_size=2, updates=6, lr=.0001, eval_every=3,
                                      initialize=None, m0_checkpoint=None, output=str(base / 'continuous'),
                                      warmup=1, weight_decay=.05, dropout=.1, mel_warmup=2, temperature=.1,
                                      contrastive_weight=1., sequence_weight=1., delta_weight=.2, duration_weight=.5,
                                      subject_layer=True, augment=True, positional=True, mix_same_content=.5, initialize_trunk=None,
                                      mix_partners=2, mix_start=1., mix_anneal_epochs=2, background_mix=.5,
                                      background_alpha=(.2, .8), shift_max=4, channel_gain=.1, throttle=0.)
            class Bank:
                def __init__(self, dataset): self.records = dataset.records
                def __getitem__(self, i): return self.records[i]['eeg'], self.records[i]['channel_mask']
            with patch.object(runner.legacy, 'dataset_for', return_value=data), \
                 patch.object(runner, 'EEGBank', Bank), \
                 patch.object(runner.legacy, 'load_payload', return_value=source), \
                 patch.object(runner.legacy, 'check_eeg_artifacts'), \
                 patch.object(runner, 'dataset_signature', return_value=dict(manifest='manifest', cache='cache')), \
                 patch.object(runner, 'train_templates', return_value=tuple(torch.zeros(80, 251) for _ in range(4))), \
                 patch.object(runner, 'metrics', return_value=report):
                runner.train(args, {})
                args.output = str(base / 'resumed')
                original = torch.optim.AdamW.step
                def interrupt_after_step(optimizer, *a, **kw):
                    result = original(optimizer, *a, **kw)
                    os.kill(os.getpid(), signal.SIGINT)
                    return result
                with patch.object(torch.optim.AdamW, 'step', interrupt_after_step):
                    with self.assertRaises(SystemExit) as cm: runner.train(args, {})
                    self.assertEqual(cm.exception.code, 130)
                runner.train(args, {})
            a = runner.load_checkpoint(base / 'continuous/training_state.pt')
            b = runner.load_checkpoint(base / 'resumed/training_state.pt')
            self.assertEqual(a['step'], b['step'])
            self.assertEqual(a['signature']['spec']['subjects'], 2)
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
            self.assertTrue((base / 'continuous/best_passed.pt').exists())


if __name__ == '__main__':
    unittest.main()

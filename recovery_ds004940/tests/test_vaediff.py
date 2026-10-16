"""Checks for the VAE + diffusion route (app/vaediff*.py); synthetic data only, no training of real models.

* losses and measures: KL identities, chunk / text InfoNCE at chance and at perfect alignment, the frame-partner
  exclusion, a level-free timbre distance, tie-averaged ranks, product-of-experts pooling;
* the weight controller: a violated threshold raises its weight, a met one lowers it, bounds, the anchor, and the
  control-file override (weights, thresholds, hold);
* data contracts: sentence pairs never cross splits, unseen people are test-only, epoch indexing, robust scaling;
* an unseen person (subject -1) takes the shared encoder path;
* an end-to-end smoke run (stage 1, stage 2, evaluate, status) on a tiny synthetic data folder.
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src'))
import vaediff_data as vd
import vaediff_model as vm


class LossTests(unittest.TestCase):
    def test_kl_zero_for_equal_and_matches_closed_form(self):
        mu = torch.randn(2, 4, 6); lv = torch.randn(2, 4, 6) * .3
        mask = torch.ones(2, 6, dtype=torch.bool)
        self.assertAlmostEqual(float(vm.gaussian_kl(mu, lv, mu, lv, mask)), 0., places=6)
        kl = vm.gaussian_kl(torch.ones(1, 1, 1), torch.zeros(1, 1, 1), torch.zeros(1, 1, 1), torch.zeros(1, 1, 1), torch.ones(1, 1, dtype=torch.bool))
        self.assertAlmostEqual(float(kl), .5, places=6)                                    # KL(N(1,1) || N(0,1)) = 1/2

    def test_widened_prior_lowers_the_kl(self):
        mu_q, lv = torch.ones(1, 3, 5), torch.full((1, 3, 5), -2.)
        mask = torch.ones(1, 5, dtype=torch.bool)
        tight = vm.gaussian_kl(mu_q, lv, torch.zeros_like(mu_q), lv, mask)
        loose = vm.gaussian_kl(mu_q, lv, torch.zeros_like(mu_q), vm.widened(lv, .5), mask)
        self.assertLess(float(loose), float(tight))

    def test_chunk_infonce_chance_and_alignment(self):
        torch.manual_seed(0)
        a = torch.randn(16, 8, 40); mask = torch.ones(16, 40, dtype=torch.bool)
        aligned, accuracy = vm.chunk_infonce(a, a, mask, size=10, temperature=.05)
        shuffled, _ = vm.chunk_infonce(a, torch.randn(16, 8, 40), mask, size=10, temperature=.05)
        self.assertLess(float(aligned), .3); self.assertGreater(accuracy, .9)
        self.assertGreater(float(shuffled), .8)

    def test_chunk_infonce_excludes_frame_partners(self):
        a = torch.randn(4, 8, 20); b = a.clone(); b[1] = b[0]                          # trial 1 has trial 0's audio
        mask = torch.ones(4, 20, dtype=torch.bool)
        exclude = torch.zeros(4, 4, dtype=torch.bool); exclude[0, 1] = exclude[1, 0] = True
        with_partner, _ = vm.chunk_infonce(a, b, mask, size=10, temperature=.05)
        without, _ = vm.chunk_infonce(a, b, mask, exclude=exclude, size=10, temperature=.05)
        self.assertLess(float(without), float(with_partner))
        alone, _ = vm.chunk_infonce(a[:2], b[:2], mask[:2], exclude=exclude[:2, :2], size=10)
        self.assertEqual(float(alone), 0.)                                                  # nothing left to contrast

    def test_text_infonce(self):
        t = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
        perfect, accuracy = vm.text_infonce(t, t, temperature=.05)
        self.assertLess(float(perfect), .2); self.assertEqual(accuracy, 1.)

    def test_timbre_distance_is_level_free(self):
        mel = torch.randn(2, 20, 30); mask = torch.ones(2, 30, dtype=torch.bool)
        self.assertLess(float(vm.timbre_distance(mel, mel + 3., mask).max()), 1e-3)
        tilt = mel + torch.linspace(-1, 1, 20)[None, :, None]
        self.assertGreater(float(vm.timbre_distance(mel, tilt, mask).min()), 1.)

    def test_rank_percentile_ties_and_extremes(self):
        scores = torch.tensor([[3., 2., 1.], [1., 1., 1.]])
        r = vm.rank_percentile(scores, torch.tensor([3., 1.]))
        self.assertEqual(r.tolist(), [0., .5])

    def test_product_of_experts_keeps_single_trial_noise(self):
        mu = torch.randn(1, 4, 5).expand(3, -1, -1); lv = torch.full((3, 4, 5), -1.)
        m, v = vm.product_of_experts(mu, lv)
        self.assertTrue(torch.allclose(m, mu[0])); self.assertTrue(torch.allclose(v, lv[0]))
        lv2 = lv.clone(); lv2[0] = -5.                                                      # the confident trial dominates
        m2, _ = vm.product_of_experts(torch.stack([torch.ones(4, 5), torch.zeros(4, 5), torch.zeros(4, 5)]), lv2)
        self.assertGreater(float(m2.mean()), .9)

    def test_untrained_decoder_predicts_zero(self):
        model = vm.LatentMelDiffusion(latent=4, bins=10, frames=12, hidden=16, blocks=2, heads=2, timesteps=100)
        x0 = torch.randn(3, 10, 12); c = model.null_stream(3)
        loss, x0_hat, ab = model.loss(x0, c)
        self.assertEqual(loss.shape, (3,)); self.assertEqual(x0_hat.shape, x0.shape)
        self.assertLess(float((model(x0, torch.ones(3, dtype=torch.long), c)).abs().max()), 1e-6)


class ControllerTests(unittest.TestCase):
    def make(self, folder=None, **kw):
        return vm.WeightController(dict(voice=.5, text=.2, diff=1.), dict(voice=dict(voice_rank=.4, timbre_db=3.), text=dict(text_rank=.45)),
                                   control=folder / 'control.json' if folder else None, **kw)

    def test_violation_raises_and_slack_lowers(self):
        c = self.make()
        row = c.step(50, dict(voice_rank=.48, timbre_db=2., text_rank=.30))
        self.assertGreater(c.weights['voice'], .5); self.assertLess(c.weights['text'], .2); self.assertEqual(c.weights['diff'], 1.)
        self.assertEqual(row['violated'], ['voice'])

    def test_bounds(self):
        c = self.make(bound=2.)
        for step in range(100):
            c.step(step, dict(voice_rank=.9, timbre_db=9., text_rank=.0))
        self.assertAlmostEqual(c.weights['voice'], 1.); self.assertAlmostEqual(c.weights['text'], .1)

    def test_control_file_override_and_hold(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder); c = self.make(folder)
            (folder / 'control.json').write_text(json.dumps(dict(weights=dict(text=.6), thresholds=dict(text=dict(text_rank=.40)), mode='hold', note='guess 1')))
            row = c.step(50, dict(voice_rank=.9, timbre_db=9., text_rank=.42))
            self.assertEqual(row['override']['note'], 'guess 1')
            self.assertEqual(c.weights['text'], .6); self.assertEqual(c.weights['voice'], .5)     # hold: nothing moves by itself
            self.assertEqual(row['violated'], ['voice', 'text'])                                  # the new threshold applies
            row = c.step(100, dict(voice_rank=.9, timbre_db=9., text_rank=.42))
            self.assertNotIn('override', row)                                                     # read once per change


class DataTests(unittest.TestCase):
    def test_pairs_stay_together_and_unseen_are_test_only(self):
        pairs = [f'p{i}' for i in range(30) for _ in (0, 1)]
        splits = vd.make_splits(pairs, [f'sub-{i:03d}' for i in range(10)], val=4, test=4, unseen=2)
        roles = splits['pair_role']
        self.assertEqual(sorted(set(roles.values())), ['test', 'train', 'val'])
        self.assertEqual(sum(r == 'test' for r in roles.values()), 4)
        self.assertEqual(vd.trial_role('train', True), 'unused'); self.assertEqual(vd.trial_role('test', True), 'test_unseen')
        self.assertEqual(vd.trial_role('val', False), 'dev')

    def test_pair_names(self):
        self.assertEqual(vd.pair_of('NPI_car(chest).wav'), vd.pair_of('NPC_car.wav'))
        with self.assertRaises(ValueError):
            vd.pair_of('(Audio_01)DidThisSentenceEndCorrectly.wav')

    def test_epoch_start_and_scaling(self):
        self.assertEqual(vd.epoch_start(10., 0.), int(round(9.5 * 128)))
        self.assertEqual(vd.epoch_start(110., 100.), int(round(9.5 * 128)))
        data = np.random.default_rng(0).normal(size=(3, 1000)) * np.array([[1.], [5.], [0.]])
        scaled = vd.robust_scale(data, np.array([True, True, False]))
        self.assertTrue(np.allclose(np.median(np.abs(scaled[:2]), 1) * 1.4826, 1., atol=.1))
        self.assertTrue((scaled[2] == 0).all())


class EncoderTests(unittest.TestCase):
    def test_unseen_person_takes_shared_path(self):
        torch.manual_seed(0)
        model = vm.EEGPosterior(channels=8, width=16, latent=4, subjects=3, samples=64, blocks=2).eval()
        with torch.no_grad():
            model.subject_delta.normal_(); model.film.normal_()
        x = torch.randn(2, 8, 64); mask = torch.ones(2, 8, dtype=torch.bool); q = vm.latent_times(.4)
        a = model(x, mask, torch.tensor([-1, -1]), q)[0]
        with torch.no_grad():
            model.subject_delta.normal_(); model.film.normal_()
        self.assertTrue(torch.allclose(a, model(x, mask, torch.tensor([-1, -1]), q)[0]))
        self.assertFalse(torch.allclose(a, model(x, mask, torch.tensor([0, 1]), q)[0]))


def make_synthetic(root: Path, sentences=16, subjects=4, rng=np.random.default_rng(3)):
    """A tiny processed-data folder with the real array layout (structure only, no signal)."""
    audio = root / 'audio'; (root / 'eeg').mkdir(parents=True); audio.mkdir()
    rows = []
    for i in range(sentences):
        pair = f'w{i // 2}'
        rows.append(dict(index=i, stim_file=f"NP{'CI'[i % 2]}_{pair}.wav", text=f'sentence {i} about {pair}', pair=pair,
                         condition='NPC', duration=float(rng.uniform(1.6, 3.0))))
    (audio / 'sentences.json').write_text(json.dumps(rows))
    np.save(audio / 'mel.npy', rng.normal(-5, 2, size=(sentences, 100, 422)).astype(np.float32))
    np.save(audio / 'mel_times.npy', (np.arange(422) * 256 / 24000).astype(np.float32))
    np.save(audio / 'wavlm.npy', rng.normal(size=(sentences, 4, 224, 768)).astype(np.float16))
    np.save(audio / 'wavlm_times.npy', vd.wavlm_frame_times(224).astype(np.float32))
    text = rng.normal(size=(sentences, 384)); np.save(audio / 'text.npy', (text / np.linalg.norm(text, axis=1, keepdims=True)).astype(np.float32))
    np.save(audio / 'wave24.npy', (rng.normal(size=(sentences, 108000)) * .05).astype(np.float16))
    for s in range(subjects):
        name = f'sub-{s + 1:03d}'
        np.save(root / 'eeg' / f'{name}_task-N400Passive.npy', rng.normal(size=(sentences, 128, 640)).astype(np.float16))
        (root / 'eeg' / f'{name}_task-N400Passive.json').write_text(json.dumps(
            [dict(subject=name, task='N400Passive', sentence=i, stim_file=rows[i]['stim_file'], peak=5.) for i in range(sentences)]))
    return vd.build_index(root, val=2, test=2, unseen=1, eligible=None)          # one task per synthetic person


@unittest.skipUnless((vd.MODELS / 'vocos-mel-24khz/config.yaml').exists(), 'local Vocos weights needed for the smoke run')
class SmokeTests(unittest.TestCase):
    def test_end_to_end(self):
        import vaediff
        with tempfile.TemporaryDirectory() as folder:
            root, outputs = Path(folder) / 'data', Path(folder) / 'outputs'
            counts = make_synthetic(root)['counts']
            self.assertEqual(set(counts), {'train', 'dev', 'test_seen', 'test_unseen', 'unused'})
            common = ['--root', str(root), '--outputs', str(outputs), '--device', 'cpu', '--batch', '4', '--quick-steps', '2',
                      '--quick-trials', '4']
            vaediff.main(['audio', *common, '--run', 'audio', '--updates', '4', '--eval-every', '2', '--hidden', '32', '--blocks', '2',
                          '--latent', '8', '--audio-hidden', '32', '--warmup', '2'])
            self.assertTrue((outputs / 'audio/audio_best.pt').exists())
            vaediff.main(['eeg', *common, '--run', 'eeg', '--updates', '4', '--eval-every', '2', '--tune-every', '2', '--width', '16',
                          '--warmup', '2', '--check-trials', '4'])
            rows = [json.loads(line) for line in (outputs / 'eeg/tune.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            for key in ('voice_rank', 'text_rank', 'timbre_db', 'diff_advantage'):
                self.assertTrue(math.isfinite(rows[-1]['measures'][key]), key)
            vaediff.main(['evaluate', '--root', str(root), '--outputs', str(outputs), '--device', 'cpu', '--checkpoint',
                          str(outputs / 'eeg/best.pt'), '--force-audio', '--smoke', '--limit', '4', '--steps', '2', '--asr-trials', '0',
                          '--permutations', '20', '--bootstrap', '20', '--listen', '1', '--batch', '4'])
            summary = json.loads((outputs / 'eeg/evaluate_best/summary.json').read_text())
            self.assertIn('test_seen', summary['roles']); self.assertIn('test_unseen', summary['roles'])
            generation = summary['roles']['test_seen']['generation']
            self.assertTrue(generation['forced'])
            self.assertEqual(set(generation['conditions']), {'vocos_ceiling', 'real', 'wrong_trial', 'zero_eeg', 'pooled', 'pooled_wrong',
                                                             'audio_ceiling', 'null'})
            self.assertTrue((outputs / 'eeg/evaluate_best/report.md').read_text().startswith('# 冒烟测试'))
            vaediff.main(['status', '--outputs', str(outputs), '--run', 'eeg'])


if __name__ == '__main__':
    unittest.main()

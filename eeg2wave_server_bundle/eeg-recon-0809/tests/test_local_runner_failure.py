from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest
import json
import sys


class LocalRunnerFailureTests(unittest.TestCase):
    def test_full_routes_new_learning_rates_report_and_output(self):
        original = Path(__file__).resolve().parents[1] / 'app/run_aligned_speech_v1.sh'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / 'app').mkdir()
            shutil.copyfile(original, root / 'app/run_aligned_speech_v1.sh')
            fake = root / 'fake-python'
            fake.write_text('#!/bin/bash\necho "CALLED $*"\ncase "$*" in *"--stage finetune"*) exit 17;; esac\n')
            fake.chmod(0o700)
            result = subprocess.run(['bash', str(root / 'app/run_aligned_speech_v1.sh'), 'full'],
                env={**os.environ, 'PYTHON_BIN': str(fake), 'ALIGNED_OBJECTIVE_ONLY': '1',
                     'ALIGNED_SEEDS': '322', 'ALIGNED_OUTPUT': str(root / 'audio'),
                     'ALIGNED_FULL_OUTPUT': str(root / 'new-full'),
                     'ALIGNED_M0_REPORT': str(root / 'refine/report.json'),
                     'ALIGNED_ALIGN_LR': '0.00001', 'ALIGNED_FINETUNE_LR': '0.000003'},
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
            self.assertIn('--lr 0.00001', result.stdout)
            self.assertIn('--lr 0.000003', result.stdout)
            self.assertIn('--m0-report ' + str(root / 'refine/report.json'), result.stdout)
            self.assertIn('--initialize ' + str(root / 'audio/adapt/best_checkpoint.pt'), result.stdout)
            self.assertIn(str(root / 'new-full/seed-322/lag-0/finetune'), result.stdout)
            self.assertNotIn('--audio-review', result.stdout)
            self.assertNotIn('select-lag', result.stdout)

    def test_failed_m0_stops_full_before_training(self):
        original = Path(__file__).resolve().parents[1] / 'app/run_aligned_local.sh'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'app').mkdir()
            shutil.copyfile(original, root / 'app/run_aligned_local.sh')
            (root / 'app/run_aligned_speech_v1.sh').write_text('echo UNEXPECTED_TRAINING\n')
            output = root / 'outputs/aligned_speech_local_v1'
            (output / 'adapt').mkdir(parents=True)
            (output / 'adapt/best_checkpoint.pt').touch()
            (output / 'm0_report.json').write_text(json.dumps({'m0_passed': False,
                'retrieval_r1': 0.1, 'prediction_variance_ratio': 1e-11}))
            result = subprocess.run(['bash', str(root / 'app/run_aligned_local.sh'), 'full'],
                                    env={**os.environ, 'PYTHON_BIN': sys.executable},
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('M0 FAILED', result.stderr)
            self.assertNotIn('UNEXPECTED_TRAINING', result.stdout)

    def test_all_stops_before_dependent_stages(self):
        original = Path(__file__).resolve().parents[1] / 'app/run_aligned_local.sh'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'app').mkdir()
            shutil.copyfile(original, root / 'app/run_aligned_local.sh')
            fake = root / 'fake-python'
            fake.write_text('#!/bin/bash\necho "CALLED $*"\ncase "$*" in *references*) exit 17;; esac\n')
            fake.chmod(0o700)
            result = subprocess.run(['bash', str(root / 'app/run_aligned_local.sh'), 'all'],
                                    env={**os.environ, 'PYTHON_BIN': str(fake)},
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
            self.assertIn('references', result.stdout)
            self.assertNotIn('train-eeg', result.stdout)
            self.assertNotIn('bootstrap_features_v2', result.stdout)
            self.assertTrue((root / 'logs/aligned_local_all.log').exists())

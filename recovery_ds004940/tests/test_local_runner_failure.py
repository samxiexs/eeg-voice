from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest


class LocalRunnerFailureTests(unittest.TestCase):
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

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
MODULE = ROOT / "scripts" / "eeg_preprocessing_qc.py"
spec = importlib.util.spec_from_file_location("eeg_preprocessing_qc", MODULE)
qc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qc)


class TestPreprocessingPSD(unittest.TestCase):
    def test_band_power_detects_high_frequency_suppression(self):
        rate = 256
        time = np.arange(rate * 4) / rate
        before = np.stack([np.sin(2*np.pi*10*time) + np.sin(2*np.pi*70*time) for _ in range(3)])
        after = np.stack([np.sin(2*np.pi*10*time) for _ in range(3)])
        bands = {"dc":[0.0,0.5], "passband":[0.5,45.0], "high":[45.0,100.0]}
        raw = qc.band_power(before, rate, bands); processed = qc.band_power(after, rate, bands)
        self.assertGreater(raw["high"] / raw["passband"], processed["high"] / processed["passband"])

    def test_band_power_rejects_nonfinite(self):
        value = np.ones((2,128)); value[0,0] = np.nan
        with self.assertRaises(ValueError):
            qc.band_power(value, 256, {"passband":[0.5,45.0]})


if __name__ == "__main__":
    unittest.main()

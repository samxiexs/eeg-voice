import collections
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
MODULE = ROOT / "scripts" / "prepare_training_data.py"
spec = importlib.util.spec_from_file_location("prepare_integration", MODULE)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class TestRealManifestAdapters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / "data/ds004940").exists():
            raise unittest.SkipTest("raw dataset is not installed")
        cls.config, _ = prepare.load_config(ROOT / "configs/training_data_v3.yaml")
        cls.qc = {"actual_subjects": {}, "exclusions": collections.Counter(), "warnings": []}
        cls.lock = {}
        cls.ds004 = prepare._ds004_trial_rows(cls.config, cls.lock, cls.qc)

    def test_release_counts_and_boundary_exclusion(self):
        self.assertEqual(len(self.ds004), 17491)
        self.assertEqual(sum(row["qc_pass"] for row in self.ds004), 17489)
        self.assertEqual(sum(row["boundary_overlap"] for row in self.ds004), 2)
        self.assertEqual(self.qc["actual_subjects"]["ds004940"], 22)

    def test_dataset_is_perception(self):
        self.assertTrue(all(row["neural_task"] == "perception" for row in self.ds004))
        self.assertFalse(any(row["production_contaminated"] for row in self.ds004))


if __name__ == "__main__":
    unittest.main()

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
MODULE = ROOT / "scripts" / "prepare_m0_artifacts.py"
spec = importlib.util.spec_from_file_location("prepare_m0_artifacts", MODULE)
m0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m0)

from prepare_training_data import load_config

TRAIN_MODULE = ROOT / "app" / "train_joint.py"
sys.path.insert(0, str(ROOT / "app" / "src"))
train_spec = importlib.util.spec_from_file_location("train_joint", TRAIN_MODULE)
train = importlib.util.module_from_spec(train_spec)
train_spec.loader.exec_module(train)


class TestRunScripts(unittest.TestCase):
    def test_all_shell_entry_points_are_executable_and_parse(self):
        scripts = sorted((ROOT / "app").glob("run_joint_*.sh"))
        scripts += [ROOT / "app/run_ds004940_large_scale_v1.sh",
                    ROOT / "app/run_ds004940_large_scale_audio_comparisons.sh",
                    ROOT / "app/run_ds004940_trial_diverse_v3.sh"]
        self.assertGreaterEqual(len(scripts), 8)
        for script in scripts:
            self.assertTrue(os.access(script, os.X_OK), script)
            subprocess.run(["bash", "-n", str(script)], check=True)
        subprocess.run(["bash", "-n", str(ROOT / "app/lib/joint_pilot_common.sh")], check=True)

    def test_registered_m0_selection_uses_real_frozen_split(self):
        manifest = ROOT / "artifacts/training_data/v3/manifests/manifest_all.csv"
        split = ROOT / "artifacts/training_data/v3/splits/joint_ood_fold-0.csv"
        if not manifest.exists() or not split.exists():
            self.skipTest("installed dataset artifacts are unavailable")
        config, _ = load_config(ROOT / "configs/training_data_v3.yaml")
        pilot = yaml.safe_load((ROOT / "configs/joint_pilot_v1.yaml").read_text())
        grids = m0.select_registered_grids(config, pilot)
        self.assertEqual(len(grids["ds004940"]), 50)
        self.assertEqual(len(grids["ds006104"]), 50)
        self.assertEqual(len(grids["ds006104_label_only"]), 30)
        for frame in grids.values():
            self.assertTrue(frame.groupby(["subject", "linguistic_content_id"]).size().eq(1).all())

    def test_complete_runner_has_no_automatic_human_approval(self):
        complete = (ROOT / "app/run_joint_pilot_all.sh").read_text()
        stage0 = (ROOT / "app/run_joint_stage0.sh").read_text()
        self.assertIn("exit 3", complete)
        self.assertIn("human_listen_transcript_status=pass only after real human verification", stage0)
        self.assertNotIn("human_listen_transcript_status=pass\"", complete)

    def test_explore_runner_is_explicitly_isolated_from_registered_outputs(self):
        explore = (ROOT / "app/run_joint_explore.sh").read_text()
        self.assertIn("--explore", explore)
        self.assertIn("$RUN_ROOT/explore", explore)
        self.assertIn("--explore --materialize", explore)
        self.assertIn("--checkpoint-every", explore)
        self.assertIn("EXPLORE_STAGE2_MAX_STEPS", explore)

    def test_eight_hour_runner_is_isolated_resumable_and_renders_figures(self):
        explore = (ROOT / "app/run_joint_explore_8h.sh").read_text()
        self.assertIn("joint_explore_8h_v1.yaml", explore)
        self.assertIn("explore_8h_v1", explore)
        self.assertIn("EXPLORE_8H_MAX_STEPS", explore)
        self.assertIn("--checkpoint-every", explore)
        self.assertIn("--output-root", explore)
        self.assertIn("plot_joint_comparison.py", explore)
        self.assertIn("export_audio_pair_comparisons.py", explore)
        self.assertIn("EXPORT_AUDIO_PAIRS", explore)
        self.assertNotIn("--stage overfit", explore)
        self.assertIn('PILOT_CONFIG="$PROJECT_ROOT/configs/joint_explore_8h_v1.yaml"', explore)
        self.assertIn("explore_8h_v1_corrected", explore)

    def test_ds004940_ten_hour_runner_is_single_dataset_and_resumable(self):
        runner = (ROOT / "app/run_ds004940_explore_10h.sh").read_text()
        config = yaml.safe_load((ROOT / "configs/ds004940_explore_10h_v1.yaml").read_text())
        self.assertIn("--mode ds004940", runner)
        self.assertIn("--checkpoint-every", runner)
        self.assertIn("--output-root", runner)
        self.assertNotIn("--mode joint", runner)
        self.assertEqual(config["stage2"]["datasets"], ["ds004940"])
        self.assertEqual(config["pilot"]["generalization_subjects_by_role"],
                         {"train": 4, "validation": 1, "test": 1})
        self.assertEqual(config["pilot"]["generalization_contents_by_role"],
                         {"train": 128, "validation": 16, "test": 16})

    def test_ds004940_large_scale_runner_is_epoch_based_and_isolated(self):
        runner = (ROOT / "app/run_ds004940_large_scale_v1.sh").read_text()
        config = yaml.safe_load((ROOT / "configs/ds004940_large_scale_v1.yaml").read_text())
        self.assertIn("--max-epochs", runner)
        self.assertIn("--mode ds004940", runner)
        self.assertIn("plot_ds004940_large_scale.py", runner)
        self.assertEqual(config["training"]["max_epochs"], 50)
        self.assertEqual(config["training"]["sampling_strategy"], "shuffled_without_replacement")
        self.assertEqual(config["training"]["validation_early_stopping"], {
            "selection_metric": "mfcc_retrieval_mrr", "interval_epochs": 2, "minimum_epochs": 20,
            "patience_validations": 8, "minimum_delta": 0.0001,
        })

    def test_trial_diverse_v3_isolated_and_requires_m0_before_m1(self):
        runner = (ROOT / "app/run_ds004940_trial_diverse_v3.sh").read_text()
        config = yaml.safe_load((ROOT / "configs/ds004940_trial_diverse_v3.yaml").read_text())
        self.assertIn("require_v3_m0_gate", runner)
        self.assertIn("--eeg-checkpoint", runner)
        self.assertIn("--sampling-mode random", runner)
        self.assertEqual(config["model"]["teacher_dimension"], 64)
        self.assertTrue(config["model"]["duration_standardized"])
        self.assertEqual(config["stage2"]["datasets"], ["ds004940"])

    def test_epoch_horizon_and_validation_selection_contract(self):
        maximum, epochs = train.epoch_horizon(
            max_steps=None, max_epochs=50, training={}, mode="ds004940", loader_count=1, steps_per_epoch=423,
        )
        self.assertEqual((maximum, epochs), (21150, 50))
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            train.epoch_horizon(max_steps=1, max_epochs=1, training={}, mode="ds004940", loader_count=1, steps_per_epoch=1)
        best = {"retrieval": {"mrr": 0.3}}
        self.assertTrue(train.validation_improved({"retrieval": {"mrr": 0.301}}, best, 0.0001))
        self.assertFalse(train.validation_improved({"retrieval": {"mrr": 0.30005}}, best, 0.0001))

    def test_shuffled_sampler_covers_each_item_once_and_is_seeded(self):
        class TinyDataset(Dataset):
            def __len__(self): return 7
            def __getitem__(self, index): return index
            def sampling_weights(self): return torch.ones(7, dtype=torch.double)
        dataset = TinyDataset()
        first = list(train.sampler_for(dataset, list(range(7)), 31, "shuffled_without_replacement"))
        repeat = list(train.sampler_for(dataset, list(range(7)), 31, "shuffled_without_replacement"))
        other = list(train.sampler_for(dataset, list(range(7)), 47, "shuffled_without_replacement"))
        self.assertEqual(first, repeat)
        self.assertEqual(sorted(first), list(range(7)))
        self.assertNotEqual(first, other)

    def test_ds004940_audio_pair_runner_uses_single_checkpoint_mode(self):
        runner = (ROOT / "app/run_ds004940_audio_pair_comparisons.sh").read_text()
        generic = (ROOT / "app/run_joint_audio_pair_comparisons.sh").read_text()
        self.assertIn("AUDIO_PAIR_SINGLE_ONLY=1", runner)
        self.assertIn("AUDIO_PAIR_DATASETS=ds004940", runner)
        self.assertIn("--single-only", generic)

    def test_large_scale_audio_runner_exports_heldout_and_train_representatives(self):
        runner = (ROOT / "app/run_ds004940_large_scale_audio_comparisons.sh").read_text()
        generic = (ROOT / "app/run_joint_audio_pair_comparisons.sh").read_text()
        self.assertIn("AUDIO_PAIR_ROLES=validation,test", runner)
        self.assertIn("AUDIO_PAIR_ROLES=train", runner)
        self.assertIn("AUDIO_PAIR_TRAIN_REPRESENTATIVE=1", runner)
        self.assertIn("--one-train-representative-per-content", generic)

    def test_atomic_training_state_and_contract_mismatch_detection(self):
        contract = {"mode": "joint", "seed": 31, "artifact_hashes": {"manifest": "abc"}}
        self.assertEqual(train.contract_mismatches(contract, dict(contract)), [])
        changed = dict(contract); changed["seed"] = 47
        self.assertEqual(train.contract_mismatches(contract, changed), ["seed"])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "training_state.pt"
            train.atomic_torch_save({"completed_steps": 25, "resume_contract": contract}, target)
            self.assertTrue(target.exists())
            self.assertFalse(target.with_suffix(".pt.tmp").exists())
            self.assertEqual(torch.load(target, map_location="cpu", weights_only=False)["completed_steps"], 25)

    def test_legacy_explore_contract_allows_only_control_plane_upgrade(self):
        legacy = {"mode": "joint", "run_kind": "explore", "artifact_hashes": {"manifest": "abc"},
                  "runtime_code_sha256": "old-control-and-model-code"}
        migrated = {"mode": "joint", "run_kind": "explore", "artifact_hashes": {"manifest": "abc"},
                    "model_code_sha256": "model-code"}
        self.assertEqual(train.contract_mismatches(
            legacy, migrated, allow_legacy_explore_control_upgrade=True,
        ), [])
        changed_artifact = dict(migrated); changed_artifact["artifact_hashes"] = {"manifest": "changed"}
        self.assertEqual(train.contract_mismatches(
            legacy, changed_artifact, allow_legacy_explore_control_upgrade=True,
        ), ["artifact_hashes"])
        self.assertEqual(train.contract_mismatches(legacy, migrated), ["legacy_runtime_code_sha256"])

    def test_resume_keeps_original_finish_line_when_new_budget_is_smaller(self):
        self.assertEqual(train.resume_maximum_steps(2700, {"completed_steps": 4050, "maximum_steps": 5000}), (5000, True))
        self.assertEqual(train.resume_maximum_steps(5000, {"completed_steps": 2500, "maximum_steps": 2700}), (5000, False))
        with self.assertRaisesRegex(RuntimeError, "invalid completed_steps"):
            train.resume_maximum_steps(5000, {"completed_steps": 8, "maximum_steps": 7})

    def test_cosine_schedule_is_optional_and_reaches_its_floor(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=3e-4)
        self.assertIsNone(train.learning_rate_scheduler(optimizer, {"learning_rate": 3e-4}, 10))
        scheduler = train.learning_rate_scheduler(
            optimizer, {"learning_rate": 3e-4, "lr_schedule": "cosine", "min_learning_rate": 3e-5}, 4,
        )
        for _ in range(4):
            optimizer.step(); scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 3e-5, places=10)


if __name__ == "__main__":
    unittest.main()

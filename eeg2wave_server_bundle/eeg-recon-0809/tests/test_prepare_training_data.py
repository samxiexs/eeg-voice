import importlib.util
import unittest
from pathlib import Path

MODULE = Path(__file__).parents[1] / "scripts" / "prepare_training_data.py"
spec = importlib.util.spec_from_file_location("prepare", MODULE)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class TestPureV3Contracts(unittest.TestCase):
    def test_round_half_up_and_integer_epoch_contract(self):
        self.assertEqual(prepare.round_half_up(0.5), 1)
        self.assertEqual(prepare.round_half_up(499.5), 500)
        self.assertEqual((500 + 2500 - (500 - 500)) * 256 // 2000, 384)
        self.assertEqual(2356 * 256 // 512, 1178)

    def test_singlephoneme_is_not_implicitly_cropped(self):
        mask = prepare.clean_perception_mask(384, 0, 384, False)
        self.assertEqual(sum(mask), 384)

    def test_acoustic_mask_is_exact_pair_and_native_duration_only(self):
        exact = prepare.acoustic_supervision_mask(1178, 900, 64, 2.0, 256, "verified_exact")
        weak = prepare.acoustic_supervision_mask(384, 384, 64, 0.6, 256, "candidate_filename_timing")
        self.assertEqual(sum(exact), 512)
        self.assertFalse(any(exact[:64]))
        self.assertTrue(all(exact[64:576]))
        self.assertFalse(any(exact[576:]))
        self.assertFalse(any(weak))

    def test_channel_hash_is_order_sensitive(self):
        self.assertNotEqual(prepare.channel_order_hash(["A1", "A2"]), prepare.channel_order_hash(["A2", "A1"]))

    def test_pinned_source_hash_positive_and_negative(self):
        payload = b"pinned-source"
        expected = prepare.sha256_bytes(payload)
        self.assertEqual(expected, prepare.sha256_bytes(payload))
        self.assertNotEqual(expected, prepare.sha256_bytes(payload + b"changed"))

    def test_audio_mel_snapshot_is_explicit(self):
        text = (Path(__file__).parents[1] / "configs" / "training_data_v2.yaml").read_text()
        for required in ("target_sample_rate: 16000", "power: 2.0", "center: false", "n_mels: 80", "log_epsilon: 1.0e-10"):
            self.assertIn(required, text)

    def test_v3_config_is_rollback_safe_and_content_first(self):
        text = (Path(__file__).parents[1] / "configs" / "training_data_v3.yaml").read_text()
        for required in ("output_root: artifacts/training_data/v3", "candidate_filename_timing", "relative_frames: 161", "fit_scope: train_fold_only"):
            self.assertIn(required, text)

    def test_ds004_split_recording_offsets_are_explicit(self):
        root = Path(__file__).parents[1]
        events = root / "data/ds004940/sub-004/eeg/sub-004_task-N400Active_events.tsv"
        if not events.exists():
            self.skipTest("raw DS004940 is not installed")
        runs = prepare._ds004_recording_runs(events, 512)
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0]["start_sample"], 0)
        self.assertEqual(runs[1]["start_sample"], runs[0]["end_sample"])

    def test_resume_rejects_any_pinned_input_change(self):
        attrs = {"preprocess_config_sha256": "config", "source_lock_sha256": "source", "channel_order_hash": "channels", "split_index_sha256": "split"}
        self.assertTrue(prepare.resume_compatible(attrs, config_sha="config", source_lock_sha="source", channel_hash="channels", split_hash="split"))
        self.assertFalse(prepare.resume_compatible(attrs, config_sha="changed", source_lock_sha="source", channel_hash="channels", split_hash="split"))
        self.assertFalse(prepare.resume_compatible(attrs, config_sha="config", source_lock_sha="changed", channel_hash="channels", split_hash="split"))
        self.assertFalse(prepare.resume_compatible(attrs, config_sha="config", source_lock_sha="source", channel_hash="channels", split_hash="changed"))

    def test_balanced_assignment_is_deterministic(self):
        weights = {"a": 20, "b": 12, "c": 9, "d": 8, "e": 7, "f": 2}
        a = prepare.balanced_group_assignment(weights, 3, "seed", "subject")
        b = prepare.balanced_group_assignment(weights, 3, "seed", "subject")
        self.assertEqual(a, b)
        totals = [sum(v["trial_weight"] for v in a.values() if v["fold"] == i) for i in range(3)]
        self.assertLessEqual(max(totals) - min(totals), max(weights.values()))

    def test_joint_train_excludes_both_held_out_axes(self):
        self.assertEqual(prepare.split_role("joint_ood", 0, 2, 3, "paired_audio"), ("train", ""))
        self.assertEqual(prepare.split_role("joint_ood", 0, 0, 2, "paired_audio")[0], "excluded")
        self.assertEqual(prepare.split_role("joint_ood", 0, 2, 1, "paired_audio")[0], "excluded")
        self.assertEqual(prepare.split_role("joint_ood", 0, 0, 0, "paired_audio"), ("test", ""))
        self.assertEqual(prepare.split_role("joint_ood", 0, 2, 3, "weak_audio"), ("train", ""))

    def test_label_only_uses_linguistic_content_fold(self):
        self.assertEqual(prepare.split_role("joint_ood", 0, 2, 3, "label_only"), ("train", ""))
        self.assertEqual(prepare.split_role("joint_ood", 0, 0, 0, "label_only"), ("test", ""))
        self.assertEqual(prepare.split_role("joint_ood", 0, 1, 1, "label_only"), ("validation", ""))
        self.assertEqual(prepare.split_role("joint_ood", 0, 2, None, "label_only"),
                         ("excluded", "missing_linguistic_content_group"))


if __name__ == "__main__":
    unittest.main()

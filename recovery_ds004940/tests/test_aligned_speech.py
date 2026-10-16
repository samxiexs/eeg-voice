"""Scientific failure modes and the complete synthetic training/export contract."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import signal
import tempfile
import types
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app/src"))
from eeg2speech.aligned import (AcousticDecoder, AlignedEEGModel, CONTRACT, alignment_loss,
                               convolution_times, fixed_wave, sample_at, sha256, two_way_bootstrap)
from eeg2speech.aligned_data import (AlignedDataset, content_split, fit_eeg_normalizer,
                                    fixed_bank, wrong_trial_indices)

spec = importlib.util.spec_from_file_location("aligned_runner", ROOT / "app/aligned_speech.py")
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


class AlignedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(31)

    def decoder(self):
        return AcousticDecoder(torch.arange(20).float() * .2, torch.arange(25).float() * .16,
                               speech_dimension=12, hidden=12, layers=2)

    def test_teacher_frame_centers_and_padding_are_physical(self):
        times = convolution_times(64000, [10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2])
        self.assertEqual(len(times), 199)
        self.assertAlmostEqual(float(times[0]), 199.5 / 16000)
        self.assertAlmostEqual(float(times[1] - times[0]), .02)
        wave = np.linspace(-.2, .2, 12000, dtype=np.float32)
        padded = fixed_wave(wave)
        np.testing.assert_array_equal(padded[:12000], wave)
        self.assertFalse(padded[12000:].any())
        with self.assertRaises(ValueError):
            fixed_wave(np.ones(64001))

    def test_cache_runs_real_hubert_frontend_without_time_warp(self):
        from scipy.io import wavfile
        from transformers import HubertConfig, HubertModel, Wav2Vec2FeatureExtractor
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); teacher = base / "tiny_random_teacher"
            tiny = HubertModel(HubertConfig(hidden_size=12, num_hidden_layers=9, num_attention_heads=3,
                                            intermediate_size=24, conv_dim=(8,) * 7,
                                            num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=3,
                                            feat_extract_norm="layer"))
            tiny.save_pretrained(teacher)
            Wav2Vec2FeatureExtractor().save_pretrained(teacher)
            wave = (.1 * np.sin(np.arange(24000) * 2 * np.pi * 220 / 16000)).astype("float32")
            wavfile.write(base / "source.wav", 16000, wave)
            pd.DataFrame([dict(audio_key="example", audio_sha256=sha256(base / "source.wav"), audio_path=str(base / "source.wav"))]).to_csv(base / "manifest.csv", index=False)
            cfg = {"artifact_root": str(base)}
            args = argparse.Namespace(manifest=None, output=None, hubert=str(teacher), device="cpu")
            # WAV I/O replacement only; real local HuBERT and SpeechT5 feature
            # extractors execute. No download or trained speech claim is made.
            sf = types.SimpleNamespace(read=lambda path, **kw: (wavfile.read(path)[1], wavfile.read(path)[0]))
            with patch.dict(sys.modules, {"soundfile": sf}):
                runner.cache_targets(args, cfg)
                # Report/training changes must not invalidate frozen audio features.
                with patch.object(runner, "runtime_hash", return_value="changed-report-code"):
                    runner.cache_targets(args, cfg)
                with patch.object(runner, "feature_hash", return_value="changed-feature-extractor"):
                    with self.assertRaisesRegex(RuntimeError, "feature_hash"):
                        runner.cache_targets(args, cfg)
                with patch.object(runner, "tree_hash", return_value="changed-teacher"):
                    with self.assertRaisesRegex(RuntimeError, "teacher_sha256"):
                        runner.cache_targets(args, cfg)
            with h5py.File(base / "targets.h5", "r") as h5:
                self.assertEqual(h5["targets/example/teacher"].shape, (199, 12))
                self.assertEqual(h5["targets/example/mel"].shape, (80, 251))
                np.testing.assert_array_equal(h5["targets/example/wave"][:24000], wave)
                self.assertFalse(h5["targets/example/wave"][24000:].any())
                self.assertAlmostEqual(float(h5["speech_times"][1] - h5["speech_times"][0]), .02)

    def test_time_interpolation_lag_and_fixed_endpoint_extension(self):
        times = torch.arange(5).float()
        value = (times * 2)[None, :, None]
        actual = sample_at(value, times, torch.tensor([-.1, .1, 1.4, 4.1]))
        torch.testing.assert_close(actual.flatten(), torch.tensor([0., .2, 2.8, 8.]))

    def test_decoder_consumes_sequence_and_cannot_update_in_eeg_training(self):
        decoder = self.decoder()
        model = AlignedEEGModel(decoder, dimension=12, heads=3, layers=1, local_layers=1, token_steps=24)
        eeg = torch.randn(2, 3, 1178); xyz = torch.randn(2, 3, 3)
        cm = torch.ones(2, 3, dtype=torch.bool); tm = torch.ones(2, 1178, dtype=torch.bool)
        state = model(eeg, xyz, cm, tm)
        teacher = torch.randn_like(state.aligned_sequence, requires_grad=True)
        loss, _ = alignment_loss(state, teacher, torch.randn_like(state.native_mel), decoder.normalizer,
                                  torch.randn(3, 12), torch.tensor([0, 1]), acoustic=True)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(all(p.grad is None for p in decoder.parameters()))
        self.assertGreater(float(model.speech_head.weight.grad.abs().sum()), 0)
        with torch.no_grad():
            a = decoder(torch.randn(1, 20, 12)); b = decoder(torch.randn(1, 20, 12))
        self.assertGreater(float((a - b).abs().max()), 1e-4)
        tm[0, -1] = False
        with self.assertRaises(ValueError):
            model(eeg, xyz, cm, tm)

    def test_content_and_waveform_aliases_cannot_cross_splits(self):
        rows = [{"trial_id": f"{s}:{c}", "subject": s, "linguistic_content_id": str(c), "audio_sha256": str(c)}
                for s in ("a", "b", "c", "d", "e") for c in range(30)]
        rows.append({"trial_id": "alias", "subject": "a", "linguistic_content_id": "other", "audio_sha256": "1"})
        frame = content_split(pd.DataFrame(rows), reserved_train_ids=["a:1"])
        self.assertTrue((frame[frame.audio_sha256 == "1"].role == "train").all())
        self.assertEqual(frame.groupby("audio_sha256").role.nunique().max(), 1)
        self.assertEqual(frame.groupby("linguistic_content_id").role.nunique().max(), 1)
        joint = content_split(pd.DataFrame(rows), joint_ood=True)
        active = joint[joint.role != "excluded"]
        self.assertEqual(active.groupby("subject").role.nunique().max(), 1)

    def fixture(self, base):
        rows = []
        manifest = base / "manifest.csv"; cache = base / "targets.h5"; norm = base / "eeg_normalizer.json"
        for index in range(12):
            content = index % 6; role = "train" if content < 2 else "validation" if content < 4 else "test"
            rows.append(dict(trial_id=f"trial-{index}", subject=f"s{index // 6}", content_group=f"c{content}",
                             linguistic_content_id=f"c{content}", audio_sha256=f"audio{content}", audio_key=f"a{content}",
                             role=role, is_m0=role == "train", shard_path=str(base / "eeg.h5"), shard_row=index,
                             preprocess_config_sha256="config", source_lock_sha256="lock", channel_order_hash="channels", split_index_sha256="split"))
        pd.DataFrame(rows).to_csv(manifest, index=False)
        with h5py.File(base / "eeg.h5", "w") as h5:
            h5.attrs.update(eeg_unit="V", model_time_mask_policy="fixed_full_epoch", preprocess_config_sha256="config",
                            source_lock_sha256="lock", channel_order_hash="channels", split_index_sha256="split")
            eeg = np.random.default_rng(31).normal(size=(12, 3, 1178)).astype("float32") * 1e-5
            # Held-out rows have enormous different scale: normalizer must ignore them.
            eeg[[i for i, r in enumerate(rows) if r["role"] != "train"]] *= 1000
            h5.create_dataset("eeg", data=eeg)
            h5.create_dataset("eeg_valid_mask", data=np.ones((12, 1178), dtype=bool))
            h5.create_dataset("channel_valid_mask", data=np.ones((12, 3), dtype=bool))
            h5.create_dataset("channel_xyz", data=np.random.default_rng(2).normal(size=(3, 3)))
            h5.create_dataset("provenance/trial_id", data=np.array([r["trial_id"] for r in rows], dtype=h5py.string_dtype()))
        for row in rows:
            row["shard_sha256"] = sha256(base / "eeg.h5")
        pd.DataFrame(rows).to_csv(manifest, index=False)
        fit_eeg_normalizer(base, manifest, norm)
        with h5py.File(cache, "w") as h5:
            h5.attrs.update(contract=CONTRACT, manifest_sha256=sha256(manifest), teacher_sha256="frozen-teacher")
            h5.create_dataset("speech_times", data=np.arange(20, dtype="float32") * .2)
            h5.create_dataset("mel_times", data=np.arange(25, dtype="float32") * .16)
            for content in range(6):
                group = h5.create_group(f"targets/a{content}")
                group.attrs["source_samples_16k"] = 24 * 256
                rng = np.random.default_rng(content)
                group.create_dataset("teacher", data=rng.normal(size=(20, 12)).astype("float32"))
                group.create_dataset("mel", data=rng.normal(size=(80, 25)).astype("float32"))
                group.create_dataset("mfcc", data=rng.normal(size=(39, 161)).astype("float32"))
                group.create_dataset("wave", data=np.zeros(64000, dtype="float32"))
        return manifest, cache, norm

    def test_normalizer_bank_and_wrong_trial_have_no_heldout_shortcuts(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); manifest, cache, norm = self.fixture(base)
            payload = json.loads(norm.read_text())
            self.assertLess(max(payload["scale"]), 2e-5)
            train = AlignedDataset(base, manifest, cache, "train", norm)
            labels, bank = fixed_bank(train, self.decoder().normalizer)
            self.assertEqual(labels, ["c0", "c1"])
            self.assertFalse(bank.requires_grad)
            indices = wrong_trial_indices(train.frame)
            for i, j in enumerate(indices):
                self.assertEqual(train.frame.iloc[i].subject, train.frame.iloc[j].subject)
                self.assertNotEqual(train.frame.iloc[i].content_group, train.frame.iloc[j].content_group)
            with h5py.File(base / "eeg.h5", "a") as h5:
                h5["eeg_valid_mask"][0, -1] = False
            with self.assertRaises(ValueError):
                train[0]

    def test_synthetic_training_resume_evaluation_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); manifest, cache, norm = self.fixture(base)
            cfg = {"schema_version": CONTRACT, "artifact_root": str(base), "decoder": {"hidden": 12, "layers": 2},
                   "encoder": {"dimension": 12, "heads": 3, "layers": 1, "local_layers": 1, "token_steps": 24},
                   "minimum_epochs": 2, "patience": 2}
            train = AlignedDataset(base, manifest, cache, "train", audio_only=True)
            validation = AlignedDataset(base, manifest, cache, "validation", audio_only=True)
            decoder, spec = runner.make_decoder(cfg, train)
            args = argparse.Namespace(manifest=None, cache=None, initialize=None, stage="adapt", seed=31,
                                      lr=.001, batch_size=2, epochs=1, max_steps=0, checkpoint_every=1)
            output = base / "audio"; output.mkdir()
            runner.train_loop(args, cfg, decoder, train, validation, output, torch.device("cpu"), spec, audio=True)
            payload = runner.load_payload(output / "best_checkpoint.pt")
            self.assertEqual(payload["stage"], "adapt")
            runner.train_loop(args, cfg, decoder, train, validation, output, torch.device("cpu"), spec, audio=True)
            args.seed = 47
            with self.assertRaises(RuntimeError):
                runner.train_loop(args, cfg, decoder, train, validation, output, torch.device("cpu"), spec, audio=True)
            args.seed = 31
            baseline_args = argparse.Namespace(**vars(args)); baseline_args.stage = "mfcc"
            baseline_args.output = str(base / "mfcc"); baseline_args.device = "cpu"
            runner.train_audio(baseline_args, cfg)
            self.assertEqual(runner.load_payload(base / "mfcc/best_checkpoint.pt")["stage"], "mfcc")
            eeg_args = argparse.Namespace(**vars(args)); eeg_args.initialize = str(output / "best_checkpoint.pt")
            eeg_args.m0 = True; eeg_args.lag_ms = 100; eeg_args.stage = "align"; eeg_args.device = "cpu"
            eeg_args.output = str(base / "align")
            runner.train_eeg(eeg_args, cfg)
            trained = runner.load_payload(base / "align/best_checkpoint.pt")
            self.assertEqual(trained["decoder_origin_sha256"], sha256(Path(eeg_args.initialize)))
            for name, value in payload["decoder"].items():
                torch.testing.assert_close(value, trained["decoder"][name])
            eeg_args.initialize = str(base / "align/best_checkpoint.pt"); eeg_args.stage = "finetune"
            eeg_args.output = str(base / "finetune"); runner.train_eeg(eeg_args, cfg)
            refine_args = argparse.Namespace(**vars(eeg_args))
            refine_args.refine_finetune = True
            refine_args.initialize = str(base / "finetune/last_checkpoint.pt")
            refine_args.output = str(base / "refine")
            runner.train_eeg(refine_args, cfg)
            refined = runner.load_payload(base / "refine/last_checkpoint.pt")
            self.assertEqual(refined["signature"]["initialize"], sha256(Path(refine_args.initialize)))
            self.assertEqual(refined["decoder_origin_sha256"], trained["decoder_origin_sha256"])
            for name, value in trained["decoder"].items():
                torch.testing.assert_close(value, refined["decoder"][name])
            refine_args.output = str(base / "finetune")
            with self.assertRaisesRegex(ValueError, "new output directory"):
                runner.train_eeg(refine_args, cfg)
            evaluation_args = argparse.Namespace(checkpoint=str(base / "finetune/last_checkpoint.pt"), role="train",
                m0=True, selection=None, device="cpu", batch_size=2, output=str(base / "evaluation.json"))
            runner.evaluate(evaluation_args, cfg)
            metrics = json.loads((base / "evaluation.json").read_text())
            self.assertEqual(metrics["pairs"], 4)
            self.assertIn("subject_content_bootstrap", metrics)
            export_args = argparse.Namespace(**vars(evaluation_args)); export_args.kind = "eeg"
            export_args.output = str(base / "export"); export_args.hifigan = str(base); export_args.mfcc_checkpoint = None
            class DummyVocoder:
                def __init__(self, *args, **kwargs): pass
                def synthesize(self, mel):
                    return F.interpolate(mel.mean(1, keepdim=True), size=64000, mode="linear", align_corners=False)[:, 0] * .01
            from torch.nn import functional as F
            references = {str(row.trial_id): {"reference_transcript": "the cat sat", "verified": True}
                          for row in pd.read_csv(manifest, keep_default_na=False).query("role == 'train' and is_m0 == True").itertuples()}
            with patch.object(runner, "SpeechT5HiFiGan", DummyVocoder), patch.object(runner, "tree_hash", return_value="test-vocoder"), \
                 patch.object(runner, "official_reference_transcripts", return_value=(references, {"kind": "synthetic"})):
                runner.export_audio(export_args, cfg)
            self.assertTrue((base / "export/blind/transcriptions.csv").exists())
            self.assertEqual(len(list((base / "export/bundles").glob("*/*.wav"))), 4 * 7)
            cfg["objective_only"] = True
            export_args.output = str(base / "objective_export")
            with patch.object(runner, "SpeechT5HiFiGan", DummyVocoder), patch.object(runner, "tree_hash", return_value="test-vocoder"), \
                 patch.object(runner, "official_reference_transcripts", return_value=(references, {"kind": "synthetic"})):
                runner.export_audio(export_args, cfg)
            self.assertFalse((base / "objective_export/blind").exists())
            self.assertFalse((base / "objective_export/private_key.csv").exists())
            self.assertEqual(len(pd.read_csv(base / "objective_export/reference_transcripts.csv")), 4)

    def test_listening_scores_cannot_fabricate_missing_responses(self):
        self.assertEqual(runner.word_accuracy("The cat sat.", "the cat sat"), 1.)
        self.assertAlmostEqual(runner.word_accuracy("the cat sat", "cat sat"), 2 / 3)
        records = [{"subject": s, "content": c, "gain": 1.} for s in ("a", "b") for c in ("x", "y")]
        result = two_way_bootstrap(records, "gain", repeats=20)
        self.assertEqual(result["ci_low"], 1.)

    def test_interrupt_resumes_exact_optimizer_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); manifest, cache, norm = self.fixture(base)
            cfg = {"artifact_root": str(base), "decoder": {"hidden": 12, "layers": 2}, "minimum_epochs": 2, "patience": 2}
            train = AlignedDataset(base, manifest, cache, "train", audio_only=True)
            validation = AlignedDataset(base, manifest, cache, "validation", audio_only=True)
            original, spec = runner.make_decoder(cfg, train)
            interrupted = copy.deepcopy(original)
            args = argparse.Namespace(manifest=None, cache=None, initialize=None, stage="adapt", seed=31,
                                      lr=.001, batch_size=1, epochs=2, max_steps=0, checkpoint_every=1)
            continuous = base / "continuous"; continuous.mkdir()
            resumed = base / "resumed"; resumed.mkdir()
            runner.train_loop(args, cfg, original, train, validation, continuous, torch.device("cpu"), spec, audio=True)
            original_step = torch.optim.AdamW.step
            def step_and_interrupt(optimizer, *a, **kw):
                value = original_step(optimizer, *a, **kw)
                os.kill(os.getpid(), signal.SIGINT)
                return value
            with patch.object(torch.optim.AdamW, "step", step_and_interrupt):
                runner.train_loop(args, cfg, interrupted, train, validation, resumed, torch.device("cpu"), spec, audio=True)
            progress = runner.load_payload(resumed / "training_state.pt")
            self.assertEqual(progress["step"], 1)
            self.assertFalse(progress["complete"])
            runner.train_loop(args, cfg, interrupted, train, validation, resumed, torch.device("cpu"), spec, audio=True)
            a = runner.load_payload(continuous / "last_checkpoint.pt")
            b = runner.load_payload(resumed / "last_checkpoint.pt")
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)

    def test_improving_ineligible_training_does_not_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); manifest, cache, norm = self.fixture(base)
            cfg = {"artifact_root": str(base), "decoder": {"hidden": 12, "layers": 2},
                   "encoder": {"dimension": 12, "heads": 3, "layers": 1,
                               "local_layers": 1, "token_steps": 30},
                   "minimum_epochs": 1, "patience": 1}
            train = AlignedDataset(base, manifest, cache, "train", norm, m0=True)
            decoder, spec = runner.make_decoder(cfg, train)
            model = AlignedEEGModel(decoder, **cfg["encoder"])
            args = argparse.Namespace(manifest=None, cache=None, initialize=None, stage="finetune", seed=31,
                                      lr=.00001, batch_size=1, epochs=3, max_steps=0, checkpoint_every=50, m0=True)
            reports = [{"native_mel_mae": value, "beats_template": False, "beats_wrong_trial": True}
                       for value in [3., 2.9, 2.8]]
            out = base / "training"; out.mkdir()
            with patch.object(runner, "evaluate_model", side_effect=reports):
                runner.train_loop(args, cfg, model, train, train, out, torch.device("cpu"), spec, audio=False)
            saved = runner.load_payload(out / "training_state.pt")
            self.assertEqual(saved["epoch"], 3)
            self.assertEqual(saved["stale"], 0)
            self.assertFalse((out / "best_checkpoint.pt").exists())
            # Better mel alone must not overwrite the model that passed M0.
            qualified = base / "qualified"; qualified.mkdir()
            reports = [{"native_mel_mae": metric, "beats_template": True,
                        "beats_wrong_trial": True, "m0_passed": passed}
                       for metric, passed in [(1., True), (.8, True), (.6, False)]]
            with patch.object(runner, "evaluate_model", side_effect=reports):
                runner.train_loop(args, cfg, model, train, train, qualified, torch.device("cpu"), spec, audio=False)
            m0_best = runner.load_payload(qualified / "best_m0_checkpoint.pt")
            mel_best = runner.load_payload(qualified / "best_checkpoint.pt")
            self.assertEqual(m0_best["epoch"], 2)
            self.assertEqual(mel_best["epoch"], 3)
            self.assertTrue(m0_best["history"][-1]["m0_passed"])
            # A completed resume keeps the qualified checkpoint intact.
            original_hash = sha256(qualified / "best_m0_checkpoint.pt")
            runner.train_loop(args, cfg, model, train, train, qualified, torch.device("cpu"), spec, audio=False)
            self.assertEqual(original_hash, sha256(qualified / "best_m0_checkpoint.pt"))

    def test_legacy_reopen_requires_progress_and_remaining_budget(self):
        args = argparse.Namespace(resume_early_stop_fix=True, epochs=100, max_steps=5000)
        cfg = {"minimum_epochs": 20, "patience": 10}
        saved = {"complete": True, "stage": "finetune", "epoch": 20, "step": 1000, "stale": 20,
                 "history": [{"native_mel_mae": 3. - i * .01} for i in range(20)]}
        self.assertTrue(runner.can_resume_early_stop_fix(saved, args, cfg))
        for changes in ({"step": 5000}, {"epoch": 100}, {"stopping_policy": "metric_progress_v2"},
                        {"history": [{"native_mel_mae": 3.}] * 20}):
            self.assertFalse(runner.can_resume_early_stop_fix({**saved, **changes}, args, cfg))
        args.resume_early_stop_fix = False
        self.assertFalse(runner.can_resume_early_stop_fix(saved, args, cfg))

    def test_blind_review_rejects_incomplete_and_scores_paired_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            key = [{"sample_id": f"{i}-{c}", "trial_id": str(i), "condition": c}
                   for i in range(40) for c in ("correct", "wrong_trial")]
            pd.DataFrame(key).to_csv(base / "private_key.csv", index=False)
            refs = [{"trial_id": str(i), "reference_transcript": "the cat sat", "verified": True} for i in range(40)]
            pd.DataFrame(refs).to_csv(base / "refs.csv", index=False)
            answers = [{"listener_id": str(listener), "sample_id": r["sample_id"],
                        "transcript": "the cat sat" if r["condition"] == "correct" else "a dog ran"}
                       for listener in range(3) for r in key]
            pd.DataFrame(answers).to_csv(base / "answers.csv", index=False)
            (base / "export_manifest.json").write_text(json.dumps({"signature": {"kind": "eeg", "checkpoint_sha256": "synthetic-test-only"},
                                                                  "blind_csv_sha256": sha256(base / "private_key.csv")}))
            args = argparse.Namespace(export_root=str(base), transcriptions=str(base / "answers.csv"), references=str(base / "refs.csv"), output=str(base / "score.json"))
            runner.score_review(args, {})
            self.assertTrue(json.loads((base / "score.json").read_text())["passed"])
            answers[0]["transcript"] = ""
            pd.DataFrame(answers).to_csv(base / "answers.csv", index=False)
            with self.assertRaises(ValueError):
                runner.score_review(args, {})
            with self.assertRaises(ValueError):
                runner.require_selection(argparse.Namespace(role="test", selection=None, m0=False), {"signature": {"m0": False}})

    def test_prepare_materializes_all_roles_once_and_repairs_legacy_hash(self):
        import prepare_training_data
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); data_root = base / "raw_artifacts"; out = base / "aligned"
            (data_root / "manifests").mkdir(parents=True); (data_root / "splits").mkdir()
            (data_root / "splits/assignment.json").write_text("{}")
            (data_root / "source_lock.json").write_text(json.dumps({"config_sha256": "config", "files": []}))
            rows = []
            for s in range(5):
                for c in range(20):
                    rows.append(dict(trial_id=f"s{s}c{c}", subject=f"s{s}", linguistic_content_id=f"c{c}", audio_sha256=f"a{c}",
                                     dataset="ds004940", task="N400Active", pairing_level="verified_exact", build_status="included", qc_pass=True,
                                     bad_channels="[]", preprocess_config_sha256="config", source_lock_sha256="lock", channel_order_hash="channels"))
            original = pd.DataFrame(rows)
            original.to_csv(data_root / "manifests/manifest_all.csv", index=False)
            original[original.linguistic_content_id.isin([f"c{i}" for i in range(10)])].to_csv(base / "old_m0.csv", index=False)
            data_cfg = {"_config_sha256": "config", "output_root": str(data_root), "harmonized": {"interpolation": {"max_bad_fraction": .15}},
                        "sources": {"ds004940": {"channel_order": ["a", "b", "c"]}}}
            cfg = {"artifact_root": str(out), "data_config": "unused", "split_seed": 31, "legacy_m0_manifest": str(base / "old_m0.csv")}
            calls = []
            def build(*args):
                calls.append(args)
                transport = pd.read_csv(data_root / "splits/aligned_v1_materialize_fold-0.csv")
                self.assertEqual(set(transport.role), {"materialize"})
                built = original.copy(); built["shard_path"] = str(base / "shard.h5")
                built["shard_row"] = range(len(built)); built["split_index_sha256"] = "legacy-global-hash"
                built.to_csv(data_root / "manifests/manifest_aligned_v1.csv", index=False)
                with h5py.File(base / "shard.h5", "w") as h5:
                    h5.attrs["split_index_sha256"] = sha256(data_root / "splits/aligned_v1_materialize_fold-0.csv")
            with patch.object(prepare_training_data, "load_config", return_value=(data_cfg, "config")), \
                 patch.object(prepare_training_data, "build", build), patch.object(runner, "fit_eeg_normalizer") as normalize:
                runner.prepare(argparse.Namespace(materialize=True), cfg)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][7], "materialize")      # build() lost its tms_condition argument
            prepared = pd.read_csv(out / "manifest.csv")
            self.assertEqual(set(prepared.role), {"train", "validation", "test"})
            self.assertEqual(set(prepared.split_index_sha256), {sha256(data_root / "splits/aligned_v1_materialize_fold-0.csv")})
            self.assertEqual(int(prepared.is_m0.sum()), 50)
            normalize.assert_called_once()


if __name__ == "__main__":
    unittest.main()

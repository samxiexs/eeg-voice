from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.alignment_retrieval import (  # noqa: E402
    RetrievalBank,
    build_retrieval_bank,
    build_waveform_distance_bank,
    compute_first_match_ranks,
    compute_rank_metrics,
    evaluate_embedding_retrieval,
    evaluate_waveform_nta,
    expected_random_retrieval_metrics,
)


def _tone(freq_hz: float, samples: int = 4096, sample_rate: int = 16000) -> np.ndarray:
    t = np.arange(samples, dtype=np.float32) / float(sample_rate)
    return np.sin(2.0 * np.pi * float(freq_hz) * t).astype(np.float32)


class _DummyDataset:
    def __init__(self, protocol: str):
        self.protocol = protocol
        self.target_kind = "hubert_sequence"
        self._metadata = {
            "01:a": {"subject_id": "01", "label": "a", "audio_relpath": "audio/01_a.wav", "audio_path": "audio/01_a.wav"},
            "01:b": {"subject_id": "01", "label": "b", "audio_relpath": "audio/01_b.wav", "audio_path": "audio/01_b.wav"},
            "02:a": {"subject_id": "02", "label": "a", "audio_relpath": "audio/02_a.wav", "audio_path": "audio/02_a.wav"},
            "03:a": {"subject_id": "03", "label": "a", "audio_relpath": "audio/03_a.wav", "audio_path": "audio/03_a.wav"},
        }
        self._targets = {
            "01:a": np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            "01:b": np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
            "02:a": np.asarray([[0.9, 0.1], [0.85, 0.15]], dtype=np.float32),
            "03:a": np.asarray([[0.88, 0.12], [0.82, 0.18]], dtype=np.float32),
        }
        self._waveforms = {
            "audio/01_a.wav": _tone(220.0),
            "audio/01_b.wav": _tone(440.0),
            "audio/02_a.wav": _tone(260.0),
            "audio/03_a.wav": _tone(300.0),
        }
        self._splits = {
            "train": ["01:a", "01:b", "02:a"],
            "test": ["03:a"],
        }

    def unique_template_ids(self, split: str | None = None):
        return list(self._splits["train" if split is None else split])

    def template_metadata(self, template_id: str):
        return dict(self._metadata[template_id])

    def get_template_target(self, template_id: str):
        sequence = self._targets[template_id]
        return {
            "speech_embedding": sequence.mean(axis=0).astype(np.float32),
            "target_sequence": sequence.astype(np.float32),
            "target_mask": np.ones(sequence.shape[0], dtype=np.float32),
            "target_summary": sequence.mean(axis=0).astype(np.float32),
            "decoder_scale": None,
        }

    def _load_audio(self, relpath: str):
        return np.asarray(self._waveforms[relpath], dtype=np.float32)


class AlignmentRetrievalTests(unittest.TestCase):
    def test_build_retrieval_bank_protocol_u_policies(self):
        dataset = _DummyDataset(protocol="U")
        strict_bank = build_retrieval_bank(dataset, "unseen_strict_seen_subjects")
        oracle_bank = build_retrieval_bank(dataset, "unseen_oracle_holdout")
        self.assertEqual(strict_bank.template_ids, ("01:a", "01:b", "02:a"))
        self.assertEqual(strict_bank.match_mode, "label")
        self.assertEqual(oracle_bank.template_ids, ("03:a",))
        self.assertEqual(oracle_bank.match_mode, "exact")

    def test_sequence_rank_metrics_and_ranks(self):
        bank = RetrievalBank(
            policy="pooled_train",
            match_mode="exact",
            target_kind="hubert_sequence",
            template_ids=("01:a", "01:b", "02:a"),
            subject_ids=("01", "01", "02"),
            labels=("a", "b", "a"),
            audio_paths=("audio/01_a.wav", "audio/01_b.wav", "audio/02_a.wav"),
            sequences=np.asarray(
                [
                    [[1.0, 0.0], [1.0, 0.0]],
                    [[0.0, 1.0], [0.0, 1.0]],
                    [[0.9, 0.1], [0.85, 0.15]],
                ],
                dtype=np.float32,
            ),
            masks=np.ones((3, 2), dtype=np.float32),
            summaries=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.875, 0.125]], dtype=np.float32),
            waveforms=np.stack([_tone(220.0), _tone(440.0), _tone(260.0)], axis=0),
        )
        predicted = np.asarray(
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[0.92, 0.08], [0.90, 0.10]],
            ],
            dtype=np.float32,
        )
        evaluation = evaluate_embedding_retrieval(
            bank=bank,
            predicted_sequences=predicted,
            predicted_summaries=predicted.mean(axis=1),
            target_template_ids=["01:a", "02:a"],
            target_labels=["a", "a"],
            top_k=3,
        )
        metrics = evaluation["metrics"]
        self.assertAlmostEqual(metrics["retrieval_top1_exact"], 1.0)
        self.assertAlmostEqual(metrics["retrieval_top3_exact"], 1.0)
        self.assertAlmostEqual(metrics["MRR"], 1.0)
        self.assertAlmostEqual(metrics["mean_rank"], 1.0)
        first_ranks = compute_first_match_ranks(
            order=evaluation["order"],
            bank=bank,
            target_template_ids=["01:a", "02:a"],
            target_labels=["a", "a"],
        )
        self.assertEqual(first_ranks, [1, 1])

    def test_exact_and_label_rank_metrics_diverge(self):
        bank = RetrievalBank(
            policy="pooled_train",
            match_mode="exact",
            target_kind="hubert_sequence",
            template_ids=("01:a", "01:b", "02:a"),
            subject_ids=("01", "01", "02"),
            labels=("a", "b", "a"),
            audio_paths=("audio/01_a.wav", "audio/01_b.wav", "audio/02_a.wav"),
            sequences=np.asarray(
                [
                    [[1.0, 0.0], [1.0, 0.0]],
                    [[0.0, 1.0], [0.0, 1.0]],
                    [[0.9, 0.1], [0.9, 0.1]],
                ],
                dtype=np.float32,
            ),
            masks=np.ones((3, 2), dtype=np.float32),
            summaries=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32),
            waveforms=np.stack([_tone(220.0), _tone(440.0), _tone(260.0)], axis=0),
        )
        predicted = np.asarray(
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[0.99, 0.01], [0.99, 0.01]],
            ],
            dtype=np.float32,
        )
        exact_eval = evaluate_embedding_retrieval(
            bank=bank,
            predicted_sequences=predicted,
            predicted_summaries=predicted.mean(axis=1),
            target_template_ids=["01:a", "02:a"],
            target_labels=["a", "a"],
            top_k=3,
        )
        label_metrics = compute_rank_metrics(
            ranked_candidates=exact_eval["ranked_candidates"],
            order=exact_eval["order"],
            bank=RetrievalBank(
                policy="unseen_strict_seen_subjects",
                match_mode="label",
                target_kind=bank.target_kind,
                template_ids=bank.template_ids,
                subject_ids=bank.subject_ids,
                labels=bank.labels,
                audio_paths=bank.audio_paths,
                sequences=bank.sequences,
                masks=bank.masks,
                summaries=bank.summaries,
                waveforms=bank.waveforms,
            ),
            target_template_ids=["03:a", "03:a"],
            target_labels=["a", "a"],
            top_k_values=(1, 3),
        )
        self.assertAlmostEqual(exact_eval["metrics"]["retrieval_top1_exact"], 0.5)
        self.assertAlmostEqual(label_metrics["retrieval_top1_label"], 1.0)

    def test_waveform_nta_exact_and_label(self):
        waveforms = np.stack([_tone(220.0), _tone(440.0), _tone(260.0)], axis=0)
        exact_bank = RetrievalBank(
            policy="pooled_train",
            match_mode="exact",
            target_kind="hubert_sequence",
            template_ids=("01:a", "01:b", "02:a"),
            subject_ids=("01", "01", "02"),
            labels=("a", "b", "a"),
            audio_paths=("audio/01_a.wav", "audio/01_b.wav", "audio/02_a.wav"),
            sequences=np.asarray(
                [
                    [[1.0, 0.0], [1.0, 0.0]],
                    [[0.0, 1.0], [0.0, 1.0]],
                    [[0.9, 0.1], [0.9, 0.1]],
                ],
                dtype=np.float32,
            ),
            masks=np.ones((3, 2), dtype=np.float32),
            summaries=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32),
            waveforms=waveforms,
        )
        exact_features = build_waveform_distance_bank(exact_bank)
        exact_eval = evaluate_waveform_nta(
            output_waveforms=[waveforms[0]],
            target_template_ids=["01:a"],
            target_labels=["a"],
            bank_features=exact_features,
        )
        self.assertAlmostEqual(exact_eval["metrics"]["NTA_exact"], 1.0)

        label_bank = RetrievalBank(
            policy="unseen_strict_seen_subjects",
            match_mode="label",
            target_kind=exact_bank.target_kind,
            template_ids=exact_bank.template_ids,
            subject_ids=exact_bank.subject_ids,
            labels=exact_bank.labels,
            audio_paths=exact_bank.audio_paths,
            sequences=exact_bank.sequences,
            masks=exact_bank.masks,
            summaries=exact_bank.summaries,
            waveforms=exact_bank.waveforms,
        )
        label_features = build_waveform_distance_bank(label_bank)
        label_eval = evaluate_waveform_nta(
            output_waveforms=[waveforms[0]],
            target_template_ids=["03:a"],
            target_labels=["a"],
            bank_features=label_features,
        )
        self.assertAlmostEqual(label_eval["metrics"]["NTA_label"], 1.0)

    def test_expected_random_metrics_include_rank_terms(self):
        bank = RetrievalBank(
            policy="unseen_strict_seen_subjects",
            match_mode="label",
            target_kind="hubert_sequence",
            template_ids=("01:a", "01:b", "02:a"),
            subject_ids=("01", "01", "02"),
            labels=("a", "b", "a"),
            audio_paths=("audio/01_a.wav", "audio/01_b.wav", "audio/02_a.wav"),
            sequences=np.asarray(
                [
                    [[1.0, 0.0], [1.0, 0.0]],
                    [[0.0, 1.0], [0.0, 1.0]],
                    [[0.9, 0.1], [0.9, 0.1]],
                ],
                dtype=np.float32,
            ),
            masks=np.ones((3, 2), dtype=np.float32),
            summaries=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32),
            waveforms=np.stack([_tone(220.0), _tone(440.0), _tone(260.0)], axis=0),
        )
        metrics = expected_random_retrieval_metrics(
            bank=bank,
            target_template_ids=["03:a"],
            target_labels=["a"],
            top_k=3,
        )
        self.assertIn("MRR", metrics)
        self.assertIn("mean_rank", metrics)
        self.assertGreater(metrics["MRR"], 0.0)
        self.assertGreater(metrics["mean_rank"], 0.0)


if __name__ == "__main__":
    unittest.main()

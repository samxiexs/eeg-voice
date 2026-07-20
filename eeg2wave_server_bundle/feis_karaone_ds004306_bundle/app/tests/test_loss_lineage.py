from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.combined_0715.lineage import (
    CHECKPOINT_SCHEMA_VERSION,
    build_run_lineage,
    file_sha256,
    preauthorize_locked_test,
    preprocessing_sha256_from_rows,
    validate_checkpoint_payload,
    validate_gate_binding,
)
from src.combined_0715.losses import multi_positive_contrastive_loss


def test_hierarchical_multi_positive_loss_is_finite_and_active() -> None:
    audio = torch.tensor(
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    good = audio.clone().requires_grad_(True)
    bad = audio[[2, 3, 0, 1]].clone().requires_grad_(True)
    labels = torch.tensor([0, 0, 1, 2])
    subjects = torch.tensor([0, 1, 2, 3])
    audio_ids = torch.tensor([10, 11, 12, 13])
    good_result = multi_positive_contrastive_loss(
        good,
        audio,
        labels,
        subjects,
        audio_ids,
        cross_subject_weight=0.25,
    )
    bad_result = multi_positive_contrastive_loss(
        bad,
        audio,
        labels,
        subjects,
        audio_ids,
        cross_subject_weight=0.25,
    )
    assert good_result["total"] < bad_result["total"]
    assert float(good_result["extra_positive_fraction"]) > 0.0
    assert float(good_result["mean_positive_count"]) > 1.0
    good_result["total"].backward()
    assert good.grad is not None and torch.isfinite(good.grad).all()


def test_exact_audio_key_is_a_strong_extra_positive() -> None:
    value = torch.eye(3, dtype=torch.float32)
    result = multi_positive_contrastive_loss(
        value,
        value,
        torch.tensor([0, 0, 1]),
        torch.tensor([0, 0, 1]),
        torch.tensor([5, 5, 6]),
        cross_subject_weight=0.25,
    )
    assert float(result["extra_positive_fraction"]) > 0.0
    assert torch.isfinite(result["total"])


def test_contrastive_masks_distinguish_neutral_soft_positive_and_negative() -> None:
    embeddings = torch.eye(3, dtype=torch.float32)
    labels = torch.tensor([0, 0, 1])

    neutral = multi_positive_contrastive_loss(
        embeddings,
        embeddings,
        labels,
        torch.tensor([0, 0, 2]),
        torch.tensor([10, 11, 12]),
        cross_subject_weight=0.25,
    )
    assert float(neutral["mean_positive_count"]) == pytest.approx(1.0)

    soft = multi_positive_contrastive_loss(
        embeddings,
        embeddings,
        labels,
        torch.tensor([0, 1, 2]),
        torch.tensor([10, 11, 12]),
        cross_subject_weight=0.25,
    )
    assert float(soft["mean_positive_count"]) == pytest.approx(5.0 / 3.0)

    distinct_labels = torch.tensor([0, 1, 2])
    identifiers = torch.arange(3)
    matched = multi_positive_contrastive_loss(
        embeddings,
        embeddings,
        distinct_labels,
        identifiers,
        identifiers,
    )
    mismatched = multi_positive_contrastive_loss(
        embeddings[[1, 2, 0]],
        embeddings,
        distinct_labels,
        identifiers,
        identifiers,
    )
    assert matched["total"] < mismatched["total"]


def _tiny_lineage(tmp_path):
    config = tmp_path / "config.yaml"
    split = tmp_path / "split.yaml"
    manifest = tmp_path / "manifest.csv"
    eeg_root = tmp_path / "eeg"
    payload = eeg_root / "subjects" / "tiny.npz"
    qc = eeg_root / "qc" / "channel_qc.csv"
    cache = tmp_path / "cache.npz"
    payload.parent.mkdir(parents=True)
    qc.parent.mkdir(parents=True)
    config.write_text("version: one\n", encoding="utf-8")
    split.write_text("version: split-one\n", encoding="utf-8")
    manifest.write_text("sample_key\na\n", encoding="utf-8")
    np.savez(payload, eeg=np.zeros((1, 1, 1), dtype=np.float32))
    qc.write_text("dataset,channel\ntiny,F3\n", encoding="utf-8")
    np.savez(cache, version=np.asarray("combined-0715-cache-v2"))
    context = SimpleNamespace(
        root=eeg_root,
        rows=({"eeg_relpath": "subjects/tiny.npz"},),
        split_path=split,
        manifest_path=manifest,
        split={"version": "split-one"},
    )
    bank = SimpleNamespace(path=cache, version="combined-0715-cache-v2")
    _refresh_verification(context)
    return config, split, manifest, payload, cache, context, bank


def _refresh_verification(context) -> None:
    verification = context.root / "qc" / "eeg_verification.jsonl"
    verification.write_text(
        json.dumps({
            "status": "passed",
            "provenance": {
                "locked_split_sha256": file_sha256(context.split_path),
                "manifest_sha256": file_sha256(context.manifest_path),
                "preprocessing_sha256": preprocessing_sha256_from_rows(context.root, context.rows),
            },
        }) + "\n",
        encoding="utf-8",
    )


def test_lineage_changes_only_in_the_mutated_component(tmp_path) -> None:
    config, split, manifest, payload, cache, context, bank = _tiny_lineage(tmp_path)
    first = build_run_lineage(config, context, bank)
    split.write_text("version: split-two\n", encoding="utf-8")
    _refresh_verification(context)
    second = build_run_lineage(config, context, bank)
    assert first["split_sha256"] != second["split_sha256"]
    assert first["preprocessing_sha256"] == second["preprocessing_sha256"]

    manifest.write_text("sample_key\nb\n", encoding="utf-8")
    _refresh_verification(context)
    third = build_run_lineage(config, context, bank)
    assert second["manifest_sha256"] != third["manifest_sha256"]
    assert second["preprocessing_sha256"] == third["preprocessing_sha256"]

    np.savez(payload, eeg=np.ones((1, 1, 1), dtype=np.float32))
    _refresh_verification(context)
    fourth = build_run_lineage(config, context, bank)
    assert third["preprocessing_sha256"] != fourth["preprocessing_sha256"]
    assert third["cache_sha256"] == fourth["cache_sha256"]

    np.savez(cache, version=np.asarray("combined-0715-cache-v2"), changed=np.asarray(1))
    fifth = build_run_lineage(config, context, bank)
    assert fourth["cache_sha256"] != fifth["cache_sha256"]
    assert fourth["preprocessing_sha256"] == fifth["preprocessing_sha256"]


def test_lineage_rejects_a_stale_preprocessing_verification(tmp_path) -> None:
    config, _, _, payload, _, context, bank = _tiny_lineage(tmp_path)
    build_run_lineage(config, context, bank)
    np.savez(payload, eeg=np.ones((1, 1, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="verification report is stale"):
        build_run_lineage(config, context, bank)


def test_checkpoint_schema_and_dependencies_are_strict(tmp_path) -> None:
    config, _, _, _, _, context, bank = _tiny_lineage(tmp_path)
    lineage = build_run_lineage(config, context, bank)
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "phase": "eeg",
        "lineage": lineage,
        "dependencies": {"audio_checkpoint_sha256": "abc"},
    }
    validate_checkpoint_payload(
        payload,
        expected_phase="eeg",
        expected_lineage=lineage,
        expected_dependencies={"audio_checkpoint_sha256": "abc"},
    )
    with pytest.raises(ValueError, match="dependency mismatch"):
        validate_checkpoint_payload(
            payload,
            expected_phase="eeg",
            expected_lineage=lineage,
            expected_dependencies={"audio_checkpoint_sha256": "different"},
        )
    with pytest.raises(ValueError, match="legacy/incompatible"):
        validate_checkpoint_payload(
            {"phase": "eeg"},
            expected_phase="eeg",
            expected_lineage=lineage,
        )


def test_validation_gate_binds_the_exact_report(tmp_path) -> None:
    config, _, _, _, _, context, bank = _tiny_lineage(tmp_path)
    lineage = build_run_lineage(config, context, bank)
    report_path = tmp_path / "validation.json"
    report_path.write_text(
        json.dumps({
            "lineage": lineage,
            "audio_checkpoint_sha256": "audio",
            "eeg_checkpoint_sha256": "eeg",
            "split": "validation",
            "test_accessed": False,
        }),
        encoding="utf-8",
    )
    gate = {
        "passed": True,
        "lineage": lineage,
        "audio_checkpoint_sha256": "audio",
        "eeg_checkpoint_sha256": "eeg",
        "validation_report": str(report_path),
        "validation_report_sha256": file_sha256(report_path),
    }
    validate_gate_binding(
        gate,
        lineage=lineage,
        audio_checkpoint_sha256="audio",
        eeg_checkpoint_sha256="eeg",
    )
    report_path.write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError, match="report SHA256 mismatch"):
        validate_gate_binding(
            gate,
            lineage=lineage,
            audio_checkpoint_sha256="audio",
            eeg_checkpoint_sha256="eeg",
        )


def test_locked_test_preauthorization_uses_only_bound_artifact_metadata(tmp_path) -> None:
    config, _, _, _, _, context, bank = _tiny_lineage(tmp_path)
    lineage = build_run_lineage(config, context, bank)
    audio = tmp_path / "audio.pt"
    eeg = tmp_path / "eeg.pt"
    audio.write_bytes(b"audio-checkpoint")
    eeg.write_bytes(b"eeg-checkpoint")
    audio_sha = file_sha256(audio)
    eeg_sha = file_sha256(eeg)
    report = tmp_path / "validation.json"
    report.write_text(json.dumps({
        "lineage": lineage,
        "audio_checkpoint_sha256": audio_sha,
        "eeg_checkpoint_sha256": eeg_sha,
        "split": "validation",
        "test_accessed": False,
    }), encoding="utf-8")
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps({
        "passed": True,
        "lineage": lineage,
        "audio_checkpoint_sha256": audio_sha,
        "eeg_checkpoint_sha256": eeg_sha,
        "validation_report": str(report),
        "validation_report_sha256": file_sha256(report),
    }), encoding="utf-8")
    gate = preauthorize_locked_test(
        gate_path,
        config_path=config,
        audio_checkpoint_path=audio,
        eeg_checkpoint_path=eeg,
    )
    assert gate["lineage"] == lineage

    config.write_text("version: changed\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="config mismatch before test access"):
        preauthorize_locked_test(
            gate_path,
            config_path=config,
            audio_checkpoint_path=audio,
            eeg_checkpoint_path=eeg,
        )

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


APP_DIR = Path(__file__).resolve().parents[1]
SCRIPT = APP_DIR / "scripts" / "synthesize_combined_0715.py"
sys.path.insert(0, str(APP_DIR))
SPEC = importlib.util.spec_from_file_location("combined_synthesis", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["combined_synthesis"] = MODULE
SPEC.loader.exec_module(MODULE)


def rows(labels: list[str]) -> list[dict[str, str]]:
    return [
        {"dataset": "feis", "label": label, "sample_key": f"feis:s{index}:{index}"}
        for index, label in enumerate(labels)
    ]


def test_label_derangement_is_deterministic_same_label_and_has_no_self_loops() -> None:
    records = rows(["a", "a", "a", "b", "b"])
    first = MODULE.deterministic_label_derangement(records, seed=15)
    second = MODULE.deterministic_label_derangement(records, seed=15)
    np.testing.assert_array_equal(first, second)
    assert all(int(source) != target for target, source in enumerate(first))
    assert all(records[target]["label"] == records[int(source)]["label"] for target, source in enumerate(first))


def test_label_derangement_rejects_singleton_instead_of_silently_self_pairing() -> None:
    with pytest.raises(ValueError, match="only one trial"):
        MODULE.deterministic_label_derangement(rows(["a", "a", "b"]), seed=15)


def test_output_layout_has_reference_plus_six_controls(tmp_path: Path) -> None:
    folders = MODULE.ensure_output_layout(tmp_path)
    assert set(folders) == set(MODULE.OUTPUT_KINDS)
    assert len(folders) == 7
    assert all(path.is_dir() for path in folders.values())


def test_control_inputs_keep_predicted_probabilities_and_isolate_conditions() -> None:
    target_condition = torch.ones(1, 2, 3)
    zero_condition = torch.full((1, 2, 3), 2.0)
    shuffled_condition = torch.full((1, 2, 3), 3.0)
    target_logits = torch.tensor([[1.0, 2.0, 3.0]])
    zero_logits = torch.tensor([[3.0, 2.0, 1.0]])
    shuffled_logits = torch.tensor([[2.0, 3.0, 1.0]])
    prior = torch.tensor([0.2, 0.3, 0.5])
    controls = MODULE.build_control_inputs(
        {"condition": target_condition, "label_logits": target_logits},
        {"condition": zero_condition, "label_logits": zero_logits},
        {"condition": shuffled_condition, "label_logits": shuffled_logits},
        prior,
    )
    assert torch.equal(controls["eeg_conditioned"][0], target_condition)
    assert torch.count_nonzero(controls["label_only"][0]) == 0
    assert torch.allclose(controls["label_only"][1], torch.softmax(target_logits, -1))
    assert torch.equal(controls["zero_eeg"][0], zero_condition)
    assert torch.allclose(controls["zero_eeg"][1], torch.softmax(zero_logits, -1))
    assert torch.equal(controls["shuffled_eeg"][0], shuffled_condition)
    assert torch.allclose(controls["shuffled_eeg"][1], torch.softmax(shuffled_logits, -1))
    assert torch.count_nonzero(controls["dataset_only"][0]) == 0
    assert torch.equal(controls["dataset_only"][1], prior.view(1, -1))


def test_waveform_metrics_are_exact_for_identical_signal() -> None:
    signal = np.sin(np.linspace(0, 20 * np.pi, 4096, dtype=np.float32))
    metrics = MODULE.waveform_metrics(signal, signal.copy(), 16000)
    assert metrics["waveform_correlation"] == pytest.approx(1.0, abs=1e-7)
    assert metrics["si_sdr_db"] > 100.0
    assert metrics["log_spectrogram_mae_db"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["envelope_correlation"] == pytest.approx(1.0, abs=1e-7)
    assert metrics["envelope_overlap"] == pytest.approx(1.0, abs=1e-7)
    assert metrics["activity_iou"] == pytest.approx(1.0, abs=1e-7)
    assert metrics["structure_score"] == pytest.approx(1.0, abs=1e-7)


def test_phase_changed_carrier_can_have_low_raw_but_high_envelope_similarity() -> None:
    sample_rate = 16000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    amplitude = np.exp(-0.5 * ((time - 0.5) / 0.12) ** 2)
    reference = amplitude * np.sin(2.0 * np.pi * 220.0 * time)
    candidate = amplitude * np.sin(2.0 * np.pi * 220.0 * time + np.pi / 2.0)
    metrics = MODULE.waveform_metrics(reference, candidate, sample_rate)
    assert abs(metrics["waveform_correlation"]) < 0.05
    assert metrics["envelope_correlation"] > 0.99
    assert metrics["structure_score"] > 0.95


def test_envelope_lag_metric_recovers_a_50_ms_shift() -> None:
    sample_rate = 16000
    reference = np.zeros(sample_rate, dtype=np.float64)
    candidate = np.zeros_like(reference)
    reference[4000:8000] = np.hanning(4000)
    shift = int(round(0.050 * sample_rate))
    candidate[4000 + shift : 8000 + shift] = np.hanning(4000)
    metrics = MODULE.waveform_metrics(reference, candidate, sample_rate)
    assert metrics["lag_aligned_envelope_correlation"] > 0.99
    assert metrics["envelope_best_lag_ms"] == pytest.approx(50.0, abs=10.0)
    assert metrics["onset_error_ms"] == pytest.approx(50.0, abs=10.0)


def test_morphology_metrics_remain_finite_for_silence() -> None:
    metrics = MODULE.waveform_metrics(np.zeros(2048), np.zeros(2048), 16000)
    assert all(np.isfinite(value) for value in metrics.values())


def test_limited_synthesis_selection_is_label_balanced_and_deterministic() -> None:
    records = rows(["a"] * 10 + ["b"] * 10 + ["c"] * 10)
    first = MODULE.stratified_evaluation_indices(records, 6, seed=15)
    second = MODULE.stratified_evaluation_indices(records, 6, seed=15)
    assert first == second
    assert [records[index]["label"] for index in first].count("a") == 2
    assert [records[index]["label"] for index in first].count("b") == 2
    assert [records[index]["label"] for index in first].count("c") == 2


def test_gate_policy_forbids_test_bypass_before_reading_gate(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(PermissionError, match="forbidden"):
        MODULE.check_access_policy(
            split="test",
            allow_final_test=True,
            allow_failed_gate=True,
            roundtrip_gate_path=missing,
            validation_gate_path=missing,
            lineage={},
            audio_checkpoint_sha256="audio",
            eeg_checkpoint_sha256="eeg",
        )


def test_main_rejects_missing_test_gate_before_loading_any_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "run:\n  seed: 15\n"
        "paths:\n"
        f"  output_root: {tmp_path / 'outputs'}\n"
        f"  cache_root: {tmp_path / 'cache-root'}\n",
        encoding="utf-8",
    )

    def forbidden_data_access(*args, **kwargs):
        raise AssertionError("locked-test dataset was touched before gate authorization")

    monkeypatch.setattr(MODULE, "load_context", forbidden_data_access)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--config", str(config),
            "--cache", str(tmp_path / "cache.npz"),
            "--audio-checkpoint", str(tmp_path / "audio.pt"),
            "--eeg-checkpoint", str(tmp_path / "eeg.pt"),
            "--dataset", "karaone",
            "--split", "test",
            "--allow-final-test",
            "--output", str(tmp_path / "synthesis"),
        ],
    )
    with pytest.raises(PermissionError, match="Validation gate is missing"):
        MODULE.main()


def test_validation_failed_roundtrip_requires_explicit_exploratory_flag(tmp_path: Path) -> None:
    gate = tmp_path / "roundtrip.json"
    gate.write_text('{"passed": false, "reason": "smoke"}', encoding="utf-8")
    with pytest.raises(PermissionError, match="round-trip"):
        MODULE.check_access_policy(
            split="validation",
            allow_final_test=False,
            allow_failed_gate=False,
            roundtrip_gate_path=gate,
            validation_gate_path=tmp_path / "unused.json",
            lineage={},
            audio_checkpoint_sha256="audio",
            eeg_checkpoint_sha256="eeg",
        )
    exploratory, reasons, payload = MODULE.check_access_policy(
        split="validation",
        allow_final_test=False,
        allow_failed_gate=True,
        roundtrip_gate_path=gate,
        validation_gate_path=tmp_path / "unused.json",
        lineage={},
        audio_checkpoint_sha256="audio",
        eeg_checkpoint_sha256="eeg",
    )
    assert exploratory
    assert reasons
    assert payload["passed"] is False


def test_stale_passed_roundtrip_gate_is_not_accepted(tmp_path: Path) -> None:
    gate = tmp_path / "roundtrip.json"
    gate.write_text(
        '{"passed": true, "version": "cache-v2", "config_sha256": "old", '
        '"cache_sha256": "cache", "test_audio_waveforms_decoded": false}',
        encoding="utf-8",
    )
    lineage = {"config_sha256": "new", "cache_sha256": "cache", "cache_version": "cache-v2"}
    with pytest.raises(PermissionError, match="binding mismatch"):
        MODULE.check_access_policy(
            split="validation",
            allow_final_test=False,
            allow_failed_gate=False,
            roundtrip_gate_path=gate,
            validation_gate_path=tmp_path / "unused.json",
            lineage=lineage,
            audio_checkpoint_sha256="audio",
            eeg_checkpoint_sha256="eeg",
        )

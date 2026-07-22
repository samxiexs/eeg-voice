from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
import torch

from src.open_vocab_0722.data import collate_openvoice
from src.open_vocab_0722.audio_gate import (
    AUDIO_FREEZE_SCHEMA,
    AUDIO_ORACLE_GATE_SCHEMA,
    require_frozen_audio_checkpoint,
)
from src.open_vocab_0722.lineage import file_sha256
from src.open_vocab_0722.losses import (
    exact_pair_contrastive_loss,
    loss_eligibility,
    router_collapse_flags,
    semantic_positive_weights,
)
from src.open_vocab_0722.metrics import reconstruction_metrics
from src.open_vocab_0722.model import (
    LabelFreeAudioConfig,
    LabelFreeAudioModel,
    OpenVoiceEEGConfig,
    OpenVoiceEEGEncoder,
    OpenVoiceGenerator,
)


def tiny_models(*, moe: bool = True) -> tuple[LabelFreeAudioModel, OpenVoiceEEGEncoder]:
    audio = LabelFreeAudioModel(
        LabelFreeAudioConfig(
            codebooks=2, code_steps=6, vocab_size=16, d_model=24,
            condition_steps=5, encoder_layers=1, decoder_layers=1,
            heads=4, dropout=0.0, text_dimension=8, xlsr_dimension=12,
        )
    ).eval()
    eeg = OpenVoiceEEGEncoder(
        OpenVoiceEEGConfig(
            eeg_samples=64, patch_size=16, patch_hop=8, d_model=24,
            condition_steps=5, code_steps=6, heads=4, latent_layers=1,
            dropout=0.0, specialists=4, specialist_bottleneck=6,
            expert_dropout=0.0, num_train_subjects=3,
            adapter_moe_enabled=moe,
        )
    ).eval()
    return audio, eeg


def eeg_inputs(channels: int = 4) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(3)
    eeg = torch.randn(2, channels, 64, generator=generator)
    xyz = torch.randn(2, channels, 3, generator=generator)
    channel = torch.ones(2, channels, dtype=torch.bool)
    time = torch.ones(2, 64, dtype=torch.bool)
    return eeg, xyz, channel, time


def test_public_generation_api_is_label_free() -> None:
    audio, eeg = tiny_models()
    signature = inspect.signature(OpenVoiceGenerator.generate)
    assert set(signature.parameters) == {"self", "eeg", "channel_xyz", "channel_mask", "time_mask", "steps", "temperature"}
    with pytest.raises(TypeError):
        OpenVoiceGenerator(eeg, audio).generate(*eeg_inputs(), label=torch.tensor([1]))  # type: ignore[call-arg]
    assert "label" not in inspect.signature(audio.decoder.forward).parameters
    assert "dataset" not in inspect.signature(eeg.forward).parameters
    assert "subject" not in inspect.signature(eeg.forward).parameters


def test_channel_permutation_and_masked_values_are_invariant() -> None:
    _, model = tiny_models()
    eeg, xyz, channel, time = eeg_inputs()
    with torch.no_grad():
        expected = model(eeg, xyz, channel, time)["condition"]
        permutation = torch.tensor([2, 0, 3, 1])
        observed = model(eeg[:, permutation], xyz[:, permutation], channel[:, permutation], time)["condition"]
        assert torch.allclose(expected, observed, atol=2e-5, rtol=2e-5)
        masked = channel.clone(); masked[:, -1] = False
        first = model(eeg, xyz, masked, time)["condition"]
        corrupted = eeg.clone(); corrupted[:, -1] = 1e6
        second = model(corrupted, xyz, masked, time)["condition"]
        assert torch.allclose(first, second, atol=2e-5, rtol=2e-5)
        truncated = time.clone(); truncated[:, 48:] = False
        first_time = model(eeg, xyz, channel, truncated)["condition"]
        corrupted_time = eeg.clone(); corrupted_time[:, :, 48:] = -1e6
        second_time = model(corrupted_time, xyz, channel, truncated)["condition"]
        assert torch.allclose(first_time, second_time, atol=2e-5, rtol=2e-5)


def test_variable_channel_collation() -> None:
    def sample(channels: int) -> dict[str, object]:
        return {
            "eeg": torch.ones(channels, 64), "channel_xyz": torch.zeros(channels, 3),
            "channel_mask": torch.ones(channels, dtype=torch.bool), "common_channel_mask": torch.ones(channels, dtype=torch.bool),
            "time_mask": torch.ones(64, dtype=torch.bool), "codes": torch.zeros(2, 6, dtype=torch.long),
            "code_valid_mask": torch.ones(2, 6, dtype=torch.bool), "audio_envelope": torch.zeros(6),
            "onset": torch.tensor(0.0), "duration": torch.tensor(1.0), "audio_idx": torch.tensor(0),
            "label_idx": torch.tensor(0), "dataset_idx": torch.tensor(0), "subject_idx": torch.tensor(0),
            "xlsr_tokens": torch.zeros(5, 12), "text_embedding": torch.zeros(8),
            "has_audio_teacher": torch.tensor(True), "has_text_teacher": torch.tensor(True),
            **{key: "x" for key in ("sample_key", "audio_key", "dataset", "subject_group_id", "label", "label_key", "pairing_confidence", "pairing_level", "eeg_relpath")},
            "channel_names": tuple(f"C{i}" for i in range(channels)), "eeg_row": 0,
        }
    batch = collate_openvoice([sample(14), sample(64), sample(128)])
    assert batch["eeg"].shape == (3, 128, 64)
    assert batch["channel_mask"].sum(dim=1).tolist() == [14, 64, 128]
    _, model = tiny_models()
    with torch.no_grad():
        output = model(batch["eeg"], batch["channel_xyz"], batch["channel_mask"], batch["time_mask"])
    assert output["condition"].shape == (3, 5, 24)


def test_pairing_policy_and_two_space_semantics() -> None:
    eligibility = loss_eligibility(
        ["karaone", "feis", "ds004306"],
        ["karaone_same_trial_overt", "feis_subject_label", "weak_category_level"],
    )
    assert eligibility.exact_acoustic.tolist() == [True, False, False]
    assert eligibility.weak_semantic.tolist() == [True, True, False]
    weights = semantic_positive_weights(torch.tensor([1, 1, 2]), eligibility.exact_acoustic, eligibility.weak_semantic)
    assert weights[0, 0] == 1.0
    assert weights[0, 1] == pytest.approx(0.15)
    assert weights[1, 0] == pytest.approx(0.15)
    assert weights[2].sum() == 0

    exact = torch.tensor([True, True])
    audio = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    good = exact_pair_contrastive_loss(audio.clone(), audio, exact)["total"]
    collapsed = exact_pair_contrastive_loss(torch.tensor([[1.0, 1.0], [1.0, 1.0]]), audio, exact)["total"]
    assert good < collapsed  # same-label non-pair remains an acoustic negative


def test_router_collapse_and_dense_control() -> None:
    assert router_collapse_flags(torch.tensor([0.01, 0.33, 0.33, 0.33]))["expert_dying"]
    assert router_collapse_flags(torch.tensor([0.70, 0.10, 0.10, 0.10]))["routing_collapse"]
    _, dense = tiny_models(moe=False)
    with torch.no_grad():
        output = dense(*eeg_inputs())
    assert torch.allclose(output["router"]["specialist_mass"], torch.full((4,), 0.25), atol=1e-6)


def test_identical_waveform_metrics() -> None:
    time = np.linspace(0, 1, 16000, endpoint=False)
    waveform = np.sin(2 * np.pi * 220 * time).astype(np.float32)
    metrics = reconstruction_metrics(waveform, waveform.copy(), 16000)
    assert metrics["waveform_correlation"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["lag_envelope_correlation"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["log_mel_mae_db"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["multi_resolution_stft_distance"] == pytest.approx(0.0, abs=1e-6)


def test_audio_freeze_binds_gate_and_checkpoint(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("version: test\n", encoding="utf-8")
    checkpoint = tmp_path / "audio.pt"
    checkpoint.write_bytes(b"audio-v1")
    gate_path = tmp_path / "gate.json"
    freeze_path = tmp_path / "freeze.json"
    lineage: dict[str, object] = {}
    gate = {
        "schema_version": AUDIO_ORACLE_GATE_SCHEMA,
        "passed": True,
        "audio_checkpoint_sha256": file_sha256(checkpoint),
        "lineage": lineage,
    }
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    freeze = {
        "schema_version": AUDIO_FREEZE_SCHEMA,
        "audio_checkpoint_sha256": file_sha256(checkpoint),
        "audio_oracle_gate_sha256": file_sha256(gate_path),
        "lineage": lineage,
    }
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    cfg = {
        "audio_oracle_gate": {"required_before_paired_eeg": True},
        "paths": {"audio_oracle_gate": "gate.json", "audio_freeze_manifest": "freeze.json"},
    }
    assert require_frozen_audio_checkpoint(config_path, cfg, lineage, checkpoint) == freeze
    checkpoint.write_bytes(b"audio-v2")
    with pytest.raises(PermissionError, match="binding mismatch"):
        require_frozen_audio_checkpoint(config_path, cfg, lineage, checkpoint)

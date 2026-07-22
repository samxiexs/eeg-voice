from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data import resolve_config_path
from .lineage import file_sha256, validate_lineage


AUDIO_ORACLE_GATE_SCHEMA = "openvoice-0722-audio-oracle-gate-v1"
AUDIO_FREEZE_SCHEMA = "openvoice-0722-audio-freeze-v1"


def audio_gate_required(cfg: dict[str, Any]) -> bool:
    return bool((cfg.get("audio_oracle_gate") or {}).get("required_before_paired_eeg", False))


def require_frozen_audio_checkpoint(
    config_path: str | Path,
    cfg: dict[str, Any],
    lineage: dict[str, Any],
    audio_checkpoint: str | Path,
) -> dict[str, Any] | None:
    """Reject an unaudited project-only audio prior before paired EEG work.

    Formal public-audio configurations may opt out by omitting or disabling
    ``audio_oracle_gate.required_before_paired_eeg``.  When enabled, both the
    oracle report and freeze manifest must bind the exact checkpoint and audio
    lineage used by the caller.
    """

    if not audio_gate_required(cfg):
        return None
    paths = cfg["paths"]
    gate_path = resolve_config_path(config_path, paths["audio_oracle_gate"])
    freeze_path = resolve_config_path(config_path, paths["audio_freeze_manifest"])
    checkpoint = Path(audio_checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Audio checkpoint is missing: {checkpoint}")
    if not gate_path.is_file():
        raise PermissionError(
            f"Audio-condition-oracle gate is missing: {gate_path}. "
            "Run `bash app/run_open_vocab_0722_v1.sh audit-audio-oracle` first."
        )
    if not freeze_path.is_file():
        raise PermissionError(f"Audited audio freeze manifest is missing: {freeze_path}")

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if gate.get("schema_version") != AUDIO_ORACLE_GATE_SCHEMA or not bool(gate.get("passed")):
        raise PermissionError(f"Audio-condition-oracle gate failed: {gate.get('failed_checks', [])}")
    if freeze.get("schema_version") != AUDIO_FREEZE_SCHEMA:
        raise ValueError(f"Unsupported audio freeze manifest: {freeze_path}")
    validate_lineage(gate.get("lineage"), lineage, source="audio-condition-oracle gate", scope="audio")
    validate_lineage(freeze.get("lineage"), lineage, source="audio freeze manifest", scope="audio")

    checkpoint_sha = file_sha256(checkpoint)
    expected = {
        "audio_checkpoint_sha256": checkpoint_sha,
        "audio_oracle_gate_sha256": file_sha256(gate_path),
    }
    observed = {
        "audio_checkpoint_sha256": freeze.get("audio_checkpoint_sha256"),
        "audio_oracle_gate_sha256": freeze.get("audio_oracle_gate_sha256"),
    }
    mismatch = {
        key: {"saved": observed[key], "current": value}
        for key, value in expected.items()
        if observed[key] != value
    }
    if gate.get("audio_checkpoint_sha256") != checkpoint_sha:
        mismatch["gate_audio_checkpoint_sha256"] = {
            "saved": gate.get("audio_checkpoint_sha256"),
            "current": checkpoint_sha,
        }
    if mismatch:
        raise PermissionError(f"Frozen audio checkpoint binding mismatch: {json.dumps(mismatch, sort_keys=True)}")
    return freeze


__all__ = [
    "AUDIO_FREEZE_SCHEMA",
    "AUDIO_ORACLE_GATE_SCHEMA",
    "audio_gate_required",
    "require_frozen_audio_checkpoint",
]

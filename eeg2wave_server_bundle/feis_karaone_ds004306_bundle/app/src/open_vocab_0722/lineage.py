from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .data import resolve_config_path


LINEAGE_SCHEMA_VERSION = "openvoice-0722-lineage-v1"
CHECKPOINT_SCHEMA_VERSION = "openvoice-0722-checkpoint-v1"
GATE_SCHEMA_VERSION = "openvoice-0722-gate-v1"
_COMPARABLE = (
    "schema_version",
    "config_sha256",
    "subject_split_sha256",
    "label_split_sha256",
    "manifest_sha256",
    "eeg_payloads_sha256",
    "montage_registry_sha256",
    "project_audio_cache_sha256",
    "public_audio_manifest_sha256",
    "public_audio_cache_sha256",
    "teacher_cache_sha256",
    "pairing_policy_version",
    "encodec_version",
    "xlsr_version",
    "text_encoder_version",
    "audio_recipe_sha256",
    "eeg_recipe_sha256",
)
_AUDIO_COMPARABLE = tuple(key for key in _COMPARABLE if key not in {
    "config_sha256", "label_split_sha256", "eeg_payloads_sha256", "montage_registry_sha256", "eeg_recipe_sha256",
})


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_sha256(path: str | Path) -> str:
    value = Path(path)
    if value.is_file():
        return file_sha256(value)
    if not value.is_dir():
        raise FileNotFoundError(value)
    index = value / "index.json"
    if index.is_file():
        metadata = json.loads(index.read_text(encoding="utf-8"))
        if metadata.get("content_sha256") and metadata.get("file_sha256"):
            # The index binds every shard digest generated at cache-build
            # time, so phase startup need not re-read an ~18 GB cache.
            return file_sha256(index)
    digest = hashlib.sha256(b"openvoice-directory-v1\0")
    for child in sorted(item for item in value.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(value)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(child).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def optional_sha256(path: str | Path | None) -> str:
    return path_sha256(path) if path and Path(path).exists() else "absent"


def eeg_payloads_sha256(root: Path, rows: Iterable[dict[str, str]]) -> str:
    digest = hashlib.sha256(b"openvoice-eeg-payloads-v1\0")
    for relative in sorted({str(row["eeg_relpath"]) for row in rows}):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Manifest-referenced EEG payload is missing: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _cache_schema(path: Path) -> str:
    if path.is_dir():
        index = path / "index.json"
        if index.is_file():
            return str(json.loads(index.read_text(encoding="utf-8")).get("schema_version", "unknown"))
        return "unknown"
    if not path.is_file():
        return "absent"
    with np.load(path, allow_pickle=False) as raw:
        for key in ("cache_schema_version", "schema_version", "version"):
            if key in raw.files and np.asarray(raw[key]).size == 1:
                return str(np.asarray(raw[key]).reshape(-1)[0])
    return "unknown"


def object_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_lineage(context: Any, *, require_optional_caches: bool = False) -> dict[str, Any]:
    cfg = context.config
    paths = cfg["paths"]
    config_path = context.config_path
    label_split = resolve_config_path(config_path, cfg["data"]["label_holdout_file"])
    registry = resolve_config_path(config_path, cfg["data"]["montage_registry"])
    project_cache = resolve_config_path(config_path, paths["project_audio_cache"])
    public_manifest = resolve_config_path(config_path, paths["public_audio_manifest"])
    public_cache = resolve_config_path(config_path, paths["public_audio_cache"])
    teacher_cache = resolve_config_path(config_path, paths["teacher_cache"])
    if not project_cache.is_file():
        raise FileNotFoundError(f"Project audio cache is missing: {project_cache}")
    if require_optional_caches:
        for name, path in (("public audio manifest", public_manifest), ("public audio cache", public_cache), ("teacher cache", teacher_cache)):
            if not path.exists():
                raise FileNotFoundError(f"Required {name} is missing: {path}")
    return {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "config_sha256": file_sha256(config_path),
        "subject_split_sha256": file_sha256(context.split_path),
        "subject_split_version": str(context.split.get("version", "unknown")),
        "label_split_sha256": file_sha256(label_split),
        "manifest_sha256": file_sha256(context.manifest_path),
        "eeg_payloads_sha256": eeg_payloads_sha256(context.eeg_root, context.rows),
        "montage_registry_sha256": file_sha256(registry),
        "project_audio_cache_sha256": file_sha256(project_cache),
        "project_audio_cache_schema": _cache_schema(project_cache),
        "public_audio_manifest_sha256": optional_sha256(public_manifest),
        "public_audio_cache_sha256": optional_sha256(public_cache),
        "public_audio_cache_schema": _cache_schema(public_cache),
        "teacher_cache_sha256": optional_sha256(teacher_cache),
        "teacher_cache_schema": _cache_schema(teacher_cache),
        "pairing_policy_version": str(cfg["data"]["pairing_policy_version"]),
        "encodec_version": str(cfg["paths"]["encodec_model"]),
        "xlsr_version": str(cfg["teachers"]["xlsr_model"]),
        "text_encoder_version": str(cfg["teachers"]["text_model"]),
        "audio_recipe_sha256": object_sha256({
            "codec": cfg["codec"], "audio_model": cfg["audio_model"], "teachers": cfg["teachers"],
            "public_audio": cfg["public_audio"], "subject_split_version": str(context.split.get("version", "unknown")),
        }),
        "eeg_recipe_sha256": object_sha256({
            "data": cfg["data"], "eeg_model": cfg["eeg_model"], "loss": cfg["loss"], "evaluation": cfg["evaluation"],
        }),
        "paths": {
            "config": str(config_path),
            "subject_split": str(context.split_path),
            "label_split": str(label_split),
            "manifest": str(context.manifest_path),
            "montage_registry": str(registry),
            "project_audio_cache": str(project_cache),
            "public_audio_manifest": str(public_manifest),
            "public_audio_cache": str(public_cache),
            "teacher_cache": str(teacher_cache),
        },
    }


def comparable(lineage: dict[str, Any], *, scope: str = "full") -> dict[str, Any]:
    keys = _AUDIO_COMPARABLE if scope == "audio" else _COMPARABLE
    return {key: lineage.get(key) for key in keys}


def validate_lineage(saved: Any, expected: dict[str, Any], *, source: str, scope: str = "full") -> None:
    if not isinstance(saved, dict):
        raise ValueError(f"{source} is a legacy artifact without OpenVoice lineage")
    differences = {
        key: {"saved": comparable(saved, scope=scope).get(key), "current": comparable(expected, scope=scope).get(key)}
        for key in comparable(expected, scope=scope)
        if comparable(saved, scope=scope).get(key) != comparable(expected, scope=scope).get(key)
    }
    if differences:
        raise ValueError(f"{source} lineage mismatch: {json.dumps(differences, sort_keys=True)}")


def checkpoint_payload(
    *,
    phase: str,
    lineage: dict[str, Any],
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any] | None,
    epoch: int,
    metrics: dict[str, Any],
    dependencies: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "phase": phase,
        "lineage": lineage,
        "dependencies": dependencies or {},
        "epoch": int(epoch),
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "metrics": metrics,
    }


def validate_checkpoint(
    payload: dict[str, Any],
    *,
    phase: str,
    lineage: dict[str, Any],
    source: str,
    dependencies: dict[str, str] | None = None,
) -> None:
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"{source} is legacy/incompatible; OpenVoice checkpoints must be retrained")
    if payload.get("phase") != phase:
        raise ValueError(f"{source} phase mismatch: {payload.get('phase')!r} != {phase!r}")
    validate_lineage(payload.get("lineage"), lineage, source=source, scope="audio" if phase == "audio" else "full")
    if (payload.get("dependencies") or {}) != (dependencies or {}):
        raise ValueError(
            f"{source} dependency mismatch: saved={payload.get('dependencies') or {}}, expected={dependencies or {}}"
        )


def preauthorize_test(
    gate_path: str | Path,
    *,
    lineage: dict[str, Any],
    audio_checkpoint: str | Path,
    eeg_checkpoint: str | Path,
) -> dict[str, Any]:
    path = Path(gate_path)
    if not path.is_file():
        raise PermissionError(f"Validation gate is missing: {path}")
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("schema_version") != GATE_SCHEMA_VERSION or not gate.get("passed"):
        raise PermissionError(f"Validation gate is not passed: {gate.get('reasons', [])}")
    validate_lineage(gate.get("lineage"), lineage, source="validation gate")
    expected = {
        "audio_checkpoint_sha256": file_sha256(audio_checkpoint),
        "eeg_checkpoint_sha256": file_sha256(eeg_checkpoint),
    }
    mismatch = {key: {"gate": gate.get(key), "current": value} for key, value in expected.items() if gate.get(key) != value}
    if mismatch:
        raise PermissionError(f"Validation gate checkpoint mismatch: {json.dumps(mismatch, sort_keys=True)}")
    report_path = Path(gate.get("validation_report", ""))
    if not report_path.is_file() or file_sha256(report_path) != gate.get("validation_report_sha256"):
        raise PermissionError("Validation report bound to the gate is missing or changed")
    return gate


def preauthorize_test_metadata(
    gate_path: str | Path,
    *,
    config_path: str | Path,
    audio_checkpoint: str | Path,
    eeg_checkpoint: str | Path,
) -> dict[str, Any]:
    """Authorize before hashing/loading any locked-test EEG or audio payload."""
    path = Path(gate_path)
    if not path.is_file():
        raise PermissionError(f"Validation gate is missing: {path}")
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("schema_version") != GATE_SCHEMA_VERSION or not gate.get("passed"):
        raise PermissionError(f"Validation gate is not passed: {gate.get('reasons', [])}")
    expected = {
        "config_sha256": file_sha256(config_path),
        "audio_checkpoint_sha256": file_sha256(audio_checkpoint),
        "eeg_checkpoint_sha256": file_sha256(eeg_checkpoint),
    }
    observed = {
        "config_sha256": (gate.get("lineage") or {}).get("config_sha256"),
        "audio_checkpoint_sha256": gate.get("audio_checkpoint_sha256"),
        "eeg_checkpoint_sha256": gate.get("eeg_checkpoint_sha256"),
    }
    mismatch = {key: {"gate": observed[key], "current": value} for key, value in expected.items() if observed[key] != value}
    if mismatch:
        raise PermissionError(f"Pre-read validation gate mismatch: {json.dumps(mismatch, sort_keys=True)}")
    report = Path(gate.get("validation_report", ""))
    if not report.is_file() or file_sha256(report) != gate.get("validation_report_sha256"):
        raise PermissionError("Validation report bound to the gate is missing or changed")
    return gate


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "GATE_SCHEMA_VERSION",
    "LINEAGE_SCHEMA_VERSION",
    "build_lineage",
    "checkpoint_payload",
    "file_sha256",
    "path_sha256",
    "preauthorize_test",
    "preauthorize_test_metadata",
    "validate_checkpoint",
    "validate_lineage",
]

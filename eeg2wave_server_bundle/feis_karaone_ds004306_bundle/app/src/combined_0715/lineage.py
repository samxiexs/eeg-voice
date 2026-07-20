from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CHECKPOINT_SCHEMA_VERSION = "combined-0715-checkpoint-v2"
LINEAGE_SCHEMA_VERSION = "combined-0715-lineage-v1"
_LINEAGE_KEYS = (
    "schema_version",
    "config_sha256",
    "split_sha256",
    "split_version",
    "manifest_sha256",
    "preprocessing_sha256",
    "cache_sha256",
    "cache_version",
)


def file_sha256(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocessing_sha256_from_rows(root: str | Path, rows: Any) -> str:
    """Hash referenced EEG payloads and channel QC using the lineage contract."""

    root = Path(root)
    digest = hashlib.sha256(b"combined-0715-preprocessing-v1\0")
    relative_paths = sorted({str(row["eeg_relpath"]) for row in rows})
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Manifest-referenced EEG payload is missing: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\0")
    qc_path = root / "qc" / "channel_qc.csv"
    if not qc_path.is_file():
        raise FileNotFoundError(f"Preprocessing QC is missing: {qc_path}")
    digest.update(b"qc/channel_qc.csv\0")
    digest.update(file_sha256(qc_path).encode("ascii"))
    return digest.hexdigest()


def preprocessing_sha256(context: Any) -> str:
    """Hash the exact preprocessed EEG payloads referenced by the manifest."""

    return preprocessing_sha256_from_rows(context.root, context.rows)


def build_run_lineage(config_path: str | Path, context: Any, bank: Any) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    verification_path = context.root / "qc" / "eeg_verification.jsonl"
    if not verification_path.is_file():
        raise FileNotFoundError(
            "EEG verification report is missing; run `bash run_preprocess.sh --verify-only` first"
        )
    lines = [line for line in verification_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    verification = json.loads(lines[-1]) if lines else {}
    if verification.get("status") != "passed":
        raise ValueError(
            "EEG preprocessing verification is not passed; run `bash run_preprocess.sh --verify-only` and fix errors"
        )
    current_split_sha = file_sha256(context.split_path)
    current_manifest_sha = file_sha256(context.manifest_path)
    current_preprocessing_sha = preprocessing_sha256(context)
    provenance = verification.get("provenance")
    expected_provenance = {
        "locked_split_sha256": current_split_sha,
        "manifest_sha256": current_manifest_sha,
        "preprocessing_sha256": current_preprocessing_sha,
    }
    if not isinstance(provenance, dict):
        raise ValueError(
            "EEG verification report predates provenance binding; rerun "
            "`bash run_preprocess.sh --verify-only`"
        )
    stale = {
        key: {"verified": provenance.get(key), "current": value}
        for key, value in expected_provenance.items()
        if provenance.get(key) != value
    }
    if stale:
        raise ValueError(
            "EEG verification report is stale; rerun `bash run_preprocess.sh --verify-only`: "
            f"{json.dumps(stale, sort_keys=True)}"
        )
    return {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "config_sha256": file_sha256(config_path),
        "split_sha256": current_split_sha,
        "split_version": str(context.split.get("version", "unknown")),
        "manifest_sha256": current_manifest_sha,
        "preprocessing_sha256": current_preprocessing_sha,
        "cache_sha256": file_sha256(bank.path),
        "cache_version": str(bank.version),
        "paths": {
            "config": str(config_path),
            "split": str(context.split_path.resolve()),
            "manifest": str(context.manifest_path.resolve()),
            "cache": str(bank.path.resolve()),
            "eeg_verification": str(verification_path.resolve()),
        },
    }


def comparable_lineage(lineage: dict[str, Any]) -> dict[str, Any]:
    return {key: lineage.get(key) for key in _LINEAGE_KEYS}


def validate_lineage(saved: dict[str, Any] | None, expected: dict[str, Any], *, source: str) -> None:
    if not isinstance(saved, dict):
        raise ValueError(f"{source} is a legacy artifact without lineage; rebuild/retrain it")
    observed = comparable_lineage(saved)
    wanted = comparable_lineage(expected)
    mismatches = {
        key: {"saved": observed.get(key), "current": wanted.get(key)}
        for key in _LINEAGE_KEYS
        if observed.get(key) != wanted.get(key)
    }
    if mismatches:
        raise ValueError(f"{source} lineage mismatch: {json.dumps(mismatches, sort_keys=True)}")


def validate_checkpoint_payload(
    payload: dict[str, Any],
    *,
    expected_phase: str,
    expected_lineage: dict[str, Any],
    expected_dependencies: dict[str, str] | None = None,
    source: str = "checkpoint",
) -> None:
    schema = payload.get("checkpoint_schema_version")
    if schema != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"{source} uses legacy/incompatible checkpoint schema {schema!r}; "
            "restart training without --resume"
        )
    if payload.get("phase") != expected_phase:
        raise ValueError(f"{source} phase mismatch: saved={payload.get('phase')!r}, expected={expected_phase!r}")
    validate_lineage(payload.get("lineage"), expected_lineage, source=source)
    observed_dependencies = payload.get("dependencies") or {}
    wanted_dependencies = expected_dependencies or {}
    if observed_dependencies != wanted_dependencies:
        raise ValueError(
            f"{source} dependency mismatch: saved={observed_dependencies!r}, expected={wanted_dependencies!r}"
        )


def validate_gate_binding(
    gate: dict[str, Any],
    *,
    lineage: dict[str, Any],
    audio_checkpoint_sha256: str,
    eeg_checkpoint_sha256: str,
) -> None:
    if not bool(gate.get("passed")):
        raise PermissionError(f"Validation gate is not passed: {gate.get('reasons') or gate.get('reason')}")
    validate_lineage(gate.get("lineage"), lineage, source="validation gate")
    expected = {
        "audio_checkpoint_sha256": audio_checkpoint_sha256,
        "eeg_checkpoint_sha256": eeg_checkpoint_sha256,
    }
    mismatches = {key: {"gate": gate.get(key), "current": value} for key, value in expected.items() if gate.get(key) != value}
    if mismatches:
        raise PermissionError(f"Validation gate checkpoint binding mismatch: {json.dumps(mismatches, sort_keys=True)}")

    report_value = gate.get("validation_report")
    report_sha = gate.get("validation_report_sha256")
    if not isinstance(report_value, str) or not report_value or not isinstance(report_sha, str):
        raise PermissionError("Validation gate does not bind a validation report and SHA256")
    report_path = Path(report_value)
    if not report_path.is_file():
        raise PermissionError(f"Validation report bound by the gate is missing: {report_path}")
    observed_report_sha = file_sha256(report_path)
    if observed_report_sha != report_sha:
        raise PermissionError(
            "Validation report SHA256 mismatch: "
            f"gate={report_sha!r}, current={observed_report_sha!r}"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PermissionError(f"Validation report is unreadable: {report_path}: {error}") from error
    if not isinstance(report, dict):
        raise PermissionError(f"Validation report must be a JSON object: {report_path}")
    validate_lineage(report.get("lineage"), lineage, source="validation report")
    report_bindings = {
        "audio_checkpoint_sha256": audio_checkpoint_sha256,
        "eeg_checkpoint_sha256": eeg_checkpoint_sha256,
        "split": "validation",
        "test_accessed": False,
    }
    report_mismatches = {
        key: {"report": report.get(key), "current": value}
        for key, value in report_bindings.items()
        if report.get(key) != value
    }
    if report_mismatches:
        raise PermissionError(
            "Validation report binding mismatch: "
            f"{json.dumps(report_mismatches, sort_keys=True)}"
        )


def preauthorize_locked_test(
    gate_path: str | Path,
    *,
    config_path: str | Path,
    audio_checkpoint_path: str | Path,
    eeg_checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Authorize opening locked-test data from artifact metadata alone.

    This first-stage check intentionally does not inspect the manifest, cache,
    or EEG NPZ files.  Callers must still build and validate the current full
    lineage after this authorization succeeds.
    """

    gate_path = Path(gate_path)
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PermissionError(f"Validation gate is missing: {gate_path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise PermissionError(f"Validation gate is unreadable: {gate_path}: {error}") from error
    if not isinstance(gate, dict):
        raise PermissionError(f"Validation gate must be a JSON object: {gate_path}")
    lineage = gate.get("lineage")
    if not isinstance(lineage, dict) or any(lineage.get(key) in (None, "") for key in _LINEAGE_KEYS):
        raise PermissionError("Validation gate has incomplete lineage and cannot unlock test data")
    current_config_sha = file_sha256(config_path)
    if lineage.get("config_sha256") != current_config_sha:
        raise PermissionError(
            "Validation gate config mismatch before test access: "
            f"gate={lineage.get('config_sha256')!r}, current={current_config_sha!r}"
        )
    audio_sha = file_sha256(audio_checkpoint_path)
    eeg_sha = file_sha256(eeg_checkpoint_path)
    validate_gate_binding(
        gate,
        lineage=lineage,
        audio_checkpoint_sha256=audio_sha,
        eeg_checkpoint_sha256=eeg_sha,
    )
    return gate


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "LINEAGE_SCHEMA_VERSION",
    "build_run_lineage",
    "file_sha256",
    "preprocessing_sha256",
    "preprocessing_sha256_from_rows",
    "preauthorize_locked_test",
    "validate_checkpoint_payload",
    "validate_gate_binding",
    "validate_lineage",
]

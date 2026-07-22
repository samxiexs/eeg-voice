#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import site
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm


APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from src.open_vocab_0722.data import (  # noqa: E402
    AudioCodeBank,
    DATASETS,
    build_montage_registry_payload,
    load_context,
    parse_asa_elc,
    read_csv,
    resolve_config_path,
)
from src.open_vocab_0722.audio_io import canonical_audio_sha256, read_wav  # noqa: E402
from src.open_vocab_0722.lineage import file_sha256  # noqa: E402


def find_standard_1005() -> Path:
    candidates: list[Path] = []
    for root in site.getsitepackages():
        candidates.append(Path(root) / "mne/channels/data/montages/standard_1005.elc")
    candidates.append(Path(sys.prefix) / "lib/python3.12/site-packages/mne/channels/data/montages/standard_1005.elc")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("MNE standard_1005.elc was not found; install the pinned requirements")


def write_registry(config_path: Path, *, overwrite: bool) -> Path:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    eeg_root = resolve_config_path(config_path, cfg["data"]["eeg_output_root"])
    manifest = eeg_root / "manifests/unified_trials.csv"
    registry_path = resolve_config_path(config_path, cfg["data"]["montage_registry"])
    if registry_path.exists() and not overwrite:
        print(f"[openvoice-0722] montage registry exists: {registry_path}")
        return registry_path
    rows = read_csv(manifest)
    coordinates = parse_asa_elc(find_standard_1005())
    payload = build_montage_registry_payload(eeg_root, tqdm(rows, desc="[0722] montage registry"), coordinates)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[openvoice-0722] wrote {registry_path}")
    return registry_path


def audit(config_path: Path) -> dict[str, object]:
    context = load_context(config_path)
    cache_path = resolve_config_path(config_path, context.config["paths"]["project_audio_cache"])
    bank = AudioCodeBank(cache_path)
    fit_by_key = {key: bool(bank.fit_split[index]) for index, key in enumerate(bank.keys)}
    counts = Counter((row["dataset"], row["pairing_confidence"]) for row in context.rows)
    expected = {
        ("karaone", "karaone_same_trial_overt"),
        ("feis", "feis_subject_label"),
        ("ds004306", "weak_category_level"),
    }
    unexpected = sorted({key for key in counts if key not in expected})
    heldout_audio_leaks: list[str] = []
    heldout_audio_hashes: set[str] = set()
    heldout_canonical_hashes: set[str] = set()
    for row in tqdm(context.rows, desc="[0722] leakage audit"):
        split = context.split_for(row)
        # ds004306 candidate sounds are explicitly ineligible for both audio
        # prior and paired generation, so their three category keys do not
        # create a speech-supervision leak.
        generation_eligible = row["dataset"] in {"karaone", "feis"}
        if split in {"validation", "test"} and generation_eligible:
            audio_path = context.audio_root / row["audio_relpath"]
            if audio_path.is_file():
                heldout_audio_hashes.add(file_sha256(audio_path))
                waveform, rate = read_wav(audio_path)
                heldout_canonical_hashes.add(canonical_audio_sha256(waveform, rate))
            if fit_by_key.get(row["audio_key"], False):
                heldout_audio_leaks.append(row["audio_key"])
    report = {
        "schema_version": "openvoice-0722-data-audit-v1",
        "status": "passed" if not unexpected and not heldout_audio_leaks else "failed",
        "test_eeg_accessed": False,
        "trial_counts": dict(Counter(row["dataset"] for row in context.rows)),
        "pairing_counts": {f"{dataset}:{pairing}": value for (dataset, pairing), value in sorted(counts.items())},
        "unexpected_pairing_routes": unexpected,
        "heldout_generation_audio_fit_leaks": sorted(set(heldout_audio_leaks)),
        "heldout_generation_audio_sha256": sorted(heldout_audio_hashes),
        "heldout_generation_audio_canonical_sha256": sorted(heldout_canonical_hashes),
        "policy": {
            "karaone": "exact_trial_generation",
            "feis": "weak_semantic_only",
            "ds004306": "eeg_self_supervision_only",
        },
        "montage_registry": str(context.montage_registry.path),
        "montage_registry_sha256": file_sha256(context.montage_registry.path),
    }
    output = resolve_config_path(config_path, context.config["paths"]["leakage_audit"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "trial_counts": report["trial_counts"],
        "pairing_counts": report["pairing_counts"],
        "heldout_generation_audio_hashes": len(heldout_audio_hashes),
        "heldout_generation_audio_fit_leaks": len(heldout_audio_leaks),
        "report": str(output),
    }, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and audit OpenVoice-EEG 0722 Track A metadata")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite-registry", action="store_true")
    parser.add_argument("--registry-only", action="store_true")
    args = parser.parse_args()
    config = args.config.resolve()
    write_registry(config, overwrite=args.overwrite_registry)
    if not args.registry_only:
        audit(config)


if __name__ == "__main__":
    main()

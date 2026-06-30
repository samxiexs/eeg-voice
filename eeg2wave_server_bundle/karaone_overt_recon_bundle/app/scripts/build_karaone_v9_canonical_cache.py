from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.karaone_v9.data import KaraOneV9Dataset, KaraOneV9TargetBank
from src.utils import ensure_dir, load_simple_yaml, resolve_bundle_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/audit KaraOne v9 canonical manifest.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone_v9.yaml"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--stages", default=None)
    parser.add_argument("--require-codec", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    stages = tuple(
        item.strip()
        for item in str(args.stages or cfg["data"].get("stages", "overt_like")).split(",")
        if item.strip()
    )
    cache_cfg = cfg.get("cache", {})
    out = resolve_bundle_path(args.out or cache_cfg.get("canonical_manifest"), BUNDLE_DIR)
    targets = KaraOneV9TargetBank(
        resolve_bundle_path(cache_cfg["semantic"], BUNDLE_DIR),
        codec_cache=resolve_bundle_path(cache_cfg["codec"], BUNDLE_DIR) if cache_cfg.get("codec") else None,
        prosody_cache=resolve_bundle_path(cache_cfg["prosody"], BUNDLE_DIR) if cache_cfg.get("prosody") else None,
        semantic_token_cache=resolve_bundle_path(cache_cfg["semantic_tokens"], BUNDLE_DIR) if cache_cfg.get("semantic_tokens") else None,
        data_root=root,
    )
    subject_val = str(cfg["data"].get("subject_val", "P02"))
    subject_test = str(cfg["data"].get("subject_test", "MM21"))
    eeg_len = int(cfg["data"].get("eeg_len", 1280))
    split_names = ("subject_train", "subject_val", "subject_test", "train", "val", "test")
    split_summaries: dict[str, Any] = {}
    split_keys: dict[str, set[str]] = {}
    for split in split_names:
        ds = KaraOneV9Dataset(
            root,
            targets,
            split,
            stages=stages,
            subject_val=subject_val,
            subject_test=subject_test,
            eeg_len=eeg_len,
            require_codec=bool(args.require_codec),
        )
        split_summaries[split] = summarize_dataset(ds)
        split_keys[split] = {f"{entry.subject}:{entry.stage}:{entry.trial_index}" for entry in ds.entries}

    coverage = coverage_summary(root, stages, targets)
    overlaps = {
        "subject_train__subject_val": len(split_keys["subject_train"] & split_keys["subject_val"]),
        "subject_train__subject_test": len(split_keys["subject_train"] & split_keys["subject_test"]),
        "subject_val__subject_test": len(split_keys["subject_val"] & split_keys["subject_test"]),
        "trial_train__subject_holdouts": len(split_keys["train"] & (split_keys["subject_val"] | split_keys["subject_test"])),
    }
    manifest = {
        "manifest_kind": "karaone_v9_canonical_manifest",
        "config": str(Path(args.config).resolve()),
        "data_root": str(root),
        "stages": list(stages),
        "subject_val": subject_val,
        "subject_test": subject_test,
        "eeg_len": eeg_len,
        "target_shapes": {
            "semantic": [targets.semantic_steps, targets.semantic_dim],
            "semantic_tokens": [targets.semantic_token_steps, targets.semantic_token_vocab],
            "codec": [targets.codec_steps, targets.codec_dim],
            "prosody_steps": targets.prosody_steps,
        },
        "cache_paths": {key: str(resolve_bundle_path(value, BUNDLE_DIR)) for key, value in cache_cfg.items() if key != "canonical_manifest"},
        "coverage": coverage,
        "splits": split_summaries,
        "overlaps": overlaps,
        "audit_pass": all(value == 0 for value in overlaps.values()) and coverage["semantic_missing"] == 0,
        "notes": [
            "subject_train excludes subject_val and subject_test",
            "test audio is referenced for evaluation only; train-bank/generation priors must use subject_train",
            "codec target exists for conditional transport but is not the first-stage EEG regression target",
        ],
    }
    ensure_dir(out.parent)
    write_json(out, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def summarize_dataset(dataset: KaraOneV9Dataset) -> dict[str, Any]:
    valid_lengths = []
    for idx in range(len(dataset)):
        entry = dataset.entries[idx]
        _, valid = dataset._eeg(entry)  # noqa: SLF001 - canonical audit needs raw valid lengths
        valid_lengths.append(int(valid))
    subjects = sorted({entry.subject for entry in dataset.entries})
    labels = sorted({entry.label for entry in dataset.entries})
    return {
        "n": len(dataset),
        "subjects": subjects,
        "labels": labels,
        "valid_len_median": float(np.median(valid_lengths)) if valid_lengths else 0.0,
        "valid_len_p05": float(np.percentile(valid_lengths, 5)) if valid_lengths else 0.0,
        "valid_len_p95": float(np.percentile(valid_lengths, 95)) if valid_lengths else 0.0,
    }


def coverage_summary(root: Path, stages: tuple[str, ...], targets: KaraOneV9TargetBank) -> dict[str, Any]:
    rows = []
    with (root / "segments.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row["segment_stage"]) in stages:
                rows.append(row)
    semantic_missing = 0
    codec_missing = 0
    prosody_missing = 0
    token_missing = 0
    for row in rows:
        subject = str(row["subject_id"])
        trial = int(row["trial_index"])
        semantic_missing += int(not targets.semantic.has_trial(subject, trial))
        codec_missing += int(targets.codec is None or not targets.codec.has_trial(subject, trial))
        prosody_missing += int(targets.prosody is None or not targets.prosody.has_trial(subject, trial))
        token_missing += int(targets.semantic_tokens is None or not targets.semantic_tokens.has_trial(subject, trial))
    return {
        "segments": len(rows),
        "semantic_missing": semantic_missing,
        "codec_missing": codec_missing,
        "prosody_missing": prosody_missing,
        "semantic_token_missing": token_missing,
    }


if __name__ == "__main__":
    main()

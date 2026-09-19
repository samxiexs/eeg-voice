#!/usr/bin/env python3
"""Select and materialize the registered M0 grids without hard-coded IDs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from prepare_training_data import (ROOT, build as build_eeg_shards, fit_normalizer,
                                   load_config, output_root, validate, write_frame)
from cache_speech_targets import cache as cache_speech_targets
from eeg_preprocessing_qc import run as run_preprocessing_qc

sys.path.insert(0, str(ROOT / "app" / "src"))
from eeg2speech.data import _complete_grid


def _buildable_rows(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Exclude rows whose recording-level bad-channel fraction cannot build.

    The audit manifest records declared bad channels, while the shard builder
    applies ``max_bad_fraction`` once per recording.  M0 selection must use the
    same contract or it can deterministically select a grid that later writes
    fewer cells than requested (for example a subject with 20/128 bad channels
    under a 0.15 limit).
    """
    if frame.empty:
        return frame
    threshold = float(config["harmonized"]["interpolation"]["max_bad_fraction"])
    canonical_count = {
        name: len(spec["channel_order"])
        for name, spec in config["sources"].items()
    }

    def keep(row: pd.Series) -> bool:
        raw = str(row.get("bad_channels", ""))
        try:
            bad = json.loads(raw) if raw.startswith("[") else []
        except json.JSONDecodeError:
            bad = []
        denominator = max(int(canonical_count.get(str(row.get("dataset", "")), 1)), 1)
        return len(set(bad)) / denominator <= threshold

    return frame.loc[frame.apply(keep, axis=1)].copy()


def _grid(frame: pd.DataFrame, subjects: int, contents: int, namespace: str) -> pd.DataFrame:
    selected = _complete_grid(frame, subjects, contents, namespace)
    if selected.groupby(["subject", "linguistic_content_id"]).size().ne(1).any():
        raise RuntimeError(f"{namespace}: selected M0 grid contains duplicate cells")
    return selected


def select_registered_grids(config: dict, pilot: dict) -> dict[str, pd.DataFrame]:
    root = output_root(config)
    manifest_path = root / "manifests" / "manifest_all.csv"
    split_path = root / "splits" / f"{pilot['split']['protocol']}_fold-{pilot['split']['fold']}.csv"
    if not manifest_path.exists() or not split_path.exists():
        raise RuntimeError("audit and make-splits must run before M0 selection")
    manifest = pd.read_csv(manifest_path, keep_default_na=False, low_memory=False)
    split = pd.read_csv(split_path, keep_default_na=False)
    train_ids = set(split[split.role == "train"].trial_id)
    eligible = manifest[
        manifest.trial_id.isin(train_ids)
        & (manifest.build_status == "included")
        & manifest.qc_pass.astype(str).str.lower().eq("true")
    ].copy()
    spec = pilot["pilot"]
    subject_count = int(spec["overfit_subjects_per_dataset"])
    content_count = int(spec["overfit_contents_per_dataset"])

    ds004 = eligible[
        (eligible.dataset == "ds004940")
        & (eligible.task == str(spec["primary_ds004940_task"]))
        & (eligible.supervision_type == "paired_audio")
    ]
    ds004 = _buildable_rows(ds004, config)
    requested = tuple(pilot.get("stage2", {}).get("datasets", ("ds004940",)))
    if requested != ("ds004940",):
        raise RuntimeError("stage2.datasets must be exactly [ds004940]; DS006104 support was removed")
    grids = {"ds004940": _grid(ds004, subject_count, content_count, "M0|ds004940")}
    expected = int(spec["overfit_pairs_per_dataset"])
    if len(grids["ds004940"]) != expected:
        raise RuntimeError(f"ds004940: selected {len(grids['ds004940'])} M0 pairs, expected {expected}")
    return grids


def _selection_payload(grids: dict[str, pd.DataFrame]) -> dict:
    return {
        name: {
            "pairs": int(len(frame)),
            "subjects": sorted(frame.subject.astype(str).unique().tolist()),
            "contents": sorted(frame.linguistic_content_id.astype(str).unique().tolist()),
            "tasks": sorted(frame.task.astype(str).unique().tolist()),
        }
        for name, frame in grids.items()
    }


def _curate_m0_manifest(config: dict, grids: dict[str, pd.DataFrame], artifact_set: str) -> pd.DataFrame:
    """Make an M0 manifest exact while retaining superseded rows as audit evidence."""
    path = output_root(config) / "manifests" / f"manifest_{artifact_set}.csv"
    frame = pd.read_csv(path, keep_default_na=False, low_memory=False)
    desired = pd.Series(False, index=frame.index)
    for name, grid in grids.items():
        desired |= (
            (frame.dataset == "ds004940")
            & (frame.supervision_type == "paired_audio")
            & frame.subject.isin(set(grid.subject.astype(str)))
            & frame.linguistic_content_id.isin(set(grid.linguistic_content_id.astype(str)))
            & frame.task.isin(set(grid.task.astype(str)))
        )
    stale = (frame.build_status == "included") & ~desired
    frame.loc[stale, "build_status"] = "excluded"
    frame.loc[stale, "exclusion_reason"] = "outside_registered_m0_artifact"

    included = frame[(frame.build_status == "included") & desired]
    expected = {name: len(grid) for name, grid in grids.items()}
    partitions = {"ds004940": included[(included.dataset == "ds004940") & (included.supervision_type == "paired_audio")]}
    for name, selected in partitions.items():
        cells = selected.groupby(["subject", "linguistic_content_id"]).size()
        if len(selected) != expected[name] or len(cells) != expected[name] or not cells.eq(1).all():
            raise RuntimeError(
                f"{name}: built artifact has {len(selected)} rows/{len(cells)} cells, expected {expected[name]}"
            )
    write_frame(frame, output_root(config) / "manifests" / f"manifest_{artifact_set}", pd)
    return included


def materialize(config: dict, pilot: dict, grids: dict[str, pd.DataFrame],
                hubert_local_path: Path, rebuild: bool, artifact_set: str = "built") -> dict:
    if not hubert_local_path.exists():
        raise RuntimeError(f"local HuBERT model is missing: {hubert_local_path}")
    if artifact_set != "built" and not artifact_set.startswith("explore_m0"):
        raise ValueError("M0 artifact_set must be built or an isolated explore_m0* namespace")
    protocol = str(pilot["split"]["protocol"])
    fold = int(pilot["split"]["fold"])
    target_name = "speech_targets" if artifact_set == "built" else f"speech_targets_{artifact_set}"
    normalizer_name = split_path_name = f"{protocol}_fold-{fold}"
    if artifact_set != "built":
        normalizer_name = f"{artifact_set}_{split_path_name}"

    calls = [("ds004940", grids["ds004940"], ",".join(sorted(grids["ds004940"].task.unique())))]
    for dataset, frame, tasks in calls:
        build_eeg_shards(
            config,
            dataset,
            ",".join(sorted(frame.subject.astype(str).unique())),
            tasks,
            None,
            None,
            ",".join(sorted(frame.linguistic_content_id.astype(str).unique())),
            "train",
            protocol,
            fold,
            not rebuild,
            False,
            artifact_set,
        )

    _curate_m0_manifest(config, grids, artifact_set)
    split_path = output_root(config) / "splits" / f"{protocol}_fold-{fold}.csv"
    fit_normalizer(config, split_path, fold, False, artifact_set, normalizer_name)
    config["audio"]["content"]["hubert_local_path"] = str(hubert_local_path.resolve())
    cache_speech_targets(config, "all", None, True, False, artifact_set, target_name)
    if artifact_set == "built":
        psd = run_preprocessing_qc(config)
        if psd["status"] != "pass":
            raise RuntimeError("preprocessing PSD gate failed")
        if validate(config, True) != 0:
            raise RuntimeError("strict Stage-0 validation failed")

    payload = {"status": "pass", "artifact_set": artifact_set, "target_name": target_name,
               "normalizer_name": normalizer_name, "selection": _selection_payload(grids)}
    target = output_root(config) / "qc" / ("m0_artifacts.json" if artifact_set == "built" else f"{artifact_set}_artifacts.json")
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, default=ROOT / "configs" / "training_data_v3.yaml")
    parser.add_argument("--pilot-config", type=Path, required=True,
                        help="legacy joint-pilot config (the joint pipeline was removed on 2026-09-18; "
                             "see ../eeg-recon-0809_explore_8h_v1_backup). The aligned route only imports _buildable_rows.")
    parser.add_argument("--hubert-local-path", type=Path)
    parser.add_argument("--check-only", action="store_true", help="print the deterministic grids without writing artifacts")
    parser.add_argument("--rebuild", action="store_true", help="rewrite the selected M0 shards instead of resuming compatible files")
    parser.add_argument("--artifact-set", default="built",
                        help="built or an isolated explore_m0* namespace")
    args = parser.parse_args()
    config, _ = load_config(args.data_config)
    pilot = yaml.safe_load(args.pilot_config.read_text())
    grids = select_registered_grids(config, pilot)
    if args.check_only:
        print(json.dumps({"status": "pass", "selection": _selection_payload(grids)}, indent=2, sort_keys=True))
        return 0
    if args.hubert_local_path is None:
        parser.error("artifact materialization requires --hubert-local-path; implicit model download is forbidden")
    print(json.dumps(materialize(config, pilot, grids, args.hubert_local_path, args.rebuild, args.artifact_set), indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)

"""Build train-only FEIS v3 EEG/channel cluster cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.feis_v3.data import CLEAN_FIELD, SUBJECT_FIELD, as_bool, norm_subject, parse_stage_spec, sample_key, stable_kmeans, assign_to_centers
from src.utils import ensure_dir, load_simple_yaml, read_csv_rows, resolve_bundle_path, resolve_feis_root, write_json


def parse_args():
    p = argparse.ArgumentParser(description="Build FEIS v3 train-only EEG/channel clusters.")
    p.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "feis_v3_tokenized_generation.yaml"))
    p.add_argument("--stage", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--allow-negative-train", action="store_true")
    return p.parse_args()


def _format_path(raw: str, stage: str) -> str:
    return str(raw).replace("{stage}", stage)


def _descriptor(eeg: np.ndarray) -> np.ndarray:
    x = eeg.astype(np.float32)
    mean = x.mean(axis=-1)
    std = x.std(axis=-1)
    logvar = np.log(np.var(x, axis=-1) + 1e-6)
    abs_mean = np.abs(x).mean(axis=-1)
    diff = np.diff(x, axis=-1)
    diff_energy = np.square(diff).mean(axis=-1)
    slope = (x[:, -1] - x[:, 0]) / max(x.shape[-1], 1)
    return np.stack([mean, std, logvar, abs_mean, diff_energy, slope], axis=-1).reshape(-1).astype(np.float32)


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    data_cfg = cfg.get("data", {})
    token_cfg = cfg.get("tokens", {})
    stage = args.stage or str(data_cfg.get("stage", "stimuli"))
    root = resolve_feis_root(resolve_bundle_path(data_cfg.get("root", "../data/feis"), BUNDLE_DIR))
    subject_val = norm_subject(data_cfg.get("subject_val", "20"))
    subject_test = norm_subject(data_cfg.get("subject_test", "21"))
    include_anomalous = bool(data_cfg.get("include_anomalous", False))
    negative_stage = str(data_cfg.get("negative_stage", "resting"))
    wanted = set(parse_stage_spec(stage, negative_stage=negative_stage))
    out_path = resolve_bundle_path(
        args.out or _format_path(token_cfg.get("cluster_cache", "../artifacts/audio_targets/feis_v3_clusters_{stage}.npz"), stage),
        BUNDLE_DIR,
    )
    audit_path = resolve_bundle_path(
        _format_path(token_cfg.get("cluster_audit", "../artifacts/audio_targets/feis_v3_cluster_audit_{stage}.json"), stage),
        BUNDLE_DIR,
    )
    rows = []
    for row in read_csv_rows(root / "segments.csv"):
        segment_stage = str(row.get("segment_stage", ""))
        if segment_stage not in wanted:
            continue
        subject_id = norm_subject(row[SUBJECT_FIELD])
        if not include_anomalous and not as_bool(row.get(CLEAN_FIELD), subject_id != "05"):
            continue
        rows.append(
            {
                "subject_id": subject_id,
                "label": str(row["label"]),
                "stage": segment_stage,
                "trial_index": int(row["trial_index"]),
            }
        )
    rows.sort(key=lambda item: (item["stage"], item["subject_id"], item["label"], item["trial_index"]))
    if not rows:
        raise ValueError(f"No FEIS rows for stage={stage}")

    descriptors = []
    fit_mask = []
    bundle_cache = {}
    for row in rows:
        subject_id = row["subject_id"]
        if subject_id not in bundle_cache:
            bundle = np.load(root / "subjects" / f"{subject_id}.npz", allow_pickle=True)
            bundle_cache[subject_id] = {
                "bundle": bundle,
                "trial_to_pos": {int(v): i for i, v in enumerate(bundle["trial_indices"].astype(int).tolist())},
            }
        pos = bundle_cache[subject_id]["trial_to_pos"].get(row["trial_index"])
        if pos is None:
            descriptors.append(np.zeros(14 * 6, dtype=np.float32))
        else:
            eeg = bundle_cache[subject_id]["bundle"][f"stage__{row['stage']}"][pos]
            descriptors.append(_descriptor(eeg))
        fit_mask.append(subject_id not in {subject_val, subject_test} and (args.allow_negative_train or row["stage"] != negative_stage))
    desc = np.stack(descriptors, axis=0)
    fit_mask_arr = np.asarray(fit_mask, dtype=bool)
    if not fit_mask_arr.any():
        raise ValueError("No train EEG rows for channel cluster fitting")
    k = int(token_cfg.get("channel_clusters", 4))
    centers, _ = stable_kmeans(
        desc[fit_mask_arr],
        k,
        seed=int(cfg.get("train", {}).get("seed", 7)) + 101,
        iters=int(token_cfg.get("kmeans_iters", 10)),
        max_fit_rows=int(token_cfg.get("max_fit_eeg_rows", 4000)),
    )
    ids = assign_to_centers(desc, centers)
    keys = [sample_key(row["subject_id"], row["label"], row["stage"], row["trial_index"]) for row in rows]
    ensure_dir(out_path.parent)
    np.savez_compressed(
        out_path,
        sample_keys=np.asarray(keys),
        sample_cluster_ids=ids.astype(np.int64),
        channel_cluster_centers=centers,
        stage=np.asarray(stage),
        fit_subjects=np.asarray(sorted({row["subject_id"] for row, ok in zip(rows, fit_mask) if ok})),
        heldout_subjects=np.asarray([subject_val, subject_test]),
    )
    audit = {
        "stage": stage,
        "cluster_cache": str(out_path),
        "n_samples": len(rows),
        "channel_clusters": int(centers.shape[0]),
        "train_fit_subjects": sorted({row["subject_id"] for row, ok in zip(rows, fit_mask) if ok}),
        "heldout_subjects": [subject_val, subject_test],
        "heldout_subjects_used_for_fit": False,
        "resting_used_for_fit": bool(args.allow_negative_train and negative_stage in wanted),
        "train_only_fit": True,
        "allow_negative_train": bool(args.allow_negative_train),
    }
    write_json(audit_path, audit)
    print(json.dumps({"cluster_cache": str(out_path), "audit": str(audit_path), "n_samples": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

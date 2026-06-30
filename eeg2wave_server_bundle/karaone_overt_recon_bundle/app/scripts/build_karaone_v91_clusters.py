from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.karaone_v9.data import KaraOneV9Dataset, KaraOneV9TargetBank
from src.karaone_v91.clusters import build_cluster_bank_arrays, cluster_audit, eeg_descriptor, speech_descriptor
from src.utils import ensure_dir, load_simple_yaml, resolve_bundle_path, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train-only KaraOne v9.1 EEG/speech/cross-modal cluster bank.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone_v91.yaml"))
    parser.add_argument("--stages", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--audit-out", default=None)
    parser.add_argument("--max-rows", type=int, default=None, help="optional smoke limit after deterministic sorting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    train_cfg = cfg.get("train", {})
    set_seed(int(train_cfg.get("seed", 7)))
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    stages = tuple(
        item.strip()
        for item in str(args.stages or cfg["data"].get("stages", "overt_like")).split(",")
        if item.strip()
    )
    subject_val = str(cfg["data"].get("subject_val", "P02"))
    subject_test = str(cfg["data"].get("subject_test", "MM21"))
    eeg_len = int(cfg["data"].get("eeg_len", 1280))
    cache_cfg = cfg.get("cache", {})
    targets = KaraOneV9TargetBank(
        resolve_bundle_path(cache_cfg["semantic"], BUNDLE_DIR),
        codec_cache=resolve_bundle_path(cache_cfg["codec"], BUNDLE_DIR) if cache_cfg.get("codec") else None,
        prosody_cache=resolve_bundle_path(cache_cfg["prosody"], BUNDLE_DIR) if cache_cfg.get("prosody") else None,
        semantic_token_cache=resolve_bundle_path(cache_cfg["semantic_tokens"], BUNDLE_DIR) if cache_cfg.get("semantic_tokens") else None,
        data_root=root,
    )
    rows: list[dict[str, Any]] = []
    for split in ("subject_train", "subject_val", "subject_test"):
        ds = KaraOneV9Dataset(root, targets, split, stages=stages, subject_val=subject_val, subject_test=subject_test, eeg_len=eeg_len)
        for idx in range(len(ds)):
            item = ds[idx]
            rows.append(row_descriptor(item, split=split, targets=targets))
    rows.sort(key=lambda row: (row["split_kind"], row["subject"], row["stage"], row["label"], int(row["trial_index"])))
    if args.max_rows:
        fit_rows = [row for row in rows if row["fit_split"]]
        heldout_rows = [row for row in rows if not row["fit_split"]]
        budget = int(args.max_rows)
        keep_fit = max(1, min(len(fit_rows), int(round(budget * 0.7))))
        keep_heldout = max(0, min(len(heldout_rows), budget - keep_fit))
        if keep_heldout == 0 and heldout_rows and budget > keep_fit:
            keep_heldout = 1
            keep_fit = max(1, min(len(fit_rows), budget - keep_heldout))
        rows = fit_rows[:keep_fit] + heldout_rows[:keep_heldout]
    cluster_cfg = cfg.get("clusters", {})
    arrays = build_cluster_bank_arrays(
        rows,
        n_eeg_clusters=int(cluster_cfg.get("n_eeg_clusters", 12)),
        n_speech_clusters=int(cluster_cfg.get("n_speech_clusters", 12)),
        n_cross_clusters=int(cluster_cfg.get("n_cross_clusters", 16)),
        seed=int(train_cfg.get("seed", 7)),
    )
    out = resolve_bundle_path(args.out or cache_cfg.get("cluster_bank", "../artifacts/audio_targets/karaone_v91_clusters_overt_like.npz"), BUNDLE_DIR)
    ensure_dir(out.parent)
    np.savez_compressed(out, **arrays)
    audit = cluster_audit(arrays, subject_val=subject_val, subject_test=subject_test)
    audit["cluster_bank"] = str(out)
    audit["stages"] = list(stages)
    audit["config"] = {
        "n_eeg_clusters": int(arrays["eeg_centroids"].shape[0]),
        "n_speech_clusters": int(arrays["speech_centroids"].shape[0]),
        "n_cross_clusters": int(arrays["cross_centroids"].shape[0]),
    }
    audit_out = resolve_bundle_path(
        args.audit_out or cache_cfg.get("cluster_audit", "../artifacts/audio_targets/karaone_v91_cluster_audit_overt_like.json"),
        BUNDLE_DIR,
    )
    write_json(audit_out, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def row_descriptor(item: dict[str, Any], *, split: str, targets: KaraOneV9TargetBank) -> dict[str, Any]:
    subject = str(item["subject"])
    stage = str(item["stage"])
    trial = int(item["trial_index"])
    prosody = {
        "active": item["prosody_active"].numpy(),
        "energy": item["prosody_energy"].numpy(),
        "duration": float(item["prosody_duration"].item()),
        "onset": float(item["prosody_onset"].item()),
    }
    return {
        "key": f"{subject}:{stage}:{trial}",
        "subject": subject,
        "label": str(item["label"]),
        "stage": stage,
        "trial_index": trial,
        "split_kind": split,
        "fit_split": split == "subject_train",
        "eeg_descriptor": eeg_descriptor(item["eeg"].numpy(), int(item["eeg_valid_len"].item())),
        "speech_descriptor": speech_descriptor(
            item["semantic_summary"].numpy(),
            item["semantic_token_targets"].numpy(),
            item["semantic_token_mask"].numpy(),
            prosody,
            token_vocab=max(2, int(targets.semantic_token_vocab)),
        ),
    }


if __name__ == "__main__":
    main()

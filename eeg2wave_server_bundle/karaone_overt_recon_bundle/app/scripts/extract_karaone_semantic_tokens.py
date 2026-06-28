from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.karaone_recon.data import KaraOneTrialDataset
from src.karaone_recon.semantic_tokens import assign_tokens_to_centroids, kmeans_tokens
from src.karaone_recon.targets import KaraOneTargets
from src.utils import load_simple_yaml, resolve_bundle_path, resolve_target_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize HuBERT speech features into KaraOne semantic tokens.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone.yaml"))
    parser.add_argument("--out", default="../artifacts/audio_targets/karaone_trial_hubert_tokens_k64.npz")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--codebook-split", choices=["all", "train"], default="all")
    parser.add_argument("--stages", default=None, help="comma list used when --codebook-split=train")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    out = resolve_bundle_path(args.out, BUNDLE_DIR)
    if out.exists() and not args.force:
        print(f"[semantic-tokens] exists: {out}")
        return
    _, hubert_cache = resolve_target_cache(cfg, BUNDLE_DIR, "hubert_sequence")
    if not hubert_cache.exists():
        raise FileNotFoundError(
            f"Missing HuBERT cache: {hubert_cache}. Build it with scripts/extract_karaone_targets.py --target hubert_sequence"
        )
    targets = KaraOneTargets(hubert_cache, data_root=root)
    features = targets.raw_seq.astype(np.float32)
    if args.codebook_split == "train":
        stages = tuple((args.stages or cfg["data"].get("stages", "overt_like")).split(","))
        train_ds = KaraOneTrialDataset(
            data_root=root,
            targets=targets,
            split="train",
            stages=stages,
            split_protocol=str(cfg["data"].get("split_protocol", "trial")),
            heldout_subjects=cfg["data"].get("heldout_subjects", ["P02", "MM21"]),
            eeg_len=int(cfg["data"].get("eeg_len", 1280)),
        )
        train_keys = {KaraOneTargets.key(entry.subject, entry.trial_index) for entry in train_ds.entries}
        train_indices = [
            idx
            for idx, (subject, trial) in enumerate(zip(targets.subject_ids.tolist(), targets.trial_indices.tolist()))
            if KaraOneTargets.key(str(subject), int(trial)) in train_keys
        ]
        if not train_indices:
            raise ValueError("no train trials found for semantic-token codebook")
        train_features = features[np.asarray(train_indices, dtype=np.int64)]
        flat_train = train_features.reshape(-1, train_features.shape[-1])
        print(
            f"[semantic-tokens] train-only kmeans K={args.k} train_trials={len(train_indices)} "
            f"frames={flat_train.shape[0]} dim={flat_train.shape[1]}"
        )
        _, centers = kmeans_tokens(flat_train, k=int(args.k), iters=int(args.iters), seed=int(args.seed))
        all_labels = assign_tokens_to_centroids(features.reshape(-1, features.shape[-1]), centers)
        token_sequences = all_labels.reshape(features.shape[0], features.shape[1]).astype(np.int64)
        codebook_train_template_ids = np.asarray(
            [KaraOneTargets.key(str(targets.subject_ids[idx]), int(targets.trial_indices[idx])) for idx in train_indices],
            dtype=str,
        )
    else:
        flat = features.reshape(-1, features.shape[-1])
        print(f"[semantic-tokens] all-split kmeans K={args.k} frames={flat.shape[0]} dim={flat.shape[1]}")
        labels, centers = kmeans_tokens(flat, k=int(args.k), iters=int(args.iters), seed=int(args.seed))
        token_sequences = labels.reshape(features.shape[0], features.shape[1]).astype(np.int64)
        codebook_train_template_ids = np.asarray([], dtype=str)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        template_ids=targets.template_ids,
        subject_ids=targets.subject_ids,
        trial_indices=targets.trial_indices,
        labels=targets.labels,
        token_sequences=token_sequences,
        token_mask=np.ones_like(token_sequences, dtype=np.float32),
        centroids=centers.astype(np.float32),
        vocab_size=np.asarray(int(args.k), dtype=np.int32),
        source_cache=str(hubert_cache),
        codebook_split=str(args.codebook_split),
        codebook_train_template_ids=codebook_train_template_ids,
    )
    counts = np.bincount(token_sequences.reshape(-1), minlength=int(args.k))
    summary = {
        "out": str(out),
        "source": str(hubert_cache),
        "k": int(args.k),
        "trials": int(token_sequences.shape[0]),
        "steps": int(token_sequences.shape[1]),
        "codebook_split": str(args.codebook_split),
        "codebook_train_trials": int(codebook_train_template_ids.shape[0]),
        "empty_clusters": int(np.sum(counts == 0)),
        "min_cluster_count": int(counts.min(initial=0)),
        "max_cluster_count": int(counts.max(initial=0)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

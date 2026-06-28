from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.karaone_recon.semantic_tokens import kmeans_tokens
from src.karaone_recon.targets import KaraOneTargets
from src.utils import load_simple_yaml, resolve_bundle_path, resolve_target_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize HuBERT speech features into KaraOne semantic tokens.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone.yaml"))
    parser.add_argument("--out", default="../artifacts/audio_targets/karaone_trial_hubert_tokens_k64.npz")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
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
    flat = features.reshape(-1, features.shape[-1])
    print(f"[semantic-tokens] kmeans K={args.k} frames={flat.shape[0]} dim={flat.shape[1]}")
    labels, centers = kmeans_tokens(flat, k=int(args.k), iters=int(args.iters), seed=int(args.seed))
    token_sequences = labels.reshape(features.shape[0], features.shape[1]).astype(np.int64)
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
    )
    counts = np.bincount(token_sequences.reshape(-1), minlength=int(args.k))
    summary = {
        "out": str(out),
        "source": str(hubert_cache),
        "k": int(args.k),
        "trials": int(token_sequences.shape[0]),
        "steps": int(token_sequences.shape[1]),
        "empty_clusters": int(np.sum(counts == 0)),
        "min_cluster_count": int(counts.min(initial=0)),
        "max_cluster_count": int(counts.max(initial=0)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

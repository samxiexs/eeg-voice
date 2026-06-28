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
from src.karaone_recon.prototypes import semantic_prototype_payload
from src.karaone_recon.semantic_tokens import KaraOneSemanticTokenTargets
from src.karaone_recon.targets import KaraOneTargets
from src.utils import load_simple_yaml, resolve_bundle_path, resolve_target_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train-split-only KaraOne semantic Mel prototype cache.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone.yaml"))
    parser.add_argument("--stages", default="overt_like")
    parser.add_argument("--token-cache", default="../artifacts/audio_targets/karaone_trial_hubert_tokens_k64_trainonly.npz")
    parser.add_argument("--alignment-cache", default="../artifacts/alignment/karaone_overt_like_alignment.npz")
    parser.add_argument("--out", default="../artifacts/audio_targets/karaone_v4_semantic_mel_prototypes_trainonly.npz")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    out = resolve_bundle_path(args.out, BUNDLE_DIR)
    if out.exists() and not args.force:
        print(f"[v4-prototypes] exists: {out}")
        return
    _, mel_cache = resolve_target_cache(cfg, BUNDLE_DIR, "mel")
    if not mel_cache.exists():
        raise FileNotFoundError(f"Missing Mel cache: {mel_cache}")
    token_cache = resolve_bundle_path(args.token_cache, BUNDLE_DIR)
    if not token_cache.exists():
        raise FileNotFoundError(f"Missing semantic-token cache: {token_cache}")
    alignment_cache = resolve_bundle_path(args.alignment_cache, BUNDLE_DIR)
    targets = KaraOneTargets(mel_cache, data_root=root)
    token_targets = KaraOneSemanticTokenTargets(token_cache)
    stages = tuple(item.strip() for item in str(args.stages).split(",") if item.strip())
    common = dict(
        data_root=root,
        targets=targets,
        stages=stages,
        split_protocol=str(cfg["data"].get("split_protocol", "trial")),
        heldout_subjects=cfg["data"].get("heldout_subjects", ["P02", "MM21"]),
        eeg_len=int(cfg["data"].get("eeg_len", 1280)),
        semantic_token_targets=token_targets,
    )
    if alignment_cache.exists():
        common["alignment_cache"] = alignment_cache
    ds = KaraOneTrialDataset(split="train", **common)
    target_rows = []
    token_rows = []
    mask_rows = []
    label_rows = []
    lag_rows = []
    template_ids = []
    for idx in range(len(ds)):
        item = ds[idx]
        entry = ds.entries[idx]
        target_rows.append(item["target_seq"].numpy())
        token_rows.append(item["semantic_token_targets"].numpy())
        mask_rows.append(item["semantic_token_mask"].numpy())
        label_rows.append(int(item["label_idx"].item()))
        lag_rows.append(float(item["lag_sec"].item()) if "lag_sec" in item else 0.0)
        template_ids.append(KaraOneTargets.key(entry.subject, entry.trial_index))
    payload = semantic_prototype_payload(
        np.stack(target_rows, axis=0),
        np.stack(token_rows, axis=0),
        np.stack(mask_rows, axis=0),
        np.asarray(label_rows, dtype=np.int64),
        targets.label_vocab,
        token_targets.vocab_size,
        np.asarray(lag_rows, dtype=np.float32),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        **payload,
        source_mel_cache=str(mel_cache),
        source_token_cache=str(token_cache),
        source_alignment_cache=str(alignment_cache) if alignment_cache.exists() else "",
        split="train",
        stages=np.asarray(stages, dtype=str),
        train_template_ids=np.asarray(template_ids, dtype=str),
        vocab_size=np.asarray(token_targets.vocab_size, dtype=np.int32),
        target_steps=np.asarray(targets.T, dtype=np.int32),
        target_dim=np.asarray(targets.D, dtype=np.int32),
    )
    summary = {
        "out": str(out),
        "split": "train",
        "stages": list(stages),
        "n_train": int(len(ds)),
        "vocab_size": int(token_targets.vocab_size),
        "target_shape": [int(targets.T), int(targets.D)],
        "token_empty": int((payload["token_counts"] <= 0).sum()),
        "label_empty": int((payload["label_counts"] <= 0).sum()),
        "global_lag_sec": float(payload["global_lag_sec"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

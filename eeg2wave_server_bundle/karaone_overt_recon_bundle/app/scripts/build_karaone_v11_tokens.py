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

from src.karaone_v9.data import KaraOneV9TargetBank
from src.karaone_v10.data import KaraOneV10ClusterBank, KaraOneV10ClusteredDataset
from src.utils import ensure_dir, load_simple_yaml, resolve_bundle_path, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train-only KaraOne v11 EEG/audio token cache and audit.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone_v11.yaml"))
    parser.add_argument("--stages", default=None)
    parser.add_argument("--cluster-bank", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--audit-out", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    train_cfg = cfg.get("train", {})
    set_seed(int(train_cfg.get("seed", 11)))
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    stages = tuple(item.strip() for item in str(args.stages or cfg["data"].get("stages", "overt_like")).split(",") if item.strip())
    cache_cfg = cfg.get("cache", {})
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
    eeg_sample_rate = float(cfg["data"].get("eeg_sample_rate", 256.0))
    cluster_path = args.cluster_bank or cache_cfg.get("cluster_bank", "")
    cluster_bank = KaraOneV10ClusterBank(resolve_bundle_path(cluster_path, BUNDLE_DIR) if cluster_path else None)
    datasets = [
        KaraOneV10ClusteredDataset(root, targets, split, cluster_bank=cluster_bank, stages=stages, subject_val=subject_val, subject_test=subject_test, eeg_len=eeg_len, require_codec=True)
        for split in ("subject_train", "subject_val", "subject_test")
    ]
    all_items = []
    for dataset in datasets:
        for idx in range(len(dataset)):
            all_items.append(dataset[idx])
            if args.max_rows and len(all_items) >= int(args.max_rows):
                break
        if args.max_rows and len(all_items) >= int(args.max_rows):
            break
    if not all_items:
        raise RuntimeError("No rows found for v11 token cache")
    train_items = [item for item in all_items if str(item["subject"]) not in {subject_val, subject_test}]
    if not train_items:
        raise RuntimeError("No train rows available for train-only v11 token fit")
    token_cfg = cfg.get("tokens", {})
    codec_vocab = int(token_cfg.get("codec_token_vocab", 64))
    channel_clusters = int(token_cfg.get("channel_clusters", 8))
    max_codec_frames = int(token_cfg.get("max_codec_fit_frames", 12000))
    kmeans_iters = int(token_cfg.get("kmeans_iters", 12))

    channel_cluster_id = fit_channel_clusters(train_items, n_clusters=channel_clusters, kmeans_iters=kmeans_iters, sample_rate=eeg_sample_rate)
    fit_frames = np.concatenate([np.asarray(item["codec_seq"], dtype=np.float32) for item in train_items], axis=0)
    if fit_frames.shape[0] > max_codec_frames:
        rng = np.random.default_rng(int(train_cfg.get("seed", 11)))
        fit_frames = fit_frames[rng.choice(fit_frames.shape[0], size=max_codec_frames, replace=False)]
    codec_codebook = kmeans(fit_frames, n_clusters=codec_vocab, n_iter=kmeans_iters)

    keys, subjects, labels, stages_out, trials, splits, fit_split = [], [], [], [], [], [], []
    codec_token_ids, codec_token_mask = [], []
    for item in all_items:
        subject = str(item["subject"])
        stage = str(item["stage"])
        trial = int(item["trial_index"])
        keys.append(f"{subject}:{stage}:{trial}")
        subjects.append(subject)
        labels.append(str(item["label"]))
        stages_out.append(stage)
        trials.append(trial)
        split = "subject_val" if subject == subject_val else "subject_test" if subject == subject_test else "subject_train"
        splits.append(split)
        fit_split.append(split == "subject_train")
        codec = np.asarray(item["codec_seq"], dtype=np.float32)
        ids = nearest_codebook(codec, codec_codebook)
        codec_token_ids.append(ids.astype(np.int64))
        codec_token_mask.append(np.ones(ids.shape[0], dtype=np.float32))

    out = resolve_bundle_path(args.out or cache_cfg.get("v11_token_bank", "../artifacts/audio_targets/karaone_v11_tokens_overt_like.npz"), BUNDLE_DIR)
    ensure_dir(out.parent)
    np.savez_compressed(
        out,
        keys=np.asarray(keys, dtype=object),
        subjects=np.asarray(subjects, dtype=object),
        labels=np.asarray(labels, dtype=object),
        stages=np.asarray(stages_out, dtype=object),
        trials=np.asarray(trials, dtype=np.int64),
        split=np.asarray(splits, dtype=object),
        fit_split=np.asarray(fit_split, dtype=bool),
        channel_cluster_id=channel_cluster_id.astype(np.int64),
        n_channel_clusters=np.asarray(channel_clusters, dtype=np.int64),
        codec_codebook=codec_codebook.astype(np.float32),
        codec_token_ids=np.stack(codec_token_ids, axis=0).astype(np.int64),
        codec_token_mask=np.stack(codec_token_mask, axis=0).astype(np.float32),
    )
    audit = {
        "audit_kind": "karaone_v11_train_only_token_bank",
        "status": "pass",
        "token_bank": str(out),
        "stages": list(stages),
        "n_rows": len(keys),
        "n_train_fit_rows": len(train_items),
        "heldout_subject_used_for_token_fit": {subject_val: False, subject_test: False},
        "codec_token_vocab": codec_vocab,
        "codec_dim": int(codec_codebook.shape[1]),
        "codec_token_steps": int(codec_token_ids[0].shape[0]),
        "channel_clusters": channel_clusters,
        "channel_cluster_counts": {str(k): int(v) for k, v in zip(*np.unique(channel_cluster_id, return_counts=True))},
    }
    audit_out = resolve_bundle_path(args.audit_out or cache_cfg.get("v11_token_audit", "../artifacts/audio_targets/karaone_v11_token_audit_overt_like.json"), BUNDLE_DIR)
    write_json(audit_out, audit)
    print(json.dumps({"token_bank": str(out), "audit": str(audit_out), **audit}, ensure_ascii=False, indent=2), flush=True)


def fit_channel_clusters(items: list[dict[str, Any]], *, n_clusters: int, kmeans_iters: int, sample_rate: float = 256.0) -> np.ndarray:
    descs = []
    for item in items:
        eeg = np.asarray(item["eeg"], dtype=np.float32)
        valid = int(item["eeg_valid_len"])
        descs.append(channel_descriptor(eeg[:, :valid], sample_rate=sample_rate))
    desc = np.stack(descs, axis=0).mean(axis=0)
    centroids = kmeans(desc, n_clusters=n_clusters, n_iter=kmeans_iters)
    return nearest_codebook(desc, centroids)


def channel_descriptor(eeg: np.ndarray, *, sample_rate: float = 256.0) -> np.ndarray:
    mean = eeg.mean(axis=1, keepdims=True)
    std = eeg.std(axis=1, keepdims=True)
    logvar = np.log(eeg.var(axis=1, keepdims=True) + 1e-5)
    abs_mean = np.abs(eeg).mean(axis=1, keepdims=True)
    half = max(1, eeg.shape[1] // 2)
    slope = eeg[:, half:].mean(axis=1, keepdims=True) - eeg[:, :half].mean(axis=1, keepdims=True)
    diff = np.diff(eeg, axis=1)
    diff_energy = np.log((diff**2).mean(axis=1, keepdims=True) + 1e-5)
    bands = log_bandpower(eeg, sample_rate=sample_rate)
    return zscore(np.concatenate([mean, std, logvar, abs_mean, slope, diff_energy, bands], axis=1))


def log_bandpower(eeg: np.ndarray, *, sample_rate: float = 256.0) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(eeg.astype(np.float32), axis=1)) ** 2
    freqs = np.fft.rfftfreq(eeg.shape[1], d=1.0 / max(float(sample_rate), 1.0))
    band_defs = ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 60.0))
    out = []
    for low, high in band_defs:
        mask = (freqs >= low) & (freqs < high)
        if mask.any():
            value = spectrum[:, mask].mean(axis=1, keepdims=True)
        else:
            value = np.zeros((eeg.shape[0], 1), dtype=np.float32)
        out.append(np.log(value + 1e-5))
    return np.concatenate(out, axis=1).astype(np.float32)


def kmeans(values: np.ndarray, *, n_clusters: int, n_iter: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"kmeans values must be [N,D], got {values.shape}")
    n_clusters = max(1, min(int(n_clusters), values.shape[0]))
    rng = np.random.default_rng(11)
    centroids = values[rng.choice(values.shape[0], size=n_clusters, replace=False)].copy()
    for _ in range(max(1, int(n_iter))):
        ids = nearest_codebook(values, centroids)
        for k in range(n_clusters):
            mask = ids == k
            if mask.any():
                centroids[k] = values[mask].mean(axis=0)
    return centroids.astype(np.float32)


def nearest_codebook(values: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    codebook = np.asarray(codebook, dtype=np.float32)
    dist = ((values[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=-1)
    return dist.argmin(axis=1).astype(np.int64)


def zscore(values: np.ndarray) -> np.ndarray:
    return (values - values.mean(axis=0, keepdims=True)) / (values.std(axis=0, keepdims=True) + 1e-6)


if __name__ == "__main__":
    main()

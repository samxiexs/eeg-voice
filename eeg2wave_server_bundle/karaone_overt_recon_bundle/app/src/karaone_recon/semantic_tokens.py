from __future__ import annotations

from pathlib import Path

import numpy as np


class KaraOneSemanticTokenTargets:
    def __init__(self, cache_path: str | Path):
        payload = np.load(Path(cache_path), allow_pickle=True)
        self.path = Path(cache_path)
        self.template_ids = payload["template_ids"].astype(str)
        self.subject_ids = payload["subject_ids"].astype(str)
        self.labels = payload["labels"].astype(str)
        self.trial_indices = payload["trial_indices"].astype(np.int32)
        self.token_sequences = payload["token_sequences"].astype(np.int64)
        self.token_mask = (
            payload["token_mask"].astype(np.float32)
            if "token_mask" in payload.files
            else np.ones(self.token_sequences.shape, dtype=np.float32)
        )
        self.centroids = payload["centroids"].astype(np.float32)
        self.vocab_size = int(payload["vocab_size"])
        self.T = int(self.token_sequences.shape[1])
        self.key_to_idx = {
            f"{subject}:{int(trial)}": idx
            for idx, (subject, trial) in enumerate(zip(self.subject_ids.tolist(), self.trial_indices.tolist()))
        }

    @staticmethod
    def key(subject: str, trial_index: int) -> str:
        return f"{subject}:{int(trial_index)}"

    def has_trial(self, subject: str, trial_index: int) -> bool:
        return self.key(subject, trial_index) in self.key_to_idx

    def index(self, subject: str, trial_index: int) -> int:
        return self.key_to_idx[self.key(subject, trial_index)]

    def tokens(self, subject: str, trial_index: int) -> np.ndarray:
        return self.token_sequences[self.index(subject, trial_index)]

    def mask(self, subject: str, trial_index: int) -> np.ndarray:
        return self.token_mask[self.index(subject, trial_index)]


def kmeans_tokens(features: np.ndarray, k: int = 64, iters: int = 30, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected [N,D] features, got {x.shape}")
    n = int(x.shape[0])
    if n < k:
        raise ValueError(f"k={k} exceeds number of feature frames n={n}")
    init = rng.choice(n, size=k, replace=False)
    centers = x[init].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max(1, int(iters))):
        # Chunked distances keep memory bounded for HuBERT caches.
        chunks = []
        for start in range(0, n, 4096):
            chunk = x[start : start + 4096]
            dist = (
                np.sum(chunk * chunk, axis=1, keepdims=True)
                - 2.0 * (chunk @ centers.T)
                + np.sum(centers * centers, axis=1, keepdims=True).T
            )
            chunks.append(np.argmin(dist, axis=1).astype(np.int64))
        labels = np.concatenate(chunks, axis=0)
        new_centers = np.zeros_like(centers)
        counts = np.bincount(labels, minlength=k).astype(np.float32)
        for idx in range(k):
            if counts[idx] > 0:
                new_centers[idx] = x[labels == idx].mean(axis=0)
            else:
                new_centers[idx] = x[int(rng.integers(0, n))]
        shift = float(np.mean(np.square(new_centers - centers)))
        centers = new_centers
        if shift < 1e-7:
            break
    return labels, centers


def assign_tokens_to_centroids(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    centers = np.asarray(centroids, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected [N,D] features, got {x.shape}")
    if centers.ndim != 2 or centers.shape[1] != x.shape[1]:
        raise ValueError(f"centroids shape {centers.shape} incompatible with features {x.shape}")
    chunks = []
    center_norm = np.sum(centers * centers, axis=1, keepdims=True).T
    for start in range(0, x.shape[0], 4096):
        chunk = x[start : start + 4096]
        dist = np.sum(chunk * chunk, axis=1, keepdims=True) - 2.0 * (chunk @ centers.T) + center_norm
        chunks.append(np.argmin(dist, axis=1).astype(np.int64))
    return np.concatenate(chunks, axis=0)

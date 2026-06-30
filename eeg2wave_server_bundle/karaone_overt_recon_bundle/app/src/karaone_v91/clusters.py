from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ClusterBuildResult:
    path: Path
    audit_path: Path
    audit: dict[str, Any]


def build_cluster_bank_arrays(
    rows: list[dict[str, Any]],
    *,
    n_eeg_clusters: int = 12,
    n_speech_clusters: int = 12,
    n_cross_clusters: int = 16,
    seed: int = 7,
) -> dict[str, np.ndarray]:
    """Fit train-only EEG/speech centroids and assign every row to fixed banks.

    `rows` must contain descriptors for all requested samples and a `fit_split`
    boolean.  Only rows with `fit_split=True` estimate centroids.  Heldout rows
    are projected to those train centroids for evaluation.
    """

    if not rows:
        raise ValueError("Cannot build a v9.1 cluster bank from zero rows")
    fit_mask = np.asarray([bool(row["fit_split"]) for row in rows], dtype=bool)
    if int(fit_mask.sum()) == 0:
        raise ValueError("Cluster fitting requires at least one subject_train row")

    eeg_desc = np.stack([np.asarray(row["eeg_descriptor"], dtype=np.float32) for row in rows], axis=0)
    speech_desc = np.stack([np.asarray(row["speech_descriptor"], dtype=np.float32) for row in rows], axis=0)
    eeg_desc = zscore_train(eeg_desc, fit_mask)
    speech_desc = zscore_train(speech_desc, fit_mask)
    cross_desc = zscore_train(np.concatenate([eeg_desc, speech_desc], axis=1), fit_mask)

    eeg_centroids = kmeans(eeg_desc[fit_mask], min(int(n_eeg_clusters), int(fit_mask.sum())), seed=seed)
    speech_centroids = kmeans(speech_desc[fit_mask], min(int(n_speech_clusters), int(fit_mask.sum())), seed=seed + 11)
    cross_centroids = kmeans(cross_desc[fit_mask], min(int(n_cross_clusters), int(fit_mask.sum())), seed=seed + 23)

    eeg_cluster_id = nearest_centroid(eeg_desc, eeg_centroids)
    speech_cluster_id = nearest_centroid(speech_desc, speech_centroids)
    cross_modal_cluster_id = nearest_centroid(cross_desc, cross_centroids)
    soft_map = soft_correspondence(
        eeg_cluster_id[fit_mask],
        speech_cluster_id[fit_mask],
        n_eeg=int(eeg_centroids.shape[0]),
        n_speech=int(speech_centroids.shape[0]),
    )
    return {
        "keys": np.asarray([str(row["key"]) for row in rows], dtype="<U128"),
        "subject_ids": np.asarray([str(row["subject"]) for row in rows], dtype="<U32"),
        "labels": np.asarray([str(row["label"]) for row in rows], dtype="<U64"),
        "stages": np.asarray([str(row["stage"]) for row in rows], dtype="<U32"),
        "trial_indices": np.asarray([int(row["trial_index"]) for row in rows], dtype=np.int32),
        "split_kind": np.asarray([str(row["split_kind"]) for row in rows], dtype="<U32"),
        "fit_split": fit_mask.astype(np.int8),
        "eeg_cluster_id": eeg_cluster_id.astype(np.int16),
        "speech_cluster_id": speech_cluster_id.astype(np.int16),
        "cross_modal_cluster_id": cross_modal_cluster_id.astype(np.int16),
        "eeg_centroids": eeg_centroids.astype(np.float32),
        "speech_centroids": speech_centroids.astype(np.float32),
        "cross_centroids": cross_centroids.astype(np.float32),
        "eeg_to_speech_soft": soft_map.astype(np.float32),
    }


def cluster_audit(arrays: dict[str, np.ndarray], *, subject_val: str, subject_test: str) -> dict[str, Any]:
    subjects = arrays["subject_ids"].astype(str)
    split = arrays["split_kind"].astype(str)
    fit = arrays["fit_split"].astype(bool)
    val_in_fit = bool(np.any((subjects == str(subject_val)) & fit))
    test_in_fit = bool(np.any((subjects == str(subject_test)) & fit))
    train_subjects = sorted(set(subjects[fit].tolist()))
    coverage: dict[str, dict[str, int]] = {}
    for name in ("eeg_cluster_id", "speech_cluster_id", "cross_modal_cluster_id"):
        values, counts = np.unique(arrays[name], return_counts=True)
        coverage[name] = {str(int(value)): int(count) for value, count in zip(values, counts)}
    return {
        "audit_kind": "karaone_v91_train_only_cluster_bank",
        "status": "pass" if not val_in_fit and not test_in_fit and int(fit.sum()) > 0 else "fail",
        "n_rows": int(subjects.shape[0]),
        "n_fit_rows": int(fit.sum()),
        "fit_subjects": train_subjects,
        "heldout_subjects": [str(subject_val), str(subject_test)],
        "heldout_subject_used_for_centroid_fit": {
            str(subject_val): val_in_fit,
            str(subject_test): test_in_fit,
        },
        "split_counts": {str(item): int(np.sum(split == item)) for item in sorted(set(split.tolist()))},
        "cluster_coverage": coverage,
    }


def eeg_descriptor(eeg: np.ndarray, valid_len: int, *, sample_rate: float = 256.0, envelope_bins: int = 32) -> np.ndarray:
    """Per-trial normalized EEG descriptor used for train-only clustering."""

    x = np.asarray(eeg, dtype=np.float32)
    valid_len = int(max(1, min(int(valid_len), x.shape[-1])))
    x = x[:, :valid_len]
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = (x - x.mean(axis=1, keepdims=True)) / np.sqrt(x.var(axis=1, keepdims=True) + 1e-5)
    logvar = np.log(np.var(x, axis=1) + 1e-5)
    band = bandpower_descriptor(x, sample_rate=sample_rate)
    env = low_frequency_envelope(x, bins=envelope_bins)
    cov = channel_covariance_sketch(x)
    return np.concatenate([logvar, band.reshape(-1), env, cov], axis=0).astype(np.float32)


def speech_descriptor(
    semantic_summary: np.ndarray,
    semantic_tokens: np.ndarray,
    semantic_token_mask: np.ndarray,
    prosody: dict[str, Any],
    *,
    token_vocab: int,
) -> np.ndarray:
    sem = np.asarray(semantic_summary, dtype=np.float32).reshape(-1)
    if sem.size > 128:
        chunks = np.array_split(sem, 128)
        sem = np.asarray([float(chunk.mean()) for chunk in chunks], dtype=np.float32)
    tokens = np.asarray(semantic_tokens, dtype=np.int64).reshape(-1)
    mask = np.asarray(semantic_token_mask, dtype=np.float32).reshape(-1) > 0.5
    vocab = max(2, int(token_vocab))
    hist = np.zeros(vocab, dtype=np.float32)
    if tokens.size and bool(mask.any()):
        valid = np.clip(tokens[mask], 0, vocab - 1)
        hist += np.bincount(valid, minlength=vocab).astype(np.float32)
        hist /= max(float(hist.sum()), 1.0)
    active = np.asarray(prosody.get("active", []), dtype=np.float32).reshape(-1)
    energy = np.asarray(prosody.get("energy", []), dtype=np.float32).reshape(-1)
    active_mean = float(active.mean()) if active.size else 0.0
    energy_mean = float(energy.mean()) if energy.size else 0.0
    energy_std = float(energy.std()) if energy.size else 0.0
    p = np.asarray(
        [
            active_mean,
            energy_mean,
            energy_std,
            float(prosody.get("duration", 0.0)),
            float(prosody.get("onset", 0.0)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([sem, hist, p], axis=0).astype(np.float32)


def bandpower_descriptor(x: np.ndarray, *, sample_rate: float) -> np.ndarray:
    bands = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 80.0)]
    spec = np.abs(np.fft.rfft(x, axis=1)) ** 2
    freqs = np.fft.rfftfreq(x.shape[1], d=1.0 / float(sample_rate))
    total = spec.sum(axis=1, keepdims=True).clip(min=1e-6)
    out = []
    for lo, hi in bands:
        band_mask = (freqs >= lo) & (freqs < hi)
        if not bool(band_mask.any()):
            out.append(np.zeros(x.shape[0], dtype=np.float32))
        else:
            out.append(np.log((spec[:, band_mask].sum(axis=1) / total[:, 0]).clip(min=1e-8)))
    return np.stack(out, axis=1).astype(np.float32)


def low_frequency_envelope(x: np.ndarray, *, bins: int) -> np.ndarray:
    env = np.mean(np.abs(x), axis=0)
    if env.size == int(bins):
        return env.astype(np.float32)
    src = np.linspace(0.0, 1.0, env.size)
    dst = np.linspace(0.0, 1.0, int(bins))
    return np.interp(dst, src, env).astype(np.float32)


def channel_covariance_sketch(x: np.ndarray, *, max_pairs: int = 256) -> np.ndarray:
    cov = np.corrcoef(x)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    upper = cov[np.triu_indices(cov.shape[0], k=1)]
    if upper.size <= int(max_pairs):
        return upper.astype(np.float32)
    idx = np.linspace(0, upper.size - 1, int(max_pairs)).round().astype(np.int64)
    return upper[idx].astype(np.float32)


def zscore_train(x: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    train = x[fit_mask]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True).clip(min=1e-5)
    return ((x - mean) / std).astype(np.float32)


def kmeans(x: np.ndarray, k: int, *, seed: int, iters: int = 40) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"kmeans expects [N,D] with N>0, got {tuple(x.shape)}")
    k = max(1, min(int(k), int(x.shape[0])))
    rng = np.random.default_rng(int(seed))
    first = int(rng.integers(0, x.shape[0]))
    centroids = [x[first]]
    dist = np.sum((x - centroids[0]) ** 2, axis=1)
    for _ in range(1, k):
        probs = dist / dist.sum() if float(dist.sum()) > 1e-8 else np.full(x.shape[0], 1.0 / x.shape[0])
        idx = int(rng.choice(x.shape[0], p=probs))
        centroids.append(x[idx])
        dist = np.minimum(dist, np.sum((x - x[idx]) ** 2, axis=1))
    c = np.stack(centroids, axis=0).astype(np.float32)
    for _ in range(int(iters)):
        labels = nearest_centroid(x, c)
        new = c.copy()
        for idx in range(k):
            mask = labels == idx
            if bool(mask.any()):
                new[idx] = x[mask].mean(axis=0)
        if np.allclose(new, c, atol=1e-5):
            break
        c = new
    return c.astype(np.float32)


def nearest_centroid(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x, axis=1, keepdims=True)
    c2 = np.sum(centroids * centroids, axis=1, keepdims=True).T
    dist = x2 + c2 - 2.0 * (x @ centroids.T)
    return np.argmin(dist, axis=1).astype(np.int16)


def soft_correspondence(eeg_ids: np.ndarray, speech_ids: np.ndarray, *, n_eeg: int, n_speech: int) -> np.ndarray:
    counts = np.ones((int(n_eeg), int(n_speech)), dtype=np.float32)
    for eeg_id, speech_id in zip(eeg_ids.tolist(), speech_ids.tolist()):
        counts[int(eeg_id), int(speech_id)] += 1.0
    return counts / counts.sum(axis=1, keepdims=True).clip(min=1e-6)

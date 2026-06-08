from __future__ import annotations

from collections import defaultdict

import numpy as np


def l2_normalize(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, eps, None)


def cosine_similarity_matrix(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return l2_normalize(lhs) @ l2_normalize(rhs).T


def retrieval_topk(
    predicted: np.ndarray,
    target_template_ids: list[str],
    bank_embeddings: np.ndarray,
    bank_template_ids: list[str],
    topk: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    sims = cosine_similarity_matrix(predicted, bank_embeddings)
    order = np.argsort(-sims, axis=1)
    metrics: dict[str, float] = {}
    for k in topk:
        correct = 0
        for row_idx, target_id in enumerate(target_template_ids):
            retrieved_ids = [bank_template_ids[col] for col in order[row_idx, :k].tolist()]
            correct += int(target_id in retrieved_ids)
        metrics[f"retrieval_top{k}"] = correct / max(len(target_template_ids), 1)
    return metrics


def nearest_centroid_subject_probe(
    train_embeddings: np.ndarray,
    train_subject_ids: list[str],
    eval_embeddings: np.ndarray,
    eval_subject_ids: list[str],
) -> float:
    subject_pool: dict[str, list[np.ndarray]] = defaultdict(list)
    for embedding, subject_id in zip(train_embeddings, train_subject_ids):
        subject_pool[str(subject_id)].append(np.asarray(embedding, dtype=np.float32))
    centroids = {
        subject_id: np.mean(np.stack(values, axis=0), axis=0)
        for subject_id, values in subject_pool.items()
        if values
    }
    if not centroids:
        return 0.0
    centroid_ids = sorted(centroids)
    centroid_matrix = np.stack([centroids[item] for item in centroid_ids], axis=0)
    sims = cosine_similarity_matrix(eval_embeddings, centroid_matrix)
    pred_ids = [centroid_ids[idx] for idx in np.argmax(sims, axis=1).tolist()]
    correct = sum(int(pred == str(target)) for pred, target in zip(pred_ids, eval_subject_ids))
    return correct / max(len(eval_subject_ids), 1)

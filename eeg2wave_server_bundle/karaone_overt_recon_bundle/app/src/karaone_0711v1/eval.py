from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import SplitManifest, write_json


@dataclass(frozen=True)
class GateResult:
    passed: bool
    semantic_retrieval_gain: float
    semantic_retrieval_ci_low: float
    token_retrieval_gain: float
    token_retrieval_ci_low: float
    semantic_over_zero_gain: float
    std_ratio: float
    median_pairwise_correlation: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "semantic_retrieval_gain": self.semantic_retrieval_gain,
            "semantic_retrieval_ci_low": self.semantic_retrieval_ci_low,
            "token_retrieval_gain": self.token_retrieval_gain,
            "token_retrieval_ci_low": self.token_retrieval_ci_low,
            "semantic_over_zero_gain": self.semantic_over_zero_gain,
            "std_ratio": self.std_ratio,
            "median_pairwise_correlation": self.median_pairwise_correlation,
            "reasons": list(self.reasons),
        }


def _norm(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def bootstrap_lower(values: np.ndarray, *, samples: int = 1000, seed: int = 11) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("-inf")
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(values, size=values.size, replace=True).mean() for _ in range(int(samples))])
    return float(np.quantile(means, 0.025))


def cross_subject_label_retrieval(
    eeg_embed: np.ndarray,
    target_labels: np.ndarray,
    train_audio_embed: np.ndarray,
    train_audio_labels: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Retrieval bank contains only subject_train audio, never P02/MM21 targets."""
    similarity = _norm(np.asarray(eeg_embed)) @ _norm(np.asarray(train_audio_embed)).T
    predicted = np.asarray(train_audio_labels)[similarity.argmax(axis=1)]
    success = (predicted == np.asarray(target_labels)).astype(np.float64)
    chance = np.asarray([(np.asarray(train_audio_labels) == label).mean() for label in target_labels], dtype=np.float64)
    gains = success - chance
    return float(gains.mean()), gains


def token_gain(logits: np.ndarray, target_ids: np.ndarray, train_token_ids: np.ndarray) -> tuple[float, np.ndarray]:
    logits = np.asarray(logits)
    target_ids = np.asarray(target_ids)
    pred = logits.argmax(axis=-1)
    acc = (pred == target_ids).mean(axis=1)
    counts = np.bincount(np.asarray(train_token_ids).reshape(-1), minlength=logits.shape[-1])
    prior = counts.argmax()
    baseline = (target_ids == prior).mean(axis=1)
    gains = acc - baseline
    return float(gains.mean()), gains


def embedding_diagnostics(embed: np.ndarray, reference_embed: np.ndarray) -> tuple[float, float]:
    embed = np.asarray(embed, dtype=np.float64)
    reference_embed = np.asarray(reference_embed, dtype=np.float64)
    std_ratio = float(np.median(embed.std(axis=0) / np.maximum(reference_embed.std(axis=0), 1e-8)))
    if embed.shape[0] < 2:
        return std_ratio, 1.0
    corr = np.corrcoef(embed)
    return std_ratio, float(np.median(corr[np.triu_indices_from(corr, k=1)]))


def semantic_over_zero(eeg_embed: np.ndarray, target_embed: np.ndarray, zero_embed: np.ndarray) -> float:
    target = _norm(np.asarray(target_embed))
    pred = _norm(np.asarray(eeg_embed))
    zero = _norm(np.asarray(zero_embed))
    return float((pred * target).sum(axis=1).mean() - (zero * target).sum(axis=1).mean())


def evaluate_global_gate(
    *,
    eeg_embed: np.ndarray,
    zero_embed: np.ndarray,
    target_embed: np.ndarray,
    target_labels: np.ndarray,
    token_logits: np.ndarray,
    target_tokens: np.ndarray,
    train_audio_embed: np.ndarray,
    train_audio_labels: np.ndarray,
    train_token_ids: np.ndarray,
) -> GateResult:
    semantic_gain, semantic_values = cross_subject_label_retrieval(eeg_embed, target_labels, train_audio_embed, train_audio_labels)
    token_retrieval, token_values = token_gain(token_logits, target_tokens, train_token_ids)
    over_zero = semantic_over_zero(eeg_embed, target_embed, zero_embed)
    std_ratio, pairwise = embedding_diagnostics(eeg_embed, target_embed)
    semantic_low = bootstrap_lower(semantic_values)
    token_low = bootstrap_lower(token_values)
    reasons = []
    if semantic_low <= 0:
        reasons.append("semantic_retrieval_ci_not_positive")
    if token_low <= 0:
        reasons.append("token_retrieval_ci_not_positive")
    if over_zero <= 0:
        reasons.append("semantic_over_zero_not_positive")
    if std_ratio < 0.10:
        reasons.append("embedding_variance_collapse")
    if pairwise > 0.95:
        reasons.append("embedding_pairwise_correlation_collapse")
    return GateResult(
        passed=not reasons,
        semantic_retrieval_gain=semantic_gain,
        semantic_retrieval_ci_low=semantic_low,
        token_retrieval_gain=token_retrieval,
        token_retrieval_ci_low=token_low,
        semantic_over_zero_gain=over_zero,
        std_ratio=std_ratio,
        median_pairwise_correlation=pairwise,
        reasons=tuple(reasons),
    )


def write_gate(path: str | Path, result: GateResult, *, split: str, manifest: SplitManifest, checkpoint: str | Path) -> Path:
    if split != "subject_val":
        raise ValueError("Only subject_val may decide a 0711v1 gate")
    payload = {
        "version": "0711v1",
        "selection_split": split,
        "test_accessed": False,
        "checkpoint": str(checkpoint),
        "split_checksum": manifest.checksum,
        **result.to_dict(),
    }
    return write_json(path, payload)


def require_flow_gate(path: str | Path, manifest: SplitManifest) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("split_checksum") != manifest.checksum:
        raise ValueError("Gate split checksum does not match current split")
    if payload.get("selection_split") != "subject_val" or payload.get("test_accessed"):
        raise ValueError("Flow requires a validation-only gate")
    if not payload.get("passed"):
        raise RuntimeError(f"Flow is blocked: {payload.get('reasons', [])}")
    return payload


def final_test_report(path: str | Path, metrics: dict[str, Any], *, manifest: SplitManifest, checkpoint: str | Path) -> Path:
    """This is the sole API that records MM21 access after a locked checkpoint."""
    payload = {
        "version": "0711v1",
        "evaluation_split": "subject_test",
        "checkpoint": str(checkpoint),
        "split_checksum": manifest.checksum,
        "test_accessed": True,
        **metrics,
    }
    return write_json(path, payload)

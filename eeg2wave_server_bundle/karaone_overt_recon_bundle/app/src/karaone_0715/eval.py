from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def balanced_accuracy(target: np.ndarray, prediction: np.ndarray, classes: int = 11) -> float:
    target = np.asarray(target, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    recalls = []
    for label in range(int(classes)):
        selected = target == label
        if selected.any():
            recalls.append(float(np.mean(prediction[selected] == label)))
    return float(np.mean(recalls)) if recalls else float("nan")


def stratified_bootstrap_accuracy(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    samples: int = 1000,
    seed: int = 15,
) -> tuple[float, float]:
    target = np.asarray(target, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(target == label) for label in sorted(set(target.tolist()))]
    values = []
    for _ in range(int(samples)):
        draw = np.concatenate([indices[rng.integers(0, len(indices), size=len(indices))] for indices in groups])
        values.append(balanced_accuracy(target[draw], prediction[draw], classes=len(groups)))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def retrieval_top1(query: np.ndarray, target: np.ndarray) -> float:
    query = np.asarray(query, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
    return float(np.mean(np.argmax(query @ target.T, axis=1) == np.arange(len(query))))


@dataclass(frozen=True)
class EEGValidationGate:
    passed: bool
    class_balanced_accuracy: float
    class_accuracy_ci_low: float
    class_accuracy_ci_high: float
    paired_retrieval_top1: float
    coarse_code_accuracy: float
    label_only_coarse_code_accuracy: float
    coarse_code_gain: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def make_eeg_gate(
    *,
    labels: np.ndarray,
    predictions: np.ndarray,
    eeg_condition: np.ndarray,
    audio_condition: np.ndarray,
    coarse_code_accuracy: float,
    label_only_coarse_code_accuracy: float,
    min_balanced_accuracy: float,
    chance_accuracy: float,
    bootstrap_samples: int,
) -> EEGValidationGate:
    accuracy = balanced_accuracy(labels, predictions)
    ci_low, ci_high = stratified_bootstrap_accuracy(labels, predictions, samples=bootstrap_samples)
    retrieval = retrieval_top1(eeg_condition, audio_condition)
    gain = float(coarse_code_accuracy - label_only_coarse_code_accuracy)
    reasons = []
    if accuracy < float(min_balanced_accuracy):
        reasons.append("class_balanced_accuracy_below_threshold")
    if ci_low <= float(chance_accuracy):
        reasons.append("class_accuracy_ci_not_above_chance")
    if gain <= 0.0:
        reasons.append("eeg_condition_does_not_improve_over_label_only_codec_prior")
    return EEGValidationGate(
        passed=not reasons,
        class_balanced_accuracy=accuracy,
        class_accuracy_ci_low=ci_low,
        class_accuracy_ci_high=ci_high,
        paired_retrieval_top1=retrieval,
        coarse_code_accuracy=float(coarse_code_accuracy),
        label_only_coarse_code_accuracy=float(label_only_coarse_code_accuracy),
        coarse_code_gain=gain,
        reasons=tuple(reasons),
    )

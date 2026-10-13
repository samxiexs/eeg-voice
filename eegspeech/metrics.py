"""Content-first scoring and subject-level inference, with no pseudo-replication."""
from __future__ import annotations
import itertools
import numpy as np


def word_errors(reference: str, hypothesis: str) -> dict:
    a, b = reference.split(), hypothesis.split()
    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    dp[:, 0] = np.arange(len(a) + 1); dp[0] = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i, j] = min(dp[i-1, j] + 1, dp[i, j-1] + 1, dp[i-1, j-1] + int(a[i-1] != b[j-1]))
    return {"errors": int(dp[-1, -1]), "words": len(a), "wer": float(dp[-1, -1] / len(a)) if a else None}


def bootstrap_mean(values, repetitions=10000, seed=11):
    values = np.asarray(values, dtype=float)
    if not len(values): raise ValueError("No independent units")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (repetitions, len(values)))].mean(1)
    lower, upper = np.quantile(means, [.025, .975])
    return {"n": len(values), "mean": float(values.mean()), "lower": float(lower), "upper": float(upper)}


def paired_test(deltas, repetitions=10000, seed=11):
    values = np.asarray(deltas, dtype=float)
    if not len(values): raise ValueError("No paired subjects")
    rng = np.random.default_rng(seed)
    if len(values) <= 19:
        signs = np.array(list(itertools.product([-1, 1], repeat=len(values))))
        null = (signs * values).mean(1)
        p = np.mean(np.abs(null) >= abs(values.mean()) - 1e-12)
    else:
        signs = rng.choice([-1, 1], (repetitions, len(values)))
        null = (signs * values).mean(1)
        p = (1 + np.sum(np.abs(null) >= abs(values.mean()))) / (repetitions + 1)
    return {**bootstrap_mean(values, repetitions, seed), "p_two_sided": float(p), "unit": "subject"}


def holm(pvalues):
    values = np.asarray(pvalues)
    order = np.argsort(values)
    adjusted = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order]).clip(0, 1)
    result = np.empty_like(adjusted); result[order] = adjusted
    return result.tolist()


def aggregate_wer(rows):
    """Sum word errors within sample seed, average samples then training seeds."""
    by_seed = {}
    for row in rows:
        key = (row["subject"], row["system"], row["control"], row["training_seed"], row["generation_seed"])
        count = by_seed.setdefault(key, [0, 0])
        count[0] += row["errors"]; count[1] += row["words"]
    subjects = {}
    for key, (errors, words) in by_seed.items():
        if not words: raise ValueError("An entire subject/control has no reference words")
        subject, system, control, _, _ = key
        subjects.setdefault((subject, system, control), []).append(errors / words)
    return [{"subject": s, "system": m, "control": c, "wer": float(np.mean(values)), "repetitions": len(values)} for (s, m, c), values in sorted(subjects.items())]


def two_way_bootstrap(rows_a, rows_b, repetitions=10000, seed=11):
    """Paired subject × stimulus resampling; only three stimuli in main test."""
    def table(rows):
        groups = {}
        for r in rows:
            key = (r["subject"], r["audio"], r["training_seed"], r["generation_seed"])
            g = groups.setdefault(key, [0, 0]); g[0] += r["errors"]; g[1] += r["words"]
        result = {}
        for (s, a, _, _), (e, n) in groups.items():
            if not n: raise ValueError("An entire stimulus has no reference words")
            result.setdefault((s, a), []).append(e/n)
        return {key: np.mean(v) for key, v in result.items()}
    a, b = table(rows_a), table(rows_b)
    people = sorted({s for s, clip in a if (s, clip) in b})
    clips = sorted({clip for s, clip in a if (s, clip) in b})
    matrix = np.array([[a.get((s,c), np.nan) - b.get((s,c), np.nan) for c in clips] for s in people])
    if np.isnan(matrix).any(): raise ValueError("Incomplete subject/audio pairs in two-way inference")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        estimates.append(matrix[np.ix_(rng.integers(len(people), size=len(people)), rng.integers(len(clips), size=len(clips)))].mean())
    lower, upper = np.quantile(estimates, [.025, .975])
    return {"mean": float(matrix.mean()), "lower": float(lower), "upper": float(upper), "subjects": len(people), "stimuli": len(clips), "limitation": "Only three independent test stimuli"}

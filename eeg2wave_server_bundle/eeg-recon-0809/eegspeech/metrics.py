"""Accuracy summaries with the statistics every result in this project is reported with."""
from __future__ import annotations

import numpy as np
from scipy import stats


def summarize(per_subject, chance):
    """Per-subject accuracies -> mean, SEM, Wilcoxon (one-sided, > chance) and the number above chance."""
    values = np.asarray([v for v in per_subject.values() if np.isfinite(v)], float)
    if not len(values):
        return dict(subjects=0)
    out = dict(subjects=len(values), mean=float(values.mean()), sem=float(values.std(ddof=1) / np.sqrt(len(values)))
               if len(values) > 1 else float('nan'), chance=float(chance), above_chance=int((values > chance).sum()))
    if len(values) >= 5 and np.any(values != chance):
        out['p_wilcoxon'] = float(stats.wilcoxon(values - chance, alternative='greater').pvalue)
    return out


def binomial_p(correct, total, chance):
    """One-sided exact binomial p of ``correct`` hits out of ``total`` at ``chance``."""
    return float(stats.binomtest(int(correct), int(total), chance, alternative='greater').pvalue)


def permutation_chance(labels, predictions, rounds=1000, seed=0):
    """Accuracy null from label permutations: (mean, 95th percentile)."""
    rng = np.random.default_rng(seed)
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    null = np.array([(rng.permutation(labels) == predictions).mean() for _ in range(rounds)])
    return float(null.mean()), float(np.percentile(null, 95))

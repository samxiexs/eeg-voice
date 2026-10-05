"""Accuracy summaries with the statistics every result in this project is reported with."""
from __future__ import annotations

import numpy as np
from scipy import stats


def summarize(per_subject, chance):
    """Per-subject accuracies -> mean, SEM, Wilcoxon (one-sided, > chance) and the number above chance."""
    values = np.asarray([v for v in per_subject.values() if np.isfinite(v)], float)
    if not len(values):
        return dict(subjects=0)
    diff = np.where(np.abs(values - chance) < 1e-9, 0., values - chance)       # float noise is a tie
    out = dict(subjects=len(values), mean=float(values.mean()), sem=float(values.std(ddof=1) / np.sqrt(len(values)))
               if len(values) > 1 else float('nan'), chance=float(chance), above_chance=int((diff > 0).sum()))
    if len(values) >= 5 and np.any(diff != 0):
        out['p_wilcoxon'] = float(stats.wilcoxon(diff, alternative='greater', zero_method='zsplit').pvalue)
    return out


def binomial_p(correct, total, chance):
    """One-sided exact binomial p of ``correct`` hits out of ``total`` at ``chance``."""
    return float(stats.binomtest(int(correct), int(total), chance, alternative='greater').pvalue)

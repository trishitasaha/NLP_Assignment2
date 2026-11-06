"""Bootstrap confidence intervals (decision D13). Seeded, so results are reproducible."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from . import config as C


def _ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.nanpercentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_spearman(x, y, n: int = C.BOOTSTRAP_RESAMPLES, seed: int = C.SEED) -> tuple[float, float]:
    """95% CI of Spearman rho, resampling (x, y) pairs with replacement."""
    x, y = np.asarray(x), np.asarray(y)
    rng = np.random.default_rng(seed)
    stats = np.empty(n)
    for b in range(n):
        idx = rng.integers(0, len(x), len(x))
        stats[b] = spearmanr(x[idx], y[idx]).statistic
    return _ci(stats)


def bootstrap_uuas(correct, total, n: int = C.BOOTSTRAP_RESAMPLES, seed: int = C.SEED) -> tuple[float, float]:
    """95% CI of micro-averaged UUAS, resampling *sentences* with replacement."""
    correct, total = np.asarray(correct, float), np.asarray(total, float)
    rng = np.random.default_rng(seed)
    stats = np.empty(n)
    for b in range(n):
        idx = rng.integers(0, len(correct), len(correct))
        stats[b] = correct[idx].sum() / max(total[idx].sum(), 1)
    return _ci(stats)


def paired_bootstrap_uuas(correct_a, correct_b, total, n: int = C.BOOTSTRAP_RESAMPLES,
                          seed: int = C.SEED) -> dict:
    """Paired bootstrap for UUAS(A) - UUAS(B) on the same sentences.

    Returns the observed difference, its 95% CI, and the fraction of resamples where A <= B
    (a one-sided p-value style quantity).
    """
    a, b, t = (np.asarray(v, float) for v in (correct_a, correct_b, total))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for k in range(n):
        idx = rng.integers(0, len(t), len(t))
        diffs[k] = (a[idx].sum() - b[idx].sum()) / max(t[idx].sum(), 1)
    obs = (a.sum() - b.sum()) / t.sum()
    lo, hi = _ci(diffs)
    return dict(diff=float(obs), ci_low=lo, ci_high=hi, frac_le_zero=float((diffs <= 0).mean()))


def paired_bootstrap_spearman(model_a, model_b, human, n: int = C.BOOTSTRAP_RESAMPLES,
                              seed: int = C.SEED) -> dict:
    """Paired bootstrap for rho(A, human) - rho(B, human) on the *same* word pairs."""
    a, b, h = (np.asarray(v, float) for v in (model_a, model_b, human))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for k in range(n):
        idx = rng.integers(0, len(h), len(h))
        diffs[k] = spearmanr(a[idx], h[idx]).statistic - spearmanr(b[idx], h[idx]).statistic
    obs = spearmanr(a, h).statistic - spearmanr(b, h).statistic
    lo, hi = _ci(diffs)
    return dict(diff=float(obs), ci_low=lo, ci_high=hi, frac_le_zero=float((diffs <= 0).mean()))


def mcnemar_exact(correct_a, correct_b) -> dict:
    """Exact (binomial) McNemar test for two classifiers on the same items (e.g. analogy questions)."""
    from scipy.stats import binomtest
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    a_only, b_only = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return dict(a_only=a_only, b_only=b_only, p_value=float(p))

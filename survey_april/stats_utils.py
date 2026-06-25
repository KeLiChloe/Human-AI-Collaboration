"""Shared bootstrap CI and Welch t-test helpers for survey figures."""

from __future__ import annotations

import numpy as np
from scipy.stats import ttest_ind, ttest_rel


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        v = float(arr[0])
        return v, v
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, n), replace=True)
    means = samples.mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def bootstrap_ci_half_width(
    sample_values,
    *,
    n_boot: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> float:
    """Symmetric half-width for bar whiskers (max distance from mean to bootstrap bounds)."""
    arr = np.asarray(sample_values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return 0.0
    mean = float(np.mean(arr))
    lo, hi = bootstrap_mean_ci(arr, n_boot=n_boot, seed=seed, alpha=alpha)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0
    return max(mean - lo, hi - mean)


def p_value_welch_ttest(a, b) -> float:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    arr_a = arr_a[~np.isnan(arr_a)]
    arr_b = arr_b[~np.isnan(arr_b)]
    if len(arr_a) < 2 or len(arr_b) < 2:
        return np.nan
    try:
        return float(ttest_ind(arr_a, arr_b, equal_var=False, nan_policy="omit").pvalue)
    except Exception:
        return np.nan


def p_value_welch_ttest_one_sided(a, b, *, alternative: str = "greater") -> float:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    arr_a = arr_a[~np.isnan(arr_a)]
    arr_b = arr_b[~np.isnan(arr_b)]
    if len(arr_a) < 2 or len(arr_b) < 2:
        return np.nan
    try:
        return float(
            ttest_ind(
                arr_a,
                arr_b,
                equal_var=False,
                nan_policy="omit",
                alternative=alternative,
            ).pvalue
        )
    except Exception:
        return np.nan


def welch_test(a: dict, b: dict, values_key: str = "sims") -> float:
    return p_value_welch_ttest(a[values_key], b[values_key])


def p_value_paired_ttest(pre, post) -> float:
    arr_pre = np.asarray(pre, dtype=float)
    arr_post = np.asarray(post, dtype=float)
    mask = np.isfinite(arr_pre) & np.isfinite(arr_post)
    arr_pre = arr_pre[mask]
    arr_post = arr_post[mask]
    if len(arr_pre) < 2:
        return np.nan
    try:
        return float(ttest_rel(arr_pre, arr_post, nan_policy="omit").pvalue)
    except Exception:
        return np.nan


def p_value_paired_ttest_pairs(pairs: list[tuple[float, float]]) -> float:
    if not pairs:
        return np.nan
    pre = [p[0] for p in pairs]
    post = [p[1] for p in pairs]
    return p_value_paired_ttest(pre, post)

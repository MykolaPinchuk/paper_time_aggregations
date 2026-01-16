from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x_sorted = x[order]
    n = x_sorted.size
    midranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = midranks
    return out


def _fast_delong(preds_sorted: np.ndarray, n_pos: int) -> Tuple[np.ndarray, np.ndarray]:
    m = int(n_pos)
    if m <= 0 or m >= preds_sorted.shape[1]:
        raise ValueError("n_pos must be in (0, n_samples).")
    n = preds_sorted.shape[1] - m
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    k = preds_sorted.shape[0]

    tx = np.empty((k, m), dtype=np.float64)
    ty = np.empty((k, n), dtype=np.float64)
    tz = np.empty((k, m + n), dtype=np.float64)
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(preds_sorted[r])

    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def delong_roc_test(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float, float, float]:
    """
    Paired DeLong test for the ROC-AUC difference between two models.

    Returns:
        delta_auc, ci_low, ci_high, p_value
    """
    y_true = np.asarray(y_true)
    p_a = np.asarray(p_a)
    p_b = np.asarray(p_b)
    if y_true.ndim != 1 or p_a.ndim != 1 or p_b.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")
    if not (y_true.size == p_a.size == p_b.size):
        raise ValueError("y_true, p_a, p_b must have the same length.")

    y_bin = (y_true > 0).astype(np.int8)
    n_pos = int(y_bin.sum())
    n_neg = int(y_bin.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need both positive and negative samples for DeLong.")

    order = np.argsort(-y_bin, kind="mergesort")
    preds_sorted = np.vstack((p_a, p_b))[:, order]

    aucs, delong_cov = _fast_delong(preds_sorted, n_pos)
    delta = float(aucs[0] - aucs[1])
    var = float(delong_cov[0, 0] + delong_cov[1, 1] - 2.0 * delong_cov[0, 1])
    var = max(var, 1e-12)
    se = math.sqrt(var)

    z = delta / se
    p_value = 2.0 * (1.0 - _norm_cdf(abs(z)))

    z_alpha = 1.959963984540054  # 97.5% quantile for N(0,1)
    ci_low = delta - z_alpha * se
    ci_high = delta + z_alpha * se
    return delta, ci_low, ci_high, float(p_value)


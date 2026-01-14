from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def _prepare_paired_preds(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    row_id_col: str,
    group_col: str,
    y_col: str,
    p_col: str,
) -> pd.DataFrame:
    for name, df in (("df_a", df_a), ("df_b", df_b)):
        for col in (row_id_col, group_col, y_col, p_col):
            if col not in df.columns:
                raise ValueError(f"{name} must contain column '{col}'.")
        if df[row_id_col].duplicated().any():
            raise ValueError(f"{name} contains duplicate row_id values.")

    merged = df_a[[row_id_col, group_col, y_col, p_col]].merge(
        df_b[[row_id_col, group_col, y_col, p_col]],
        on=row_id_col,
        how="inner",
        suffixes=("_a", "_b"),
    )
    if len(merged) != len(df_a) or len(merged) != len(df_b):
        raise ValueError("df_a and df_b must have the same row_id set.")

    if not np.array_equal(merged[f"{group_col}_a"].to_numpy(), merged[f"{group_col}_b"].to_numpy()):
        raise ValueError("group_col values do not match between df_a and df_b.")
    if not np.array_equal(merged[f"{y_col}_a"].to_numpy(), merged[f"{y_col}_b"].to_numpy()):
        raise ValueError("y values do not match between df_a and df_b.")

    merged = merged.rename(
        columns={
            f"{group_col}_a": group_col,
            f"{y_col}_a": y_col,
            f"{p_col}_a": "p_a",
            f"{p_col}_b": "p_b",
        }
    )
    return merged[[row_id_col, group_col, y_col, "p_a", "p_b"]]


def paired_block_bootstrap_pr_auc(
    df_pred_a: pd.DataFrame,
    df_pred_b: pd.DataFrame,
    group_col: str = "hour_dt",
    y_col: str = "y",
    p_col: str = "p",
    row_id_col: str = "row_id",
    B: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Paired block bootstrap for PR-AUC deltas (A - B) using grouped resampling.

    Returns:
        delta_mean, ci_low, ci_high
    """
    if B <= 0:
        raise ValueError("B must be > 0.")
    paired = _prepare_paired_preds(df_pred_a, df_pred_b, row_id_col, group_col, y_col, p_col)

    groups = paired[group_col].to_numpy()
    y = paired[y_col].to_numpy()
    p_a = paired["p_a"].to_numpy()
    p_b = paired["p_b"].to_numpy()

    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError("Need at least 2 groups for block bootstrap.")

    group_to_idx = {g: np.flatnonzero(groups == g) for g in unique_groups}
    rng = np.random.default_rng(seed)

    deltas = np.empty(B, dtype=np.float64)
    for b in range(B):
        sampled_groups = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sampled_groups])
        y_b = y[idx]
        if y_b.min() == y_b.max():
            deltas[b] = np.nan
            continue
        pr_a = average_precision_score(y_b, p_a[idx])
        pr_b = average_precision_score(y_b, p_b[idx])
        deltas[b] = pr_a - pr_b

    valid = deltas[~np.isnan(deltas)]
    if valid.size == 0:
        raise RuntimeError("All bootstrap replicates had single-class outcomes.")

    delta_mean = float(valid.mean())
    ci_low = float(np.quantile(valid, alpha / 2.0))
    ci_high = float(np.quantile(valid, 1.0 - alpha / 2.0))
    return delta_mean, ci_low, ci_high


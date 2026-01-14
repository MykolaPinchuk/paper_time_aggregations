from __future__ import annotations

import pandas as pd


def deterministic_pct_mask(values: pd.Series, pct: int) -> pd.Series:
    if not (0 < pct <= 100):
        raise ValueError(f"pct must be in [1,100], got {pct}")
    h = pd.util.hash_pandas_object(values, index=False).astype("uint64")
    return (h % 100) < pct


def deterministic_frac_mask(values: pd.Series, frac: float, denom: int = 10000) -> pd.Series:
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"frac must be in (0,1], got {frac}")
    if denom <= 0:
        raise ValueError(f"denom must be >0, got {denom}")
    threshold = int(round(frac * denom))
    if threshold <= 0:
        raise ValueError(f"frac too small for denom={denom} (threshold=0)")
    h = pd.util.hash_pandas_object(values, index=False).astype("uint64")
    return (h % denom) < threshold

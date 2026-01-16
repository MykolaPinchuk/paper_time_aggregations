from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TEConfig:
    m: float = 100.0
    min_imps: int = 0
    prior_alpha: float = 1.0
    prior_beta: float = 10.0


def _global_prior_by_hour(df: pd.DataFrame, label_col: str, cfg: TEConfig) -> pd.Series:
    per_hour = (
        df.groupby("hour_dt", sort=True)[label_col]
        .agg(imps="count", clicks="sum")
        .reset_index()
        .sort_values("hour_dt")
    )
    cum_imps = per_hour["imps"].cumsum().shift(1, fill_value=0)
    cum_clicks = per_hour["clicks"].cumsum().shift(1, fill_value=0)
    prior = (cum_clicks + cfg.prior_alpha) / (cum_imps + cfg.prior_alpha + cfg.prior_beta)
    return pd.Series(prior.to_numpy(), index=pd.Index(per_hour["hour_dt"]), name="prior")


def add_time_target_encoding(
    df: pd.DataFrame,
    cat_cols: list[str],
    label_col: str = "click",
    cfg: TEConfig | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    if cfg is None:
        cfg = TEConfig()
    if "hour_dt" not in df.columns:
        raise ValueError("df must contain 'hour_dt' (use add_time_columns first).")

    prior_by_hour = _global_prior_by_hour(df, label_col=label_col, cfg=cfg)
    prior_for_rows = df["hour_dt"].map(prior_by_hour).astype("float32")

    out = df if inplace else df.copy()
    out["prior_ctr"] = prior_for_rows

    for col in cat_cols:
        g = (
            df.groupby([col, "hour_dt"], sort=True)[label_col]
            .agg(imps="count", clicks="sum")
            .reset_index()
            .sort_values([col, "hour_dt"])
        )
        g["cum_imps"] = g.groupby(col, sort=False)["imps"].cumsum() - g["imps"]
        g["cum_clicks"] = g.groupby(col, sort=False)["clicks"].cumsum() - g["clicks"]

        if cfg.min_imps > 0:
            g.loc[g["cum_imps"] < cfg.min_imps, ["cum_imps", "cum_clicks"]] = 0

        # Join-free alignment back to rows.
        te_idx = pd.MultiIndex.from_frame(g[[col, "hour_dt"]])
        row_idx = pd.MultiIndex.from_frame(df[[col, "hour_dt"]])

        cum_imps = pd.Series(g["cum_imps"].to_numpy(), index=te_idx).reindex(row_idx).fillna(0.0).to_numpy()
        cum_clicks = (
            pd.Series(g["cum_clicks"].to_numpy(), index=te_idx).reindex(row_idx).fillna(0.0).to_numpy()
        )

        te = (cum_clicks + cfg.m * prior_for_rows.to_numpy()) / (cum_imps + cfg.m)
        out[f"{col}__te"] = te.astype("float32")
        out[f"{col}__hist_imps"] = np.log1p(cum_imps).astype("float32")

    return out


def add_time_prior_ctr(
    df: pd.DataFrame,
    label_col: str = "click",
    cfg: TEConfig | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Add only the time-aware global prior CTR feature (`prior_ctr`) without per-category TE columns.

    This is useful for running time-aggregation-only experiments while keeping the same online-safe prior.
    """
    if cfg is None:
        cfg = TEConfig()
    if "hour_dt" not in df.columns:
        raise ValueError("df must contain 'hour_dt' (use add_time_columns first).")
    prior_by_hour = _global_prior_by_hour(df, label_col=label_col, cfg=cfg)
    prior_for_rows = df["hour_dt"].map(prior_by_hour).astype("float32")
    out = df if inplace else df.copy()
    out["prior_ctr"] = prior_for_rows
    return out

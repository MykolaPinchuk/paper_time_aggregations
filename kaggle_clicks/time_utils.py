from __future__ import annotations

import pandas as pd


def parse_avazu_hour(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.zfill(8)
    return pd.to_datetime(s, format="%y%m%d%H", errors="raise")


def add_time_columns(df: pd.DataFrame, hour_col: str = "hour") -> pd.DataFrame:
    hour_dt = parse_avazu_hour(df[hour_col])
    out = df.copy()
    out["hour_dt"] = hour_dt
    out["day"] = hour_dt.dt.floor("D")
    out["hour_of_day"] = hour_dt.dt.hour.astype("int16")
    return out


def make_oot_splits_by_day(df: pd.DataFrame) -> dict[str, pd.Index]:
    days = pd.Index(df["day"].unique()).sort_values()
    if len(days) < 3:
        hours = pd.Index(df["hour_dt"].unique()).sort_values()
        if len(hours) < 3:
            raise ValueError(
                f"Need >=3 distinct time buckets for train/val/test, got days={len(days)} hours={len(hours)}."
            )

        n = len(hours)
        train_end = hours[int(n * 0.6)]
        val_end = hours[int(n * 0.8)]

        train_idx = df.index[df["hour_dt"] < train_end]
        val_idx = df.index[(df["hour_dt"] >= train_end) & (df["hour_dt"] < val_end)]
        test_idx = df.index[df["hour_dt"] >= val_end]
        return {"train": train_idx, "val": val_idx, "test": test_idx}

    test_day = days[-1]
    val_day = days[-2]
    train_days = days[:-2]

    train_idx = df.index[df["day"].isin(train_days)]
    val_idx = df.index[df["day"] == val_day]
    test_idx = df.index[df["day"] == test_day]

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def make_rolling_tail_folds_by_day(df: pd.DataFrame) -> dict[str, dict[str, pd.Index]]:
    """
    Create two "rolling tail" folds using the last 3 distinct days in the dataset.

    Let the final three days be: D-2, D-1, D (chronological).

    Fold A:
      - train: days < D-1
      - val:   day  D-1
      - test:  day  D

    Fold B:
      - train: days < D-2
      - val:   day  D-2
      - test:  day  D-1

    This is intended as a lightweight stability check for OOT evaluation while keeping runtime bounded.
    """
    if "day" not in df.columns:
        if "hour_dt" not in df.columns:
            raise ValueError("df must contain 'day' or 'hour_dt' to construct rolling tail folds.")
        day = df["hour_dt"].dt.floor("D")
        days = pd.Index(day.unique()).sort_values()
        day_series = day
    else:
        days = pd.Index(df["day"].unique()).sort_values()
        day_series = df["day"]

    if len(days) < 4:
        raise ValueError(f"Need >=4 distinct days for rolling tail folds, got {len(days)}.")

    d_minus2, d_minus1, d0 = days[-3], days[-2], days[-1]

    fold_a = {
        "train": df.index[day_series < d_minus1],
        "val": df.index[day_series == d_minus1],
        "test": df.index[day_series == d0],
    }
    fold_b = {
        "train": df.index[day_series < d_minus2],
        "val": df.index[day_series == d_minus2],
        "test": df.index[day_series == d_minus1],
    }
    return {"A": fold_a, "B": fold_b}


def assert_strict_oot(df: pd.DataFrame, splits: dict[str, pd.Index]) -> None:
    def bounds(name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        s = df.loc[splits[name], "hour_dt"]
        return s.min(), s.max()

    tr_min, tr_max = bounds("train")
    va_min, va_max = bounds("val")
    te_min, te_max = bounds("test")

    if not (tr_max < va_min and va_max < te_min):
        raise AssertionError(
            "Not strictly OOT:\n"
            f"train: [{tr_min}, {tr_max}]\n"
            f"  val: [{va_min}, {va_max}]\n"
            f" test: [{te_min}, {te_max}]"
        )

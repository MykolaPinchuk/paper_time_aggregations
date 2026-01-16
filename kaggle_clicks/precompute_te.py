from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

from kaggle_clicks.paths import get_paths
from kaggle_clicks.te import TEConfig, add_time_target_encoding
from kaggle_clicks.time_utils import add_time_columns


def _infer_cat_cols(df: pd.DataFrame, label_col: str) -> list[str]:
    drop_cols = {"id", "hour", "hour_dt", "day", "hour_of_day", label_col}
    return [c for c in df.columns if c not in drop_cols]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-parquet", required=True, help="Input parquet sample (must include `hour` and `click`).")
    ap.add_argument("--out-parquet", required=True, help="Output parquet for cached TE features.")
    ap.add_argument("--m", type=float, default=100.0, help="TE smoothing strength (larger = more shrinkage).")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    args = ap.parse_args()

    out_path = os.path.abspath(args.out_parquet)
    if os.path.exists(out_path) and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file (use --overwrite): {out_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = pd.read_parquet(args.sample_parquet)
    df = add_time_columns(df, hour_col="hour").sort_values("hour_dt").reset_index(drop=True)

    label_col = "click"
    cat_cols = _infer_cat_cols(df, label_col=label_col)

    # Speed/memory: factorize object columns to compact int codes (deterministic given row order).
    for col in cat_cols:
        if df[col].dtype == object:
            codes, _ = pd.factorize(df[col], sort=False)
            df[col] = codes.astype(np.int32, copy=False)

    fe = add_time_target_encoding(df, cat_cols=cat_cols, label_col=label_col, cfg=TEConfig(m=float(args.m)), inplace=True)

    # Export only what downstream needs; align by row order (row_id).
    te_cols: list[str] = []
    for c in cat_cols:
        te_cols.append(f"{c}__te")
        te_cols.append(f"{c}__hist_imps")

    out = pd.DataFrame({"row_id": np.arange(len(fe), dtype=np.int64)})
    out["prior_ctr"] = fe["prior_ctr"].to_numpy(dtype=np.float32, copy=False)
    for col in te_cols:
        out[col] = fe[col].to_numpy(dtype=np.float32, copy=False)

    # Optional: include hour_dt for sanity checks (cheap, helps detect mismatched ordering).
    out["hour_dt"] = fe["hour_dt"].to_numpy()

    out.to_parquet(out_path, index=False)

    meta_path = os.path.splitext(out_path)[0] + ".meta.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"created_utc={datetime.utcnow().isoformat()}Z\n")
        f.write(f"sample_parquet={os.path.abspath(args.sample_parquet)}\n")
        f.write(f"n_rows={len(out)}\n")
        f.write(f"te_m={float(args.m)}\n")
        f.write(f"n_cat_cols={len(cat_cols)}\n")
        f.write("cat_cols=" + ",".join(cat_cols) + "\n")

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


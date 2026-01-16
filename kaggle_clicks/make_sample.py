from __future__ import annotations

import argparse
import os

import pandas as pd

from kaggle_clicks.sampling import deterministic_frac_mask, deterministic_pct_mask


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a deterministic sample parquet from Avazu train.csv.")
    ap.add_argument("--train-csv", default="data/raw/train.csv")
    ap.add_argument("--sample-pct", type=int, default=10)
    ap.add_argument("--sample-frac", type=float, default=None, help="Optional fractional sample in (0,1].")
    ap.add_argument("--out-parquet", required=True)
    args = ap.parse_args()

    if not os.path.exists(args.train_csv):
        raise SystemExit(f"Missing train csv: {args.train_csv}")
    if args.sample_frac is not None and not (0.0 < float(args.sample_frac) <= 1.0):
        raise SystemExit("--sample-frac must be in (0,1].")
    if args.sample_frac is None and not (1 <= int(args.sample_pct) <= 100):
        raise SystemExit("--sample-pct must be in [1,100].")

    out_path = os.path.abspath(args.out_parquet)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(args.train_csv, chunksize=500_000):
        if "id" not in chunk.columns:
            raise SystemExit("Expected 'id' column in train.csv.")
        if args.sample_frac is not None:
            mask = deterministic_frac_mask(chunk["id"], frac=float(args.sample_frac))
        else:
            mask = deterministic_pct_mask(chunk["id"], pct=int(args.sample_pct))
        if mask.any():
            chunks.append(chunk.loc[mask])

    if not chunks:
        raise SystemExit("Sampling produced 0 rows; increase sample size.")

    df = pd.concat(chunks, axis=0, ignore_index=True)
    df.to_parquet(out_path, index=False)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


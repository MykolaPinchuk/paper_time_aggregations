from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kaggle_clicks.inference_auc import delong_roc_test
from kaggle_clicks.inference_bootstrap import paired_block_bootstrap_pr_auc


def _load_preds(run_dir: str, split: str) -> pd.DataFrame | None:
    path = Path(run_dir) / f"preds_{split}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _align_for_inference(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    needed = {"row_id", "hour_dt", "y", "p"}
    for name, df in (("df_a", df_a), ("df_b", df_b)):
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
    if df_a["row_id"].duplicated().any() or df_b["row_id"].duplicated().any():
        raise ValueError("row_id must be unique within each preds file.")

    merged = df_a[["row_id", "hour_dt", "y", "p"]].merge(
        df_b[["row_id", "hour_dt", "y", "p"]],
        on="row_id",
        how="inner",
        suffixes=("_a", "_b"),
    )
    if len(merged) != len(df_a) or len(merged) != len(df_b):
        raise ValueError("Predictions must share the same row_id set.")
    if not np.array_equal(merged["hour_dt_a"].to_numpy(), merged["hour_dt_b"].to_numpy()):
        raise ValueError("hour_dt mismatch between paired predictions.")
    if not np.array_equal(merged["y_a"].to_numpy(), merged["y_b"].to_numpy()):
        raise ValueError("y mismatch between paired predictions.")
    return merged.rename(
        columns={
            "hour_dt_a": "hour_dt",
            "y_a": "y",
            "p_a": "p_a",
            "p_b": "p_b",
        }
    )


def _infer_one(
    df_candidate: pd.DataFrame,
    df_baseline: pd.DataFrame,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, Any]:
    merged = _align_for_inference(df_candidate, df_baseline)
    y = merged["y"].to_numpy()
    p_a = merged["p_a"].to_numpy()
    p_b = merged["p_b"].to_numpy()

    delta_auc, auc_ci_low, auc_ci_high, auc_p = delong_roc_test(y, p_a, p_b)
    if int(bootstrap_reps) > 0:
        delta_pr, pr_ci_low, pr_ci_high = paired_block_bootstrap_pr_auc(
            merged.rename(columns={"p_a": "p"}),
            merged.rename(columns={"p_b": "p"}),
            group_col="hour_dt",
            y_col="y",
            p_col="p",
            row_id_col="row_id",
            B=int(bootstrap_reps),
            seed=int(seed),
        )
    else:
        delta_pr, pr_ci_low, pr_ci_high = (float("nan"), float("nan"), float("nan"))
    return {
        "delta_roc_auc": delta_auc,
        "delta_roc_ci_low": auc_ci_low,
        "delta_roc_ci_high": auc_ci_high,
        "delta_roc_p_value": auc_p,
        "delta_pr_auc": delta_pr,
        "delta_pr_ci_low": pr_ci_low,
        "delta_pr_ci_high": pr_ci_high,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True, help="Path to a sweep dir containing summary.csv.")
    ap.add_argument("--baseline-run-id", required=True, help="Run id to use as baseline (e.g. R3).")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--bootstrap-reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    summary_path = sweep_dir / "summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary.csv at {summary_path}")

    summary = pd.read_csv(summary_path)
    if "fold_id" not in summary.columns:
        summary["fold_id"] = "single"

    results: list[dict[str, Any]] = []
    for fold_id in sorted(summary["fold_id"].unique()):
        fold_rows = summary[summary["fold_id"] == fold_id]
        baseline_rows = fold_rows[fold_rows["run_id"] == args.baseline_run_id]
        if len(baseline_rows) != 1:
            print(f"Skipping fold {fold_id}: expected 1 baseline row, got {len(baseline_rows)}")
            continue
        baseline_dir = baseline_rows.iloc[0]["run_dir"]
        if not isinstance(baseline_dir, str) or not baseline_dir:
            print(f"Skipping fold {fold_id}: baseline run_dir missing")
            continue

        baseline_preds = _load_preds(baseline_dir, args.split)
        if baseline_preds is None:
            print(f"Skipping fold {fold_id}: baseline preds_{args.split}.parquet missing")
            continue

        for _, row in fold_rows.iterrows():
            run_id = row["run_id"]
            run_dir = row["run_dir"]
            if run_id == args.baseline_run_id:
                continue
            if not isinstance(run_dir, str) or not run_dir:
                continue
            candidate_preds = _load_preds(run_dir, args.split)
            if candidate_preds is None:
                continue
            inf = _infer_one(candidate_preds, baseline_preds, args.bootstrap_reps, args.seed)
            results.append(
                {
                    "run_id": run_id,
                    "fold_id": fold_id,
                    "run_dir": run_dir,
                    "baseline_run_id": args.baseline_run_id,
                    "baseline_run_dir": baseline_dir,
                    "split": args.split,
                    **inf,
                }
            )

    out_path = sweep_dir / "inference_vs_baseline.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

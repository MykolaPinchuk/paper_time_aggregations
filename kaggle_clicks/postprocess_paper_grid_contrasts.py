from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kaggle_clicks.inference_auc import delong_roc_test
from kaggle_clicks.inference_bootstrap import paired_block_bootstrap_pr_auc


def _parse_run_id(run_id: str) -> tuple[str, str]:
    parts = str(run_id).split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Expected run_id like A3_trailing, got: {run_id!r}")
    return parts[0], parts[1]


def _load_preds(run_dir: str, split: str) -> pd.DataFrame:
    path = Path(run_dir) / f"preds_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(str(path))
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
    *,
    df_candidate: pd.DataFrame,
    df_baseline: pd.DataFrame,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, Any]:
    merged = _align_for_inference(df_candidate, df_baseline)
    y = merged["y"].to_numpy()
    p_c = merged["p_a"].to_numpy()
    p_b = merged["p_b"].to_numpy()

    delta_auc, auc_ci_low, auc_ci_high, auc_p = delong_roc_test(y, p_c, p_b)
    out: dict[str, Any] = {
        "delta_roc_auc": delta_auc,
        "delta_roc_ci_low": auc_ci_low,
        "delta_roc_ci_high": auc_ci_high,
        "delta_roc_p_value": auc_p,
    }

    if bootstrap_reps > 0:
        # PR-AUC bootstrap expects a DF with y/p/hour_dt/row_id; do it paired by hour.
        df_pa = merged.rename(columns={"p_a": "p"})[["row_id", "hour_dt", "y", "p"]]
        df_pb = merged.rename(columns={"p_b": "p"})[["row_id", "hour_dt", "y", "p"]]
        delta_pr, pr_ci_low, pr_ci_high = paired_block_bootstrap_pr_auc(
            df_pa,
            df_pb,
            group_col="hour_dt",
            y_col="y",
            p_col="p",
            row_id_col="row_id",
            B=bootstrap_reps,
            seed=seed,
        )
        out.update(
            {
                "delta_pr_auc": delta_pr,
                "delta_pr_ci_low": pr_ci_low,
                "delta_pr_ci_high": pr_ci_high,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True, help="Sweep dir containing summary.csv.")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument(
        "--bootstrap-reps",
        type=int,
        default=0,
        help="If >0, also compute paired block-bootstrap CIs for PR-AUC deltas (slower).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--baseline-shape",
        default="trailing",
        help="Shape to use as baseline within each length set (default: trailing).",
    )
    ap.add_argument(
        "--shapes",
        nargs="*",
        default=["gap1", "bucket", "calendar", "event50"],
        help="Shapes to compare against baseline within each length set.",
    )
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    summary_path = sweep_dir / "summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary.csv at {summary_path}")

    summary = pd.read_csv(summary_path)
    if "fold_id" not in summary.columns:
        summary["fold_id"] = "single"
    if summary["fold_id"].nunique() != 2 or set(summary["fold_id"].unique()) != {"A", "B"}:
        raise SystemExit("Expected rolling-tail folds A/B in summary.csv.")

    summary[["length_set", "shape"]] = summary["run_id"].apply(lambda x: pd.Series(_parse_run_id(x)))

    rows: list[dict[str, Any]] = []
    for fold_id in ("A", "B"):
        fold_rows = summary[summary["fold_id"] == fold_id]

        for length_set in sorted(fold_rows["length_set"].unique()):
            base_id = f"{length_set}_{args.baseline_shape}"
            base_row = fold_rows[fold_rows["run_id"] == base_id]
            if len(base_row) != 1:
                raise SystemExit(f"Expected exactly 1 baseline row for {base_id} fold {fold_id}, got {len(base_row)}")
            base_dir = str(base_row.iloc[0]["run_dir"])
            base_preds = _load_preds(base_dir, args.split)

            for shape in args.shapes:
                cand_id = f"{length_set}_{shape}"
                cand_row = fold_rows[fold_rows["run_id"] == cand_id]
                if len(cand_row) != 1:
                    raise SystemExit(f"Expected exactly 1 candidate row for {cand_id} fold {fold_id}, got {len(cand_row)}")
                cand_dir = str(cand_row.iloc[0]["run_dir"])
                cand_preds = _load_preds(cand_dir, args.split)

                inf = _infer_one(
                    df_candidate=cand_preds,
                    df_baseline=base_preds,
                    bootstrap_reps=int(args.bootstrap_reps),
                    seed=int(args.seed),
                )
                rows.append(
                    {
                        "fold_id": fold_id,
                        "length_set": length_set,
                        "shape": shape,
                        "candidate_run_id": cand_id,
                        "baseline_run_id": base_id,
                        "candidate_run_dir": cand_dir,
                        "baseline_run_dir": base_dir,
                        "split": args.split,
                        **inf,
                    }
                )

    out_df = pd.DataFrame(rows)
    out_csv = sweep_dir / f"contrasts_vs_{args.baseline_shape}_{args.split}.csv"
    out_df.to_csv(out_csv, index=False)

    # Also write a compact markdown table: averaged across folds for each (length_set, shape).
    gcols = ["length_set", "shape"]
    cols = ["delta_roc_auc", "delta_roc_ci_low", "delta_roc_ci_high"]
    md_df = (
        out_df.groupby(gcols)[cols]
        .mean()
        .reset_index()
        .sort_values(["shape", "length_set"])
    )
    lines: list[str] = []
    lines.append(f"# Contrasts vs `{args.baseline_shape}` ({args.split}, mean over folds)")
    lines.append("")
    lines.append(f"- Sweep: `{sweep_dir.name}`")
    lines.append(f"- Baseline within each length set: `{args.baseline_shape}`")
    lines.append(f"- Shapes compared: {', '.join(args.shapes)}")
    if int(args.bootstrap_reps) > 0:
        lines.append(f"- PR-AUC bootstrap reps: {int(args.bootstrap_reps)}")
    else:
        lines.append("- PR-AUC bootstrap: not computed")
    lines.append("")
    lines.append("| Length set | Shape | ΔROC-AUC | 95% CI |")
    lines.append("| --- | --- | ---: | --- |")
    for _, r in md_df.iterrows():
        lines.append(
            f"| {r['length_set']} | {r['shape']} | {r['delta_roc_auc']:.6f} | "
            f"[{r['delta_roc_ci_low']:.6f}, {r['delta_roc_ci_high']:.6f}] |"
        )

    out_md = sweep_dir / f"contrasts_vs_{args.baseline_shape}_{args.split}.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(str(out_csv))
    print(str(out_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

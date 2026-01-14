from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from kaggle_clicks.inference_auc import delong_roc_test
from kaggle_clicks.inference_bootstrap import paired_block_bootstrap_pr_auc


def _find_complete_run_dir(run_tag: str) -> Path:
    runs_root = Path("runs")
    matches = sorted(runs_root.glob(f"*_{run_tag}"), reverse=True)
    for p in matches:
        if not p.is_dir():
            continue
        if not (p / "metrics.json").exists():
            continue
        if not (p / "preds_test.parquet").exists():
            continue
        if not (p / "preds_val.parquet").exists():
            continue
        return p
    raise SystemExit(f"Missing completed run for tag: {run_tag}")


def _read_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except Exception:
        return str(path)


def _infer_pair(df_a: pd.DataFrame, df_b: pd.DataFrame, bootstrap_reps: int, seed: int) -> dict[str, float]:
    needed = {"row_id", "hour_dt", "y", "p"}
    if not needed.issubset(df_a.columns) or not needed.issubset(df_b.columns):
        raise ValueError("preds parquet must include row_id,hour_dt,y,p")
    merged = df_a[["row_id", "hour_dt", "y", "p"]].merge(
        df_b[["row_id", "hour_dt", "y", "p"]],
        on="row_id",
        how="inner",
        suffixes=("_a", "_b"),
    )
    if len(merged) != len(df_a) or len(merged) != len(df_b):
        raise ValueError("preds files must share the same row_id set")
    y = merged["y_a"].to_numpy()
    p_a = merged["p_a"].to_numpy()
    p_b = merged["p_b"].to_numpy()

    delta_roc, roc_lo, roc_hi, roc_p = delong_roc_test(y, p_a, p_b)
    delta_pr, pr_lo, pr_hi = paired_block_bootstrap_pr_auc(
        merged.rename(columns={"p_a": "p", "y_a": "y", "hour_dt_a": "hour_dt"})[["row_id", "hour_dt", "y", "p"]],
        merged.rename(columns={"p_b": "p", "y_b": "y", "hour_dt_b": "hour_dt"})[["row_id", "hour_dt", "y", "p"]],
        group_col="hour_dt",
        y_col="y",
        p_col="p",
        row_id_col="row_id",
        B=int(bootstrap_reps),
        seed=int(seed),
    )
    return {
        "delta_roc_auc": float(delta_roc),
        "delta_roc_ci_low": float(roc_lo),
        "delta_roc_ci_high": float(roc_hi),
        "delta_roc_p_value": float(roc_p),
        "delta_pr_auc": float(delta_pr),
        "delta_pr_ci_low": float(pr_lo),
        "delta_pr_ci_high": float(pr_hi),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="Output directory (e.g., artifacts/v1/10pct_te_lift)")
    ap.add_argument("--bootstrap-reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-pct", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_metrics: list[dict] = []
    rows_inf: list[dict] = []

    for fold_id in ("A", "B"):
        tags = {
            "TE_only": f"paper_te_lift_TE_only_{args.sample_pct}pct_fold{fold_id}",
            "A3_trailing": f"paper_te_lift_A3_trailing_{args.sample_pct}pct_fold{fold_id}",
            "A3_event50": f"paper_te_lift_A3_event50_{args.sample_pct}pct_fold{fold_id}",
        }
        run_dirs = {k: _find_complete_run_dir(v) for k, v in tags.items()}

        for spec, run_dir in run_dirs.items():
            m = _read_metrics(run_dir)
            rows_metrics.append(
                {
                    "run_key": f"{spec}_fold{fold_id}",
                    "test_roc_auc": m["splits"]["test"]["roc_auc"],
                    "test_pr_auc": m["splits"]["test"]["pr_auc"],
                    "val_roc_auc": m["splits"]["val"]["roc_auc"],
                    "val_pr_auc": m["splits"]["val"]["pr_auc"],
                    "best_iteration": m.get("best_iteration"),
                    "run_dir": _rel(run_dir),
                }
            )

        preds = {k: pd.read_parquet(run_dirs[k] / "preds_test.parquet") for k in run_dirs}
        for cand, base in (("A3_trailing", "TE_only"), ("A3_event50", "TE_only"), ("A3_event50", "A3_trailing")):
            inf = _infer_pair(preds[cand], preds[base], bootstrap_reps=int(args.bootstrap_reps), seed=int(args.seed))
            rows_inf.append({"fold_id": fold_id, "candidate": cand, "baseline": base, **inf})

    metrics_df = pd.DataFrame(rows_metrics)
    metrics_df.to_csv(out_dir / "metrics_selected_runs.csv", index=False)

    inf_df = pd.DataFrame(rows_inf)
    inf_df.to_csv(out_dir / "inference_test_te_vs_timeagg.csv", index=False)

    # A compact markdown view (writer-facing).
    lines: list[str] = []
    lines.append("# TE-only Lift vs Time Aggregation (Fold A/B)")
    lines.append("")
    lines.append("This folder contains the TE-only vs time-aggregation comparisons used in the paper.")
    lines.append("")
    lines.append("## Files")
    lines.append("- `metrics_selected_runs.csv`")
    lines.append("- `inference_test_te_vs_timeagg.csv`")
    lines.append("")
    lines.append("## Summary (test set)")
    for fold_id in ("A", "B"):
        sub = inf_df[inf_df["fold_id"] == fold_id]
        if sub.empty:
            continue
        a = sub[sub["candidate"].eq("A3_trailing") & sub["baseline"].eq("TE_only")].iloc[0]
        lines.append(
            f"- Fold {fold_id}: ΔROC(A3_trailing − TE_only)={a['delta_roc_auc']:.4f}, "
            f"ΔPR(A3_trailing − TE_only)={a['delta_pr_auc']:.4f}"
        )
    (out_dir / "te_vs_timeagg_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


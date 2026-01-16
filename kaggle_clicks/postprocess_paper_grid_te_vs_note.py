from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt_mean_std(mean: float, std: float) -> str:
    if not np.isfinite(mean) or not np.isfinite(std):
        return ""
    return f"{mean:.6f}+/-{std:.6f}"


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{name} missing columns: {missing}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--te-summary-csv", required=True)
    ap.add_argument("--note-summary-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    te = pd.read_csv(Path(args.te_summary_csv))
    note = pd.read_csv(Path(args.note_summary_csv))

    req = ["run_id", "fold_id", "description", "test_roc_auc", "test_pr_auc", "val_roc_auc", "val_pr_auc"]
    _require(te, req, "TE summary.csv")
    _require(note, req, "no-TE summary.csv")

    te = te[req].rename(
        columns={
            "description": "description_te",
            "test_roc_auc": "test_roc_auc_te",
            "test_pr_auc": "test_pr_auc_te",
            "val_roc_auc": "val_roc_auc_te",
            "val_pr_auc": "val_pr_auc_te",
        }
    )
    note = note[req].rename(
        columns={
            "description": "description_note",
            "test_roc_auc": "test_roc_auc_note",
            "test_pr_auc": "test_pr_auc_note",
            "val_roc_auc": "val_roc_auc_note",
            "val_pr_auc": "val_pr_auc_note",
        }
    )

    merged = te.merge(note, on=["run_id", "fold_id"], how="inner")
    if len(merged) != len(te) or len(merged) != len(note):
        raise SystemExit("TE and no-TE summaries must contain the same (run_id, fold_id) pairs.")

    merged["description"] = merged["description_te"].where(
        merged["description_te"] == merged["description_note"], merged["description_te"]
    )
    merged = merged.drop(columns=["description_te", "description_note"])

    for metric in ["test_roc_auc", "test_pr_auc", "val_roc_auc", "val_pr_auc"]:
        merged[f"delta_{metric}_te_minus_note"] = merged[f"{metric}_te"] - merged[f"{metric}_note"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "te_vs_note_by_fold.csv", index=False)

    # Aggregate across folds (A/B).
    agg_rows: list[dict[str, object]] = []
    for run_id, g in merged.groupby("run_id", sort=False):
        row: dict[str, object] = {"run_id": run_id, "description": g["description"].iloc[0]}
        for col in [
            "test_roc_auc_te",
            "test_roc_auc_note",
            "delta_test_roc_auc_te_minus_note",
            "test_pr_auc_te",
            "test_pr_auc_note",
            "delta_test_pr_auc_te_minus_note",
        ]:
            vals = g[col].to_numpy(dtype=np.float64)
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan")
        agg_rows.append(row)

    summary = pd.DataFrame(agg_rows).sort_values(
        ["delta_test_roc_auc_te_minus_note_mean", "delta_test_pr_auc_te_minus_note_mean"], ascending=False
    )
    summary.to_csv(out_dir / "te_vs_note_summary.csv", index=False)

    md_lines: list[str] = []
    md_lines.append("# TE vs no-TE (10% sample) — Summary")
    md_lines.append("")
    md_lines.append("Δ = TE+time-agg − no-TE time-agg (same run_id / same fold).")
    md_lines.append("")
    md_lines.append("| Run | ΔTest ROC mean+/-std | ΔTest PR mean+/-std | TE Test ROC mean+/-std | no-TE Test ROC mean+/-std |")
    md_lines.append("| --- | --- | --- | --- | --- |")
    for _, r in summary.iterrows():
        droc = _fmt_mean_std(float(r["delta_test_roc_auc_te_minus_note_mean"]), float(r["delta_test_roc_auc_te_minus_note_std"]))
        dpr = _fmt_mean_std(float(r["delta_test_pr_auc_te_minus_note_mean"]), float(r["delta_test_pr_auc_te_minus_note_std"]))
        te_roc = _fmt_mean_std(float(r["test_roc_auc_te_mean"]), float(r["test_roc_auc_te_std"]))
        note_roc = _fmt_mean_std(float(r["test_roc_auc_note_mean"]), float(r["test_roc_auc_note_std"]))
        md_lines.append(f"| {r['run_id']} | {droc} | {dpr} | {te_roc} | {note_roc} |")

    (out_dir / "te_vs_note_summary.md").write_text("\n".join(md_lines).rstrip() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


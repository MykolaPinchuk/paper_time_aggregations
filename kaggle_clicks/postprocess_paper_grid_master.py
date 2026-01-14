from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


_RUN_ID_RE = re.compile(r"^(A[1-4])_(.+)$")


def _parse_run_id(run_id: str) -> tuple[str, str]:
    m = _RUN_ID_RE.match(run_id)
    if not m:
        return ("", "")
    return (m.group(1), m.group(2))


def _fmt_mean_std(mean: float, std: float) -> str:
    if not np.isfinite(mean) or not np.isfinite(std):
        return ""
    return f"{mean:.4f}+/-{std:.4f}"


def _fmt_delta(delta: float, p_value: float | None) -> str:
    if not np.isfinite(delta):
        return ""
    if p_value is None or not np.isfinite(p_value):
        return f"{delta:.6f}"
    p_str = f"{p_value:.3g}" if p_value < 0.001 else f"{p_value:.4f}".rstrip("0").rstrip(".")
    return f"{delta:.6f} (p={p_str})"


def _assert_expected_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{name} missing columns: {missing}")


def _build_master(summary: pd.DataFrame, inference: pd.DataFrame | None, baseline_run_id: str) -> pd.DataFrame:
    summary = summary.copy()
    summary["length_set"], summary["shape"] = zip(*summary["run_id"].map(_parse_run_id), strict=True)

    metrics = [
        "test_roc_auc",
        "test_pr_auc",
        "val_roc_auc",
        "val_pr_auc",
    ]
    keep = ["run_id", "description", "fold_id", "length_set", "shape", *metrics]
    summary = summary[keep]

    fold_ids = sorted(summary["fold_id"].unique())
    if any(fid not in {"A", "B"} for fid in fold_ids):
        # Still works, but the markdown table assumes A/B.
        fold_ids = sorted(fold_ids)

    wide: dict[str, pd.DataFrame] = {}
    for col in metrics:
        pivot = summary.pivot(index=["run_id", "description", "length_set", "shape"], columns="fold_id", values=col)
        pivot = pivot.rename(columns={fid: f"{col}_{fid}" for fid in pivot.columns})
        wide[col] = pivot

    master = pd.concat([wide[c] for c in metrics], axis=1).reset_index()

    for metric, base in (
        ("test_roc_auc", "test_roc_auc"),
        ("test_pr_auc", "test_pr_auc"),
        ("val_roc_auc", "val_roc_auc"),
        ("val_pr_auc", "val_pr_auc"),
    ):
        vals = [f"{base}_{fid}" for fid in fold_ids if f"{base}_{fid}" in master.columns]
        arr = master[vals].to_numpy(dtype=np.float64)
        master[f"{metric}_mean"] = np.nanmean(arr, axis=1)
        master[f"{metric}_std"] = np.nanstd(arr, axis=1, ddof=1)

    if inference is None or inference.empty:
        return master

    inf = inference.copy()
    _assert_expected_columns(
        inf,
        [
            "run_id",
            "fold_id",
            "delta_roc_auc",
            "delta_roc_ci_low",
            "delta_roc_ci_high",
            "delta_roc_p_value",
            "delta_pr_auc",
            "delta_pr_ci_low",
            "delta_pr_ci_high",
        ],
        name="inference_vs_baseline.csv",
    )
    inf = inf[inf["run_id"] != baseline_run_id]

    # Wide-format inference columns by fold.
    inf_cols = [
        "delta_roc_auc",
        "delta_roc_ci_low",
        "delta_roc_ci_high",
        "delta_roc_p_value",
        "delta_pr_auc",
        "delta_pr_ci_low",
        "delta_pr_ci_high",
    ]
    inf_wide = inf.pivot(index="run_id", columns="fold_id", values=inf_cols)
    inf_wide.columns = [f"{col}_{fold}" for col, fold in inf_wide.columns]
    inf_wide = inf_wide.reset_index()

    master = master.merge(inf_wide, on="run_id", how="left")
    return master


def _write_md(master: pd.DataFrame, out_path: Path, sweep_name: str, baseline_run_id: str) -> None:
    # Sort by mean ROC, best first.
    master = master.sort_values(["test_roc_auc_mean", "test_pr_auc_mean"], ascending=False).reset_index(drop=True)

    cols = [
        "run_id",
        "description",
        "test_roc_auc_mean",
        "test_roc_auc_std",
        "test_pr_auc_mean",
        "test_pr_auc_std",
    ]
    has_inference = any(c.startswith("delta_roc_auc_") for c in master.columns)
    if has_inference:
        cols += [
            "delta_roc_auc_A",
            "delta_roc_p_value_A",
            "delta_roc_auc_B",
            "delta_roc_p_value_B",
            "delta_pr_auc_A",
            "delta_pr_auc_B",
        ]

    lines: list[str] = []
    title = f"# Master Results: {sweep_name}"
    if has_inference:
        title += " (with inference)"
    lines.append(title)
    lines.append("")
    lines.append(f"- Baseline for inference: `{baseline_run_id}`" if has_inference else "- Inference: not included")
    lines.append("")

    if has_inference:
        header = (
            "| Run | Description | Test ROC mean+/-std | Test PR mean+/-std | ΔROC A | ΔROC B | ΔPR A | ΔPR B |"
        )
        sep = "| --- | --- | --- | --- | --- | --- | --- | --- |"
    else:
        header = "| Run | Description | Test ROC mean+/-std | Test PR mean+/-std |"
        sep = "| --- | --- | --- | --- |"
    lines += [header, sep]

    for _, r in master.iterrows():
        roc_ms = _fmt_mean_std(float(r["test_roc_auc_mean"]), float(r["test_roc_auc_std"]))
        pr_ms = _fmt_mean_std(float(r["test_pr_auc_mean"]), float(r["test_pr_auc_std"]))
        if not has_inference:
            lines.append(f"| {r['run_id']} | {r['description']} | {roc_ms} | {pr_ms} |")
            continue

        dA = _fmt_delta(float(r.get("delta_roc_auc_A", math.nan)), r.get("delta_roc_p_value_A"))
        dB = _fmt_delta(float(r.get("delta_roc_auc_B", math.nan)), r.get("delta_roc_p_value_B"))
        dpA = r.get("delta_pr_auc_A", math.nan)
        dpB = r.get("delta_pr_auc_B", math.nan)
        dpA_s = f"{float(dpA):.6f}" if np.isfinite(dpA) else ""
        dpB_s = f"{float(dpB):.6f}" if np.isfinite(dpB) else ""
        lines.append(f"| {r['run_id']} | {r['description']} | {roc_ms} | {pr_ms} | {dA} | {dB} | {dpA_s} | {dpB_s} |")

    out_path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True, help="Sweep dir containing summary.csv (and optionally inference).")
    ap.add_argument("--out-dir", required=True, help="Output directory for master_results.*")
    ap.add_argument("--baseline-run-id", default="A3_trailing")
    ap.add_argument("--sweep-name", default="Paper Grid")
    ap.add_argument("--inference-csv", default="", help="If set, use this file; else sweep_dir/inference_vs_baseline.csv")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    summary_path = sweep_dir / "summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary.csv at {summary_path}")
    summary = pd.read_csv(summary_path)
    _assert_expected_columns(summary, ["run_id", "description", "fold_id"], name="summary.csv")

    inference: pd.DataFrame | None = None
    inf_path = Path(args.inference_csv) if args.inference_csv else (sweep_dir / "inference_vs_baseline.csv")
    if inf_path.exists():
        inference = pd.read_csv(inf_path)

    master = _build_master(summary, inference, baseline_run_id=str(args.baseline_run_id))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "master_results.csv").write_text(master.to_csv(index=False))
    _write_md(master, out_dir / "master_results.md", sweep_name=str(args.sweep_name), baseline_run_id=str(args.baseline_run_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


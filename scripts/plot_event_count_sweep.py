from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_both(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot event-count sweep deltas vs baseline (paired CIs).")
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--baseline-run-id", default="event0")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_base = Path(args.out_base)

    summary = pd.read_csv(sweep_dir / "summary.csv")
    inf = pd.read_csv(sweep_dir / "inference_vs_baseline.csv")

    if "event_count" not in summary.columns:
        raise SystemExit("summary.csv missing event_count")

    base_rows = summary[summary["run_id"] == args.baseline_run_id]
    if len(base_rows) == 0:
        raise SystemExit(f"Baseline run_id not found in summary.csv: {args.baseline_run_id}")

    merge_cols = ["run_id", "event_count"]
    if "fold_id" in summary.columns and "fold_id" in inf.columns:
        merge_cols.append("fold_id")
    merged = inf.merge(summary[merge_cols], on=[c for c in merge_cols if c != "event_count"], how="left")
    merged = merged.sort_values(["fold_id", "event_count"] if "fold_id" in merged.columns else ["event_count"])

    x = merged["event_count"].to_numpy(dtype=float)
    x = np.where(x <= 0, 1.0, x)  # avoid log(0) if present

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.0), sharex=True)

    def plot_delta(ax: plt.Axes, metric: str, title: str) -> None:
        ci_prefix = metric.split("_auc", 1)[0]
        if "fold_id" in merged.columns:
            colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
            for i, fold_id in enumerate(sorted(merged["fold_id"].unique())):
                d = merged[merged["fold_id"] == fold_id].sort_values("event_count")
                ax.errorbar(
                    d["event_count"],
                    d[metric],
                    yerr=[d[metric] - d[f"{ci_prefix}_ci_low"], d[f"{ci_prefix}_ci_high"] - d[metric]],
                    fmt="o",
                    capsize=3,
                    color=colors[i % len(colors)],
                    label=f"Fold {fold_id}",
                )
            ax.legend(loc="best", frameon=False)
        else:
            d = merged.copy()
            ax.errorbar(
                d["event_count"],
                d[metric],
                yerr=[d[metric] - d[f"{ci_prefix}_ci_low"], d[f"{ci_prefix}_ci_high"] - d[metric]],
                fmt="o",
                capsize=3,
                color="#4C72B0",
            )
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
        ax.set_title(title)
        ax.set_ylabel("Δ (candidate − baseline)")
        ax.grid(axis="y", alpha=0.3)

    plot_delta(axes[0], "delta_roc_auc", "Paired Δ Test ROC-AUC vs baseline (DeLong 95% CI)")
    plot_delta(axes[1], "delta_pr_auc", "Paired Δ Test PR-AUC vs baseline (hour-block bootstrap 95% CI)")

    axes[1].set_xlabel("Event-count window size N (last N impressions)")
    axes[1].set_xscale("log")
    axes[1].set_xticks(sorted(merged["event_count"].unique()))
    axes[1].get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())

    _save_both(fig, out_base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

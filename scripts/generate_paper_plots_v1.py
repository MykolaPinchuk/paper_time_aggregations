from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_te_lift_metrics(metrics_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(metrics_csv)
    df["fold"] = df["run_key"].str.extract(r"_fold([AB])$", expand=False)
    df["spec"] = df["run_key"].str.replace(r"_fold[AB]$", "", regex=True)
    df["spec"] = df["spec"].replace(
        {"TE_only": "TE only", "A3_trailing": "A3 trailing", "A3_event50": "A3 + event50"}
    )

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    sns.pointplot(
        data=df,
        x="spec",
        y="test_roc_auc",
        hue="fold",
        markers=["o", "s"],
        linestyles=["-", "-"],
        ax=axes[0],
    )
    axes[0].set_title("Test ROC-AUC (absolute)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].tick_params(axis="x", rotation=15)

    sns.pointplot(
        data=df,
        x="spec",
        y="test_pr_auc",
        hue="fold",
        markers=["o", "s"],
        linestyles=["-", "-"],
        ax=axes[1],
        legend=True,
    )
    axes[1].set_title("Test PR-AUC (absolute)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("PR-AUC")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(title="Fold", loc="best")

    _save(fig, out_path)


def plot_te_lift_deltas(inf_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(inf_csv)
    df["comparison"] = df["candidate"] + " vs " + df["baseline"]

    labels = [
        ("A3_trailing vs TE_only", "A3 trailing − TE only"),
        ("A3_event50 vs TE_only", "A3 + event50 − TE only"),
        ("A3_event50 vs A3_trailing", "A3 + event50 − A3 trailing"),
    ]
    label_map = dict(labels)
    df = df[df["comparison"].isin(label_map.keys())].copy()
    df["comparison_label"] = df["comparison"].map(label_map)

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)

    def _forest(ax: plt.Axes, metric: str, ci_low: str, ci_high: str, title: str) -> None:
        d = df.copy()
        d = d.sort_values(["comparison_label", "fold_id"])

        y_order = list(reversed([lab for _, lab in labels]))
        d["y"] = pd.Categorical(d["comparison_label"], categories=y_order, ordered=True)

        for fold, marker in [("A", "o"), ("B", "s")]:
            sub = d[d["fold_id"] == fold]
            ax.errorbar(
                sub[metric],
                sub["y"],
                xerr=[
                    sub[metric] - sub[ci_low],
                    sub[ci_high] - sub[metric],
                ],
                fmt=marker,
                capsize=3,
                label=f"Fold {fold}",
            )
        ax.axvline(0.0, color="black", linewidth=1, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Δ (candidate − baseline)")
        ax.set_ylabel("")
        ax.legend(loc="best")

    _forest(
        axes[0],
        metric="delta_roc_auc",
        ci_low="delta_roc_ci_low",
        ci_high="delta_roc_ci_high",
        title="Paired ΔROC-AUC (95% CI, DeLong)",
    )
    _forest(
        axes[1],
        metric="delta_pr_auc",
        ci_low="delta_pr_ci_low",
        ci_high="delta_pr_ci_high",
        title="Paired ΔPR-AUC (95% CI, hour-block bootstrap)",
    )

    _save(fig, out_path)


def plot_heatmap_compare(
    te_master_csv: Path, note_master_csv: Path, out_path: Path, value_col: str = "test_roc_auc_mean"
) -> None:
    te = pd.read_csv(te_master_csv)
    note = pd.read_csv(note_master_csv)

    def pivot(df: pd.DataFrame) -> pd.DataFrame:
        p = df.pivot(index="length_set", columns="shape", values=value_col)
        return p.loc[["A1", "A2", "A3", "A4"], ["trailing", "gap1", "bucket", "calendar", "event50"]]

    te_p = pivot(te)
    note_p = pivot(note)

    vmin = min(te_p.min().min(), note_p.min().min())
    vmax = max(te_p.max().max(), note_p.max().max())

    sns.set_theme(style="white", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8))

    sns.heatmap(te_p, ax=axes[0], cmap="viridis", vmin=vmin, vmax=vmax, annot=True, fmt=".4f")
    axes[0].set_title("TE + time-agg")
    axes[0].set_xlabel("Shape (A.1)")
    axes[0].set_ylabel("Length set (A.2)")

    sns.heatmap(note_p, ax=axes[1], cmap="viridis", vmin=vmin, vmax=vmax, annot=True, fmt=".4f")
    axes[1].set_title("No TE (time-agg only)")
    axes[1].set_xlabel("Shape (A.1)")
    axes[1].set_ylabel("")

    _save(fig, out_path)


def plot_shape_contrasts_compare(te_contrasts_csv: Path, note_contrasts_csv: Path, out_path: Path) -> None:
    te = pd.read_csv(te_contrasts_csv)
    note = pd.read_csv(note_contrasts_csv)

    def summarize(df: pd.DataFrame, setting: str) -> pd.DataFrame:
        out = (
            df.groupby(["length_set", "shape"], as_index=False)["delta_roc_auc"]
            .mean()
            .rename(columns={"delta_roc_auc": "mean_delta_roc_auc"})
        )
        out["setting"] = setting
        return out

    df = pd.concat([summarize(te, "TE"), summarize(note, "no-TE")], ignore_index=True)

    shape_order = ["event50", "calendar", "bucket", "gap1"]
    df["shape"] = pd.Categorical(df["shape"], categories=shape_order, ordered=True)
    df["length_set"] = pd.Categorical(df["length_set"], categories=["A1", "A2", "A3", "A4"], ordered=True)

    sns.set_theme(style="whitegrid", context="talk")
    g = sns.catplot(
        data=df,
        kind="bar",
        x="shape",
        y="mean_delta_roc_auc",
        hue="setting",
        col="length_set",
        col_order=["A1", "A2", "A3", "A4"],
        height=3.8,
        aspect=0.9,
        sharey=True,
        errorbar=None,
    )
    g.set_titles("Length set {col_name}")
    g.set_axis_labels("Shape (vs trailing)", "Mean ΔROC-AUC (over folds)")
    for ax in g.axes.flatten():
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
        ax.tick_params(axis="x", rotation=20)

    _save(g.figure, out_path)


def plot_te_vs_note_uplift(summary_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(summary_csv)
    df = df.sort_values("delta_test_roc_auc_te_minus_note_mean", ascending=True).copy()

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.5), sharey=True)

    sns.barplot(
        data=df,
        x="delta_test_roc_auc_te_minus_note_mean",
        y="run_id",
        color="#4C72B0",
        ax=axes[0],
    )
    axes[0].set_title("TE uplift vs no-TE (ΔTest ROC-AUC)")
    axes[0].set_xlabel("Δ (TE − no-TE)")
    axes[0].set_ylabel("")

    sns.barplot(
        data=df,
        x="delta_test_pr_auc_te_minus_note_mean",
        y="run_id",
        color="#55A868",
        ax=axes[1],
    )
    axes[1].set_title("TE uplift vs no-TE (ΔTest PR-AUC)")
    axes[1].set_xlabel("Δ (TE − no-TE)")
    axes[1].set_ylabel("")

    _save(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a small set of paper-ready plots for v1.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/v1"),
        help="Root directory containing postprocessed tables (e.g., artifacts/v1).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for figures.",
    )
    args = parser.parse_args()

    artifacts_dir: Path = args.artifacts_dir
    out_dir: Path = args.out_dir or (artifacts_dir / "plots")

    plot_te_lift_metrics(
        metrics_csv=artifacts_dir / "10pct_te_lift" / "metrics_selected_runs.csv",
        out_path=out_dir / "te_lift_metrics.png",
    )
    plot_te_lift_deltas(
        inf_csv=artifacts_dir / "10pct_te_lift" / "inference_test_te_vs_timeagg.csv",
        out_path=out_dir / "te_lift_deltas.png",
    )
    plot_heatmap_compare(
        te_master_csv=artifacts_dir / "10pct_paper_grid" / "master_results.csv",
        note_master_csv=artifacts_dir / "10pct_paper_grid_noTE" / "master_results.csv",
        out_path=out_dir / "heatmap_test_roc_te_vs_note.png",
        value_col="test_roc_auc_mean",
    )
    plot_shape_contrasts_compare(
        te_contrasts_csv=artifacts_dir / "10pct_paper_grid" / "contrasts_vs_trailing_test.csv",
        note_contrasts_csv=artifacts_dir / "10pct_paper_grid_noTE" / "contrasts_vs_trailing_test.csv",
        out_path=out_dir / "shape_contrasts_te_vs_note.png",
    )
    plot_te_vs_note_uplift(
        summary_csv=artifacts_dir / "combined" / "te_vs_note_summary.csv",
        out_path=out_dir / "te_vs_note_uplift.png",
    )

    print(f"Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()

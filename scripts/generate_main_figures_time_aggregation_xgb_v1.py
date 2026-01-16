from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LENGTH_TUPLES: dict[str, tuple[int, ...]] = {
    "A1": (1, 6, 24),
    "A2": (1, 3, 6, 12, 24),
    "A3": (1, 6, 24, 48, 168),
    "A4": (1, 2, 4, 8, 16, 24, 48, 96, 168),
}

SHAPE_ORDER = ["event50", "calendar", "bucket", "gap1"]
SHAPE_ORDER_WITH_TRAILING = ["trailing", "event50", "calendar", "bucket", "gap1"]


def length_tuple_label(length_set: str) -> str:
    tup = LENGTH_TUPLES.get(length_set)
    if tup is None:
        return str(length_set)
    return "(" + ", ".join(str(x) for x in tup) + ")"


def _save_both(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _errorbar(ax: plt.Axes, x: float, y: float, lo: float, hi: float, marker: str, label: str | None) -> None:
    ax.errorbar(
        [x],
        [y],
        xerr=[[x - lo], [hi - x]],
        fmt=marker,
        capsize=3,
        markersize=6,
        linewidth=1.5,
        label=label,
    )


@dataclass(frozen=True)
class FoldDelta:
    delta: float
    ci_low: float
    ci_high: float


def fig1_decision_lengths_then_shapes(te_master_csv: Path, te_contrasts_csv: Path, out_base: Path) -> None:
    te_master = pd.read_csv(te_master_csv)
    te_contrasts = pd.read_csv(te_contrasts_csv)

    # Panel A: length set comparison (trailing only), vs reference (1,6,24,48,168) trailing.
    trailing = te_master[te_master["shape"] == "trailing"].copy()
    trailing["length_tuple"] = trailing["length_set"].map(length_tuple_label)
    trailing = trailing[trailing["length_set"].isin(["A1", "A2", "A3", "A4"])].copy()
    trailing["is_ref"] = trailing["length_set"] == "A3"

    # For reference row, deltas are missing in the CSV (baseline). Fill 0.
    for fold in ["A", "B"]:
        for col in [f"delta_roc_auc_{fold}", f"delta_roc_ci_low_{fold}", f"delta_roc_ci_high_{fold}"]:
            if col not in trailing.columns:
                trailing[col] = np.nan
    trailing.loc[trailing["is_ref"], ["delta_roc_auc_A", "delta_roc_auc_B"]] = 0.0
    trailing.loc[trailing["is_ref"], ["delta_roc_ci_low_A", "delta_roc_ci_low_B"]] = 0.0
    trailing.loc[trailing["is_ref"], ["delta_roc_ci_high_A", "delta_roc_ci_high_B"]] = 0.0

    # Panel B: shape contrasts within each length tuple, vs trailing of that length tuple.
    c = te_contrasts[te_contrasts["split"] == "test"].copy()
    c = c[c["shape"].isin(SHAPE_ORDER)].copy()
    c["length_tuple"] = c["length_set"].map(length_tuple_label)
    c["shape"] = pd.Categorical(c["shape"], categories=SHAPE_ORDER, ordered=True)
    c["length_set"] = pd.Categorical(c["length_set"], categories=["A1", "A2", "A3", "A4"], ordered=True)

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(nrows=2, ncols=4, height_ratios=[1.0, 1.45], hspace=0.35, wspace=0.25)

    axA = fig.add_subplot(gs[0, :])
    axA.axvline(0.0, color="black", linewidth=1, alpha=0.7)
    axA.set_title(
        "Panel A: Choose window lengths first (trailing only)\n"
        "Test ROC-AUC delta vs reference trailing (1, 6, 24, 48, 168)"
    )
    axA.set_xlabel("Δ Test ROC-AUC (candidate trailing − reference trailing)")
    axA.set_ylabel("Window length tuple")

    y_labels = [length_tuple_label(k) for k in ["A1", "A2", "A3", "A4"]]
    y_pos = {lab: i for i, lab in enumerate(y_labels)}

    for _, row in trailing.sort_values("length_set").iterrows():
        y = y_pos[row["length_tuple"]]
        for fold, marker in [("A", "o"), ("B", "s")]:
            _errorbar(
                axA,
                x=float(row[f"delta_roc_auc_{fold}"]),
                y=y + (-0.12 if fold == "A" else 0.12),
                lo=float(row[f"delta_roc_ci_low_{fold}"]),
                hi=float(row[f"delta_roc_ci_high_{fold}"]),
                marker=marker,
                label=f"Fold {fold}" if row["length_set"] == "A1" else None,
            )

    axA.set_yticks(list(y_pos.values()), y_labels)
    axA.legend(loc="best")

    # Panel B: one subplot per length tuple
    x_min = float(c["delta_roc_ci_low"].min())
    x_max = float(c["delta_roc_ci_high"].max())
    pad = 0.0003
    x_lim = (x_min - pad, x_max + pad)

    for j, length_set in enumerate(["A1", "A2", "A3", "A4"]):
        ax = fig.add_subplot(gs[1, j])
        ax.axvline(0.0, color="black", linewidth=1, alpha=0.7)
        ax.set_title(f"Panel B{j+1}: Shape deltas within {length_tuple_label(length_set)}")
        ax.set_xlabel("Δ Test ROC-AUC (shape − trailing)")
        ax.set_xlim(*x_lim)
        if j == 0:
            ax.set_ylabel("Shape (vs trailing)")
        else:
            ax.set_ylabel("")

        sub = c[c["length_set"] == length_set].copy()
        sub = sub.sort_values(["shape", "fold_id"])

        y_map = {shape: i for i, shape in enumerate(SHAPE_ORDER)}
        for _, r in sub.iterrows():
            y = y_map[str(r["shape"])]
            marker = "o" if r["fold_id"] == "A" else "s"
            _errorbar(
                ax,
                x=float(r["delta_roc_auc"]),
                y=y + (-0.12 if r["fold_id"] == "A" else 0.12),
                lo=float(r["delta_roc_ci_low"]),
                hi=float(r["delta_roc_ci_high"]),
                marker=marker,
                label=None,
            )
        ax.set_yticks(list(y_map.values()), SHAPE_ORDER)

    _save_both(fig, out_base)


def fig2_league_table(te_master_csv: Path, out_base: Path) -> None:
    df = pd.read_csv(te_master_csv)
    df["length_tuple"] = df["length_set"].map(length_tuple_label)

    # Fill baseline (reference trailing) deltas to 0 for plotting.
    is_ref = (df["run_id"] == "A3_trailing") | ((df["shape"] == "trailing") & (df["length_set"] == "A3"))
    for fold in ["A", "B"]:
        for col in [f"delta_roc_auc_{fold}", f"delta_roc_ci_low_{fold}", f"delta_roc_ci_high_{fold}"]:
            if col not in df.columns:
                df[col] = np.nan
    df.loc[is_ref, ["delta_roc_auc_A", "delta_roc_auc_B"]] = 0.0
    df.loc[is_ref, ["delta_roc_ci_low_A", "delta_roc_ci_low_B"]] = 0.0
    df.loc[is_ref, ["delta_roc_ci_high_A", "delta_roc_ci_high_B"]] = 0.0

    # Rank by mean test ROC-AUC across folds and select top/bottom.
    df = df.sort_values("test_roc_auc_mean", ascending=False).copy()
    top = df.head(6)
    bottom = df.tail(6)
    pick = pd.concat([top, bottom], ignore_index=True)

    pick["label"] = pick.apply(lambda r: f'{r["shape"]} {r["length_tuple"]}', axis=1)
    pick = pick.sort_values("test_roc_auc_mean", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    ax.axvline(0.0, color="black", linewidth=1, alpha=0.7)
    ax.set_title(
        "League table (top 6 + bottom 6): Test ROC-AUC deltas vs reference\n"
        "Reference is trailing (1, 6, 24, 48, 168); deltas are paired, fold-by-fold"
    )
    ax.set_xlabel("Δ Test ROC-AUC (candidate − reference)")
    ax.set_ylabel("")

    y = np.arange(len(pick))
    ax.set_yticks(y, pick["label"].tolist())
    ax.invert_yaxis()

    for i, row in pick.iterrows():
        for fold, marker, offset in [("A", "o", -0.12), ("B", "s", 0.12)]:
            _errorbar(
                ax,
                x=float(row[f"delta_roc_auc_{fold}"]),
                y=float(i) + offset,
                lo=float(row[f"delta_roc_ci_low_{fold}"]),
                hi=float(row[f"delta_roc_ci_high_{fold}"]),
                marker=marker,
                label=f"Fold {fold}" if i == 0 else None,
            )

    ax.legend(loc="best")
    _save_both(fig, out_base)


def _traffic_light_matrix(contrasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = contrasts[contrasts["split"] == "test"].copy()
    c = c[c["shape"].isin(SHAPE_ORDER)].copy()
    c["length_tuple"] = c["length_set"].map(length_tuple_label)

    def summarize(group: pd.DataFrame) -> tuple[int, float]:
        # Better if both folds CI_low > 0. Worse if both folds CI_high < 0. Else unclear.
        folds = {}
        for _, r in group.iterrows():
            folds[str(r["fold_id"])] = (float(r["delta_roc_ci_low"]), float(r["delta_roc_ci_high"]), float(r["delta_roc_auc"]))
        if "A" not in folds or "B" not in folds:
            return 0, float(group["delta_roc_auc"].mean())
        (loA, hiA, dA) = folds["A"]
        (loB, hiB, dB) = folds["B"]
        if loA > 0 and loB > 0:
            return 1, (dA + dB) / 2.0
        if hiA < 0 and hiB < 0:
            return -1, (dA + dB) / 2.0
        return 0, (dA + dB) / 2.0

    rows = []
    for (length_set, shape), group in c.groupby(["length_set", "shape"], sort=False):
        code, mean_delta = summarize(group)
        rows.append(
            {
                "length_set": length_set,
                "shape": shape,
                "code": code,
                "mean_delta": mean_delta,
                "length_tuple": length_tuple_label(length_set),
            }
        )
    out = pd.DataFrame(rows)
    out["shape"] = pd.Categorical(out["shape"], categories=SHAPE_ORDER, ordered=True)
    out["length_set"] = pd.Categorical(out["length_set"], categories=["A1", "A2", "A3", "A4"], ordered=True)

    mat = out.pivot(index="length_tuple", columns="shape", values="code").loc[
        [length_tuple_label(k) for k in ["A1", "A2", "A3", "A4"]],
        SHAPE_ORDER,
    ]
    mat_delta = out.pivot(index="length_tuple", columns="shape", values="mean_delta").loc[
        [length_tuple_label(k) for k in ["A1", "A2", "A3", "A4"]],
        SHAPE_ORDER,
    ]
    return mat, mat_delta


def fig3_traffic_light(te_contrasts_csv: Path, note_contrasts_csv: Path, out_base: Path) -> None:
    te = pd.read_csv(te_contrasts_csv)
    note = pd.read_csv(note_contrasts_csv)

    mat_te, delta_te = _traffic_light_matrix(te)
    mat_note, delta_note = _traffic_light_matrix(note)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)

    def draw(ax: plt.Axes, mat: pd.DataFrame, delta: pd.DataFrame, title: str) -> None:
        # Map -1,0,1 to colors.
        cmap = plt.matplotlib.colors.ListedColormap(["#d73027", "#dddddd", "#1a9850"])
        norm = plt.matplotlib.colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
        im = ax.imshow(mat.to_numpy(dtype=float), cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(mat.columns)), mat.columns.tolist())
        ax.set_yticks(np.arange(len(mat.index)), mat.index.tolist())
        ax.set_xlabel("Shape (compared to trailing)")
        ax.set_ylabel("Window length tuple")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                d = float(delta.iloc[i, j])
                ax.text(j, i, f"{d:+.4f}", ha="center", va="center", fontsize=10)
        return im

    draw(
        axes[0],
        mat_te,
        delta_te,
        "Traffic light: shape effect vs trailing (TE + time-agg)\n"
        "Green=better, Red=worse, Gray=unclear (paired DeLong CI, both folds)",
    )
    draw(
        axes[1],
        mat_note,
        delta_note,
        "Traffic light: shape effect vs trailing (no TE)\n"
        "Green=better, Red=worse, Gray=unclear (paired DeLong CI, both folds)",
    )

    _save_both(fig, out_base)


def fig4_protocol_schematic(out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.5, 4.5))
    ax.axis("off")

    # Timeline axis.
    ax.annotate("", xy=(0.95, 0.75), xytext=(0.05, 0.75), arrowprops={"arrowstyle": "->", "linewidth": 2})
    ax.text(0.5, 0.83, "Time (increases to the right)", ha="center", va="center", fontsize=14)

    def box(x0: float, x1: float, y: float, label: str, color: str) -> None:
        ax.add_patch(plt.Rectangle((x0, y), x1 - x0, 0.12, facecolor=color, edgecolor="black", linewidth=1))
        ax.text((x0 + x1) / 2, y + 0.06, label, ha="center", va="center", fontsize=12)

    # Fold A: train < val < test
    ax.text(0.03, 0.60, "Fold A", ha="left", va="center", fontsize=12)
    box(0.08, 0.62, 0.54, "train", "#c6dbef")
    box(0.62, 0.78, 0.54, "val", "#9ecae1")
    box(0.78, 0.92, 0.54, "test", "#6baed6")

    # Fold B: shifted earlier
    ax.text(0.03, 0.38, "Fold B", ha="left", va="center", fontsize=12)
    box(0.08, 0.50, 0.32, "train", "#c6dbef")
    box(0.50, 0.66, 0.32, "val", "#9ecae1")
    box(0.66, 0.80, 0.32, "test", "#6baed6")

    # Online-safe feature rule callout.
    ax.add_patch(plt.Rectangle((0.08, 0.08), 0.84, 0.16, facecolor="#f7f7f7", edgecolor="black", linewidth=1))
    ax.text(
        0.50,
        0.16,
        "Online-safe features: for an impression at hour H, all time-aggregation features use history from hours < H.\n"
        "No same-hour leakage (updates are applied at hour boundaries).",
        ha="center",
        va="center",
        fontsize=11,
    )

    _save_both(fig, out_base)


def _long_from_master(master: pd.DataFrame, setting: str) -> pd.DataFrame:
    df = master.copy()
    df = df[df["length_set"].isin(["A1", "A2", "A3", "A4"])].copy()
    df["length_tuple"] = df["length_set"].map(length_tuple_label)
    df["setting"] = setting

    rows = []
    for _, r in df.iterrows():
        for fold in ["A", "B"]:
            rows.append(
                {
                    "setting": setting,
                    "shape": r["shape"],
                    "length_set": r["length_set"],
                    "length_tuple": r["length_tuple"],
                    "fold": fold,
                    "test_roc_auc": float(r[f"test_roc_auc_{fold}"]),
                    "test_pr_auc": float(r[f"test_pr_auc_{fold}"]),
                    "val_roc_auc": float(r[f"val_roc_auc_{fold}"]),
                    "val_pr_auc": float(r[f"val_pr_auc_{fold}"]),
                }
            )
    out = pd.DataFrame(rows)
    out["shape"] = pd.Categorical(out["shape"], categories=SHAPE_ORDER_WITH_TRAILING, ordered=True)
    out["length_tuple"] = pd.Categorical(
        out["length_tuple"],
        categories=[length_tuple_label(k) for k in ["A1", "A2", "A3", "A4"]],
        ordered=True,
    )
    return out


def fig5_barplots_avg_by_shape(te_master_csv: Path, note_master_csv: Path, out_base: Path) -> None:
    te = _long_from_master(pd.read_csv(te_master_csv), "TE")
    note = _long_from_master(pd.read_csv(note_master_csv), "no-TE")
    df = pd.concat([te, note], ignore_index=True)

    metrics = [
        ("test_roc_auc", "Test ROC-AUC"),
        ("test_pr_auc", "Test PR-AUC"),
        ("val_roc_auc", "Val ROC-AUC"),
        ("val_pr_auc", "Val PR-AUC"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    axes = axes.flatten()
    fig.suptitle("Average performance by shape (averaged over length tuples and folds)", fontsize=16)

    shapes = SHAPE_ORDER_WITH_TRAILING
    x = np.arange(len(shapes))
    width = 0.38

    for ax, (metric, title) in zip(axes, metrics, strict=True):
        summary = (
            df.groupby(["setting", "shape"], as_index=False)[metric]
            .agg(mean="mean", std="std", n="count")
            .copy()
        )
        summary["sem"] = summary["std"] / np.sqrt(summary["n"].clip(lower=1))

        te_s = summary[summary["setting"] == "TE"].set_index("shape").reindex(shapes)
        nt_s = summary[summary["setting"] == "no-TE"].set_index("shape").reindex(shapes)

        ax.bar(x - width / 2, te_s["mean"].to_numpy(), width, yerr=te_s["sem"].to_numpy(), label="TE", capsize=3)
        ax.bar(
            x + width / 2,
            nt_s["mean"].to_numpy(),
            width,
            yerr=nt_s["sem"].to_numpy(),
            label="no-TE",
            capsize=3,
        )
        ax.set_title(title)
        ax.set_ylabel("AUC")
        ax.grid(axis="y", alpha=0.3)

    for ax in axes[2:]:
        ax.set_xticks(x, shapes, rotation=0)
        ax.set_xlabel("Shape")

    axes[0].legend(loc="best", title="Setting")
    fig.text(0.5, 0.02, "Error bars: ±1 SEM over (length tuple × fold) points", ha="center", fontsize=11)
    _save_both(fig, out_base)


def fig6_barplots_avg_by_length_tuple(te_master_csv: Path, note_master_csv: Path, out_base: Path) -> None:
    te = _long_from_master(pd.read_csv(te_master_csv), "TE")
    note = _long_from_master(pd.read_csv(note_master_csv), "no-TE")
    df = pd.concat([te, note], ignore_index=True)

    metrics = [
        ("test_roc_auc", "Test ROC-AUC"),
        ("test_pr_auc", "Test PR-AUC"),
        ("val_roc_auc", "Val ROC-AUC"),
        ("val_pr_auc", "Val PR-AUC"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    axes = axes.flatten()
    fig.suptitle("Average performance by window-length tuple (averaged over shapes and folds)", fontsize=16)

    length_tuples = [length_tuple_label(k) for k in ["A1", "A2", "A3", "A4"]]
    x = np.arange(len(length_tuples))
    width = 0.38

    for ax, (metric, title) in zip(axes, metrics, strict=True):
        summary = (
            df.groupby(["setting", "length_tuple"], as_index=False)[metric]
            .agg(mean="mean", std="std", n="count")
            .copy()
        )
        summary["sem"] = summary["std"] / np.sqrt(summary["n"].clip(lower=1))

        te_s = summary[summary["setting"] == "TE"].set_index("length_tuple").reindex(length_tuples)
        nt_s = summary[summary["setting"] == "no-TE"].set_index("length_tuple").reindex(length_tuples)

        ax.bar(x - width / 2, te_s["mean"].to_numpy(), width, yerr=te_s["sem"].to_numpy(), label="TE", capsize=3)
        ax.bar(
            x + width / 2,
            nt_s["mean"].to_numpy(),
            width,
            yerr=nt_s["sem"].to_numpy(),
            label="no-TE",
            capsize=3,
        )
        ax.set_title(title)
        ax.set_ylabel("AUC")
        ax.grid(axis="y", alpha=0.3)

    for ax in axes[2:]:
        ax.set_xticks(x, length_tuples, rotation=15, ha="right")
        ax.set_xlabel("Window length tuple (hours)")

    axes[0].legend(loc="best", title="Setting")
    fig.text(0.5, 0.02, "Error bars: ±1 SEM over (shape × fold) points", ha="center", fontsize=11)
    _save_both(fig, out_base)


def fig7_trimmed_note_test_pr_by_shape(note_master_csv: Path, out_base: Path, y_min: float = 0.30) -> None:
    note = _long_from_master(pd.read_csv(note_master_csv), "no-TE")
    df = note.copy()

    shapes = SHAPE_ORDER_WITH_TRAILING
    x = np.arange(len(shapes))

    summary = df.groupby(["shape"], as_index=False)["test_pr_auc"].mean().rename(columns={"test_pr_auc": "mean"})
    s = summary.set_index("shape").reindex(shapes)

    y_max = float(s["mean"].max())
    y_max = max(y_min + 0.01, y_max + 0.002)

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.bar(x, s["mean"].to_numpy(), color="#55A868")
    ax.set_title("no-TE: Test PR-AUC by shape (avg over length tuples and folds)")
    ax.set_xlabel("Shape")
    ax.set_ylabel("Test PR-AUC")
    ax.set_xticks(x, shapes)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", alpha=0.3)
    _save_both(fig, out_base)


def fig8_trimmed_note_test_pr_by_length_tuple(note_master_csv: Path, out_base: Path, y_min: float = 0.30) -> None:
    note = _long_from_master(pd.read_csv(note_master_csv), "no-TE")
    df = note.copy()

    length_tuples = [length_tuple_label(k) for k in ["A1", "A2", "A3", "A4"]]
    x = np.arange(len(length_tuples))

    summary = (
        df.groupby(["length_tuple"], as_index=False)["test_pr_auc"]
        .mean()
        .rename(columns={"test_pr_auc": "mean"})
    )
    s = summary.set_index("length_tuple").reindex(length_tuples)

    y_max = float(s["mean"].max())
    y_max = max(y_min + 0.01, y_max + 0.002)

    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    ax.bar(x, s["mean"].to_numpy(), color="#4C72B0")
    ax.set_title("no-TE: Test PR-AUC by window-length tuple (avg over shapes and folds)")
    ax.set_xlabel("Window length tuple (hours)")
    ax.set_ylabel("Test PR-AUC")
    ax.set_xticks(x, length_tuples, rotation=15, ha="right")
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", alpha=0.3)
    _save_both(fig, out_base)


def fig9_trimmed_note_test_roc_by_shape(note_master_csv: Path, out_base: Path, y_min: float = 0.73) -> None:
    note = _long_from_master(pd.read_csv(note_master_csv), "no-TE")
    df = note.copy()

    shapes = SHAPE_ORDER_WITH_TRAILING
    x = np.arange(len(shapes))

    summary = df.groupby(["shape"], as_index=False)["test_roc_auc"].mean().rename(columns={"test_roc_auc": "mean"})
    s = summary.set_index("shape").reindex(shapes)

    y_max = float(s["mean"].max())
    y_max = max(y_min + 0.01, y_max + 0.0015)

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.bar(x, s["mean"].to_numpy(), color="#55A868")
    ax.set_title("no-TE: Test ROC-AUC by shape (avg over length tuples and folds)")
    ax.set_xlabel("Shape")
    ax.set_ylabel("Test ROC-AUC")
    ax.set_xticks(x, shapes)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", alpha=0.3)
    _save_both(fig, out_base)


def fig10_trimmed_note_test_roc_by_length_tuple(
    note_master_csv: Path, out_base: Path, y_min: float = 0.73
) -> None:
    note = _long_from_master(pd.read_csv(note_master_csv), "no-TE")
    df = note.copy()

    length_tuples = [length_tuple_label(k) for k in ["A1", "A2", "A3", "A4"]]
    x = np.arange(len(length_tuples))

    summary = (
        df.groupby(["length_tuple"], as_index=False)["test_roc_auc"]
        .mean()
        .rename(columns={"test_roc_auc": "mean"})
    )
    s = summary.set_index("length_tuple").reindex(length_tuples)

    y_max = float(s["mean"].max())
    y_max = max(y_min + 0.01, y_max + 0.0015)

    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    ax.bar(x, s["mean"].to_numpy(), color="#4C72B0")
    ax.set_title("no-TE: Test ROC-AUC by window-length tuple (avg over shapes and folds)")
    ax.set_xlabel("Window length tuple (hours)")
    ax.set_ylabel("Test ROC-AUC")
    ax.set_xticks(x, length_tuples, rotation=15, ha="right")
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", alpha=0.3)
    _save_both(fig, out_base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate main figures for time_aggregation_xgb_v1 (PNG+PDF).")
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
    out_dir: Path = args.out_dir or (artifacts_dir / "figures_main")

    te_master_csv = artifacts_dir / "10pct_paper_grid" / "master_results.csv"
    te_contrasts_csv = artifacts_dir / "10pct_paper_grid" / "contrasts_vs_trailing_test.csv"
    note_master_csv = artifacts_dir / "10pct_paper_grid_noTE" / "master_results.csv"
    note_contrasts_csv = artifacts_dir / "10pct_paper_grid_noTE" / "contrasts_vs_trailing_test.csv"

    fig1_decision_lengths_then_shapes(
        te_master_csv=te_master_csv,
        te_contrasts_csv=te_contrasts_csv,
        out_base=out_dir / "fig_decision_lengths_then_shapes",
    )
    fig2_league_table(
        te_master_csv=te_master_csv,
        out_base=out_dir / "fig_league_table_top_bottom",
    )
    fig3_traffic_light(
        te_contrasts_csv=te_contrasts_csv,
        note_contrasts_csv=note_contrasts_csv,
        out_base=out_dir / "fig_traffic_light_shape_contrasts",
    )
    fig4_protocol_schematic(
        out_base=out_dir / "fig_protocol_schematic",
    )
    fig5_barplots_avg_by_shape(
        te_master_csv=te_master_csv,
        note_master_csv=note_master_csv,
        out_base=out_dir / "fig_avg_performance_by_shape_panels",
    )
    fig6_barplots_avg_by_length_tuple(
        te_master_csv=te_master_csv,
        note_master_csv=note_master_csv,
        out_base=out_dir / "fig_avg_performance_by_length_tuple_panels",
    )
    fig7_trimmed_note_test_pr_by_shape(
        note_master_csv=note_master_csv,
        out_base=out_dir / "fig_trimmed_note_test_prauc_by_shape",
        y_min=0.34,
    )
    fig8_trimmed_note_test_pr_by_length_tuple(
        note_master_csv=note_master_csv,
        out_base=out_dir / "fig_trimmed_note_test_prauc_by_length_tuple",
        y_min=0.34,
    )
    fig9_trimmed_note_test_roc_by_shape(
        note_master_csv=note_master_csv,
        out_base=out_dir / "fig_trimmed_note_test_rocauc_by_shape",
        y_min=0.73,
    )
    fig10_trimmed_note_test_roc_by_length_tuple(
        note_master_csv=note_master_csv,
        out_base=out_dir / "fig_trimmed_note_test_rocauc_by_length_tuple",
        y_min=0.73,
    )

    print(f"Wrote figures to: {out_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle_clicks.time_utils import add_time_columns, make_oot_splits_by_day


@dataclass(frozen=True)
class EDAPaths:
    sample_parquet: Path
    out_md: Path
    assets_dir: Path


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines: list[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _cat_cols(df: pd.DataFrame, label_col: str = "click") -> list[str]:
    drop_cols = {"id", "hour", "hour_dt", "day", "hour_of_day", label_col}
    return [c for c in df.columns if c not in drop_cols]


def _describe_time(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "min_hour": str(df["hour_dt"].min()),
        "max_hour": str(df["hour_dt"].max()),
        "n_days": int(df["day"].nunique()),
        "n_hours": int(df["hour_dt"].nunique()),
    }


def _plot_ctr_by_hour_of_day(df: pd.DataFrame, out: Path) -> None:
    g = df.groupby("hour_of_day")["click"].mean()
    plt.figure(figsize=(8, 4))
    sns.lineplot(x=g.index, y=g.values)
    plt.title("CTR by hour-of-day (sample)")
    plt.xlabel("hour_of_day")
    plt.ylabel("click rate")
    _savefig(out)


def _plot_daily_volume_and_ctr(df: pd.DataFrame, out_assets: Path) -> tuple[Path, Path]:
    daily = df.groupby("day")["click"].agg(imps="count", clicks="sum")
    daily["ctr"] = daily["clicks"] / daily["imps"]

    ctr_path = out_assets / "ctr_by_day.png"
    plt.figure(figsize=(10, 4))
    sns.lineplot(x=daily.index, y=daily["ctr"].values)
    plt.title("CTR by day (sample)")
    plt.xlabel("day")
    plt.ylabel("click rate")
    plt.xticks(rotation=45, ha="right")
    _savefig(ctr_path)

    vol_path = out_assets / "imps_by_day.png"
    plt.figure(figsize=(10, 4))
    sns.barplot(x=daily.index.astype(str), y=daily["imps"].values)
    plt.title("Impressions by day (sample)")
    plt.xlabel("day")
    plt.ylabel("impressions")
    plt.xticks(rotation=45, ha="right")
    _savefig(vol_path)

    return ctr_path, vol_path


def _compute_cardinality(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = len(df)
    for c in cols:
        vc = df[c].value_counts(dropna=False)
        top1 = int(vc.iloc[0]) if len(vc) else 0
        top10 = int(vc.iloc[:10].sum()) if len(vc) else 0
        rows.append(
            {
                "col": c,
                "nunique": int(df[c].nunique(dropna=False)),
                "top1_share": float(top1 / n) if n else 0.0,
                "top10_share": float(top10 / n) if n else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("nunique", ascending=False, kind="stable")


def _plot_cardinality(card: pd.DataFrame, out: Path, top_n: int = 20) -> None:
    top = card.head(top_n).sort_values("nunique", ascending=True, kind="stable")
    plt.figure(figsize=(9, 6))
    sns.barplot(x=top["nunique"].values, y=top["col"].values, orient="h")
    plt.title(f"Cardinality (nunique) — top {min(top_n, len(top))}")
    plt.xlabel("nunique")
    plt.ylabel("")
    _savefig(out)


def _plot_frequency_rank(df: pd.DataFrame, col: str, out: Path, max_points: int = 200_000) -> None:
    vc = df[col].value_counts()
    counts = vc.values.astype(np.float64)
    if len(counts) > max_points:
        counts = counts[:max_points]
    ranks = np.arange(1, len(counts) + 1, dtype=np.float64)
    plt.figure(figsize=(7, 5))
    plt.loglog(ranks, counts, marker=".", linestyle="none", markersize=2)
    plt.title(f"Frequency rank plot: `{col}` (sample)")
    plt.xlabel("rank (log)")
    plt.ylabel("count (log)")
    _savefig(out)


def _unseen_rate_by_split(df: pd.DataFrame, splits: dict[str, pd.Index], cols: list[str]) -> pd.DataFrame:
    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    rows: list[dict[str, Any]] = []
    for c in cols:
        seen = set(df.loc[train_idx, c].astype(str).unique())
        for split_name, idx in (("val", val_idx), ("test", test_idx)):
            x = df.loc[idx, c].astype(str)
            unseen = (~x.isin(seen)).mean()
            rows.append({"col": c, "split": split_name, "unseen_rate": float(unseen)})
    return pd.DataFrame(rows)


def _plot_unseen_rates(unseen: pd.DataFrame, out: Path, top_n: int = 20) -> None:
    pivot = unseen.pivot(index="col", columns="split", values="unseen_rate").fillna(0.0)
    pivot["max_rate"] = pivot.max(axis=1)
    top = pivot.sort_values("max_rate", ascending=False).head(top_n).sort_values("max_rate", ascending=True)

    plt.figure(figsize=(9, 6))
    y = np.arange(len(top))
    plt.barh(y - 0.2, top.get("val", pd.Series(0.0, index=top.index)).values, height=0.4, label="val")
    plt.barh(y + 0.2, top.get("test", pd.Series(0.0, index=top.index)).values, height=0.4, label="test")
    plt.yticks(y, top.index)
    plt.xlabel("unseen rate vs train")
    plt.title(f"Cold-start rate (val/test) — top {min(top_n, len(top))} columns")
    plt.legend()
    _savefig(out)


def _topk_ctr_table(df: pd.DataFrame, col: str, k: int = 15) -> pd.DataFrame:
    vc = df[col].value_counts()
    top_vals = vc.index[:k]
    g = df[df[col].isin(top_vals)].groupby(col)["click"].agg(imps="count", ctr="mean").reset_index()
    return g.sort_values("imps", ascending=False, kind="stable")


def generate_report(paths: EDAPaths, max_topk: int = 15) -> None:
    df = pd.read_parquet(paths.sample_parquet)
    df = add_time_columns(df, hour_col="hour").sort_values("hour_dt").reset_index(drop=True)

    splits = make_oot_splits_by_day(df)
    cat_cols = _cat_cols(df)
    time_desc = _describe_time(df)

    overall_ctr = float(df["click"].mean())
    total_rows = int(len(df))

    # Figures
    ctr_hod_path = paths.assets_dir / "ctr_by_hour_of_day.png"
    _plot_ctr_by_hour_of_day(df, ctr_hod_path)
    ctr_day_path, vol_day_path = _plot_daily_volume_and_ctr(df, paths.assets_dir)

    card = _compute_cardinality(df, cat_cols)
    card_path = paths.assets_dir / "cardinality_top.png"
    _plot_cardinality(card, card_path)

    freq_paths: list[Path] = []
    for key_col in ["device_ip", "device_id", "app_id", "site_id"]:
        if key_col in df.columns:
            p = paths.assets_dir / f"freq_rank_{key_col}.png"
            _plot_frequency_rank(df, key_col, p)
            freq_paths.append(p)

    unseen = _unseen_rate_by_split(df, splits, cat_cols)
    unseen_path = paths.assets_dir / "unseen_rate_top.png"
    _plot_unseen_rates(unseen, unseen_path)

    # Tables
    split_rows: list[list[Any]] = []
    for name in ("train", "val", "test"):
        idx = splits[name]
        s = df.loc[idx, "hour_dt"]
        split_rows.append([name, f"{len(idx):,}", str(s.min()), str(s.max())])

    card_top = card.head(20)
    card_rows = [
        [r["col"], int(r["nunique"]), f"{r['top1_share']:.3f}", f"{r['top10_share']:.3f}"]
        for _, r in card_top.iterrows()
    ]

    unseen_pivot = unseen.pivot(index="col", columns="split", values="unseen_rate").fillna(0.0)
    unseen_pivot["max_rate"] = unseen_pivot.max(axis=1)
    unseen_top = unseen_pivot.sort_values("max_rate", ascending=False).head(20)
    unseen_rows = [
        [c, f"{float(row.get('val', 0.0)):.3f}", f"{float(row.get('test', 0.0)):.3f}"]
        for c, row in unseen_top.iterrows()
    ]

    topk_tables: dict[str, pd.DataFrame] = {}
    for col in ["app_id", "site_id", "site_domain", "app_domain", "device_model"]:
        if col in df.columns:
            topk_tables[col] = _topk_ctr_table(df, col, k=max_topk)

    # Write markdown report.
    md: list[str] = []
    md.append("# Avazu CTR — EDA Report (sample-based)")
    md.append("")
    md.append("This report is generated from a deterministic subsample to keep iteration fast and preserve time structure.")
    md.append("")
    md.append("## Dataset snapshot")
    md.append(f"- Sample parquet: `{paths.sample_parquet}`")
    md.append(f"- Rows: `{total_rows:,}`")
    md.append(f"- Overall CTR: `{overall_ctr:.4f}`")
    md.append(f"- Time range: `{time_desc['min_hour']}` → `{time_desc['max_hour']}`")
    md.append(f"- Distinct days / hours: `{time_desc['n_days']}` / `{time_desc['n_hours']}`")
    md.append("")
    md.append("## OOT splits (by day)")
    md.append(_md_table(["split", "rows", "min hour_dt", "max hour_dt"], split_rows))
    md.append("")
    md.append("## Target behavior over time")
    md.append(f"![CTR by day]({paths.assets_dir.name}/{ctr_day_path.name})")
    md.append("")
    md.append(f"![Impressions by day]({paths.assets_dir.name}/{vol_day_path.name})")
    md.append("")
    md.append(f"![CTR by hour-of-day]({paths.assets_dir.name}/{ctr_hod_path.name})")
    md.append("")
    md.append("## Categorical cardinality + concentration")
    md.append("High-cardinality columns are important for TE and entity-history features, but also drive cold-start in OOT splits.")
    md.append("")
    md.append(f"![Cardinality]({paths.assets_dir.name}/{card_path.name})")
    md.append("")
    md.append(_md_table(["col", "nunique", "top1_share", "top10_share"], card_rows))
    md.append("")
    if freq_paths:
        md.append("## Long-tail structure (frequency rank plots)")
        for p in freq_paths:
            md.append(f"![Frequency rank plot]({paths.assets_dir.name}/{p.name})")
            md.append("")
    md.append("## Cold-start (unseen categories in val/test vs train)")
    md.append("This estimates how often a category value appears in val/test that never appears in train (on this sample).")
    md.append("")
    md.append(f"![Unseen rates]({paths.assets_dir.name}/{unseen_path.name})")
    md.append("")
    md.append(_md_table(["col", "val_unseen_rate", "test_unseen_rate"], unseen_rows))
    md.append("")
    md.append("## CTR by popular categories (top-K)")
    for col, tab in topk_tables.items():
        md.append(f"### `{col}`")
        rows = [[r[col], int(r["imps"]), f"{float(r['ctr']):.4f}"] for _, r in tab.iterrows()]
        md.append(_md_table([col, "imps", "ctr"], rows))
        md.append("")
    md.append("## Modeling implications (for this repo)")
    md.append("- OOT drift is visible (CTR and volume vary by day); keep strict chronological splits for selection.")
    md.append("- High-cardinality columns have long tails; TE (with time-aware shifting) is a compact baseline.")
    md.append("- Entity history features (Family A) help because they summarize recent behavior while respecting the online constraint.")
    md.append("")
    md.append("## Reproducibility")
    md.append("Regenerate this report via:")
    md.append("")
    md.append(f"- `python human_src/generate_eda_report.py --sample-parquet {paths.sample_parquet} --out-md {paths.out_md}`")
    md.append("")

    paths.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-parquet", default="data/interim/train_sample.parquet")
    ap.add_argument("--out-md", default="docs/EDA_REPORT.md")
    ap.add_argument("--assets-dir", default="docs/eda_assets")
    ap.add_argument("--max-topk", type=int, default=15)
    args = ap.parse_args()

    paths = EDAPaths(
        sample_parquet=Path(args.sample_parquet),
        out_md=Path(args.out_md),
        assets_dir=Path(args.assets_dir),
    )
    generate_report(paths, max_topk=int(args.max_topk))
    print(str(paths.out_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

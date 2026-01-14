from __future__ import annotations

import argparse
import json
import os
import gc
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb

from kaggle_clicks.metrics import compute_binary_metrics
from kaggle_clicks.paths import get_paths
from kaggle_clicks.sampling import deterministic_frac_mask, deterministic_pct_mask
from kaggle_clicks.te import TEConfig, add_time_prior_ctr, add_time_target_encoding
from kaggle_clicks.time_agg import TimeAggConfig, add_time_agg_features
from kaggle_clicks.time_utils import (
    add_time_columns,
    assert_strict_oot,
    make_oot_splits_by_day,
    make_rolling_tail_folds_by_day,
)


def _fmt_ts(ts: pd.Timestamp | None) -> str:
    if ts is None or pd.isna(ts):
        return "NA"
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")


def _format_sample_label(sample_pct: int, sample_frac: float | None) -> str:
    if sample_frac is None:
        return f"{sample_pct}%"
    pct = sample_frac * 100.0
    label = f"{pct:.4f}".rstrip("0").rstrip(".")
    return f"{label}%"


def _split_summary(df: pd.DataFrame, splits: dict[str, pd.Index]) -> list[tuple[str, int, str, str]]:
    rows: list[tuple[str, int, str, str]] = []
    for name in ("train", "val", "test"):
        idx = splits[name]
        subset = df.loc[idx, "hour_dt"]
        rows.append((name, int(len(idx)), _fmt_ts(subset.min()), _fmt_ts(subset.max())))
    return rows


def _write_report(
    path: str,
    run_tag: str,
    args: argparse.Namespace,
    df: pd.DataFrame,
    splits: dict[str, pd.Index],
    feature_cols: list[str],
    te_cols: list[str],
    metrics: dict[str, dict[str, float]],
    feature_importance: list[tuple[str, float]],
    time_agg_cols: list[str],
    best_iteration: int,
) -> None:
    lines: list[str] = []
    lines.append(f"# Run Report: `{run_tag}`")
    lines.append("")
    lines.append("## Data")
    sample_label = _format_sample_label(args.sample_pct, getattr(args, "sample_frac", None))
    lines.append(f"- Sample fraction: {sample_label} (deterministic hash on `id`)")
    lines.append(f"- Rows: {len(df):,}")
    if getattr(args, "rolling_tail_fold", None):
        lines.append(f"- Split mode: rolling tail (fold {args.rolling_tail_fold})")
    else:
        lines.append("- Split mode: default (test=last day, val=day before test)")
    lines.append("")
    lines.append("| Split | Rows | Start hour | End hour |")
    lines.append("| --- | ---: | --- | --- |")
    for name, count, start, end in _split_summary(df, splits):
        lines.append(f"| {name} | {count:,} | {start} | {end} |")

    lines.append("")
    lines.append("## Features")
    lines.append(f"- Numeric base features: 2 (`hour_of_day`, `prior_ctr`)")
    lines.append(f"- Target-encoded features: {len(te_cols)}")
    if time_agg_cols:
        lines.append(f"- Time aggregation: enabled (alpha={args.time_agg_alpha}, beta={args.time_agg_beta})")
        lines.append(f"  - Entities: {', '.join(args.time_agg_entities)}")
        if args.time_agg_windows:
            lines.append(f"  - Trailing windows (hours): {', '.join(map(str, args.time_agg_windows))}")
            lines.append(f"  - Gap (hours): {args.time_agg_gap_hours}")
        if args.time_agg_bucket_edges:
            lines.append(f"  - Bucket edges (hours): {', '.join(map(str, args.time_agg_bucket_edges))}")
        if args.time_agg_calendar:
            lines.append("  - Calendar windows: today + yesterday")
        if args.time_agg_event_windows:
            lines.append(f"  - Event windows (last N imps): {', '.join(map(str, args.time_agg_event_windows))}")
    else:
        lines.append("- Time aggregation: disabled")

    lines.append("")
    lines.append("## Model")
    lines.append(
        f"- XGBoost (hist) with learning_rate={args.learning_rate}, max_depth={args.max_depth}, "
        f"n_estimators={args.n_estimators}, subsample={args.subsample}, "
        f"colsample_bytree={args.colsample_bytree}, reg_lambda={args.reg_lambda}, "
        f"min_child_weight={args.min_child_weight}"
    )
    lines.append(f"- Early stopping on validation with patience 50, best_iteration={best_iteration}")

    lines.append("")
    lines.append("## Metrics")
    lines.append("| Split | ROC-AUC | PR-AUC |")
    lines.append("| --- | ---: | ---: |")
    for split in ("train", "val", "test"):
        m = metrics.get(split)
        if m:
            lines.append(f"| {split} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} |")
    if getattr(args, "export_preds", False):
        export_splits = getattr(args, "export_preds_splits", None) or ["val", "test"]
        lines.append("")
        lines.append("## Prediction Exports")
        for split in export_splits:
            lines.append(f"- preds_{split}.parquet")

    if feature_importance:
        lines.append("")
        lines.append("## Top Features (gain)")
        lines.append("| Rank | Feature | Gain share |")
        lines.append("| ---: | --- | ---: |")
        for rank, (feat, gain) in enumerate(feature_importance[:15], start=1):
            lines.append(f"| {rank} | `{feat}` | {gain:.4f} |")

    content = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _load_or_make_sample(
    train_csv: str, sample_pct: int, sample_frac: float | None, out_parquet: str
) -> pd.DataFrame:
    p = os.path.abspath(out_parquet)
    if os.path.exists(p):
        return pd.read_parquet(p)

    chunks = []
    for chunk in pd.read_csv(train_csv, chunksize=500_000):
        if "id" not in chunk.columns:
            raise ValueError("Expected 'id' column in train.csv for deterministic sampling.")
        if sample_frac is not None:
            mask = deterministic_frac_mask(chunk["id"], frac=float(sample_frac))
        else:
            mask = deterministic_pct_mask(chunk["id"], pct=sample_pct)
        if mask.any():
            chunks.append(chunk.loc[mask])
    if not chunks:
        raise RuntimeError("Sampling produced 0 rows; increase sample_pct or check data.")
    df = pd.concat(chunks, axis=0, ignore_index=True)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    df.to_parquet(p, index=False)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="data/raw/train.csv")
    ap.add_argument("--sample-pct", type=int, default=1)
    ap.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Optional fractional sample in (0,1], e.g. 0.001 for 0.1%%. If set, overrides sample-pct.",
    )
    ap.add_argument("--sample-parquet", default="data/interim/train_sample.parquet")
    ap.add_argument("--m", type=float, default=100.0, help="TE smoothing strength (larger = more shrinkage).")
    ap.add_argument("--no-te", action="store_true", help="If set, do not include per-category TE features.")
    ap.add_argument(
        "--te-parquet",
        default=None,
        help="Optional cached TE parquet produced by `python -m kaggle_clicks.precompute_te` (skips TE recomputation).",
    )
    ap.add_argument(
        "--rolling-tail-fold",
        choices=["A", "B"],
        default=None,
        help="If set, use rolling-tail Fold A/B splits (train/val/test are strict OOT by day).",
    )
    ap.add_argument("--export-preds", action="store_true", help="If set, export preds_val/test.parquet.")
    ap.add_argument(
        "--export-preds-splits",
        nargs="*",
        default=["val", "test"],
        choices=["val", "test"],
        help="Which splits to export when --export-preds is set (default: val test).",
    )
    ap.add_argument(
        "--preds-fold-id",
        default=None,
        help="Optional fold id to write into prediction exports (default: rolling-tail fold or 'single').",
    )
    ap.add_argument("--n-estimators", type=int, default=2000)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--reg-lambda", type=float, default=5.0)
    ap.add_argument("--min-child-weight", type=float, default=10.0)
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--run-tag", default="baseline_te")
    ap.add_argument(
        "--time-agg-entities",
        nargs="*",
        default=["device_ip", "device_id", "app_id", "site_id"],
        help="Entity columns for time aggregation features (empty list to disable).",
    )
    ap.add_argument(
        "--time-agg-windows",
        nargs="*",
        type=int,
        default=[1, 6, 24],
        help="Window sizes in hours for time aggregation features.",
    )
    ap.add_argument(
        "--time-agg-gap-hours",
        type=int,
        default=0,
        help="Exclude the most recent g hours: use [H-w-g, H-g) instead of [H-w, H).",
    )
    ap.add_argument(
        "--time-agg-bucket-edges",
        nargs="*",
        type=int,
        default=[],
        help="If set, add disjoint bucket windows with these cumulative edges (e.g. 1 6 24 168).",
    )
    ap.add_argument(
        "--time-agg-calendar",
        action="store_true",
        help="If set, add calendar-aligned windows: since start-of-day (today) and yesterday.",
    )
    ap.add_argument(
        "--time-agg-event-windows",
        nargs="*",
        type=int,
        default=[],
        help="If set, add event-count windows using last N impressions (approx, by-hour blocks).",
    )
    ap.add_argument("--time-agg-alpha", type=float, default=1.0)
    ap.add_argument("--time-agg-beta", type=float, default=10.0)
    args = ap.parse_args()

    paths = get_paths()
    run_dir = paths.runs / f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{args.run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.train_csv):
        raise SystemExit(f"Missing data file: {args.train_csv} (place it at data/raw/train.csv)")

    raw = _load_or_make_sample(args.train_csv, args.sample_pct, args.sample_frac, args.sample_parquet)
    df = add_time_columns(raw, hour_col="hour").sort_values("hour_dt").reset_index(drop=True)
    df_report = df[["hour_dt"]].copy()
    del raw
    gc.collect()

    # Drop unused large columns early (reduces memory pressure on large samples).
    df = df.drop(columns=["id", "hour"], errors="ignore")

    if args.rolling_tail_fold:
        folds = make_rolling_tail_folds_by_day(df)
        splits = folds[str(args.rolling_tail_fold)]
    else:
        splits = make_oot_splits_by_day(df)
    assert_strict_oot(df, splits)

    label_col = "click"
    te_parquet = getattr(args, "te_parquet", None)
    using_te_cache = bool(te_parquet)
    df_entities: pd.DataFrame | None = None

    if using_te_cache:
        # With a TE cache, we do not need the full raw categorical columns in memory.
        # Keeping only the minimal set significantly reduces peak RAM usage, especially for 10% samples.
        keep_cols = {label_col, "hour_dt", "hour_of_day"}
        keep_cols.update(getattr(args, "time_agg_entities", []) or [])
        present_keep = [c for c in df.columns if c in keep_cols]
        df = df[present_keep].copy()

        if not os.path.exists(str(te_parquet)):
            raise SystemExit(f"--te-parquet not found: {te_parquet}")
        if args.no_te:
            te_df = pd.read_parquet(str(te_parquet), columns=["row_id", "hour_dt", "prior_ctr"])
        else:
            te_df = pd.read_parquet(str(te_parquet))
        if "prior_ctr" not in te_df.columns:
            raise SystemExit(f"--te-parquet missing required column 'prior_ctr': {te_parquet}")

        if "row_id" in te_df.columns:
            te_df = te_df.sort_values("row_id").reset_index(drop=True)
            if int(te_df["row_id"].iloc[0]) != 0 or int(te_df["row_id"].iloc[-1]) != len(te_df) - 1:
                raise SystemExit(f"--te-parquet row_id must be 0..n-1: {te_parquet}")

        if len(te_df) != len(df):
            raise SystemExit(f"--te-parquet length {len(te_df)} != data length {len(df)}")

        if "hour_dt" in te_df.columns:
            if not (te_df["hour_dt"].to_numpy() == df["hour_dt"].to_numpy()).all():
                raise SystemExit("--te-parquet hour_dt does not match sample ordering; regenerate cache for this sample.")

        te_feature_cols = [c for c in te_df.columns if c not in {"row_id", "hour_dt"}]
        # Infer cat_cols ordering from TE column naming (for reporting/config).
        cat_cols: list[str] = []
        for c in te_feature_cols:
            if c.endswith("__te") or c.endswith("__hist_imps"):
                base = c.split("__", 1)[0]
                if base not in cat_cols:
                    cat_cols.append(base)

        te_cols: list[str] = []
        if not args.no_te:
            for c in cat_cols:
                for suffix in ("__te", "__hist_imps"):
                    name = f"{c}{suffix}"
                    if name in te_df.columns:
                        te_cols.append(name)

        if args.time_agg_entities:
            # Prepare a minimal frame for time-agg aligned to the same row order.
            for ent in args.time_agg_entities:
                if ent in df.columns and df[ent].dtype == object:
                    codes, _ = pd.factorize(df[ent], sort=False)
                    df[ent] = codes.astype(np.int32, copy=False)
            df_entities = df[["hour_dt", label_col, *args.time_agg_entities]].copy()

        # Build minimal feature frame: label + base cols + TE cols (avoid keeping raw categorical columns).
        fe = pd.DataFrame(
            {
                label_col: df[label_col].to_numpy(dtype=np.int8, copy=False),
                "hour_of_day": df["hour_of_day"].to_numpy(dtype=np.float32, copy=False),
                "prior_ctr": te_df["prior_ctr"].to_numpy(dtype=np.float32, copy=False),
            }
        )
        for col in te_cols:
            fe[col] = te_df[col].to_numpy(dtype=np.float32, copy=False)

        # Keep only hour_dt for reporting/export; release large categorical columns ASAP.
        df = df_report
        del df_report
        del te_df
        gc.collect()
    else:
        drop_cols = {"id", "hour", "hour_dt", "day", "hour_of_day", label_col}
        cat_cols = [c for c in df.columns if c not in drop_cols]

        if args.no_te:
            # No-TE ablation only needs the global time prior + optional time-aggregation.
            # Avoid factorizing all categorical columns (can be very RAM-heavy on 10% samples).
            keep_cols = {label_col, "hour_dt", "hour_of_day"}
            keep_cols.update(getattr(args, "time_agg_entities", []) or [])
            present_keep = [c for c in df.columns if c in keep_cols]
            df = df[present_keep].copy()

            if args.time_agg_entities:
                for ent in args.time_agg_entities:
                    if ent in df.columns and df[ent].dtype == object:
                        codes, _ = pd.factorize(df[ent], sort=False)
                        df[ent] = codes.astype(np.int32, copy=False)
                df_entities = df[["hour_dt", label_col, *args.time_agg_entities]].copy()

            fe = add_time_prior_ctr(df, label_col=label_col, cfg=TEConfig(m=args.m), inplace=True)
            te_cols = []
        else:
            # Convert categorical keys to compact int codes to reduce memory (and usually speed up groupby).
            for col in cat_cols:
                if df[col].dtype == object:
                    codes, _ = pd.factorize(df[col], sort=False)
                    df[col] = codes.astype(np.int32, copy=False)

            if args.time_agg_entities:
                df_entities = df[["hour_dt", label_col, *args.time_agg_entities]].copy()

            fe = add_time_target_encoding(
                df, cat_cols=cat_cols, label_col=label_col, cfg=TEConfig(m=args.m), inplace=True
            )
            te_cols = [f"{c}__te" for c in cat_cols] + [f"{c}__hist_imps" for c in cat_cols]

        # Keep only hour_dt for reporting/export; release large categorical columns ASAP.
        df = df_report
        del df_report
        gc.collect()

    time_agg_cols: list[str] = []
    if args.time_agg_entities:
        if df_entities is None:
            raise RuntimeError("Internal error: expected df_entities for time aggregation.")
        ta_out, time_agg_cols = add_time_agg_features(
            df_entities,
            cfg=TimeAggConfig(
                entities=tuple(args.time_agg_entities),
                windows=tuple(args.time_agg_windows),
                alpha=args.time_agg_alpha,
                beta=args.time_agg_beta,
                gap_hours=int(args.time_agg_gap_hours),
                bucket_edges=tuple(args.time_agg_bucket_edges),
                add_calendar_windows=bool(args.time_agg_calendar),
                event_windows=tuple(args.time_agg_event_windows),
            ),
            inplace=True,
        )
        for col in time_agg_cols:
            fe[col] = ta_out[col].to_numpy(dtype=np.float32, copy=False)
        del df_entities, ta_out
        gc.collect()

    base_cols = ["hour_of_day", "prior_ctr"]
    feature_cols = base_cols + te_cols + time_agg_cols

    # Reduce memory: drop large original categorical columns before building dense matrices.
    # Avoid copying the full feature matrix here; we only materialize the dense float32 array below.
    fe = fe[[label_col, *feature_cols]]
    gc.collect()

    X_all = np.empty((len(fe), len(feature_cols)), dtype=np.float32)
    for j, col in enumerate(feature_cols):
        X_all[:, j] = fe[col].to_numpy(dtype=np.float32, copy=False)
    y_all = fe[label_col].to_numpy(dtype=np.int8, copy=False)
    del fe
    gc.collect()

    def _as_slice_or_idx(idx: pd.Index) -> slice | np.ndarray:
        arr = idx.to_numpy(dtype=np.int64, copy=False)
        if arr.size == 0:
            return slice(0, 0)
        if arr[0] == 0 and np.all(arr[1:] == arr[:-1] + 1):
            return slice(int(arr[0]), int(arr[-1]) + 1)
        if np.all(arr[1:] == arr[:-1] + 1):
            return slice(int(arr[0]), int(arr[-1]) + 1)
        return arr

    tr_sel = _as_slice_or_idx(splits["train"])
    va_sel = _as_slice_or_idx(splits["val"])
    te_sel = _as_slice_or_idx(splits["test"])

    X_train = X_all[tr_sel]
    y_train = y_all[tr_sel]
    X_val = X_all[va_sel]
    y_val = y_all[va_sel]
    X_test = X_all[te_sel]
    y_test = y_all[te_sel]

    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        min_child_weight=args.min_child_weight,
        objective="binary:logistic",
        tree_method="hist",
        n_jobs=args.n_jobs,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["auc", "aucpr"],
        verbose=False,
        early_stopping_rounds=50,
    )

    train_prob = model.predict_proba(X_train)[:, 1]
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    train_metrics = compute_binary_metrics(y_train, train_prob)
    val_metrics = compute_binary_metrics(y_val, val_prob)
    test_metrics = compute_binary_metrics(y_test, test_prob)

    config = vars(args) | {
        "cat_cols": cat_cols,
        "n_rows": int(len(df)),
        "feature_cols": feature_cols,
    }

    metrics_by_split = {
        "train": {"roc_auc": train_metrics.roc_auc, "pr_auc": train_metrics.pr_auc},
        "val": {"roc_auc": val_metrics.roc_auc, "pr_auc": val_metrics.pr_auc},
        "test": {"roc_auc": test_metrics.roc_auc, "pr_auc": test_metrics.pr_auc},
    }
    best_iteration = int(getattr(model, "best_iteration", -1))
    metrics_payload = {"splits": metrics_by_split, "best_iteration": best_iteration}

    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, sort_keys=True))

    booster = model.get_booster()
    booster.save_model(str(run_dir / "model.json"))

    raw_importance = booster.get_score(importance_type="gain")
    feature_importance: list[tuple[str, float]] = []
    if raw_importance:
        total_gain = sum(raw_importance.values())
        # XGBoost returns feature keys as f0..fN when trained on NumPy arrays.
        # Use our known feature order for stable, human-readable reporting.
        name_map = {f"f{i}": name for i, name in enumerate(feature_cols)}
        feature_importance = sorted(
            [(name_map.get(key, key), gain / total_gain) for key, gain in raw_importance.items()],
            key=lambda x: x[1],
            reverse=True,
        )
    report_path = run_dir / "report.md"
    _write_report(
        str(report_path),
        args.run_tag,
        args,
        df,
        splits,
        feature_cols,
        te_cols,
        metrics_by_split,
        feature_importance,
        time_agg_cols,
        best_iteration,
    )

    if args.export_preds:
        fold_id = args.preds_fold_id or args.rolling_tail_fold or "single"
        export_splits = getattr(args, "export_preds_splits", None) or ["val", "test"]
        split_map = {
            "val": (splits["val"], y_val, val_prob),
            "test": (splits["test"], y_test, test_prob),
        }
        for split_name in export_splits:
            idx, y_arr, p_arr = split_map[str(split_name)]
            export = pd.DataFrame(
                {
                    "row_id": idx.to_numpy(dtype=np.int64, copy=False),
                    "hour_dt": df.loc[idx, "hour_dt"].to_numpy(),
                    "y": y_arr.astype(np.int8, copy=False),
                    "p": p_arr.astype(np.float32, copy=False),
                    "fold_id": np.full(len(idx), fold_id, dtype=object),
                }
            )
            export_path = run_dir / f"preds_{split_name}.parquet"
            export.to_parquet(export_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

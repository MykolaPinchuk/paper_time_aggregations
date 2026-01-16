from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class EventCountSpec:
    run_id: str
    description: str
    event_count: int | None  # None means "no event window"


def _find_run_dir_for_tag(run_tag: str) -> Path | None:
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return None
    matches = sorted(runs_dir.glob(f"*_{run_tag}"), reverse=True)
    return matches[0] if matches else None


def _is_complete_run_dir(run_dir: Path, export_preds: bool, export_preds_splits: tuple[str, ...] | None) -> bool:
    if not run_dir.exists():
        return False
    if not (run_dir / "metrics.json").exists():
        return False
    if not (run_dir / "config.json").exists():
        return False
    if export_preds:
        splits = export_preds_splits or ("val", "test")
        for split in tuple(splits):
            if not (run_dir / f"preds_{split}.parquet").exists():
                return False
    return True


def _attach_metrics(row: dict, run_dir: Path) -> None:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    splits = metrics.get("splits") or {}
    for split in ["train", "val", "test"]:
        m = splits.get(split) or {}
        row[f"{split}_roc_auc"] = m.get("roc_auc")
        row[f"{split}_pr_auc"] = m.get("pr_auc")
    row["best_iteration"] = metrics.get("best_iteration")


def _write_outputs(sweep_dir: Path, rows: list[dict], sweep_config: dict) -> None:
    df = pd.DataFrame(rows)
    (sweep_dir / "summary.csv").write_text(df.to_csv(index=False))

    def _to_md_table(frame: pd.DataFrame) -> str:
        cols = list(frame.columns)
        lines = []
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, r in frame.iterrows():
            vals = []
            for c in cols:
                v = r.get(c)
                if pd.isna(v):
                    vals.append("")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines: list[str] = []
    lines.append(f"# Event-Count Window Sweep — Summary ({sweep_config['created_utc']})")
    lines.append("")
    lines.append(f"- Sweep tag: `{sweep_config['sweep_tag']}`")
    lines.append(f"- Sample: `{sweep_config['sample_label']}`")
    lines.append(f"- Windows (hours): `{sweep_config['time_agg_windows']}`")
    lines.append(f"- Entities: `{', '.join(sweep_config['time_agg_entities'])}`")
    lines.append(f"- TE cache: `{sweep_config.get('te_parquet') or 'none'}`")
    lines.append(f"- no-TE: `{bool(sweep_config.get('no_te', False))}`")
    lines.append(f"- Rolling-tail: `{bool(sweep_config.get('rolling_tail'))}`")
    lines.append("")
    if not df.empty:
        cols = [
            "event_count",
            "fold_id",
            "returncode",
            "test_roc_auc",
            "test_pr_auc",
            "val_roc_auc",
            "val_pr_auc",
            "best_iteration",
            "run_dir",
        ]
        present = [c for c in cols if c in df.columns]
        preview = df[present].copy()
        lines.append(_to_md_table(preview))
        lines.append("")
    (sweep_dir / "summary.md").write_text("\n".join(lines) + "\n")

    (sweep_dir / "sweep_config.json").write_text(json.dumps(sweep_config, indent=2, sort_keys=True))


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep event-count window sizes for a fixed window-length spec.")
    ap.add_argument("--train-csv", default=str(Path("data") / "raw" / "train.csv"))
    ap.add_argument("--sample-pct", type=int, default=1)
    ap.add_argument("--sample-frac", type=float, default=None, help="Optional fractional sample in (0,1].")
    ap.add_argument("--sample-parquet", default=str(Path("data") / "interim" / "train_sample.parquet"))
    ap.add_argument("--te-parquet", default=None, help="Optional TE cache parquet to reuse across runs.")
    ap.add_argument("--m", type=float, default=100.0)
    ap.add_argument("--no-te", action="store_true")

    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--reg-lambda", type=float, default=5.0)
    ap.add_argument("--min-child-weight", type=float, default=10.0)
    ap.add_argument("--n-jobs", type=int, default=1)

    ap.add_argument("--time-agg-entities", nargs="*", default=["device_ip", "device_id", "app_id", "site_id"])
    ap.add_argument("--time-agg-windows", nargs="*", type=int, default=[1, 6, 24, 48, 168])
    ap.add_argument("--time-agg-alpha", type=float, default=1.0)
    ap.add_argument("--time-agg-beta", type=float, default=10.0)

    ap.add_argument("--event-counts", nargs="*", type=int, default=[10, 20, 50, 100, 200, 500, 1000])
    ap.add_argument(
        "--include-baseline",
        action="store_true",
        help="If set, include a baseline run with no event-count window (event_count=None).",
    )
    ap.add_argument("--rolling-tail", action="store_true", help="If set, run Fold A and Fold B.")
    ap.add_argument("--export-preds", action="store_true", help="If set, pass --export-preds to each run.")
    ap.add_argument(
        "--export-preds-splits",
        nargs="*",
        default=["val", "test"],
        choices=["val", "test"],
        help="Which splits to export when --export-preds is set (default: val test).",
    )

    ap.add_argument("--sweep-tag", default="event_count_sweep")
    ap.add_argument("--sweep-dir", default=None, help="Optional explicit sweep directory.")
    ap.add_argument("--per-run-timeout-seconds", type=int, default=120)
    ap.add_argument("--resume", action="store_true", help="If set, skip specs that already have complete outputs.")
    args = ap.parse_args()

    if not os.path.exists(args.train_csv):
        raise SystemExit(f"Missing train csv: {args.train_csv}")

    if args.sample_frac is not None and not (0.0 < float(args.sample_frac) <= 1.0):
        raise SystemExit("--sample-frac must be in (0,1].")

    counts = [int(x) for x in args.event_counts]
    if any(c <= 0 for c in counts):
        raise SystemExit("--event-counts must be positive integers.")
    if len(set(counts)) != len(counts):
        raise SystemExit("--event-counts must be unique.")

    specs: list[EventCountSpec] = []
    if args.include_baseline:
        specs.append(EventCountSpec(run_id="event0", description="no event-count window", event_count=None))
    for c in counts:
        specs.append(EventCountSpec(run_id=f"event{c}", description=f"event window last{c}", event_count=c))

    sweep_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    sample_label = f"{args.sample_pct}pct" if args.sample_frac is None else f"{args.sample_frac:.4g}frac"
    sweep_dir = (
        Path(args.sweep_dir)
        if args.sweep_dir
        else (Path("runs") / "sweeps" / f"{sweep_ts}_{args.sweep_tag}_event_counts_{sample_label}")
    )
    sweep_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = sweep_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    sweep_config = {
        "created_utc": sweep_ts,
        "sweep_tag": args.sweep_tag,
        "sweep_dir": str(sweep_dir),
        "sample_pct": int(args.sample_pct),
        "sample_frac": args.sample_frac,
        "sample_parquet": args.sample_parquet,
        "sample_label": sample_label,
        "train_csv": args.train_csv,
        "te_parquet": args.te_parquet,
        "m": float(args.m),
        "no_te": bool(args.no_te),
        "rolling_tail": bool(args.rolling_tail),
        "export_preds": bool(args.export_preds),
        "export_preds_splits": list(args.export_preds_splits) if bool(args.export_preds) else [],
        "per_run_timeout_seconds": int(args.per_run_timeout_seconds),
        "model_params": {
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
            "max_depth": int(args.max_depth),
            "subsample": float(args.subsample),
            "colsample_bytree": float(args.colsample_bytree),
            "reg_lambda": float(args.reg_lambda),
            "min_child_weight": float(args.min_child_weight),
            "n_jobs": int(args.n_jobs),
        },
        "time_agg_entities": list(args.time_agg_entities),
        "time_agg_windows": [int(x) for x in args.time_agg_windows],
        "time_agg_alpha": float(args.time_agg_alpha),
        "time_agg_beta": float(args.time_agg_beta),
        "event_counts": counts,
        "include_baseline": bool(args.include_baseline),
        "specs": [{"run_id": s.run_id, "description": s.description, "event_count": s.event_count} for s in specs],
    }

    rows: list[dict] = []

    def build_cmd(spec: EventCountSpec, fold_id: str | None, run_tag: str) -> list[str]:
        cmd: list[str] = [
            sys.executable,
            "-m",
            "kaggle_clicks.run_baseline_te",
            "--train-csv",
            args.train_csv,
            "--sample-pct",
            str(int(args.sample_pct)),
            "--sample-parquet",
            str(args.sample_parquet),
            "--run-tag",
            run_tag,
            "--m",
            str(float(args.m)),
            "--n-estimators",
            str(int(args.n_estimators)),
            "--learning-rate",
            str(float(args.learning_rate)),
            "--max-depth",
            str(int(args.max_depth)),
            "--subsample",
            str(float(args.subsample)),
            "--colsample-bytree",
            str(float(args.colsample_bytree)),
            "--reg-lambda",
            str(float(args.reg_lambda)),
            "--min-child-weight",
            str(float(args.min_child_weight)),
            "--n-jobs",
            str(int(args.n_jobs)),
            "--time-agg-alpha",
            str(float(args.time_agg_alpha)),
            "--time-agg-beta",
            str(float(args.time_agg_beta)),
        ]

        if args.sample_frac is not None:
            cmd += ["--sample-frac", str(float(args.sample_frac))]

        if args.no_te:
            cmd.append("--no-te")
        if args.te_parquet:
            cmd += ["--te-parquet", str(args.te_parquet)]

        if args.time_agg_entities:
            cmd += ["--time-agg-entities", *args.time_agg_entities]
        if args.time_agg_windows:
            cmd += ["--time-agg-windows", *[str(int(x)) for x in args.time_agg_windows]]

        if spec.event_count is not None:
            cmd += ["--time-agg-event-windows", str(int(spec.event_count))]

        if fold_id is not None:
            cmd += ["--rolling-tail-fold", fold_id]

        if args.export_preds:
            cmd += ["--export-preds", "--preds-fold-id", (fold_id or "single")]
            if getattr(args, "export_preds_splits", None):
                cmd += ["--export-preds-splits", *list(args.export_preds_splits)]

        return cmd

    for spec in specs:
        for fold_id in (("A", "B") if args.rolling_tail else (None,)):
            fold_tag = f"_fold{fold_id}" if fold_id else ""
            run_tag = f"{args.sweep_tag}_{spec.run_id}{fold_tag}"

            row: dict = {
                "event_count": spec.event_count if spec.event_count is not None else 0,
                "run_id": spec.run_id,
                "description": spec.description,
                "fold_id": fold_id or "single",
                "run_tag": run_tag,
                "returncode": None,
                "runtime_seconds": None,
                "run_dir": "",
                "skipped": False,
            }

            existing = _find_run_dir_for_tag(run_tag)
            if args.resume and existing and _is_complete_run_dir(
                existing,
                export_preds=bool(args.export_preds),
                export_preds_splits=tuple(args.export_preds_splits) if bool(args.export_preds) else None,
            ):
                row["skipped"] = True
                row["returncode"] = 0
                row["run_dir"] = str(existing)
                _attach_metrics(row, existing)
                rows.append(row)
                _write_outputs(sweep_dir, rows, sweep_config)
                continue

            cmd = build_cmd(spec, fold_id, run_tag)
            (sweep_dir / "commands.sh").open("a", encoding="utf-8").write(" ".join(cmd) + "\n")

            stdout_path = logs_dir / f"{run_tag}_stdout.log"
            stderr_path = logs_dir / f"{run_tag}_stderr.log"
            start = time.time()
            try:
                proc = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=max(30, int(args.per_run_timeout_seconds)),
                )
            except subprocess.TimeoutExpired as e:
                row["returncode"] = 124
                row["runtime_seconds"] = round(time.time() - start, 3)
                stdout_path.write_text((e.stdout or "") if isinstance(e.stdout, str) else "")
                stderr_path.write_text((e.stderr or "") if isinstance(e.stderr, str) else "")
                rows.append(row)
                _write_outputs(sweep_dir, rows, sweep_config)
                return 1

            row["returncode"] = int(proc.returncode)
            row["runtime_seconds"] = round(time.time() - start, 3)
            stdout_path.write_text(proc.stdout or "")
            stderr_path.write_text(proc.stderr or "")

            run_dir = _find_run_dir_for_tag(run_tag)
            if run_dir and _is_complete_run_dir(
                run_dir,
                export_preds=bool(args.export_preds),
                export_preds_splits=tuple(args.export_preds_splits) if bool(args.export_preds) else None,
            ):
                row["run_dir"] = str(run_dir)
                _attach_metrics(row, run_dir)
            rows.append(row)
            _write_outputs(sweep_dir, rows, sweep_config)

            if int(proc.returncode) != 0:
                return int(proc.returncode)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

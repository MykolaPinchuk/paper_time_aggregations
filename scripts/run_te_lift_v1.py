from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _find_complete_run_dir(run_tag: str, export_preds: bool) -> Path | None:
    runs_root = Path("runs")
    matches = sorted(runs_root.glob(f"*_{run_tag}"), reverse=True)
    for p in matches:
        if not p.is_dir():
            continue
        if not (p / "metrics.json").exists():
            continue
        if export_preds and (not (p / "preds_test.parquet").exists() or not (p / "preds_val.parquet").exists()):
            continue
        return p
    return None


def _run(cmd: list[str]) -> int:
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the 10% TE-only lift comparison runs (Fold A/B).")
    ap.add_argument("--train-csv", default="data/raw/train.csv")
    ap.add_argument("--sample-pct", type=int, default=10)
    ap.add_argument("--sample-parquet", default="data/interim/train_sample_10pct.parquet")
    ap.add_argument("--te-parquet", default="data/interim/te_cache/train_sample_10pct_te_m100.parquet")
    ap.add_argument("--m", type=float, default=100.0)
    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--reg-lambda", type=float, default=5.0)
    ap.add_argument("--min-child-weight", type=float, default=10.0)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--export-preds", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Skip runs that already have complete outputs.")
    args = ap.parse_args()

    entities = ["device_ip", "device_id", "app_id", "site_id"]
    a3_windows = ["1", "6", "24", "48", "168"]

    specs: list[tuple[str, list[str]]] = [
        ("TE_only", ["--time-agg-entities"]),
        ("A3_trailing", ["--time-agg-entities", *entities, "--time-agg-windows", *a3_windows]),
        (
            "A3_event50",
            ["--time-agg-entities", *entities, "--time-agg-windows", *a3_windows, "--time-agg-event-windows", "50"],
        ),
    ]

    for fold_id in ("A", "B"):
        for spec_name, extra_args in specs:
            run_tag = f"paper_te_lift_{spec_name}_10pct_fold{fold_id}"
            if args.resume:
                existing = _find_complete_run_dir(run_tag, export_preds=bool(args.export_preds))
                if existing is not None:
                    print(f"SKIP {run_tag} -> {existing}")
                    continue

            cmd: list[str] = [
                sys.executable,
                "-m",
                "kaggle_clicks.run_baseline_te",
                "--train-csv",
                str(args.train_csv),
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
                "--rolling-tail-fold",
                fold_id,
                *extra_args,
            ]
            if args.te_parquet:
                cmd += ["--te-parquet", str(args.te_parquet)]
            if args.export_preds:
                cmd += ["--export-preds", "--preds-fold-id", fold_id]

            print("RUN", " ".join(cmd))
            rc = _run(cmd)
            if rc != 0:
                raise SystemExit(rc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


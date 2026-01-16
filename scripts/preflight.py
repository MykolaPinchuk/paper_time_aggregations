from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path


def _run(*args: str) -> None:
    subprocess.check_call([sys.executable, *args])


def _disk_free_gib(path: Path) -> float:
    usage = shutil.disk_usage(str(path))
    return float(usage.free) / (1024.0**3)


def _ensure_dirs() -> None:
    for p in [Path("data") / "raw", Path("data") / "interim", Path("data") / "interim" / "te_cache", Path("runs")]:
        p.mkdir(parents=True, exist_ok=True)


def _check_files() -> int:
    root = Path.cwd()
    train_csv = root / "data" / "raw" / "train.csv"
    sample = root / "data" / "interim" / "train_sample_10pct.parquet"
    te_cache = root / "data" / "interim" / "te_cache" / "train_sample_10pct_te_m100.parquet"

    free_gib = _disk_free_gib(root)
    print(f"Disk free: {free_gib:.1f} GiB at {root}")
    if free_gib < 30:
        print("Warning: low free disk; full runs can produce multi-GB outputs under data/, runs/, artifacts/.")

    ok = True
    if not train_csv.exists():
        ok = False
        print(f"Missing: {train_csv} (download Avazu train.csv from Kaggle)")
    else:
        size_gib = train_csv.stat().st_size / (1024.0**3)
        print(f"Found: {train_csv} ({size_gib:.2f} GiB)")

    if not sample.exists():
        print(f"Missing: {sample} (run `make sample10`)")
    else:
        print(f"Found: {sample}")

    if not te_cache.exists():
        print(f"Missing: {te_cache} (run `make tecache10`)")
    else:
        print(f"Found: {te_cache}")

    if not ok:
        return 2
    return 0


def _write_synthetic_train_csv(path: Path, n_rows: int, seed: int) -> None:
    import numpy as np

    rng = np.random.default_rng(int(seed))
    start = datetime(2014, 10, 21, 0, 0, 0)
    hours = [start + timedelta(hours=i) for i in range(24 * 5)]  # 5 days => rolling-tail ok
    hour_codes = [int(dt.strftime("%y%m%d%H")) for dt in hours]

    device_ips = [f"ip_{i}" for i in range(200)]
    device_ids = [f"did_{i}" for i in range(100)]
    app_ids = [f"app_{i}" for i in range(50)]
    site_ids = [f"site_{i}" for i in range(50)]

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "hour", "click", "device_ip", "device_id", "app_id", "site_id"])
        for i in range(int(n_rows)):
            h = int(rng.choice(hour_codes))
            click = int(rng.random() < 0.17)
            w.writerow(
                [
                    int(i + 1),
                    h,
                    click,
                    str(rng.choice(device_ips)),
                    str(rng.choice(device_ids)),
                    str(rng.choice(app_ids)),
                    str(rng.choice(site_ids)),
                ]
            )


def _find_run_dir_by_tag(tag: str) -> Path | None:
    runs = Path("runs")
    if not runs.exists():
        return None
    matches = sorted(runs.glob(f"*_{tag}"), reverse=True)
    return matches[0] if matches else None


def _synthetic_smoke(seed: int, cleanup: bool) -> int:
    _ensure_dirs()
    with tempfile.TemporaryDirectory(prefix="pta_preflight_") as td:
        tmp = Path(td)
        train_csv = tmp / "train.csv"
        sample = tmp / "sample.parquet"
        te_cache = tmp / "te_cache.parquet"

        print(f"Writing synthetic train.csv: {train_csv}")
        _write_synthetic_train_csv(train_csv, n_rows=40_000, seed=seed)

        print("Running: make_sample (100%)")
        _run("-m", "kaggle_clicks.make_sample", "--train-csv", str(train_csv), "--sample-pct", "100", "--out-parquet", str(sample))

        print("Running: precompute_te")
        _run("-m", "kaggle_clicks.precompute_te", "--sample-parquet", str(sample), "--out-parquet", str(te_cache), "--m", "100", "--overwrite")

        common = [
            "--train-csv",
            str(train_csv),
            "--sample-pct",
            "100",
            "--sample-parquet",
            str(sample),
            "--te-parquet",
            str(te_cache),
            "--rolling-tail-fold",
            "A",
            "--n-estimators",
            "30",
            "--n-jobs",
            "1",
            "--export-preds",
            "--export-preds-splits",
            "test",
            "--time-agg-entities",
            "device_ip",
            "device_id",
            "app_id",
            "site_id",
            "--time-agg-windows",
            "1",
            "6",
            "24",
            "48",
            "168",
        ]

        tag_note = f"preflight_note_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        tag_te = f"preflight_te_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        print("Running: run_baseline_te (no-TE, exports test preds only)")
        _run(
            "-m",
            "kaggle_clicks.run_baseline_te",
            "--run-tag",
            tag_note,
            "--no-te",
            *common,
        )

        print("Running: run_baseline_te (TE, exports test preds only)")
        _run(
            "-m",
            "kaggle_clicks.run_baseline_te",
            "--run-tag",
            tag_te,
            *common,
        )

    if cleanup:
        for tag in (tag_note, tag_te):
            rd = _find_run_dir_by_tag(tag)
            if rd and rd.exists():
                shutil.rmtree(rd)

    print("Synthetic smoke: OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Preflight checks for paper sweeps.")
    ap.add_argument("--check-files", action="store_true", help="Check for data/raw/train.csv, 10% sample, TE cache.")
    ap.add_argument("--synthetic-smoke", action="store_true", help="Run a tiny synthetic end-to-end smoke test.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-cleanup", action="store_true", help="Keep synthetic run dirs under runs/ for inspection.")
    args = ap.parse_args()

    if not args.check_files and not args.synthetic_smoke:
        args.check_files = True

    rc = 0
    if args.check_files:
        rc = max(rc, _check_files())
    if args.synthetic_smoke:
        rc = max(rc, _synthetic_smoke(seed=int(args.seed), cleanup=not bool(args.no_cleanup)))
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())


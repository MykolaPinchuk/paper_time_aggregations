from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from kaggle_clicks.paths import get_paths


@dataclass(frozen=True)
class ExperimentSpec:
    run_id: str
    description: str
    cli_args: tuple[str, ...]


def _default_sample_parquet(sample_pct: int) -> str:
    if sample_pct == 5:
        return "data/interim/train_sample_5pct.parquet"
    return "data/interim/train_sample.parquet"


def _rel_to_repo(path: Path) -> str:
    root = get_paths().root.resolve()
    try:
        return str(path.resolve().relative_to(root))
    except Exception:
        return str(path)


def _family_a_specs(
    entities: tuple[str, ...],
    include_a4: bool,
    enable_phase2: bool,
    base_windows_for_phase2: tuple[int, ...],
) -> list[ExperimentSpec]:
    ent_args = ("--time-agg-entities", *entities)

    specs: list[ExperimentSpec] = [
        ExperimentSpec(
            run_id="R0",
            description="TE only",
            cli_args=("--time-agg-entities",),
        ),
        ExperimentSpec(
            run_id="R1",
            description="Trailing A1 windows {1,6,24}",
            cli_args=(
                *ent_args,
                "--time-agg-windows",
                "1",
                "6",
                "24",
            ),
        ),
        ExperimentSpec(
            run_id="R2",
            description="Trailing A2 windows {1,3,6,12,24}",
            cli_args=(
                *ent_args,
                "--time-agg-windows",
                "1",
                "3",
                "6",
                "12",
                "24",
            ),
        ),
        ExperimentSpec(
            run_id="R3",
            description="Trailing A3 windows {1,6,24,48,168}",
            cli_args=(
                *ent_args,
                "--time-agg-windows",
                "1",
                "6",
                "24",
                "48",
                "168",
            ),
        ),
    ]

    if include_a4:
        specs.append(
            ExperimentSpec(
                run_id="R4",
                description="Trailing A4 windows {1,2,4,8,16,24,48,96,168}",
                cli_args=(
                    *ent_args,
                    "--time-agg-windows",
                    "1",
                    "2",
                    "4",
                    "8",
                    "16",
                    "24",
                    "48",
                    "96",
                    "168",
                ),
            )
        )

    if enable_phase2:
        bw = [str(x) for x in base_windows_for_phase2]
        specs.extend(
            [
                ExperimentSpec(
                    run_id="R5",
                    description="Gap windows g=1 (all base windows)",
                    cli_args=(
                        *ent_args,
                        "--time-agg-windows",
                        *bw,
                        "--time-agg-gap-hours",
                        "1",
                    ),
                ),
                ExperimentSpec(
                    run_id="R6",
                    description="Gap windows g=6 (all base windows)",
                    cli_args=(
                        *ent_args,
                        "--time-agg-windows",
                        *bw,
                        "--time-agg-gap-hours",
                        "6",
                    ),
                ),
                ExperimentSpec(
                    run_id="R7",
                    description="Bucketized windows (0-1,1-6,6-24,24-168)",
                    cli_args=(
                        *ent_args,
                        "--time-agg-windows",
                        "--time-agg-bucket-edges",
                        "1",
                        "6",
                        "24",
                        "168",
                    ),
                ),
                ExperimentSpec(
                    run_id="R8",
                    description="Calendar windows (today + yesterday)",
                    cli_args=(
                        *ent_args,
                        "--time-agg-windows",
                        *bw,
                        "--time-agg-calendar",
                    ),
                ),
                ExperimentSpec(
                    run_id="R9",
                    description="Event-count windows (last N imps = 50)",
                    cli_args=(
                        *ent_args,
                        "--time-agg-windows",
                        *bw,
                        "--time-agg-event-windows",
                        "50",
                    ),
                ),
            ]
        )

    return specs


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_run_dir_for_tag(tag: str) -> Path | None:
    runs_root = get_paths().runs
    if not runs_root.exists():
        return None
    matches = sorted(runs_root.glob(f"*_{tag}"), reverse=True)
    return matches[0] if matches else None


def _write_sweep_outputs(
    sweep_dir: Path,
    sweep_ts: str,
    sweep_tag: str,
    sample_pct: int,
    sample_parquet: str,
    train_csv: str,
    rows: list[dict[str, Any]],
) -> None:
    if rows:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with (sweep_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    lines: list[str] = []
    lines.append(f"# Sweep Summary: `{sweep_tag}`")
    lines.append("")
    lines.append(f"- Created (UTC): `{sweep_ts}`")
    lines.append(f"- Sample: `{sample_pct}%` (`{sample_parquet}`)")
    lines.append(f"- Train CSV: `{train_csv}`")
    lines.append("")
    lines.append("| Run | Fold | Description | Return | Val ROC-AUC | Test ROC-AUC | Run dir |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | --- |")
    for r in rows:
        v = r.get("val_roc_auc")
        t = r.get("test_roc_auc")
        v_s = f"{float(v):.4f}" if v is not None else "NA"
        t_s = f"{float(t):.4f}" if t is not None else "NA"
        fold = r.get("fold_id", "single")
        lines.append(
            f"| {r['run_id']} | {fold} | {r['description']} | {r['returncode']} | {v_s} | {t_s} | `{r.get('run_dir','')}` |"
        )
    (sweep_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="data/raw/train.csv")
    ap.add_argument("--sample-pct", type=int, default=1)
    ap.add_argument("--sample-parquet", default=None)
    ap.add_argument(
        "--te-parquet",
        default=None,
        help="Optional cached TE parquet to pass through to each run (--te-parquet in run_baseline_te).",
    )
    ap.add_argument("--sweep-tag", default="familyA_sweep")
    ap.add_argument("--max-wall-seconds", type=int, default=280, help="Hard cap to keep the full command <5 minutes.")

    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--reg-lambda", type=float, default=5.0)
    ap.add_argument("--min-child-weight", type=float, default=10.0)
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--m", type=float, default=100.0, help="TE smoothing strength.")

    ap.add_argument(
        "--entities",
        nargs="*",
        default=["device_ip", "device_id", "app_id", "site_id"],
        help="Entities to use for time-agg runs (R1+).",
    )
    ap.add_argument("--include-a4", action="store_true")
    ap.add_argument("--enable-phase2", action="store_true")
    ap.add_argument("--rolling-tail", action="store_true", help="Run Fold A/B rolling-tail splits per spec.")
    ap.add_argument("--export-preds", action="store_true", help="Pass --export-preds to each run.")
    ap.add_argument(
        "--phase2-base-windows",
        nargs="*",
        type=int,
        default=[1, 6, 24],
        help="Base trailing windows used for the Phase 2 shape sweep runs.",
    )

    args = ap.parse_args()

    sample_parquet = args.sample_parquet or _default_sample_parquet(int(args.sample_pct))
    sweep_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path("runs") / "sweeps" / f"{sweep_ts}_{args.sweep_tag}_s{int(args.sample_pct)}pct"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    specs = _family_a_specs(
        entities=tuple(args.entities),
        include_a4=bool(args.include_a4),
        enable_phase2=bool(args.enable_phase2),
        base_windows_for_phase2=tuple(args.phase2_base_windows),
    )

    sweep_config = {
        "created_utc": sweep_ts,
        "sweep_tag": args.sweep_tag,
        "sample_pct": int(args.sample_pct),
        "sample_parquet": sample_parquet,
        "te_parquet": args.te_parquet,
        "train_csv": args.train_csv,
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
        "te_m": float(args.m),
        "rolling_tail": bool(args.rolling_tail),
        "export_preds": bool(args.export_preds),
        "specs": [{"run_id": s.run_id, "description": s.description, "cli_args": list(s.cli_args)} for s in specs],
    }
    (sweep_dir / "sweep_config.json").write_text(json.dumps(sweep_config, indent=2, sort_keys=True))

    start = time.time()
    rows: list[dict[str, Any]] = []

    try:
        stop = False
        for spec in specs:
            for fold_id in (("A", "B") if args.rolling_tail else (None,)):
                elapsed = time.time() - start
                remaining = int(args.max_wall_seconds) - elapsed
                if remaining <= 0:
                    stop = True
                    break

                fold_tag = f"_fold{fold_id}" if fold_id else ""
                run_tag = f"{args.sweep_tag}_{spec.run_id}{fold_tag}"
                cmd = [
                    sys.executable,
                    "-m",
                    "kaggle_clicks.run_baseline_te",
                    "--train-csv",
                    args.train_csv,
                    "--sample-pct",
                    str(int(args.sample_pct)),
                    "--sample-parquet",
                    sample_parquet,
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
                    *spec.cli_args,
                ]
                if args.te_parquet:
                    cmd += ["--te-parquet", str(args.te_parquet)]
                if fold_id:
                    cmd += ["--rolling-tail-fold", fold_id]
                if args.export_preds:
                    cmd += ["--export-preds"]
                    if fold_id:
                        cmd += ["--preds-fold-id", fold_id]

                with (sweep_dir / "commands.sh").open("a", encoding="utf-8") as f:
                    portable = cmd.copy()
                    portable[0] = "python"
                    f.write(" ".join(portable) + "\n")

                per_run_timeout = int(min(300, max(30, remaining + 10)))
                try:
                    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=per_run_timeout)
                except subprocess.TimeoutExpired as e:
                    log_prefix = f"{spec.run_id}{fold_tag}"
                    (sweep_dir / f"{log_prefix}_stdout.log").write_text(e.stdout or "", encoding="utf-8")
                    (sweep_dir / f"{log_prefix}_stderr.log").write_text(e.stderr or "", encoding="utf-8")
                    rows.append(
                        {
                            "run_id": spec.run_id,
                            "fold_id": fold_id or "single",
                            "run_tag": run_tag,
                            "description": spec.description,
                            "returncode": 124,
                            "run_dir": _rel_to_repo(rd) if (rd := _find_run_dir_for_tag(run_tag)) else "",
                        }
                    )
                    stop = True
                    break

                log_prefix = f"{spec.run_id}{fold_tag}"
                (sweep_dir / f"{log_prefix}_stdout.log").write_text(proc.stdout, encoding="utf-8")
                (sweep_dir / f"{log_prefix}_stderr.log").write_text(proc.stderr, encoding="utf-8")

                run_dir = _find_run_dir_for_tag(run_tag)
                row: dict[str, Any] = {
                    "run_id": spec.run_id,
                    "fold_id": fold_id or "single",
                    "run_tag": run_tag,
                    "description": spec.description,
                    "returncode": proc.returncode,
                    "run_dir": _rel_to_repo(run_dir) if run_dir else "",
                }

                if run_dir and (run_dir / "metrics.json").exists():
                    metrics = _read_json(run_dir / "metrics.json")
                    row["best_iteration"] = metrics.get("best_iteration")
                    for split in ("train", "val", "test"):
                        s = metrics["splits"][split]
                        row[f"{split}_roc_auc"] = s["roc_auc"]
                        row[f"{split}_pr_auc"] = s["pr_auc"]

                rows.append(row)
                _write_sweep_outputs(
                    sweep_dir=sweep_dir,
                    sweep_ts=sweep_ts,
                    sweep_tag=args.sweep_tag,
                    sample_pct=int(args.sample_pct),
                    sample_parquet=sample_parquet,
                    train_csv=args.train_csv,
                    rows=rows,
                )

                if proc.returncode != 0:
                    stop = True
                    break
            if stop:
                break
    finally:
        _write_sweep_outputs(
            sweep_dir=sweep_dir,
            sweep_ts=sweep_ts,
            sweep_tag=args.sweep_tag,
            sample_pct=int(args.sample_pct),
            sample_parquet=sample_parquet,
            train_csv=args.train_csv,
            rows=rows,
        )

    print(str(sweep_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

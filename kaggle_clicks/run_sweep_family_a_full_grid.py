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
class GridSpec:
    run_id: str
    description: str
    cli_args: tuple[str, ...]


def _format_sample_label(sample_pct: int, sample_frac: float | None) -> str:
    if sample_frac is None:
        return f"{sample_pct}%"
    pct = sample_frac * 100.0
    label = f"{pct:.4f}".rstrip("0").rstrip(".")
    return f"{label}%"


def _sample_tag(sample_pct: int, sample_frac: float | None) -> str:
    if sample_frac is None:
        return f"{sample_pct}pct"
    pct = sample_frac * 100.0
    label = f"{pct:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{label}pct"


def _default_sample_parquet(sample_pct: int) -> str:
    if sample_pct == 5:
        return "data/interim/train_sample_5pct.parquet"
    if sample_pct == 10:
        return "data/interim/train_sample_10pct.parquet"
    return "data/interim/train_sample.parquet"


def _rel_to_repo(path: Path) -> str:
    root = get_paths().root.resolve()
    try:
        return str(path.resolve().relative_to(root))
    except Exception:
        return str(path)


def _length_sets() -> dict[str, tuple[int, ...]]:
    return {
        "A1": (1, 6, 24),
        "A2": (1, 3, 6, 12, 24),
        "A3": (1, 6, 24, 48, 168),
        "A4": (1, 2, 4, 8, 16, 24, 48, 96, 168),
    }


def _grid_specs(entities: tuple[str, ...]) -> list[GridSpec]:
    ent_args = ("--time-agg-entities", *entities)
    specs: list[GridSpec] = []
    for set_id, windows in _length_sets().items():
        win_args = tuple(str(w) for w in windows)
        edges = win_args

        specs.append(
            GridSpec(
                run_id=f"{set_id}_trailing",
                description=f"{set_id} trailing windows {windows}",
                cli_args=(
                    *ent_args,
                    "--time-agg-windows",
                    *win_args,
                ),
            )
        )
        specs.append(
            GridSpec(
                run_id=f"{set_id}_gap1",
                description=f"{set_id} gap g=1 windows {windows}",
                cli_args=(
                    *ent_args,
                    "--time-agg-windows",
                    *win_args,
                    "--time-agg-gap-hours",
                    "1",
                ),
            )
        )
        specs.append(
            GridSpec(
                run_id=f"{set_id}_bucket",
                description=f"{set_id} bucket edges {windows}",
                cli_args=(
                    *ent_args,
                    "--time-agg-windows",
                    "--time-agg-bucket-edges",
                    *edges,
                ),
            )
        )
        specs.append(
            GridSpec(
                run_id=f"{set_id}_calendar",
                description=f"{set_id} calendar windows on trailing {windows}",
                cli_args=(
                    *ent_args,
                    "--time-agg-windows",
                    *win_args,
                    "--time-agg-calendar",
                ),
            )
        )
        specs.append(
            GridSpec(
                run_id=f"{set_id}_event50",
                description=f"{set_id} event windows last50",
                cli_args=(
                    *ent_args,
                    "--time-agg-windows",
                    *win_args,
                    "--time-agg-event-windows",
                    "50",
                ),
            )
        )
    return specs


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_meminfo_kb() -> dict[str, int]:
    """
    Lightweight /proc-based memory probe (no external deps).
    Returns values in kB (Linux /proc/meminfo units).
    """
    out: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, rest = line.split(":", 1)
                parts = rest.strip().split()
                if not parts:
                    continue
                try:
                    out[key] = int(parts[0])
                except ValueError:
                    continue
    except OSError:
        return {}
    return out


def _kb_to_gib(kb: int) -> float:
    return float(kb) / (1024.0 * 1024.0)


def _memory_gate_ok(
    *,
    ram_budget_gib: float | None,
    min_mem_available_gib: float,
    min_swap_free_gib: float,
    mem_safety_margin_gib: float,
) -> tuple[bool, dict[str, float]]:
    """
    Decide whether it is safe to launch another run based on current system memory.

    Policy:
    - Always require MemAvailable >= min_mem_available_gib.
    - If ram_budget_gib is set, also require MemAvailable >= (MemTotal - ram_budget_gib).
      This approximates: total_used <= ram_budget_gib.
    - Optionally require SwapFree >= min_swap_free_gib to avoid swap thrash.
    """
    mi = _read_meminfo_kb()
    mem_total_kb = int(mi.get("MemTotal", 0))
    mem_avail_kb = int(mi.get("MemAvailable", 0))
    swap_free_kb = int(mi.get("SwapFree", 0))

    mem_total = _kb_to_gib(mem_total_kb) if mem_total_kb > 0 else 0.0
    mem_avail = _kb_to_gib(mem_avail_kb) if mem_avail_kb > 0 else 0.0
    swap_free = _kb_to_gib(swap_free_kb) if swap_free_kb > 0 else 0.0

    budget_headroom = 0.0
    if ram_budget_gib is not None and mem_total > 0:
        budget_headroom = max(0.0, mem_total - float(ram_budget_gib))

    threshold = max(float(min_mem_available_gib), float(budget_headroom) + float(mem_safety_margin_gib))
    ok = mem_avail >= threshold and swap_free >= float(min_swap_free_gib)
    return ok, {
        "mem_total_gib": mem_total,
        "mem_available_gib": mem_avail,
        "swap_free_gib": swap_free,
        "threshold_gib": float(threshold),
        "budget_headroom_gib": float(budget_headroom),
        "mem_safety_margin_gib": float(mem_safety_margin_gib),
    }


def _find_run_dir_for_tag(tag: str) -> Path | None:
    runs_root = get_paths().runs
    if not runs_root.exists():
        return None
    matches = sorted(runs_root.glob(f"*_{tag}"), reverse=True)
    return matches[0] if matches else None


def _run_complete(run_dir: Path, export_preds: bool, export_preds_splits: tuple[str, ...] | None) -> bool:
    if not (run_dir / "metrics.json").exists():
        return False
    if export_preds:
        splits = export_preds_splits or ("val", "test")
        for split in tuple(splits):
            if not (run_dir / f"preds_{split}.parquet").exists():
                return False
    return True


def _find_complete_run_dir_for_tag(tag: str, export_preds: bool, export_preds_splits: tuple[str, ...] | None) -> Path | None:
    runs_root = get_paths().runs
    if not runs_root.exists():
        return None
    matches = sorted(runs_root.glob(f"*_{tag}"), reverse=True)
    for p in matches:
        if p.is_dir() and _run_complete(p, export_preds=export_preds, export_preds_splits=export_preds_splits):
            return p
    return matches[0] if matches else None


def _attach_metrics(row: dict[str, Any], run_dir: Path) -> None:
    if not (run_dir / "metrics.json").exists():
        return
    metrics = _read_json(run_dir / "metrics.json")
    row["best_iteration"] = metrics.get("best_iteration")
    for split in ("train", "val", "test"):
        s = metrics["splits"][split]
        row[f"{split}_roc_auc"] = s["roc_auc"]
        row[f"{split}_pr_auc"] = s["pr_auc"]


def _write_sweep_outputs(
    sweep_dir: Path,
    sweep_ts: str,
    sweep_tag: str,
    sample_label: str,
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
    lines.append(f"- Sample: `{sample_label}` (`{sample_parquet}`)")
    lines.append(f"- Train CSV: `{train_csv}`")
    lines.append("")
    lines.append("| Run | Fold | Description | Return | Skipped | Val ROC-AUC | Test ROC-AUC | Run dir |")
    lines.append("| --- | --- | --- | ---: | --- | ---: | ---: | --- |")
    for r in rows:
        v = r.get("val_roc_auc")
        t = r.get("test_roc_auc")
        v_s = f"{float(v):.4f}" if v is not None else "NA"
        t_s = f"{float(t):.4f}" if t is not None else "NA"
        fold = r.get("fold_id", "single")
        skipped = "yes" if r.get("skipped") else "no"
        lines.append(
            f"| {r['run_id']} | {fold} | {r['description']} | {r['returncode']} | {skipped} | {v_s} | {t_s} | `{r.get('run_dir','')}` |"
        )
    (sweep_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="data/raw/train.csv")
    ap.add_argument("--sample-pct", type=int, default=1)
    ap.add_argument("--sample-frac", type=float, default=None)
    ap.add_argument("--sample-parquet", default=None)
    ap.add_argument(
        "--te-parquet",
        default=None,
        help="Optional cached TE parquet to pass through to each run (--te-parquet in run_baseline_te).",
    )
    ap.add_argument("--sweep-tag", default="familyA_full_grid")
    ap.add_argument(
        "--sweep-dir",
        default=None,
        help="Optional explicit sweep directory path (if omitted, uses runs/sweeps/<timestamp>_<tag>_s<sample>).",
    )
    ap.add_argument("--max-wall-seconds", type=int, default=280, help="Hard cap to keep the full command <5 minutes.")
    ap.add_argument(
        "--per-run-timeout-seconds",
        type=int,
        default=1800,
        help="Timeout per run (seconds). Use a large value for overnight runs.",
    )
    ap.add_argument(
        "--max-parallel-runs",
        type=int,
        default=1,
        help="Run up to this many specs concurrently (useful if feature engineering is CPU-bound and single-threaded).",
    )
    ap.add_argument(
        "--ram-budget-gib",
        type=float,
        default=None,
        help="If set, throttle launches to keep total RAM usage roughly within this budget (GiB).",
    )
    ap.add_argument(
        "--min-mem-available-gib",
        type=float,
        default=2.0,
        help="Always keep at least this much MemAvailable before launching another run (GiB).",
    )
    ap.add_argument(
        "--min-swap-free-gib",
        type=float,
        default=1.0,
        help="Require at least this much SwapFree before launching another run (GiB).",
    )
    ap.add_argument(
        "--mem-safety-margin-gib",
        type=float,
        default=6.0,
        help="Extra MemAvailable cushion to reduce risk of system OOM killing other apps (GiB).",
    )

    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--reg-lambda", type=float, default=5.0)
    ap.add_argument("--min-child-weight", type=float, default=10.0)
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--m", type=float, default=100.0, help="TE smoothing strength.")
    ap.add_argument("--no-te", action="store_true", help="Pass --no-te to each run (time-agg-only ablation).")

    ap.add_argument(
        "--entities",
        nargs="*",
        default=["device_ip", "device_id", "app_id", "site_id"],
        help="Entities to use for time-agg runs.",
    )
    ap.add_argument("--rolling-tail", action="store_true", help="Run Fold A/B rolling-tail splits per spec.")
    ap.add_argument("--export-preds", action="store_true", help="Pass --export-preds to each run.")
    ap.add_argument(
        "--export-preds-splits",
        nargs="*",
        default=["val", "test"],
        choices=["val", "test"],
        help="Which splits to export when --export-preds is set (default: val test).",
    )

    args = ap.parse_args()
    max_parallel_runs = max(1, int(args.max_parallel_runs))
    ram_budget_gib = float(args.ram_budget_gib) if args.ram_budget_gib is not None else None
    min_mem_available_gib = max(0.0, float(args.min_mem_available_gib))
    min_swap_free_gib = max(0.0, float(args.min_swap_free_gib))
    mem_safety_margin_gib = max(0.0, float(args.mem_safety_margin_gib))

    if args.sample_parquet is None:
        if args.sample_frac is not None:
            raise SystemExit("When using --sample-frac, you must also set --sample-parquet.")
        sample_parquet = _default_sample_parquet(int(args.sample_pct))
    else:
        sample_parquet = args.sample_parquet

    sample_label = _format_sample_label(args.sample_pct, args.sample_frac)
    sweep_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    sample_tag = _sample_tag(args.sample_pct, args.sample_frac)
    sweep_dir = (
        Path(args.sweep_dir)
        if args.sweep_dir
        else (Path("runs") / "sweeps" / f"{sweep_ts}_{args.sweep_tag}_s{sample_tag}")
    )
    sweep_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = sweep_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    specs = _grid_specs(entities=tuple(args.entities))

    sweep_config = {
        "created_utc": sweep_ts,
        "sweep_tag": args.sweep_tag,
        "sample_pct": int(args.sample_pct),
        "sample_frac": args.sample_frac,
        "sample_parquet": sample_parquet,
        "te_parquet": args.te_parquet,
        "train_csv": args.train_csv,
        "sweep_dir": str(sweep_dir),
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
        "export_preds_splits": list(args.export_preds_splits) if bool(args.export_preds) else [],
        "per_run_timeout_seconds": int(args.per_run_timeout_seconds),
        "max_parallel_runs": int(max_parallel_runs),
        "ram_budget_gib": ram_budget_gib,
        "min_mem_available_gib": float(min_mem_available_gib),
        "min_swap_free_gib": float(min_swap_free_gib),
        "mem_safety_margin_gib": float(mem_safety_margin_gib),
        "specs": [{"run_id": s.run_id, "description": s.description, "cli_args": list(s.cli_args)} for s in specs],
    }
    (sweep_dir / "sweep_config.json").write_text(json.dumps(sweep_config, indent=2, sort_keys=True))

    def _iter_tasks() -> list[tuple[GridSpec, str | None]]:
        tasks: list[tuple[GridSpec, str | None]] = []
        for spec in specs:
            for fold_id in (("A", "B") if args.rolling_tail else (None,)):
                tasks.append((spec, fold_id))
        return tasks

    def _build_cmd(spec: GridSpec, fold_id: str | None, run_tag: str) -> list[str]:
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
            *(
                ["--no-te"]
                if bool(getattr(args, "no_te", False))
                else []
            ),
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
        if args.sample_frac is not None:
            cmd += ["--sample-frac", str(float(args.sample_frac))]
        if fold_id:
            cmd += ["--rolling-tail-fold", fold_id]
        if args.export_preds:
            cmd += ["--export-preds"]
            if fold_id:
                cmd += ["--preds-fold-id", fold_id]
            if getattr(args, "export_preds_splits", None):
                cmd += ["--export-preds-splits", *list(args.export_preds_splits)]
        return cmd

    start = time.time()
    rows: list[dict[str, Any]] = []
    sweep_log = (sweep_dir / "sweep.log").open("a", encoding="utf-8")

    try:
        per_run_timeout = int(max(30, args.per_run_timeout_seconds))

        if max_parallel_runs == 1:
            stop = False
            for spec, fold_id in _iter_tasks():
                elapsed = time.time() - start
                remaining = int(args.max_wall_seconds) - elapsed
                if remaining <= 0:
                    stop = True
                    break

                fold_tag = f"_fold{fold_id}" if fold_id else ""
                run_tag = f"{args.sweep_tag}_{spec.run_id}{fold_tag}"
                run_dir = _find_complete_run_dir_for_tag(
                    run_tag,
                    export_preds=bool(args.export_preds),
                    export_preds_splits=tuple(args.export_preds_splits) if bool(args.export_preds) else None,
                )
                ts_now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

                if run_dir and _run_complete(
                    run_dir,
                    bool(args.export_preds),
                    tuple(args.export_preds_splits) if bool(args.export_preds) else None,
                ):
                    row: dict[str, Any] = {
                        "run_id": spec.run_id,
                        "fold_id": fold_id or "single",
                        "run_tag": run_tag,
                        "description": spec.description,
                        "returncode": 0,
                        "run_dir": _rel_to_repo(run_dir),
                        "skipped": True,
                    }
                    _attach_metrics(row, run_dir)
                    rows.append(row)
                    msg = f"[{ts_now}] SKIP {spec.run_id}{fold_tag} -> {run_dir}\n"
                    sweep_log.write(msg)
                    sweep_log.flush()
                    _write_sweep_outputs(
                        sweep_dir=sweep_dir,
                        sweep_ts=sweep_ts,
                        sweep_tag=args.sweep_tag,
                        sample_label=sample_label,
                        sample_parquet=sample_parquet,
                        train_csv=args.train_csv,
                        rows=rows,
                    )
                    continue

                cmd = _build_cmd(spec=spec, fold_id=fold_id, run_tag=run_tag)

                with (sweep_dir / "commands.sh").open("a", encoding="utf-8") as f:
                    portable = cmd.copy()
                    portable[0] = "python"
                    f.write(" ".join(portable) + "\n")

                msg = f"[{ts_now}] RUN {spec.run_id}{fold_tag} timeout={per_run_timeout}s\n"
                sweep_log.write(msg)
                sweep_log.flush()
                try:
                    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=per_run_timeout)
                except subprocess.TimeoutExpired as e:
                    log_prefix = f"{spec.run_id}{fold_tag}"
                    (logs_dir / f"{log_prefix}_stdout.log").write_text(e.stdout or "", encoding="utf-8")
                    (logs_dir / f"{log_prefix}_stderr.log").write_text(e.stderr or "", encoding="utf-8")
                    rows.append(
                        {
                            "run_id": spec.run_id,
                            "fold_id": fold_id or "single",
                            "run_tag": run_tag,
                            "description": spec.description,
                            "returncode": 124,
                            "run_dir": _rel_to_repo(rd) if (rd := _find_run_dir_for_tag(run_tag)) else "",
                            "skipped": False,
                        }
                    )
                    sweep_log.write(f"[{ts_now}] TIMEOUT {spec.run_id}{fold_tag}\n")
                    sweep_log.flush()
                    stop = True
                    break

                log_prefix = f"{spec.run_id}{fold_tag}"
                (logs_dir / f"{log_prefix}_stdout.log").write_text(proc.stdout, encoding="utf-8")
                (logs_dir / f"{log_prefix}_stderr.log").write_text(proc.stderr, encoding="utf-8")

                run_dir = _find_complete_run_dir_for_tag(
                    run_tag,
                    export_preds=bool(args.export_preds),
                    export_preds_splits=tuple(args.export_preds_splits) if bool(args.export_preds) else None,
                )
                row = {
                    "run_id": spec.run_id,
                    "fold_id": fold_id or "single",
                    "run_tag": run_tag,
                    "description": spec.description,
                    "returncode": proc.returncode,
                    "run_dir": _rel_to_repo(run_dir) if run_dir else "",
                    "skipped": False,
                }
                if run_dir:
                    _attach_metrics(row, run_dir)

                rows.append(row)
                _write_sweep_outputs(
                    sweep_dir=sweep_dir,
                    sweep_ts=sweep_ts,
                    sweep_tag=args.sweep_tag,
                    sample_label=sample_label,
                    sample_parquet=sample_parquet,
                    train_csv=args.train_csv,
                    rows=rows,
                )

                if proc.returncode != 0:
                    sweep_log.write(f"[{ts_now}] FAIL {spec.run_id}{fold_tag} rc={proc.returncode}\n")
                    sweep_log.flush()
                    stop = True
                    break
                sweep_log.write(f"[{ts_now}] OK {spec.run_id}{fold_tag}\n")
                sweep_log.flush()
            if stop:
                pass
        else:
            tasks = _iter_tasks()
            running: dict[str, dict[str, Any]] = {}
            stop_launching = False

            def _maybe_launch_next() -> None:
                nonlocal stop_launching
                while not stop_launching and tasks and len(running) < max_parallel_runs:
                    mem_ok, mem_stats = _memory_gate_ok(
                        ram_budget_gib=ram_budget_gib,
                        min_mem_available_gib=min_mem_available_gib,
                        min_swap_free_gib=min_swap_free_gib,
                        mem_safety_margin_gib=mem_safety_margin_gib,
                    )
                    if not mem_ok:
                        ts_now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                        sweep_log.write(
                            f"[{ts_now}] PAUSE_MEM avail={mem_stats['mem_available_gib']:.2f}GiB "
                            f"swap_free={mem_stats['swap_free_gib']:.2f}GiB "
                            f"threshold={mem_stats['threshold_gib']:.2f}GiB "
                            f"(budget_headroom={mem_stats['budget_headroom_gib']:.2f}GiB "
                            f"margin={mem_stats['mem_safety_margin_gib']:.2f}GiB)\n"
                        )
                        sweep_log.flush()
                        return

                    elapsed = time.time() - start
                    remaining = int(args.max_wall_seconds) - elapsed
                    if remaining <= 0:
                        stop_launching = True
                        return

                    spec, fold_id = tasks.pop(0)
                    fold_tag = f"_fold{fold_id}" if fold_id else ""
                    run_tag = f"{args.sweep_tag}_{spec.run_id}{fold_tag}"
                    ts_now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

                    run_dir = _find_complete_run_dir_for_tag(
                        run_tag,
                        export_preds=bool(args.export_preds),
                        export_preds_splits=tuple(args.export_preds_splits) if bool(args.export_preds) else None,
                    )
                    if run_dir and _run_complete(
                        run_dir,
                        bool(args.export_preds),
                        tuple(args.export_preds_splits) if bool(args.export_preds) else None,
                    ):
                        row: dict[str, Any] = {
                            "run_id": spec.run_id,
                            "fold_id": fold_id or "single",
                            "run_tag": run_tag,
                            "description": spec.description,
                            "returncode": 0,
                            "run_dir": _rel_to_repo(run_dir),
                            "skipped": True,
                        }
                        _attach_metrics(row, run_dir)
                        rows.append(row)
                        sweep_log.write(f"[{ts_now}] SKIP {spec.run_id}{fold_tag} -> {run_dir}\n")
                        sweep_log.flush()
                        _write_sweep_outputs(
                            sweep_dir=sweep_dir,
                            sweep_ts=sweep_ts,
                            sweep_tag=args.sweep_tag,
                            sample_label=sample_label,
                            sample_parquet=sample_parquet,
                            train_csv=args.train_csv,
                            rows=rows,
                        )
                        continue

                    cmd = _build_cmd(spec=spec, fold_id=fold_id, run_tag=run_tag)
                    with (sweep_dir / "commands.sh").open("a", encoding="utf-8") as f:
                        portable = cmd.copy()
                        portable[0] = "python"
                        f.write(" ".join(portable) + "\n")

                    log_prefix = f"{spec.run_id}{fold_tag}"
                    stdout_path = logs_dir / f"{log_prefix}_stdout.log"
                    stderr_path = logs_dir / f"{log_prefix}_stderr.log"
                    stdout_f = stdout_path.open("w", encoding="utf-8")
                    stderr_f = stderr_path.open("w", encoding="utf-8")
                    sweep_log.write(f"[{ts_now}] RUN {spec.run_id}{fold_tag} timeout={per_run_timeout}s\n")
                    sweep_log.flush()
                    proc = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, text=True)
                    running[run_tag] = {
                        "spec": spec,
                        "fold_id": fold_id,
                        "fold_tag": fold_tag,
                        "run_tag": run_tag,
                        "start_time": time.time(),
                        "proc": proc,
                        "stdout_f": stdout_f,
                        "stderr_f": stderr_f,
                        "stdout_path": str(stdout_path),
                        "stderr_path": str(stderr_path),
                    }

            _maybe_launch_next()
            while running or (tasks and not stop_launching):
                ts_now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                elapsed = time.time() - start
                if elapsed > int(args.max_wall_seconds):
                    stop_launching = True

                for run_tag, info in list(running.items()):
                    proc: subprocess.Popen[str] = info["proc"]
                    runtime = time.time() - float(info["start_time"])
                    if runtime > per_run_timeout and proc.poll() is None:
                        proc.kill()
                        info["stdout_f"].close()
                        info["stderr_f"].close()
                        spec: GridSpec = info["spec"]
                        fold_id = info["fold_id"]
                        fold_tag = info["fold_tag"]
                        rows.append(
                            {
                                "run_id": spec.run_id,
                                "fold_id": fold_id or "single",
                                "run_tag": run_tag,
                                "description": spec.description,
                                "returncode": 124,
                                "run_dir": _rel_to_repo(rd) if (rd := _find_run_dir_for_tag(run_tag)) else "",
                                "skipped": False,
                            }
                        )
                        sweep_log.write(f"[{ts_now}] TIMEOUT {spec.run_id}{fold_tag}\n")
                        sweep_log.flush()
                        del running[run_tag]
                        stop_launching = True
                        continue

                    ret = proc.poll()
                    if ret is None:
                        continue

                    info["stdout_f"].close()
                    info["stderr_f"].close()
                    spec = info["spec"]
                    fold_id = info["fold_id"]
                    fold_tag = info["fold_tag"]
                    run_dir = _find_complete_run_dir_for_tag(
                        run_tag,
                        export_preds=bool(args.export_preds),
                        export_preds_splits=tuple(args.export_preds_splits) if bool(args.export_preds) else None,
                    )
                    row = {
                        "run_id": spec.run_id,
                        "fold_id": fold_id or "single",
                        "run_tag": run_tag,
                        "description": spec.description,
                        "returncode": int(ret),
                        "run_dir": _rel_to_repo(run_dir) if run_dir else "",
                        "skipped": False,
                    }
                    if run_dir:
                        _attach_metrics(row, run_dir)
                    rows.append(row)

                    _write_sweep_outputs(
                        sweep_dir=sweep_dir,
                        sweep_ts=sweep_ts,
                        sweep_tag=args.sweep_tag,
                        sample_label=sample_label,
                        sample_parquet=sample_parquet,
                        train_csv=args.train_csv,
                        rows=rows,
                    )
                    if int(ret) != 0:
                        sweep_log.write(f"[{ts_now}] FAIL {spec.run_id}{fold_tag} rc={ret}\n")
                        sweep_log.flush()
                        stop_launching = True
                    else:
                        sweep_log.write(f"[{ts_now}] OK {spec.run_id}{fold_tag}\n")
                        sweep_log.flush()
                    del running[run_tag]

                if not stop_launching:
                    _maybe_launch_next()
                elif not running:
                    # Nothing running and we are not allowed to launch more.
                    break

                time.sleep(0.2)
    finally:
        sweep_log.close()
        _write_sweep_outputs(
            sweep_dir=sweep_dir,
            sweep_ts=sweep_ts,
            sweep_tag=args.sweep_tag,
            sample_label=sample_label,
            sample_parquet=sample_parquet,
            train_csv=args.train_csv,
            rows=rows,
        )

    print(str(sweep_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

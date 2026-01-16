from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import shutil


def _latest_sweep_dir(tag: str) -> Path:
    sweeps_root = Path("runs") / "sweeps"
    if not sweeps_root.exists():
        raise SystemExit("Missing runs/sweeps; run a sweep first.")
    matches = sorted([p for p in sweeps_root.iterdir() if p.is_dir() and tag in p.name])
    if not matches:
        raise SystemExit(f"No sweep dirs matching tag '{tag}' under runs/sweeps/")
    return matches[-1]


def _run(*args: str) -> None:
    subprocess.check_call([sys.executable, *args])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="Output root (e.g., artifacts/v1)")
    ap.add_argument("--te-sweep-tag", default="paper_full_grid_10pct")
    ap.add_argument("--note-sweep-tag", default="paper_full_grid_10pct_noTE_with_preds")
    ap.add_argument("--baseline-run-id", default="A3_trailing")
    ap.add_argument("--bootstrap-reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    te_sweep = _latest_sweep_dir(str(args.te_sweep_tag))
    note_sweep = _latest_sweep_dir(str(args.note_sweep_tag))

    # Paired inference vs baseline for both sweeps (writes inference_vs_baseline.csv into each sweep dir).
    _run(
        "-m",
        "kaggle_clicks.postprocess_sweep_inference",
        "--sweep-dir",
        str(te_sweep),
        "--baseline-run-id",
        str(args.baseline_run_id),
        "--split",
        "test",
        "--bootstrap-reps",
        str(int(args.bootstrap_reps)),
        "--seed",
        str(int(args.seed)),
    )
    _run(
        "-m",
        "kaggle_clicks.postprocess_sweep_inference",
        "--sweep-dir",
        str(note_sweep),
        "--baseline-run-id",
        str(args.baseline_run_id),
        "--split",
        "test",
        "--bootstrap-reps",
        str(int(args.bootstrap_reps)),
        "--seed",
        str(int(args.seed)),
    )

    # Master tables.
    te_out = out_root / "10pct_paper_grid"
    note_out = out_root / "10pct_paper_grid_noTE"
    _run("-m", "kaggle_clicks.postprocess_paper_grid_master", "--sweep-dir", str(te_sweep), "--out-dir", str(te_out))
    _run("-m", "kaggle_clicks.postprocess_paper_grid_master", "--sweep-dir", str(note_sweep), "--out-dir", str(note_out))

    # Shape-vs-trailing contrasts (ROC only by default; writes into sweep dir).
    _run(
        "-m",
        "kaggle_clicks.postprocess_paper_grid_contrasts",
        "--sweep-dir",
        str(te_sweep),
        "--split",
        "test",
    )
    _run(
        "-m",
        "kaggle_clicks.postprocess_paper_grid_contrasts",
        "--sweep-dir",
        str(note_sweep),
        "--split",
        "test",
    )
    # Copy key sweep outputs into artifacts (keeps artifacts stable even if sweep dirs are rotated).
    for sweep_dir, out_dir in ((te_sweep, te_out), (note_sweep, note_out)):
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "summary.csv",
            "summary.md",
            "sweep_config.json",
            "inference_vs_baseline.csv",
            "contrasts_vs_trailing_test.csv",
            "contrasts_vs_trailing_test.md",
        ):
            src = sweep_dir / name
            if src.exists():
                shutil.copy2(src, out_dir / name)

    # TE vs no-TE comparison tables.
    combined_out = out_root / "combined"
    _run(
        "-m",
        "kaggle_clicks.postprocess_paper_grid_te_vs_note",
        "--te-summary-csv",
        str(te_sweep / "summary.csv"),
        "--note-summary-csv",
        str(note_sweep / "summary.csv"),
        "--out-dir",
        str(combined_out),
    )

    # TE-only lift summary (requires the three runs to exist; computed from the TE sweep outputs).
    lift_out = out_root / "10pct_te_lift"
    _run(
        "scripts/postprocess_te_lift_v1.py",
        "--out-dir",
        str(lift_out),
        "--bootstrap-reps",
        str(int(args.bootstrap_reps)),
        "--seed",
        str(int(args.seed)),
        "--sample-pct",
        str(int(10)),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

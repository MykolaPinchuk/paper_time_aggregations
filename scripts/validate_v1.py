from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _latest_sweep_dir(tag: str) -> Path:
    sweeps_root = Path("runs") / "sweeps"
    if not sweeps_root.exists():
        raise SystemExit("Missing runs/sweeps.")
    matches = sorted([p for p in sweeps_root.iterdir() if p.is_dir() and tag in p.name])
    if not matches:
        raise SystemExit(f"No sweep dirs matching tag '{tag}' under runs/sweeps/")
    return matches[-1]


def _require(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Missing: {path}")


def _check_sweep(sweep_dir: Path, expected_rows: int = 40) -> None:
    _require(sweep_dir / "summary.csv")
    df = pd.read_csv(sweep_dir / "summary.csv")
    if len(df) != expected_rows:
        raise SystemExit(f"Unexpected rows in {sweep_dir}/summary.csv: {len(df)} (expected {expected_rows})")
    if "returncode" not in df.columns or not (df["returncode"] == 0).all():
        bad = df[df.get("returncode", 1) != 0][["run_id", "fold_id", "returncode"]]
        raise SystemExit(f"Sweep has failures: {sweep_dir}\n{bad.to_string(index=False)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate that v1 paper artifacts/figures can be regenerated.")
    ap.add_argument("--artifacts-dir", default="artifacts/v1")
    ap.add_argument("--bundle-dir", default="paper_bundle/arxiv_v1")
    ap.add_argument("--te-sweep-tag", default="paper_full_grid_10pct")
    ap.add_argument("--note-sweep-tag", default="paper_full_grid_10pct_noTE_with_preds")
    args = ap.parse_args()

    artifacts = Path(args.artifacts_dir)
    bundle = Path(args.bundle_dir)

    te_sweep = _latest_sweep_dir(str(args.te_sweep_tag))
    note_sweep = _latest_sweep_dir(str(args.note_sweep_tag))
    _check_sweep(te_sweep)
    _check_sweep(note_sweep)

    # Core artifacts.
    for p in [
        artifacts / "10pct_paper_grid" / "master_results.csv",
        artifacts / "10pct_paper_grid" / "contrasts_vs_trailing_test.csv",
        artifacts / "10pct_paper_grid" / "inference_vs_baseline.csv",
        artifacts / "10pct_paper_grid_noTE" / "master_results.csv",
        artifacts / "10pct_paper_grid_noTE" / "contrasts_vs_trailing_test.csv",
        artifacts / "10pct_te_lift" / "metrics_selected_runs.csv",
        artifacts / "10pct_te_lift" / "inference_test_te_vs_timeagg.csv",
        artifacts / "combined" / "te_vs_note_summary.md",
        artifacts / "plots" / "te_lift_metrics.png",
        artifacts / "plots" / "te_lift_deltas.png",
        artifacts / "figures_main" / "fig_event_count_sweep_A3_10pct.pdf",
    ]:
        _require(p)

    # Bundle outputs (paper upload folder).
    _require(bundle / "paper.pdf")
    for p in [
        bundle / "figures" / "fig_league_table_top_bottom.pdf",
        bundle / "figures" / "fig_traffic_light_shape_contrasts.pdf",
        bundle / "figures" / "fig_event_count_sweep_A3_10pct.pdf",
        bundle / "plots" / "te_lift_metrics.png",
        bundle / "plots" / "te_lift_deltas.png",
    ]:
        _require(p)

    print("OK")
    print(f"TE sweep: {te_sweep}")
    print(f"no-TE sweep: {note_sweep}")
    print(f"Artifacts: {artifacts}")
    print(f"Bundle: {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


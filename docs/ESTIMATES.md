# Execution time estimates (rough)

These are rough wall-time estimates for reproducing the full v1 results on a typical workstation. Actual time varies with CPU/RAM/disk speed and whether you run specs in parallel.

## One-time setup

- Install deps (`make venv`): ~2–10 minutes (depends on wheels/network).
- Download Kaggle data: depends on network (dataset not included).

## Data prep

- Build deterministic 10% sample (`make sample10`):
  - Reads `data/raw/train.csv` (~6GB) and writes `data/interim/train_sample_10pct.parquet`.
  - Estimate: ~10–40 minutes.
- Precompute TE cache (`make tecache10`):
  - Estimate: ~5–20 minutes.

## Main results (long)

All runs below use rolling-tail folds A/B (2 folds).

- TE-only lift runs (`make te_lift`): 6 runs total (3 specs × 2 folds).
  - Estimate: ~1–3 hours total.
- TE+time-agg full grid (`make grid10_te`): 40 runs (20 specs × 2 folds).
  - Estimate (serial): ~5–12 hours.
  - If you increase `--max-parallel-runs` (and have enough RAM): ~2–6 hours.
- no-TE full grid (`make grid10_note`): 40 runs.
  - Similar to TE grid; estimate: ~4–10 hours serial.

## Postprocess + figures

- Postprocess (`make postprocess`):
  - Inference includes paired DeLong ROC-AUC and PR-AUC hour-block bootstrap (default 200 reps).
  - Estimate: ~10–60 minutes (depends mainly on bootstrap + size of preds parquet).
- Figures (`make figures`): ~1–5 minutes.
- LaTeX paper build (`make paper`): ~1–5 minutes once TeX is installed.

## Optional sensitivity sweep

- Event-count N sweep (`make event_count_sweep`): 16 runs (8 event-count specs including baseline × 2 folds).
  - Estimate: ~2–6 hours serial.


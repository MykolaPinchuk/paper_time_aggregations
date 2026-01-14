# Reproducing the results (v1)

## What you will reproduce

- 10% deterministic sample (from Kaggle Avazu `train.csv`)
- Rolling-tail Fold A/B evaluation
- Two 20×2 grids (length tuple × window shape):
  - TE + time-aggregation
  - no-TE (time-aggregation only)
- Postprocessed tables and figures written under `artifacts/v1/`
- An arXiv bundle assembled under `paper_bundle/arxiv_v1/`

## Dependencies

- Python 3.10.x
- Install pinned packages: `pip install -r requirements.txt`

For the closest match to the reference numbers, use the pinned versions and run with `--n-jobs 1` (as in the provided commands).

## Data

This repo does not distribute the dataset. You must download it from Kaggle and place:
- `data/raw/train.csv`

## End-to-end commands

All commands should be run from the repo root.

### 1) Deterministic 10% sample

- `python -m kaggle_clicks.make_sample --train-csv data/raw/train.csv --sample-pct 10 --out-parquet data/interim/train_sample_10pct.parquet`

### 2) Precompute TE cache (recommended)

- `python -m kaggle_clicks.precompute_te --sample-parquet data/interim/train_sample_10pct.parquet --out-parquet data/interim/te_cache/train_sample_10pct_te_m100.parquet --m 100`

### 3) Run the 10% grids (long)

Note: the sweep runner has a safety default `--max-wall-seconds 280` (so it doesn’t accidentally run for hours). For an unattended overnight run, pass a large value (e.g. `--max-wall-seconds 999999`) and set a generous per-run timeout (e.g. `--per-run-timeout-seconds 7200`). If a run is interrupted, re-running the same command will skip already-completed specs.

TE grid:
- `python -m kaggle_clicks.run_sweep_family_a_full_grid --sample-pct 10 --sample-parquet data/interim/train_sample_10pct.parquet --sweep-tag paper_full_grid_10pct --rolling-tail --export-preds --n-jobs 1 --te-parquet data/interim/te_cache/train_sample_10pct_te_m100.parquet`

no-TE grid:
- `python -m kaggle_clicks.run_sweep_family_a_full_grid --sample-pct 10 --sample-parquet data/interim/train_sample_10pct.parquet --sweep-tag paper_full_grid_10pct_noTE_with_preds --rolling-tail --export-preds --n-jobs 1 --no-te --te-parquet data/interim/te_cache/train_sample_10pct_te_m100.parquet`

Tip: for the paper’s paired inference/contrast scripts you only need `preds_test.parquet` (not validation preds). To reduce disk usage and I/O, pass `--export-preds-splits test` to the sweep runner.

### 3.1) Run the TE-only lift runs (long)

These runs generate the TE-only baseline and the two selected time-aggregation specs used for the headline lift figure:

- `python scripts/run_te_lift_v1.py --train-csv data/raw/train.csv --sample-pct 10 --sample-parquet data/interim/train_sample_10pct.parquet --te-parquet data/interim/te_cache/train_sample_10pct_te_m100.parquet --m 100 --n-jobs 1 --export-preds`

### 4) Postprocess into stable artifacts

- `python scripts/postprocess_v1.py --out-dir artifacts/v1`

This step finds the most recent sweep dirs matching the tags above, computes paired inference, and writes:
- `artifacts/v1/10pct_paper_grid/*`
- `artifacts/v1/10pct_paper_grid_noTE/*`
- `artifacts/v1/10pct_te_lift/*`
- `artifacts/v1/combined/*`
- `artifacts/v1/plots/*`

### 5) Generate figures

- `python scripts/generate_paper_plots_v1.py --artifacts-dir artifacts/v1`
- `python scripts/generate_main_figures_time_aggregation_xgb_v1.py --artifacts-dir artifacts/v1 --out-dir artifacts/v1/figures_main`

### 6) Assemble an arXiv bundle folder

- `python scripts/build_arxiv_bundle.py --artifacts-dir artifacts/v1 --out-dir paper_bundle/arxiv_v1`

## Building the paper PDF from sources

This repo includes a prebuilt `paper_bundle/arxiv_v1/paper.pdf`. If you also want to compile the PDF from LaTeX sources, place the LaTeX source tree at:
- `paper_src/` (must contain a `main.tex`)

Then run:
- `python scripts/build_paper.py --paper-src paper_src --out-pdf paper_bundle/arxiv_v1/paper_from_source.pdf`

You will need a LaTeX toolchain available on your machine (e.g., `latexmk`) or run inside a container that provides it.

## Optional: event-count window sensitivity sweep

To reproduce the “vary N” event-count sweep (A3 length tuple), run:
- `make event_count_sweep`

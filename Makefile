.PHONY: help venv verify_env preflight preflight_data validate_v1 data sample10 tecache10 te_lift event_count_sweep grid10_te grid10_note postprocess postprocess_fig234 figures paper arxiv_bundle smoke clean

PY ?= python
ART ?= artifacts/v1
LONG_WALL ?= 999999
PER_RUN_TIMEOUT ?= 7200
TE_N_JOBS ?= 15
NOTE_N_JOBS ?= 1

help:
	@echo "Targets:"
	@echo "  make venv          Create .venv and install pinned deps"
	@echo "  make verify_env    Check installed versions"
	@echo "  make data          Print data download instructions"
	@echo "  make sample10      Build deterministic 10% sample parquet"
	@echo "  make tecache10     Build TE cache parquet for 10% sample"
	@echo "  make te_lift       Run TE-only lift runs (long)"
	@echo "  make event_count_sweep  Run event-count N sweep (long)"
	@echo "  make grid10_te     Run 10% rolling-tail TE+time-agg grid (long)"
	@echo "  make grid10_note   Run 10% rolling-tail no-TE grid (long)"
	@echo "  make postprocess   Build tables from sweep outputs"
	@echo "  make figures       Generate figures from artifacts"
	@echo "  make paper         Build paper PDF (requires LaTeX sources; see docs/REPRODUCE.md)"
	@echo "  make arxiv_bundle  Build an arXiv upload folder (PDF + figures + tables)"
	@echo "  make smoke         Fast 0.1% smoke run (sanity check)"

venv:
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

verify_env:
	$(PY) scripts/verify_env.py

preflight:
	$(PY) scripts/preflight.py --synthetic-smoke

preflight_data:
	$(PY) scripts/preflight.py --check-files

validate_v1:
	$(PY) scripts/validate_v1.py

data:
	@echo "Download Avazu CTR data from Kaggle and place train.csv at data/raw/train.csv"
	@echo "See docs/REPRODUCE.md for details."

sample10:
	$(PY) -m kaggle_clicks.make_sample --train-csv data/raw/train.csv --sample-pct 10 --out-parquet data/interim/train_sample_10pct.parquet

tecache10:
	$(PY) -m kaggle_clicks.precompute_te --sample-parquet data/interim/train_sample_10pct.parquet --out-parquet data/interim/te_cache/train_sample_10pct_te_m100.parquet --m 100

te_lift:
	$(PY) scripts/run_te_lift_v1.py --train-csv data/raw/train.csv --sample-pct 10 --sample-parquet data/interim/train_sample_10pct.parquet --te-parquet data/interim/te_cache/train_sample_10pct_te_m100.parquet --m 100 --n-jobs $(TE_N_JOBS) --export-preds

event_count_sweep:
	$(PY) -m kaggle_clicks.run_sweep_event_counts --train-csv data/raw/train.csv --sample-pct 10 --sample-parquet data/interim/train_sample_10pct.parquet --te-parquet data/interim/te_cache/train_sample_10pct_te_m100.parquet --m 100 --time-agg-windows 1 6 24 48 168 --include-baseline --rolling-tail --export-preds --sweep-tag event_counts_A3_10pct --n-jobs $(TE_N_JOBS) --per-run-timeout-seconds 7200 --resume
	$(PY) -m kaggle_clicks.postprocess_sweep_inference --sweep-dir $$(ls -1dt runs/sweeps/*event_counts_A3_10pct* | head -n 1) --baseline-run-id event0 --split test --bootstrap-reps 200 --seed 42
	$(PY) scripts/plot_event_count_sweep.py --sweep-dir $$(ls -1dt runs/sweeps/*event_counts_A3_10pct* | head -n 1) --out-base artifacts/v1/figures_main/fig_event_count_sweep_A3_10pct

grid10_te:
	$(PY) -m kaggle_clicks.run_sweep_family_a_full_grid --sample-pct 10 --sample-parquet data/interim/train_sample_10pct.parquet --sweep-tag paper_full_grid_10pct --rolling-tail --export-preds --export-preds-splits test --n-jobs $(TE_N_JOBS) --te-parquet data/interim/te_cache/train_sample_10pct_te_m100.parquet --max-wall-seconds $(LONG_WALL) --per-run-timeout-seconds $(PER_RUN_TIMEOUT)

grid10_note:
	$(PY) -m kaggle_clicks.run_sweep_family_a_full_grid --sample-pct 10 --sample-parquet data/interim/train_sample_10pct.parquet --sweep-tag paper_full_grid_10pct_noTE_with_preds --rolling-tail --export-preds --export-preds-splits test --n-jobs $(NOTE_N_JOBS) --no-te --te-parquet data/interim/te_cache/train_sample_10pct_te_m100.parquet --max-wall-seconds $(LONG_WALL) --per-run-timeout-seconds $(PER_RUN_TIMEOUT)

postprocess:
	$(PY) scripts/postprocess_v1.py --out-dir $(ART)

postprocess_fig234:
	$(PY) scripts/postprocess_fig234_v1.py --out-dir $(ART)

figures:
	$(PY) scripts/generate_paper_plots_v1.py --artifacts-dir $(ART)
	$(PY) scripts/generate_main_figures_time_aggregation_xgb_v1.py --artifacts-dir $(ART) --out-dir paper_bundle/arxiv_v1/figures

paper:
	$(PY) scripts/build_paper.py --paper-src paper_src --out-pdf paper_bundle/arxiv_v1/paper_from_source.pdf

arxiv_bundle:
	$(PY) scripts/build_arxiv_bundle.py --artifacts-dir $(ART) --out-dir paper_bundle/arxiv_v1

smoke:
	$(PY) -m kaggle_clicks.run_baseline_te --sample-frac 0.001 --sample-parquet data/interim/train_sample_0p1pct.parquet --run-tag smoke_0p1pct --n-estimators 50 --no-te --time-agg-entities device_ip device_id app_id site_id --time-agg-windows 1 6 24

clean:
	rm -rf artifacts/v1

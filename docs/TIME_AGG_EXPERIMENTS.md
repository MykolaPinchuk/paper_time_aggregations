# Time Aggregation Experiments Plan (Family A)

## Status

Finalized plan (ready for implementation).

## Objective

Systematically test **different constructions of entity-level time aggregation features** for CTR prediction on Avazu with **strict OOT** evaluation and **online** feature simulation (features for hour `H` use only data from `< H`; within-hour updates are batched, but no same-hour leakage).

We will keep the model family fixed (XGBoost) while iterating on feature design to attribute gains to feature choices rather than modeling noise.

Primary metric: ROC-AUC  
Secondary: PR-AUC  

## Fixed Evaluation Protocol (to keep comparisons fair)

### Splits
- Maintain strict chronological splits: `train < val < test`.
- Default: `test = last day`, `val = day before test`, `train = all earlier days`.
- For feature sweeps (optional, recommended): `val = last 2 days before test` (averaged metrics), still OOT w.r.t train.

### Online simulation rule (no leakage)
- For any row at hour `H`, all time aggregation features must be computed from events with hour `< H`.
- During validation/test, features continue to update “online” using earlier hours in the same period (still shifted by 1 hour).

### Run budgets
- Fast iteration: 1% sample for logic, 5% for feature selection.
- Per-run wall time target: < 5 minutes on 5% sample.

## Baseline to Compare Against

We compare every time-aggregation variant against:

1. TE-only baseline (time-aware TE in `kaggle_clicks/te.py`).
2. TE + current Family A implementation (streaming time-aggregations in `kaggle_clicks/time_agg.py`).

All experiments should include a run report (`runs/.../report.md`) and use consistent XGB params unless a change is explicitly part of the experiment.

## Entities (initial scope)

Exactly these four to start:
- `device_ip`
- `device_id`
- `app_id`
- `site_id`

Optional later expansion (only if/when needed): `site_domain`, `app_domain`, or selected `C*` features, but not in the initial time-agg sweep.

## Family A (expanded): “Windowed” History Features

Family A is any feature construction that summarizes an entity’s history using **explicit windows** over the past, under the online constraint (at hour `H`, use only `< H`). This includes more than “trailing `w` hours”.

We keep the initial entities fixed:
- `device_ip`, `device_id`, `app_id`, `site_id`

### A.1 Window *shapes* to test (what gets included)

All windows are expressed in hours because Avazu timestamps are hourly.

1. **Trailing contiguous windows** (current default)
   - Include all events in `[H-w, H)`.
2. **Gap windows** (handles delayed feedback / avoids ultra-recent noise)
   - Include `[H-w-g, H-g)` where `g` is a gap in hours.
   - Test `g ∈ {0, 1, 6}` with a small set of `w` (e.g., 24h and 168h) first.
3. **Bucketized recency windows (piecewise)** (more expressive than a single window)
   - Disjoint buckets like:
     - `0–1h`, `1–6h`, `6–24h`, `1–7d`
   - This acts like a hand-crafted kernel and can outperform overlapping windows while keeping feature count small.
4. **Calendar-aligned windows** (still “windows”, but aligned to boundaries)
   - Examples:
     - “Since start of day”: `[floor_day(H), H)` (intraday accumulation)
     - “Yesterday”: `[floor_day(H)-24h, floor_day(H))`
   - These can capture daily cycles without explicitly conditioning on hour-of-day (kept within Family A).
5. **Event-count windows** (window by last N impressions, not time)
   - For each entity: last `N` impressions (or last `N` hours with activity) prior to `H`.
   - Test small `N` first (e.g., 10, 50, 200) to see if recency-by-activity matters more than wall-clock time.

### A.2 Window *lengths* to test (how far back)

We want a principled schedule that covers short/mid/long recency without exploding feature count:

- Short: `1h, 3h, 6h`
- Medium: `12h, 24h, 48h`
- Long: `96h, 168h (7d)`

Suggested candidate sets (for trailing contiguous windows):
- A1 (compact): `{1, 6, 24}`
- A2 (balanced): `{1, 3, 6, 12, 24}`
- A3 (multi-scale): `{1, 6, 24, 48, 168}`
- A4 (log-spaced): `{1, 2, 4, 8, 16, 24, 48, 96, 168}` (optional; only if budget allows)

### A.3 Aggregation *targets* inside windows (what statistics)

For each entity/window, there are a few “must-have” and several optional stats.

Must-have:
- `imps_w` (or `log1p(imps_w)`): volume / trust
- `ctr_w` (smoothed): `(clicks_w + α) / (imps_w + α + β)`

Optional (test selectively, not all at once):
- `clicks_w` (raw): sometimes helpful beyond CTR when combined with smoothing
- `logit_ctr_w`: `log(ctr_w / (1-ctr_w))` (if stable; clamp)
- Uncertainty / confidence proxy:
  - `inv_sqrt_imps = 1/sqrt(1+imps_w)` or `imps_w` itself may suffice
- “Active hours” within the window:
  - number of distinct hours with impressions (captures burstiness)

Recency (entity-level, independent of window length):
- `recency_seen_hours` (already)
- (later) `recency_click_hours` (arguably Family D, but can be treated as “windowed last-event”)

### A.4 Regularization choices (how to avoid overfitting)

Within Family A we want to test how aggressive smoothing/backoff should be:

1. **Beta prior smoothing** (current): `(clicks+α)/(imps+α+β)`
2. **Adaptive shrinkage with pseudo-count `m`**:
   - `ctr = (clicks + m*prior) / (imps + m)` where `prior` is global CTR to date (online).
3. **Backoff thresholds**:
   - If `imps_w < k`, replace `ctr_w` with `prior` (or a blended version).
   - Test `k ∈ {5, 20, 100}` on 5% sample.

We should not tune all regularization knobs at once; do it after the best window shapes/lengths are known.

## Family A Experiment Design (what runs we do)

We focus exclusively on Family A for now.

### Phase 0 — Reconfirm baseline(s)
- R0: TE only
- R1: TE + A1 trailing windows (current default stats)

### Phase 1 — Window length set sweep (trailing contiguous, no gap)
- R2: TE + A2
- R3: TE + A3
- R4: TE + A4 (only if runtime/memory is acceptable on 5%)

Decision: pick best set by validation ROC-AUC (PR-AUC as tiebreaker).

### Phase 2 — Window shape sweep (using the best length set)
- R5: add **gap** (g=1h) for all windows, compare to g=0
- R6: add **gap** (g=6h) for longer windows only (24h/168h)
- R7: replace overlapping windows with **bucketized** windows (0–1h, 1–6h, 6–24h, 24–168h)
- R8: add **calendar windows** (since-start-of-day, yesterday) in addition to the best trailing set
- R9: add **event-count windows** (N impressions) if we can implement without breaking runtime

### Phase 3 — Regularization sweep (only on the top 1–2 shapes)
- R10: tune smoothing (α/β) coarse grid: `(1,10)`, `(1,50)`, `(5,50)`
- R11: test backoff thresholds `k ∈ {5, 20, 100}`
- R12: test adaptive shrinkage with `m ∈ {20, 100, 500}` (if implemented)

## Correctness / Leakage Checks (mandatory for Family A variants)

For each new window shape:
- Confirm “no same-hour leakage” (features for hour `H` do not include outcomes from hour `H`).
- For gap windows: confirm features at `H` do not include outcomes from `[H-g, H)`.
- For bucketized windows: confirm buckets partition the intended interval with no overlap and no leakage.

## Promotion (when to scale up)

After selecting the top 1–2 Family A variants on 5%:
- Promote to 10% and then full dataset only if runtime remains reasonable and results persist.
- Run any robustness checks (rolling backtest / TSCV) only after the feature design is finalized.

## Reporting Requirements (each run)

Every run must emit `runs/<timestamp>_<tag>/report.md` including:
- dataset sample fraction + row counts
- split time ranges and sizes
- feature family configuration (entities, windows, smoothing)
- train/val/test ROC-AUC and PR-AUC
- top feature importance (gain share)

## Validation policy during the sweep (still applicable)

- 1-day validation (faster) is fine for early screening.
- Once we have a top 2–3 candidates, rerun them with 2-day validation (average metric) before scaling beyond 5%.

## Implementation Notes (for later)

- Prefer a single streaming feature-store API that supports:
  - trailing windows (A1–A4),
  - a gap parameter `g`,
  - bucketized disjoint windows,
  - calendar-aligned windows,
  - event-count windows (optional; may require a separate data structure per entity).
- Keep runtime < 5 minutes at 5% sample; if not, reduce window count and rerun the sweep.

### Current CLI hooks (implemented)

The baseline runner supports Family A variants via `python -m kaggle_clicks.run_baseline_te`:

- Trailing windows: `--time-agg-windows 1 6 24`
- Gap windows: `--time-agg-gap-hours 1` (uses `[H-w-g, H-g)`; feature names get a `_g{g}h` suffix)
- Bucketized windows: `--time-agg-bucket-edges 1 6 24 168` (emits disjoint `b0to1h`, `b1to6h`, …)
- Calendar windows: `--time-agg-calendar` (emits `*_today` and `*_yday`)
- Event-count windows: `--time-agg-event-windows 10 50 200` (approx by-hour blocks; still no same-hour leakage)

### Sweep runner (implemented)

For quick, comparable runs (frozen model params) use:

- `python -m kaggle_clicks.run_sweep_family_a --sample-pct 1 --sweep-tag familyA_phase01_1pct`

This writes a sweep folder under `runs/sweeps/` with `summary.md`, `summary.csv`, and the exact commands used.

# TE-only vs Time-Aggregation Lift (10% sample, rolling-tail Fold A/B)

Selected runs:
- TE-only Fold A: `runs/20251222_040821_paper_teonly_10pct_foldA`
- TE-only Fold B: `runs/20251222_041044_paper_teonly_10pct_foldB`
- A3 trailing Fold A: `runs/20251221_055523_paper_full_grid_10pct_A3_trailing_foldA`
- A3 trailing Fold B: `runs/20251221_060137_paper_full_grid_10pct_A3_trailing_foldB`
- A3 event50 Fold A: `runs/20251221_064852_paper_full_grid_10pct_A3_event50_foldA`
- A3 event50 Fold B: `runs/20251221_065648_paper_full_grid_10pct_A3_event50_foldB`

## Test metrics
| Spec | Fold | Test ROC-AUC | Test PR-AUC |
| --- | --- | ---: | ---: |
| TE_only | A | 0.7417 | 0.3693 |
| TE_only | B | 0.7438 | 0.3480 |
| A3_trailing | A | 0.7483 | 0.3787 |
| A3_trailing | B | 0.7520 | 0.3563 |
| A3_event50 | A | 0.7486 | 0.3784 |
| A3_event50 | B | 0.7524 | 0.3574 |

## Inference (paired, test split)
Columns: Δ = candidate - baseline
| Fold | Candidate | Baseline | ΔROC-AUC (95% CI) | p | ΔPR-AUC (95% CI, bootstrap) |
| --- | --- | --- | --- | ---: | --- |
| A | A3_trailing | TE_only | 0.006560 [0.006001,0.007119] | 0 | 0.009388 [0.007640,0.011148] |
| A | A3_event50 | TE_only | 0.006891 [0.006320,0.007461] | 0 | 0.009193 [0.007614,0.010622] |
| A | A3_event50 | A3_trailing | 0.000331 [0.000042,0.000620] | 0.0249 | -0.000195 [-0.000994,0.000553] |
| B | A3_trailing | TE_only | 0.008159 [0.007476,0.008841] | 0 | 0.008404 [0.006204,0.011044] |
| B | A3_event50 | TE_only | 0.008631 [0.007926,0.009336] | 0 | 0.009562 [0.007194,0.012040] |
| B | A3_event50 | A3_trailing | 0.000472 [0.000182,0.000762] | 0.00141 | 0.001157 [0.000389,0.002093] |

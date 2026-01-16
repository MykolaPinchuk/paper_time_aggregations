# Sweep Summary: `paper_full_grid_10pct`

- Created (UTC): `20260114_131742`
- Sample: `10%` (`data/interim/train_sample_10pct.parquet`)
- Train CSV: `data/raw/train.csv`

| Run | Fold | Description | Return | Skipped | Val ROC-AUC | Test ROC-AUC | Run dir |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
| A1_trailing | A | A1 trailing windows (1, 6, 24) | 0 | no | 0.7544 | 0.7477 | `runs/20260114_131742_paper_full_grid_10pct_A1_trailing_foldA` |
| A1_trailing | B | A1 trailing windows (1, 6, 24) | 0 | no | 0.7636 | 0.7513 | `runs/20260114_132749_paper_full_grid_10pct_A1_trailing_foldB` |
| A1_gap1 | A | A1 gap g=1 windows (1, 6, 24) | 0 | no | 0.7531 | 0.7460 | `runs/20260114_133523_paper_full_grid_10pct_A1_gap1_foldA` |
| A1_gap1 | B | A1 gap g=1 windows (1, 6, 24) | 0 | no | 0.7609 | 0.7500 | `runs/20260114_134428_paper_full_grid_10pct_A1_gap1_foldB` |
| A1_bucket | A | A1 bucket edges (1, 6, 24) | 0 | no | 0.7539 | 0.7467 | `runs/20260114_135239_paper_full_grid_10pct_A1_bucket_foldA` |
| A1_bucket | B | A1 bucket edges (1, 6, 24) | 0 | no | 0.7631 | 0.7509 | `runs/20260114_140221_paper_full_grid_10pct_A1_bucket_foldB` |
| A1_calendar | A | A1 calendar windows on trailing (1, 6, 24) | 0 | no | 0.7533 | 0.7475 | `runs/20260114_141024_paper_full_grid_10pct_A1_calendar_foldA` |
| A1_calendar | B | A1 calendar windows on trailing (1, 6, 24) | 0 | no | 0.7628 | 0.7509 | `runs/20260114_142000_paper_full_grid_10pct_A1_calendar_foldB` |
| A1_event50 | A | A1 event windows last50 | 0 | no | 0.7541 | 0.7480 | `runs/20260114_142854_paper_full_grid_10pct_A1_event50_foldA` |
| A1_event50 | B | A1 event windows last50 | 0 | no | 0.7630 | 0.7527 | `runs/20260114_143900_paper_full_grid_10pct_A1_event50_foldB` |
| A2_trailing | A | A2 trailing windows (1, 3, 6, 12, 24) | 0 | no | 0.7537 | 0.7477 | `runs/20260114_144742_paper_full_grid_10pct_A2_trailing_foldA` |
| A2_trailing | B | A2 trailing windows (1, 3, 6, 12, 24) | 0 | no | 0.7630 | 0.7519 | `runs/20260114_145814_paper_full_grid_10pct_A2_trailing_foldB` |
| A2_gap1 | A | A2 gap g=1 windows (1, 3, 6, 12, 24) | 0 | no | 0.7521 | 0.7447 | `runs/20260114_150749_paper_full_grid_10pct_A2_gap1_foldA` |
| A2_gap1 | B | A2 gap g=1 windows (1, 3, 6, 12, 24) | 0 | no | 0.7612 | 0.7493 | `runs/20260114_151744_paper_full_grid_10pct_A2_gap1_foldB` |
| A2_bucket | A | A2 bucket edges (1, 3, 6, 12, 24) | 0 | no | 0.7536 | 0.7460 | `runs/20260114_152640_paper_full_grid_10pct_A2_bucket_foldA` |
| A2_bucket | B | A2 bucket edges (1, 3, 6, 12, 24) | 0 | no | 0.7623 | 0.7494 | `runs/20260114_153732_paper_full_grid_10pct_A2_bucket_foldB` |
| A2_calendar | A | A2 calendar windows on trailing (1, 3, 6, 12, 24) | 0 | no | 0.7539 | 0.7465 | `runs/20260114_154635_paper_full_grid_10pct_A2_calendar_foldA` |
| A2_calendar | B | A2 calendar windows on trailing (1, 3, 6, 12, 24) | 0 | no | 0.7630 | 0.7512 | `runs/20260114_155928_paper_full_grid_10pct_A2_calendar_foldB` |
| A2_event50 | A | A2 event windows last50 | 0 | no | 0.7539 | 0.7478 | `runs/20260114_160908_paper_full_grid_10pct_A2_event50_foldA` |
| A2_event50 | B | A2 event windows last50 | 0 | no | 0.7626 | 0.7524 | `runs/20260114_161915_paper_full_grid_10pct_A2_event50_foldB` |
| A3_trailing | A | A3 trailing windows (1, 6, 24, 48, 168) | 0 | no | 0.7538 | 0.7483 | `runs/20260114_162915_paper_full_grid_10pct_A3_trailing_foldA` |
| A3_trailing | B | A3 trailing windows (1, 6, 24, 48, 168) | 0 | no | 0.7640 | 0.7515 | `runs/20260114_163941_paper_full_grid_10pct_A3_trailing_foldB` |
| A3_gap1 | A | A3 gap g=1 windows (1, 6, 24, 48, 168) | 0 | no | 0.7524 | 0.7448 | `runs/20260114_164749_paper_full_grid_10pct_A3_gap1_foldA` |
| A3_gap1 | B | A3 gap g=1 windows (1, 6, 24, 48, 168) | 0 | no | 0.7625 | 0.7508 | `runs/20260114_165905_paper_full_grid_10pct_A3_gap1_foldB` |
| A3_bucket | A | A3 bucket edges (1, 6, 24, 48, 168) | 0 | no | 0.7537 | 0.7467 | `runs/20260114_170846_paper_full_grid_10pct_A3_bucket_foldA` |
| A3_bucket | B | A3 bucket edges (1, 6, 24, 48, 168) | 0 | no | 0.7625 | 0.7508 | `runs/20260114_172113_paper_full_grid_10pct_A3_bucket_foldB` |
| A3_calendar | A | A3 calendar windows on trailing (1, 6, 24, 48, 168) | 0 | no | 0.7536 | 0.7480 | `runs/20260114_172951_paper_full_grid_10pct_A3_calendar_foldA` |
| A3_calendar | B | A3 calendar windows on trailing (1, 6, 24, 48, 168) | 0 | no | 0.7642 | 0.7518 | `runs/20260114_174117_paper_full_grid_10pct_A3_calendar_foldB` |
| A3_event50 | A | A3 event windows last50 | 0 | no | 0.7551 | 0.7486 | `runs/20260114_175249_paper_full_grid_10pct_A3_event50_foldA` |
| A3_event50 | B | A3 event windows last50 | 0 | no | 0.7638 | 0.7524 | `runs/20260114_180622_paper_full_grid_10pct_A3_event50_foldB` |
| A4_trailing | A | A4 trailing windows (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7538 | 0.7464 | `runs/20260114_181732_paper_full_grid_10pct_A4_trailing_foldA` |
| A4_trailing | B | A4 trailing windows (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7635 | 0.7521 | `runs/20260114_183102_paper_full_grid_10pct_A4_trailing_foldB` |
| A4_gap1 | A | A4 gap g=1 windows (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7525 | 0.7451 | `runs/20260114_184258_paper_full_grid_10pct_A4_gap1_foldA` |
| A4_gap1 | B | A4 gap g=1 windows (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7626 | 0.7510 | `runs/20260114_185654_paper_full_grid_10pct_A4_gap1_foldB` |
| A4_bucket | A | A4 bucket edges (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7537 | 0.7465 | `runs/20260114_191104_paper_full_grid_10pct_A4_bucket_foldA` |
| A4_bucket | B | A4 bucket edges (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7619 | 0.7508 | `runs/20260114_192850_paper_full_grid_10pct_A4_bucket_foldB` |
| A4_calendar | A | A4 calendar windows on trailing (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7536 | 0.7473 | `runs/20260114_194245_paper_full_grid_10pct_A4_calendar_foldA` |
| A4_calendar | B | A4 calendar windows on trailing (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7639 | 0.7519 | `runs/20260114_195709_paper_full_grid_10pct_A4_calendar_foldB` |
| A4_event50 | A | A4 event windows last50 | 0 | no | 0.7535 | 0.7475 | `runs/20260114_201044_paper_full_grid_10pct_A4_event50_foldA` |
| A4_event50 | B | A4 event windows last50 | 0 | no | 0.7639 | 0.7517 | `runs/20260114_202325_paper_full_grid_10pct_A4_event50_foldB` |

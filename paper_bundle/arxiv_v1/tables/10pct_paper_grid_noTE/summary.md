# Sweep Summary: `paper_full_grid_10pct_noTE_with_preds`

- Created (UTC): `20260114_041514`
- Sample: `10%` (`data/interim/train_sample_10pct.parquet`)
- Train CSV: `data/raw/train.csv`

| Run | Fold | Description | Return | Skipped | Val ROC-AUC | Test ROC-AUC | Run dir |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
| A1_trailing | A | A1 trailing windows (1, 6, 24) | 0 | no | 0.7388 | 0.7371 | `runs/20260114_041514_paper_full_grid_10pct_noTE_with_preds_A1_trailing_foldA` |
| A1_trailing | B | A1 trailing windows (1, 6, 24) | 0 | no | 0.7498 | 0.7386 | `runs/20260114_041924_paper_full_grid_10pct_noTE_with_preds_A1_trailing_foldB` |
| A1_gap1 | A | A1 gap g=1 windows (1, 6, 24) | 0 | no | 0.7363 | 0.7340 | `runs/20260114_042354_paper_full_grid_10pct_noTE_with_preds_A1_gap1_foldA` |
| A1_gap1 | B | A1 gap g=1 windows (1, 6, 24) | 0 | no | 0.7458 | 0.7354 | `runs/20260114_042916_paper_full_grid_10pct_noTE_with_preds_A1_gap1_foldB` |
| A1_bucket | A | A1 bucket edges (1, 6, 24) | 0 | no | 0.7386 | 0.7362 | `runs/20260114_043330_paper_full_grid_10pct_noTE_with_preds_A1_bucket_foldA` |
| A1_bucket | B | A1 bucket edges (1, 6, 24) | 0 | no | 0.7485 | 0.7381 | `runs/20260114_043826_paper_full_grid_10pct_noTE_with_preds_A1_bucket_foldB` |
| A1_calendar | A | A1 calendar windows on trailing (1, 6, 24) | 0 | no | 0.7393 | 0.7379 | `runs/20260114_044320_paper_full_grid_10pct_noTE_with_preds_A1_calendar_foldA` |
| A1_calendar | B | A1 calendar windows on trailing (1, 6, 24) | 0 | no | 0.7502 | 0.7384 | `runs/20260114_044851_paper_full_grid_10pct_noTE_with_preds_A1_calendar_foldB` |
| A1_event50 | A | A1 event windows last50 | 0 | no | 0.7420 | 0.7394 | `runs/20260114_045340_paper_full_grid_10pct_noTE_with_preds_A1_event50_foldA` |
| A1_event50 | B | A1 event windows last50 | 0 | no | 0.7535 | 0.7416 | `runs/20260114_045857_paper_full_grid_10pct_noTE_with_preds_A1_event50_foldB` |
| A2_trailing | A | A2 trailing windows (1, 3, 6, 12, 24) | 0 | no | 0.7386 | 0.7370 | `runs/20260114_050341_paper_full_grid_10pct_noTE_with_preds_A2_trailing_foldA` |
| A2_trailing | B | A2 trailing windows (1, 3, 6, 12, 24) | 0 | no | 0.7501 | 0.7383 | `runs/20260114_050902_paper_full_grid_10pct_noTE_with_preds_A2_trailing_foldB` |
| A2_gap1 | A | A2 gap g=1 windows (1, 3, 6, 12, 24) | 0 | no | 0.7357 | 0.7336 | `runs/20260114_051416_paper_full_grid_10pct_noTE_with_preds_A2_gap1_foldA` |
| A2_gap1 | B | A2 gap g=1 windows (1, 3, 6, 12, 24) | 0 | no | 0.7446 | 0.7345 | `runs/20260114_052014_paper_full_grid_10pct_noTE_with_preds_A2_gap1_foldB` |
| A2_bucket | A | A2 bucket edges (1, 3, 6, 12, 24) | 0 | no | 0.7377 | 0.7358 | `runs/20260114_052547_paper_full_grid_10pct_noTE_with_preds_A2_bucket_foldA` |
| A2_bucket | B | A2 bucket edges (1, 3, 6, 12, 24) | 0 | no | 0.7465 | 0.7370 | `runs/20260114_053215_paper_full_grid_10pct_noTE_with_preds_A2_bucket_foldB` |
| A2_calendar | A | A2 calendar windows on trailing (1, 3, 6, 12, 24) | 0 | no | 0.7401 | 0.7382 | `runs/20260114_053759_paper_full_grid_10pct_noTE_with_preds_A2_calendar_foldA` |
| A2_calendar | B | A2 calendar windows on trailing (1, 3, 6, 12, 24) | 0 | no | 0.7499 | 0.7385 | `runs/20260114_054646_paper_full_grid_10pct_noTE_with_preds_A2_calendar_foldB` |
| A2_event50 | A | A2 event windows last50 | 0 | no | 0.7425 | 0.7397 | `runs/20260114_055258_paper_full_grid_10pct_noTE_with_preds_A2_event50_foldA` |
| A2_event50 | B | A2 event windows last50 | 0 | no | 0.7537 | 0.7418 | `runs/20260114_060117_paper_full_grid_10pct_noTE_with_preds_A2_event50_foldB` |
| A3_trailing | A | A3 trailing windows (1, 6, 24, 48, 168) | 0 | no | 0.7420 | 0.7392 | `runs/20260114_060751_paper_full_grid_10pct_noTE_with_preds_A3_trailing_foldA` |
| A3_trailing | B | A3 trailing windows (1, 6, 24, 48, 168) | 0 | no | 0.7548 | 0.7407 | `runs/20260114_061342_paper_full_grid_10pct_noTE_with_preds_A3_trailing_foldB` |
| A3_gap1 | A | A3 gap g=1 windows (1, 6, 24, 48, 168) | 0 | no | 0.7395 | 0.7365 | `runs/20260114_061907_paper_full_grid_10pct_noTE_with_preds_A3_gap1_foldA` |
| A3_gap1 | B | A3 gap g=1 windows (1, 6, 24, 48, 168) | 0 | no | 0.7514 | 0.7391 | `runs/20260114_062521_paper_full_grid_10pct_noTE_with_preds_A3_gap1_foldB` |
| A3_bucket | A | A3 bucket edges (1, 6, 24, 48, 168) | 0 | no | 0.7411 | 0.7386 | `runs/20260114_063119_paper_full_grid_10pct_noTE_with_preds_A3_bucket_foldA` |
| A3_bucket | B | A3 bucket edges (1, 6, 24, 48, 168) | 0 | no | 0.7519 | 0.7402 | `runs/20260114_063836_paper_full_grid_10pct_noTE_with_preds_A3_bucket_foldB` |
| A3_calendar | A | A3 calendar windows on trailing (1, 6, 24, 48, 168) | 0 | no | 0.7418 | 0.7391 | `runs/20260114_064533_paper_full_grid_10pct_noTE_with_preds_A3_calendar_foldA` |
| A3_calendar | B | A3 calendar windows on trailing (1, 6, 24, 48, 168) | 0 | no | 0.7548 | 0.7414 | `runs/20260114_065209_paper_full_grid_10pct_noTE_with_preds_A3_calendar_foldB` |
| A3_event50 | A | A3 event windows last50 | 0 | no | 0.7427 | 0.7399 | `runs/20260114_065916_paper_full_grid_10pct_noTE_with_preds_A3_event50_foldA` |
| A3_event50 | B | A3 event windows last50 | 0 | no | 0.7551 | 0.7422 | `runs/20260114_070556_paper_full_grid_10pct_noTE_with_preds_A3_event50_foldB` |
| A4_trailing | A | A4 trailing windows (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7417 | 0.7392 | `runs/20260114_071229_paper_full_grid_10pct_noTE_with_preds_A4_trailing_foldA` |
| A4_trailing | B | A4 trailing windows (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7546 | 0.7414 | `runs/20260114_072046_paper_full_grid_10pct_noTE_with_preds_A4_trailing_foldB` |
| A4_gap1 | A | A4 gap g=1 windows (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7396 | 0.7367 | `runs/20260114_072902_paper_full_grid_10pct_noTE_with_preds_A4_gap1_foldA` |
| A4_gap1 | B | A4 gap g=1 windows (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7510 | 0.7394 | `runs/20260114_073819_paper_full_grid_10pct_noTE_with_preds_A4_gap1_foldB` |
| A4_bucket | A | A4 bucket edges (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7412 | 0.7380 | `runs/20260114_074732_paper_full_grid_10pct_noTE_with_preds_A4_bucket_foldA` |
| A4_bucket | B | A4 bucket edges (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7498 | 0.7391 | `runs/20260114_080043_paper_full_grid_10pct_noTE_with_preds_A4_bucket_foldB` |
| A4_calendar | A | A4 calendar windows on trailing (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7416 | 0.7391 | `runs/20260114_081017_paper_full_grid_10pct_noTE_with_preds_A4_calendar_foldA` |
| A4_calendar | B | A4 calendar windows on trailing (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7547 | 0.7410 | `runs/20260114_082001_paper_full_grid_10pct_noTE_with_preds_A4_calendar_foldB` |
| A4_event50 | A | A4 event windows last50 | 0 | no | 0.7429 | 0.7399 | `runs/20260114_082938_paper_full_grid_10pct_noTE_with_preds_A4_event50_foldA` |
| A4_event50 | B | A4 event windows last50 | 0 | no | 0.7546 | 0.7425 | `runs/20260114_083911_paper_full_grid_10pct_noTE_with_preds_A4_event50_foldB` |

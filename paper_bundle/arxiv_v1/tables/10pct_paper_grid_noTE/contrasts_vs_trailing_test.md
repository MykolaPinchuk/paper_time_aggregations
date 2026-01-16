# Contrasts vs `trailing` (test, mean over folds)

- Sweep: `20260114_041514_paper_full_grid_10pct_noTE_with_preds_s10pct`
- Baseline within each length set: `trailing`
- Shapes compared: gap1, bucket, calendar, event50
- PR-AUC bootstrap: not computed

| Length set | Shape | ΔROC-AUC | 95% CI |
| --- | --- | ---: | --- |
| A1 | bucket | -0.000712 | [-0.001021, -0.000402] |
| A2 | bucket | -0.001233 | [-0.001613, -0.000852] |
| A3 | bucket | -0.000537 | [-0.000953, -0.000121] |
| A4 | bucket | -0.001746 | [-0.002211, -0.001280] |
| A1 | calendar | 0.000301 | [0.000085, 0.000516] |
| A2 | calendar | 0.000741 | [0.000510, 0.000972] |
| A3 | calendar | 0.000313 | [0.000152, 0.000473] |
| A4 | calendar | -0.000215 | [-0.000393, -0.000037] |
| A1 | event50 | 0.002594 | [0.002213, 0.002975] |
| A2 | event50 | 0.003114 | [0.002702, 0.003526] |
| A3 | event50 | 0.001126 | [0.000886, 0.001366] |
| A4 | event50 | 0.000905 | [0.000661, 0.001149] |
| A1 | gap1 | -0.003172 | [-0.003615, -0.002728] |
| A2 | gap1 | -0.003578 | [-0.003991, -0.003165] |
| A3 | gap1 | -0.002170 | [-0.002555, -0.001785] |
| A4 | gap1 | -0.002215 | [-0.002594, -0.001837] |

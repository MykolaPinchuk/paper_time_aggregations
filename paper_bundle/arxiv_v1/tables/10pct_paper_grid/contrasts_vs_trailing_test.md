# Contrasts vs `trailing` (test, mean over folds)

- Sweep: `20260114_131742_paper_full_grid_10pct_s10pct`
- Baseline within each length set: `trailing`
- Shapes compared: gap1, bucket, calendar, event50
- PR-AUC bootstrap: not computed

| Length set | Shape | ΔROC-AUC | 95% CI |
| --- | --- | ---: | --- |
| A1 | bucket | -0.000684 | [-0.001023, -0.000345] |
| A2 | bucket | -0.002102 | [-0.002487, -0.001716] |
| A3 | bucket | -0.001151 | [-0.001524, -0.000778] |
| A4 | bucket | -0.000638 | [-0.001087, -0.000189] |
| A1 | calendar | -0.000323 | [-0.000607, -0.000040] |
| A2 | calendar | -0.000916 | [-0.001204, -0.000628] |
| A3 | calendar | 0.000015 | [-0.000226, 0.000255] |
| A4 | calendar | 0.000361 | [0.000094, 0.000628] |
| A1 | event50 | 0.000830 | [0.000504, 0.001157] |
| A2 | event50 | 0.000329 | [0.000019, 0.000638] |
| A3 | event50 | 0.000574 | [0.000286, 0.000862] |
| A4 | event50 | 0.000354 | [0.000023, 0.000685] |
| A1 | gap1 | -0.001492 | [-0.001857, -0.001128] |
| A2 | gap1 | -0.002756 | [-0.003129, -0.002383] |
| A3 | gap1 | -0.002056 | [-0.002440, -0.001672] |
| A4 | gap1 | -0.001184 | [-0.001579, -0.000789] |

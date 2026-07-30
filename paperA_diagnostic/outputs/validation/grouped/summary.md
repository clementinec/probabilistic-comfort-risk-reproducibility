# Exact-spec contributor-grouped validation

This revision analysis repeats five contributor-disjoint train/calibration/test splits with the same 400-tree ordinal LightGBM specification used by the Paper A application model. A predicted tail event is defined consistently with the manuscript as \(p_{\mathrm{tail}}\ge0.20\).

Run command:

```text
python paperA_R01/04_analysis/scripts/grouped_tsv_validation_exact.py
```

## Aggregate results

| Feature set | Exact | Within 1 | Class MAE | Tail Brier | Tail ECE | Tail AUROC | Tail AP | F1 at 0.20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Primary | 42.1% (2.8) | 79.8% (2.1) | 0.832 (0.044) | 0.174 (0.012) | 0.061 (0.025) | 0.577 (0.021) | 0.290 (0.035) | 32.5% (4.3) |
| No PMV | 41.9% (2.9) | 79.4% (1.7) | 0.839 (0.038) | 0.174 (0.012) | 0.058 (0.022) | 0.580 (0.020) | 0.285 (0.026) | 34.2% (4.5) |

Values are mean (sample standard deviation) across five splits. Each split holds out 10 complete contributor groups; test-set sizes vary because contributor groups vary substantially in size.

## Interpretation

Contributor transfer is weaker than the pooled record-level split, particularly for tail discrimination. The primary and no-PMV feature sets remain similar. These results support conditional use of the predicted probabilities and do not establish unrestricted external validity.

The earlier exploratory table used a different tree count and classified tail events from the predicted argmax category. This output supersedes that table by matching both the deployed model specification and the manuscript's probability-screen definition.

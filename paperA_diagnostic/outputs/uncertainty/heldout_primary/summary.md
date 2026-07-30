# Contributor-clustered held-out uncertainty

- Exact held-out records: 22,223
- Contributor clusters: 63
- Bootstrap draws: 5,000
- Seed: `20260729`
- Intervals: empirical 2.5th and 97.5th percentiles.
- Resampling: contributors are sampled with replacement; every held-out record belonging to a selected contributor receives the same multiplicity.

## Metrics

| Predictor | Brier | ECE | AUROC | Average precision | Observed prevalence | Mean predicted |
|---|---:|---:|---:|---:|---:|---:|
| Primary TSV model | 0.137 [0.117, 0.156] | 0.011 [0.008, 0.021] | 0.747 [0.713, 0.778] | 0.487 [0.420, 0.547] | 0.206 [0.169, 0.238] | 0.207 [0.183, 0.229] |
| No-PMV TSV model | 0.138 [0.117, 0.156] | 0.006 [0.006, 0.020] | 0.747 [0.712, 0.778] | 0.483 [0.417, 0.541] | 0.206 [0.169, 0.238] | 0.207 [0.182, 0.229] |
| PMV-only calibrated tail baseline | 0.161 [0.140, 0.180] | 0.003 [0.004, 0.044] | 0.571 [0.535, 0.623] | 0.260 [0.212, 0.317] | 0.206 [0.169, 0.238] | 0.206 [0.198, 0.215] |

## Paired differences

- Primary minus PMV-only Brier (negative favors primary): -0.023 [-0.029, -0.018].
- Primary minus PMV-only AUROC (positive favors primary): 0.177 [0.138, 0.203].
- Primary minus PMV-only average precision (positive favors primary): 0.227 [0.186, 0.260].
- Primary minus no-PMV Brier: -0.000 [-0.001, 0.000].
- Primary minus no-PMV AUROC: 0.001 [-0.002, 0.003].
- Primary minus no-PMV average precision: 0.004 [-0.001, 0.010].

## Interpretation boundary

The original split is stratified at the record level, so the same contributors can occur in fitting and held-out partitions. Contributor-clustered intervals prevent repeated votes from being treated as independent for interval calculation, but they do not establish contributor-level transport. The separate grouped and cross-corpus validations address that harder question.

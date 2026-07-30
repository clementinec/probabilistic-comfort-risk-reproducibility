# Contributor-clustered endpoint-only held-out uncertainty

- Event: observed or predicted probability of `TSV in {-3,+3}`.
- Exact held-out records: 22,223
- Observed endpoint records: 1,350 (`TSV=-3`: 443; `TSV=+3`: 907)
- Contributor clusters: 63
- Bootstrap draws: 5,000
- Seed: `20260729`
- Interpretation: support-limited bounding sensitivity; not a replacement outcome and not an occupant-dissatisfaction rate.

## Metrics

| Predictor | Brier | ECE | AUROC | Average precision | Observed prevalence | Mean predicted |
|---|---:|---:|---:|---:|---:|---:|
| Primary TSV model | 0.047 [0.034, 0.057] | 0.003 [0.002, 0.011] | 0.819 [0.773, 0.858] | 0.370 [0.250, 0.481] | 0.061 [0.043, 0.077] | 0.061 [0.049, 0.073] |
| No-PMV TSV model | 0.047 [0.035, 0.057] | 0.004 [0.002, 0.010] | 0.822 [0.775, 0.862] | 0.369 [0.247, 0.485] | 0.061 [0.043, 0.077] | 0.061 [0.049, 0.074] |
| PMV-only calibrated tail baseline | 0.056 [0.041, 0.069] | 0.003 [0.003, 0.021] | 0.638 [0.571, 0.713] | 0.110 [0.068, 0.172] | 0.061 [0.043, 0.077] | 0.061 [0.055, 0.067] |

## Paired differences

- Primary minus PMV-only Brier (negative favors primary): -0.009 [-0.015, -0.005].
- Primary minus PMV-only AUROC (positive favors primary): 0.181 [0.131, 0.221].
- Primary minus PMV-only average precision (positive favors primary): 0.260 [0.171, 0.328].
- Primary minus no-PMV Brier: 0.000 [-0.000, 0.000].
- Primary minus no-PMV AUROC: -0.003 [-0.008, 0.001].
- Primary minus no-PMV average precision: 0.001 [-0.008, 0.009].

## Interpretation boundary

The endpoints have materially less support than the primary outer-category event. Contributor clustering addresses repeated records within the test set but does not propagate model-refit uncertainty or establish transport to a new corpus.

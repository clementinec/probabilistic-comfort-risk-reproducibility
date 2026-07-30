# Corrected same-state endpoint panel uncertainty

- Event: `P(TSV=-3) + P(TSV=+3)`.
- Timing provenance: corrected same-state inference from each recorded row.
- Bootstrap draws: 10,000
- Seed: `20260729`
- Role-weighted resampling: 48 city×scenario×time-slice blocks, with all three selector roles retained.
- Unique-year sensitivity: 119 distinct source years receive equal weight and remain nested within the 48 parent design blocks.

## Spatial screen results

| Endpoint probability screen | Mean-zone exceedance (%) | Any-zone exceedance (%) | Any-minus-mean gap (pp) | Hidden any-zone (%) | Gap lower bound > 0? |
|---:|---:|---:|---:|---:|:---:|
| 0.025 | 33.30 [30.22, 36.37] | 69.33 [67.04, 71.59] | 36.03 [34.58, 37.48] | 36.03 [34.58, 37.48] | yes |
| 0.050 | 10.99 [9.14, 12.98] | 30.34 [26.65, 34.00] | 19.35 [17.26, 21.37] | 19.35 [17.26, 21.37] | yes |
| 0.075 | 3.79 [2.76, 4.96] | 23.87 [20.61, 27.12] | 20.08 [17.51, 22.58] | 20.08 [17.51, 22.58] | yes |
| 0.100 | 1.42 [0.89, 2.04] | 20.30 [17.24, 23.38] | 18.88 [16.14, 21.56] | 18.88 [16.14, 21.56] | yes |
| 0.150 | 0.12 [0.06, 0.19] | 7.65 [5.94, 9.42] | 7.53 [5.84, 9.28] | 7.53 [5.84, 9.28] | yes |
| 0.200 | 0.04 [0.02, 0.07] | 0.74 [0.44, 1.09] | 0.70 [0.41, 1.04] | 0.70 [0.41, 1.04] | yes |

## Equal-source-year check

| Screen | Role-weighted gap (pp) | Equal-source-year gap (pp) |
|---:|---:|---:|
| 0.025 | 36.03 [34.58, 37.48] | 36.21 [34.89, 37.57] |
| 0.050 | 19.35 [17.26, 21.37] | 18.90 [16.97, 20.91] |
| 0.075 | 20.08 [17.51, 22.58] | 19.49 [17.16, 21.92] |
| 0.100 | 18.88 [16.14, 21.56] | 18.13 [15.72, 20.70] |
| 0.150 | 7.53 [5.84, 9.28] | 6.93 [5.37, 8.72] |
| 0.200 | 0.70 [0.41, 1.04] | 0.63 [0.35, 1.01] |

## Late-2080s minus baseline contrast

| Screen | Mean-zone contrast (pp) | Any-zone contrast (pp) |
|---:|---:|---:|
| 0.025 | 11.81 [7.60, 16.69] | 7.72 [4.19, 11.67] |
| 0.050 | 9.75 [6.56, 13.47] | 12.67 [8.07, 18.06] |
| 0.075 | 6.39 [4.23, 8.83] | 11.66 [7.29, 16.78] |
| 0.100 | 3.21 [1.82, 4.80] | 11.43 [7.17, 16.27] |
| 0.150 | 0.32 [0.11, 0.53] | 5.25 [2.80, 7.92] |
| 0.200 | 0.07 [-0.01, 0.16] | 1.23 [0.32, 2.31] |

## Interpretation boundary

This is a support-limited bounding sensitivity. The conditional block-bootstrap intervals quantify variation across the selected case design; they do not turn endpoint probability into observed dissatisfaction or quantify climate-model, building, or occupant structural uncertainty.

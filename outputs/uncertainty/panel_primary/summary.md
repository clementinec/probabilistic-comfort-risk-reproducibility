# Clustered panel uncertainty audit

- Bootstrap draws: 10,000
- Seed: `20260729`
- Timing provenance: `corrected_same_state`
- Intervals: empirical 2.5th and 97.5th percentiles.
- Primary global/spatial resampling unit: 48 city×scenario×time-slice blocks; all three selector roles travel together.
- Paired time-slice resampling unit: 12 city×scenario blocks; all four slices and selector roles travel together.
- Separate sensitivity: 119 distinct source-year trajectories with equal source-year weight, retained within 48 parent design blocks during resampling.

## Headline role-weighted results

- Mean-zone high-tail exceedance (%): 13.15 [11.02, 15.45].
- Any-zone high-tail exceedance (%): 36.64 [32.39, 40.87].
- Any-zone minus mean-zone gap (percentage points): 23.50 [20.79, 26.12].
- Hidden any-zone exceedance (%): 23.50 [20.79, 26.12].

## Paired contrasts in high-tail exceedance

- Near-2030s minus baseline (percentage points): -0.02 [-2.00, 1.70].
- Mid-2050s minus baseline (percentage points): 5.05 [1.81, 9.03].
- Late-2080s minus baseline (percentage points): 11.93 [8.10, 16.31].
- SSP5-8.5 minus SSP2-4.5 (percentage points): 3.40 [2.28, 4.37]. The scenario interval resamples only six paired city blocks and must not be read as regional climate uncertainty.

## Equal-source-year sensitivity

- Mean-zone high-tail exceedance (%): 12.44 [10.60, 14.47].
- Any-zone high-tail exceedance (%): 35.41 [31.66, 39.31].
- Hidden any-zone exceedance (%): 22.98 [20.55, 25.52].

## Interpretation boundary

These intervals quantify sensitivity to which existing case blocks or source years receive weight. They do not include climate-model ensemble spread, weather-file construction error, building-model structural error, predictor transport error, or future occupant adaptation.

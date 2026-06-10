# R2 Reproducibility Manifest

Date: 2026-06-10

This addendum records the second-revision diagnostic additions prepared for `BAE-D-26-04563R1`.

## New Diagnostic Script

- `scripts/build_zone_and_tail_diagnostics.py`: builds the zone-aggregation, hidden-tail, and tail-steering coherence diagnostics from the Medium Office future-weather traces. In this public repository layout, it reads `traces/*_ordinal.csv.gz` directly.
- `scripts/compute_oat_sensitivity_summary.py`: verifies the compact annual OAT max-elasticity ranking and reports the archived EUI coefficient of variation used in Supplementary Section S1.3.

## Modified Trace Generator

- `scripts/rerun_medium_office_temporal_R2_trace_export.py`: extended to save per-zone expected TSV, discomfort probability, warm-tail probability, cold-tail probability, and directional tail dominance before mean aggregation.

## New Diagnostic Outputs

- `summary_outputs/annual_oat_elasticity_summary.csv`: compact archived OAT elasticity matrix for the Los Angeles annual controller sensitivity sweep.
- `summary_outputs/annual_oat_eui_cv_summary.csv`: compact archived EUI coefficient-of-variation summary for the Los Angeles annual OAT sweep.
- `summary_outputs/zone_aggregation_summary.csv`: city/year summary comparing mean-zone, any-zone, and 90th-percentile zone aggregation.
- `summary_outputs/zone_risk_ranking.csv`: zone-level risk ranking by discomfort-tail exposure.
- `summary_outputs/hidden_zone_tail_examples.csv`: examples where mean aggregation hides at least one high-tail zone.
- `summary_outputs/zone_aggregation_rows_compact.csv.gz`: gzip-compressed compact per-timestep diagnostic table used to generate the R2 aggregation summaries. Running `scripts/build_zone_and_tail_diagnostics.py` regenerates the uncompressed CSV.
- `summary_outputs/tail_steering_action_conditioned_summary.csv`: action-conditioned `mu_TSV` versus `d_tail` sign-coherence summary.
- `summary_outputs/stress_window_warm_tail_threshold_sensitivity.csv`: offline sensitivity check for the Phoenix stress-window warm-tail limiter at lower thresholds 0.15, 0.20, and 0.25 with the rejection threshold held at 0.35.
- `summary_outputs/zone_tail_diagnostic_summary.md`: human-readable diagnostic summary.
- `figures/zone_aggregation_diagnostic.pdf`: manuscript figure source for the zone-aggregation diagnostic.
- `figures/zone_aggregation_diagnostic.png`: raster copy of the zone-aggregation diagnostic.

## R2 Trace Files

- `traces/*_ordinal.csv.gz`: gzip-compressed city-year ordinal traces for Beijing, Houston, and Phoenix in 2025, 2050, and 2075. These files include the per-zone probabilistic outputs needed by the R2 diagnostics.
- The redundant combined file `medium_office_control_traces.csv` is not tracked; it is reconstructable from the nine city-year trace files and exceeded the desired single-file size for normal GitHub storage.

## Model Artifacts

- `models/control_predictors.joblib`: trained control predictor bundle used by the R2 trace rerun workflow.
- `models/control_predictor_metrics.json`: predictor/controller summary metrics.

Third-party ASHRAE Global Thermal Comfort Database II records are not redistributed; access remains subject to the original database terms of use.

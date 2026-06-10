# Reproducibility Package Manifest

Date: 2026-06-10

Package size: approximately 81 MB before Git metadata.

This package is suitable for a public GitHub repository. It contains processed diagnostic outputs, plotting/diagnostic scripts, generated figures, configuration artifacts, the R2 trained model artifact, and gzip-compressed R2 Medium Office city-year traces. It does not redistribute third-party thermal-comfort data or source weather archives.

## Included Files

### Root

- `README.md`: package overview, data restrictions, and repository plan.
- `MANIFEST.md`: this manifest.
- `R2_REPRODUCIBILITY_MANIFEST.md`: second-revision addendum for the zone-aggregation and tail-steering diagnostics.
- `CHECKSUMS_SHA256.txt`: SHA-256 checksums for tracked package files.
- `.gitignore`: excludes local plotting cache artifacts.

### `configs/`

- `control_predictor_metrics.json`: predictor/controller summary metrics.
- `medium_office_otc_control.idf`: Medium Office control IDF/configuration artifact used in the diagnostic workflow.

### `figures/`

- `city_year_tail_trend.png`: city-year high-tail trend figure.
- `probability_necessity_manuscript_figure.pdf`: manuscript probability-necessity diagnostic figure.
- `probability_necessity_manuscript_figure.png`: PNG version of the same diagnostic figure.
- `tail_spread_by_mean_bin.png`: mean-bin tail-spread diagnostic figure.
- `zone_aggregation_diagnostic.pdf`: R2 zone-aggregation diagnostic figure.
- `zone_aggregation_diagnostic.png`: PNG version of the same diagnostic figure.

### `scripts/`

- `convert_cmip_csv_to_epw.py`: CMIP-derived CSV to EPW conversion utility.
- `probability_necessity_diagnostic.py`: probability-threshold diagnostic and figure-generation script.
- `build_zone_and_tail_diagnostics.py`: R2 zone-aggregation and tail-steering diagnostic script.
- `rerun_medium_office_temporal_R2_trace_export.py`: R2 Medium Office trace-generation workflow with per-zone probability export.

### `summary_outputs/`

- `boundary_region_summary.csv`: boundary-region summary for the expected-TSV risk band.
- `directional_tail_diagnostics.csv`: directional warm/cold tail diagnostics.
- `matched_mean_tail_divergence.csv`: matched-mean tail-divergence examples.
- `mean_threshold_decision_diagnostics.csv`: mean-only threshold disagreement diagnostics.
- `medium_office_trace_summary.csv`: Medium Office trace summary.
- `microgrid_event_summary_compact.tex`: compact stress-window event summary table.
- `probability_necessity_summary.csv`: city-year probability-necessity summary.
- `same_city_2025_2075_same_mean_examples.csv`: same-city matched-mean examples across years.
- `tail_spread_by_mean_bin.csv`: tail-spread summaries by expected-TSV bin.
- `hidden_zone_tail_examples.csv`: R2 examples where mean aggregation hides a high-tail zone.
- `tail_steering_action_conditioned_summary.csv`: R2 action-conditioned mean-TSV and `d_tail` sign-coherence summary.
- `zone_aggregation_rows_compact.csv.gz`: gzip-compressed R2 compact per-timestep zone-aggregation diagnostic table.
- `zone_aggregation_summary.csv`: R2 city/year mean-zone, p90-zone, and any-zone aggregation summary.
- `zone_risk_ranking.csv`: R2 zone-level risk ranking by discomfort-tail exposure.
- `zone_tail_diagnostic_summary.md`: R2 human-readable diagnostic summary.

### `models/`

- `control_predictors.joblib`: trained control predictor bundle used by the R2 trace rerun workflow.
- `control_predictor_metrics.json`: predictor/controller summary metrics.

### `traces/`

- `README.md`: trace-file notes and reconstruction guidance.
- `beijing_ssp585_cmip_direct_2025_ordinal.csv.gz`: compressed R2 Beijing 2025 ordinal trace.
- `beijing_ssp585_cmip_direct_2050_ordinal.csv.gz`: compressed R2 Beijing 2050 ordinal trace.
- `beijing_ssp585_cmip_direct_2075_ordinal.csv.gz`: compressed R2 Beijing 2075 ordinal trace.
- `houston_ssp585_cmip_direct_2025_ordinal.csv.gz`: compressed R2 Houston 2025 ordinal trace.
- `houston_ssp585_cmip_direct_2050_ordinal.csv.gz`: compressed R2 Houston 2050 ordinal trace.
- `houston_ssp585_cmip_direct_2075_ordinal.csv.gz`: compressed R2 Houston 2075 ordinal trace.
- `phoenix_ssp585_cmip_direct_2025_ordinal.csv.gz`: compressed R2 Phoenix 2025 ordinal trace.
- `phoenix_ssp585_cmip_direct_2050_ordinal.csv.gz`: compressed R2 Phoenix 2050 ordinal trace.
- `phoenix_ssp585_cmip_direct_2075_ordinal.csv.gz`: compressed R2 Phoenix 2075 ordinal trace.

## Not Included

- ASHRAE Global Thermal Comfort Database II: third-party dataset; users must obtain it from the original source and comply with its terms.
- CMIP-derived source weather CSVs and generated EPWs: not bundled in this lightweight package; provenance and conversion assumptions are documented in the manuscript and README.
- Redundant combined R2 trace file `medium_office_control_traces.csv`: not tracked because it can be reconstructed from `traces/*_ordinal.csv.gz`.
- Full EnergyPlus/Sinergym working directories and intermediate simulator output folders: available locally but better suited for a Zenodo/OSF DOI archive if public raw working-directory deposition is required.

The R2 generated city-year trace CSVs are included here in gzip-compressed form.

## SHA-256 Checksums

Current checksums are stored in `CHECKSUMS_SHA256.txt`.

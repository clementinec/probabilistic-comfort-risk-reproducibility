# Reproducibility Package

This package contains the processed outputs and scripts needed to reproduce the probability-necessity diagnostics and manuscript figures for:

**Probabilistic Thermal Comfort Risk Under Future Weather: Separating Expected Sensation from Discomfort-Tail Exposure for HVAC Control**

## Contents

- `summary_outputs/`: processed diagnostic summaries used in the manuscript.
  - `probability_necessity_summary.csv`
  - `boundary_region_summary.csv`
  - `mean_threshold_decision_diagnostics.csv`
  - `tail_spread_by_mean_bin.csv`
  - `directional_tail_diagnostics.csv`
  - `matched_mean_tail_divergence.csv`
  - `same_city_2025_2075_same_mean_examples.csv`
  - `medium_office_trace_summary.csv`
  - `microgrid_event_summary_compact.tex`
  - `zone_aggregation_summary.csv`
  - `zone_risk_ranking.csv`
  - `hidden_zone_tail_examples.csv`
  - `zone_aggregation_rows_compact.csv.gz`
  - `tail_steering_action_conditioned_summary.csv`
  - `stress_window_warm_tail_threshold_sensitivity.csv`
  - `zone_tail_diagnostic_summary.md`
- `scripts/`: analysis and weather-conversion scripts.
  - `probability_necessity_diagnostic.py`
  - `convert_cmip_csv_to_epw.py`
  - `build_zone_and_tail_diagnostics.py`
  - `rerun_medium_office_temporal_R2_trace_export.py`
- `configs/`: simulation/model configuration artifacts.
  - `medium_office_otc_control.idf`
  - `control_predictor_metrics.json`
- `models/`: trained control predictor artifacts used by the R2 trace rerun.
  - `control_predictors.joblib`
  - `control_predictor_metrics.json`
- `figures/`: generated probability-necessity and R2 zone-aggregation diagnostic figures.
- `traces/`: gzip-compressed R2 Medium Office city-year traces with per-zone probabilistic outputs.

## Data Restrictions

The ASHRAE Global Thermal Comfort Database II is a third-party dataset and is not redistributed here. Users must obtain it from the original source and comply with its terms of use.

The CMIP-derived weather CSVs and generated EPWs used for the diagnostic are not bundled in this lightweight package. The EPW conversion script records the source path pattern and conversion assumptions. The manuscript reports that the local source files identify the archive as `CMIP6_MPI_0515_ssp585`, but do not encode a separate ensemble-member identifier or upstream bias-correction metadata.

The R2 trace files in `traces/` are generated simulation outputs, not third-party source weather files. They are stored as `.csv.gz` files to keep the public repository within ordinary GitHub file-size limits. The redundant concatenated file `medium_office_control_traces.csv` is not tracked because it can be reconstructed by concatenating the nine city-year trace files.

The compact row-level zone-aggregation table is also stored as `summary_outputs/zone_aggregation_rows_compact.csv.gz`; running `scripts/build_zone_and_tail_diagnostics.py` regenerates the uncompressed CSV.

## Repository Plan

This folder is prepared as the submission-time Supplementary Data package and is small enough to be committed directly to a public GitHub repository. It contains scripts, configuration files, processed summary outputs, and generated diagnostic figures needed to inspect the probability-threshold, zone-aggregation, tail-steering, and stress-window threshold analyses reported in the manuscript.

The second-revision R2 zone-level traces are now bundled in compressed form for direct inspection of the zone-aggregation and tail-steering diagnostics. Larger EnergyPlus working directories, source weather archives, and third-party thermal-comfort records remain excluded; if publication requires full raw working directories, they should be deposited separately in a DOI-bearing repository such as Zenodo or OSF, with this GitHub package linking to that archive.

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
- `scripts/`: analysis and weather-conversion scripts.
  - `probability_necessity_diagnostic.py`
  - `convert_cmip_csv_to_epw.py`
- `configs/`: simulation/model configuration artifacts.
  - `medium_office_otc_control.idf`
  - `control_predictor_metrics.json`
- `figures/`: generated probability-necessity diagnostic figures.

## Data Restrictions

The ASHRAE Global Thermal Comfort Database II is a third-party dataset and is not redistributed here. Users must obtain it from the original source and comply with its terms of use.

The CMIP-derived weather CSVs and generated EPWs used for the diagnostic are not bundled in this lightweight package. The EPW conversion script records the source path pattern and conversion assumptions. The manuscript reports that the local source files identify the archive as `CMIP6_MPI_0515_ssp585`, but do not encode a separate ensemble-member identifier or upstream bias-correction metadata.

## Repository Plan

This folder is prepared as the submission-time Supplementary Data package and is small enough to be committed directly to a public GitHub repository. It contains scripts, configuration files, processed summary outputs, and generated diagnostic figures needed to inspect the probability-threshold and stress-window analyses reported in the manuscript.

Large raw co-simulation traces are intentionally not bundled in this lightweight GitHub package. If full raw traces are required for publication or post-acceptance archiving, they should be deposited separately in a DOI-bearing repository such as Zenodo or OSF, with this GitHub package linking to that archive. The third-party ASHRAE Global Thermal Comfort Database II should remain externally referenced rather than redistributed.

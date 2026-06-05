# Reproducibility Package Manifest

Date: 2026-06-06

Package size: approximately 4.3 MB before Git metadata.

This lightweight package is suitable for a public GitHub repository. It contains processed diagnostic outputs, plotting/diagnostic scripts, generated figures, and configuration artifacts used by the manuscript revision. It does not redistribute third-party thermal-comfort data or large raw co-simulation traces.

## Included Files

### Root

- `README.md`: package overview, data restrictions, and repository plan.
- `MANIFEST.md`: this manifest.
- `.gitignore`: excludes local plotting cache artifacts.

### `configs/`

- `control_predictor_metrics.json`: predictor/controller summary metrics.
- `medium_office_otc_control.idf`: Medium Office control IDF/configuration artifact used in the diagnostic workflow.

### `figures/`

- `city_year_tail_trend.png`: city-year high-tail trend figure.
- `probability_necessity_manuscript_figure.pdf`: manuscript probability-necessity diagnostic figure.
- `probability_necessity_manuscript_figure.png`: PNG version of the same diagnostic figure.
- `tail_spread_by_mean_bin.png`: mean-bin tail-spread diagnostic figure.

### `scripts/`

- `convert_cmip_csv_to_epw.py`: CMIP-derived CSV to EPW conversion utility.
- `probability_necessity_diagnostic.py`: probability-threshold diagnostic and figure-generation script.

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

## Not Included

- ASHRAE Global Thermal Comfort Database II: third-party dataset; users must obtain it from the original source and comply with its terms.
- CMIP-derived source weather CSVs and generated EPWs: not bundled in this lightweight package; provenance and conversion assumptions are documented in the manuscript and README.
- Full raw EnergyPlus/Sinergym traces: available locally but better suited for a Zenodo/OSF DOI archive if public raw-trace deposition is required.

Local raw-trace directories available outside this lightweight package:

- `city_year_probability_necessity_diagnostic/`: approximately 87 MB.
- `rerun_medium_office_city_year_probability_diagnostic/`: approximately 351 MB.
- `rerun_medium_office_microgrid_ssp585/`: approximately 345 MB.
- `rerun_medium_office_temporal/`: approximately 91 MB.

## SHA-256 Checksums

Checksums below are for the tracked package files listed above, excluding this manifest.

```text
1f4a50eebb08308bddea96d21129b5bc33bf4612b8431b3af257ee1cbc27dcdc  .gitignore
505698522590d74019e1830565426d669463f7a196370fd6c7c8778976892a6f  README.md
b5922472a72727d71d263e5968dd474ab13ee8ec630570578d199d2275494713  configs/control_predictor_metrics.json
7e0a1415a7cb2ea2cdeb7356477c56ac5462ed8c83cf2241b3ed0541db53ba0a  configs/medium_office_otc_control.idf
a618f6162a15d82f2d06046b1920bdc9dfc4f3d32574a9581ce3570ec7b97165  figures/city_year_tail_trend.png
69415fc00890e37c6393741a22527e2dbb149fea0ff2d49c78fd27e57612a370  figures/probability_necessity_manuscript_figure.pdf
1c58ab77484017d05968f611a340fdc85f9b3d565c78bf9665916e7920117dab  figures/probability_necessity_manuscript_figure.png
3be2585b49fc9c4e7d85d5766b192db18930c16f14c69ba046762e4a89d6afe6  figures/tail_spread_by_mean_bin.png
86278bf4262a6e12fbde124179893960bebc1786c019238cc7fbab5153afb74d  scripts/convert_cmip_csv_to_epw.py
96db51e482e1c1bdf5f32ffd1a529970de75d83c1a9a7c96016b648b3ddabbf4  scripts/probability_necessity_diagnostic.py
4c9377cda5864200f0f0cfcef567e75df9657b8fcbf1f01e6077caae38e3ac54  summary_outputs/boundary_region_summary.csv
071b219b6e5a61b0379cc2d9455161c16373c1d4b9d05f15d557a02d7b927eca  summary_outputs/directional_tail_diagnostics.csv
1b36a6b09606be87ec736d7856834ff7f06a2c1c7301af1cb4e917e9b59f3576  summary_outputs/matched_mean_tail_divergence.csv
962ddb668f4a9b3e6a868fbec379036c5206778e832361dd2fe7c8ce2f87779b  summary_outputs/mean_threshold_decision_diagnostics.csv
4a0fc860da90cdc9cb1ab78b4b86997fe5c4c280a1b8dd76b7465957e03b2070  summary_outputs/medium_office_trace_summary.csv
608081d19853acde750ed6821879b44f20f88e90fc4aee354660faf3ef84eb35  summary_outputs/microgrid_event_summary_compact.tex
80eee7ae94a8da96165b5f591c60517532c7d0910c7fe43c3f0fc03111e4f8ea  summary_outputs/probability_necessity_summary.csv
aa2f7ea3271c17d1b8c00458d5899a9204ef0f240353ff137985bf39d029d8e9  summary_outputs/same_city_2025_2075_same_mean_examples.csv
840830ee16a08803331ae79b2824f62c64fdb5ea2a335b40c3439901a23d53e4  summary_outputs/tail_spread_by_mean_bin.csv
```

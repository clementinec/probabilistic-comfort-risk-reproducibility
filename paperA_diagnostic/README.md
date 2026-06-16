# Paper A Diagnostic Rebuild Reproducibility Bundle

This folder contains compact reproducibility materials for:

**Hidden Behind the Mean: Probabilistic Thermal-Sensation Diagnostics of Discomfort-Tail Risk Under Future Weather**

The bundle supports the diagnostic rebuild of the manuscript. It is intentionally not a full EnergyPlus trace archive.

## Included

- `scripts/`: analysis, validation, plotting, manifest, and run-audit scripts used for the rebuild.
- `data/`: compact case manifests and warmup-warning manifests.
- `diagnostics/`: generated CSV, JSON, Markdown, and LaTeX summary outputs used to report model validation, calibration, scalar comparison, zone aggregation, threshold sensitivity, floor-position diagnostics, and metabolic-spread diagnostics.
- `figures/`: manuscript and supplement figures generated from the compact outputs.
- `manuscript_snapshot/`: compiled main-manuscript and supplement PDFs from this cleanup pass.

## Not Included

- Full 15-minute EnergyPlus trace files.
- EnergyPlus run directories and simulation intermediates.
- Third-party ASHRAE Global Thermal Comfort Database II records.
- Large generated weather files where redistribution is not practical or not permitted.

Large or restricted artifacts should be audited through the manifest, provenance tables, and checksum records rather than by direct redistribution in this repository.

## Scope

The study is a diagnostic future-weather stress test. The included outputs do not report closed-loop control, energy savings, demand response, or equipment-facing performance. The central probability variable is:

`p_tail = Pr(TSV <= -2) + Pr(TSV >= +2)`

The reported high-tail quantities are modeled occupied-timestep exceedance summaries, including zone-state and any-zone screens; they should not be read as measured occupant-exposure rates.

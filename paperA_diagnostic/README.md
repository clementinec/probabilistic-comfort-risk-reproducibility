# R01 Diagnostic Bundle

This directory is the live R01 package for **Hidden Behind the Mean:
Probabilistic Diagnostics of Discomfort-Relevant Thermal-Sensation Tails Under
Future Weather**.

## Directory map

- `manuscript_snapshot/`
  - `main.pdf`: frozen clean manuscript.
  - `supplement.pdf`: frozen Supplement.
  - `source/`: sanitized production source and the eight final figures.
- `figures/`: exact copies of the eight final manuscript/Supplement figures.
- `data/`: path-neutral panel and warmup-warning manifests.
- `scripts/`
  - baseline predictor, trace, validation, metabolic, and QA helpers retained
    from the preceding public bundle;
  - `revision/`: R01 synchronization, robustness, uncertainty, validation,
    weather-provenance, and rendering scripts.
- `outputs/`
  - `core/`: synchronized 144-role/119-state headline, scalar, threshold,
    nominal-model, endpoint, and legacy-versus-corrected summaries.
  - `manuscript_replacement/`: compact figure/table inputs and the replacement
    ledger used to update the manuscript.
  - `uncertainty/`: panel-conditional and contributor-clustered uncertainty
    summaries. Restricted row-level held-out predictions are excluded.
  - `validation/`: record-level, grouped, and overlap-excluded reciprocal
    source-shift summaries.
  - `weather/`: provenance, variable-support, and descriptive physical-driver
    audits.
  - `metabolic_profile/` and `metabolic_spread/`: source summaries supporting
    the metabolic-input figure and table.
  - `simulation_qa/`: trace and EnergyPlus run-quality audits.

## What can be reproduced from this compact packet

The checked-in summaries support direct auditing of all reported rounded
headline values, threshold curves, scalar comparisons, spatial aggregation
results, uncertainty intervals, validation tables, weather-provenance claims,
and metabolic-sensitivity values. The final source compiles the manuscript and
Supplement, subject to a compatible TeX installation.

Several revision scripts require explicit command-line paths to restricted or
checksum-only inputs. Full same-state inference requires the original zone
traces, the primary and no-PMV model artifacts, the pooled TSV corpus for
held-out validation, and the synchronized NPZ arrays. Those inputs are not all
redistributed in this compact Git repository. The processed CSV summaries are
therefore the authoritative public intermediates for the reported numerical
results.

The archived `run_endpoint_nominal_robustness.py` is the final retained
revision-stage script, but it was edited after the completed synchronized run.
Its current SHA-256 differs from the producing-script SHA-256 recorded in
`outputs/core/run_config.json`. Both hashes are reported in the R01 manifest;
the frozen outputs and their public checksum manifest are unchanged.

## Interpretation boundaries

- `p_tail` is predicted TSV-category probability mass, not PPD, observed
  dissatisfaction, acceptability, prevalence, or health risk.
- The `0.20` screen is illustrative; the spatial conclusion uses the complete
  threshold sweep.
- PMV is a useful but non-equivalent scalar comparator.
- Future weather is evaluated as a fixed-building, no-adaptation stress test,
  not as a forecast.
- The package contains no validated control policy and makes no energy-savings
  or operational-effectiveness claim.

## Data and artifact status

See the repository-level
[`R01_REPRODUCIBILITY_MANIFEST.md`](../R01_REPRODUCIBILITY_MANIFEST.md) for the
distinction among redistributed, checksum-only, third-party-restricted, and
unavailable inputs.

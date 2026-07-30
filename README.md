# Hidden Behind the Mean

This repository accompanies **Hidden Behind the Mean: Probabilistic Diagnostics
of Discomfort-Relevant Thermal-Sensation Tails Under Future Weather**.

Start with the
[`REPRODUCIBILITY_MANIFEST.md`](REPRODUCIBILITY_MANIFEST.md), which distinguishes
redistributed artifacts from checksum-only, restricted, and unavailable inputs.

## Study and package scope

The paper is a diagnostic information-loss audit. It examines what becomes
invisible when a modeled seven-category thermal-sensation distribution is
replaced by scalar summaries, spatially aggregated, or evaluated under a single
representative occupant assumption. Future weather supplies a structured,
fixed-building, no-adaptation stress test.

The principal diagnostic quantity is

```text
p_tail = Pr(TSV <= -2) + Pr(TSV >= +2)
```

This is model-estimated TSV-category mass. It is not PPD, observed
dissatisfaction, acceptability probability, prevalence, health risk, or a
standards threshold.

This repository does not report or validate closed-loop control, energy savings,
operational effectiveness, or demand-response performance.

## Repository map

- [`manuscript/`](manuscript/): frozen clean manuscript and Supplement PDFs,
  sanitized production source, and source-linked figures.
- [`figures/`](figures/): exact copies of the eight final manuscript and
  Supplement figures.
- [`data/`](data/): path-neutral panel and warmup-warning manifests.
- [`outputs/`](outputs/): synchronized core results, manuscript figure/table
  inputs, uncertainty summaries, validation results, weather audits, metabolic
  sensitivity summaries, and simulation QA.
- [`scripts/`](scripts/): analysis, validation, uncertainty, weather-audit,
  rendering, and trace-QA code.
- [`models/`](models/): the retained primary TSV predictor bundle and its
  metrics.
- [`configs/`](configs/): the fixed-building EnergyPlus input used by the
  simulation workflow.
- [`CHECKSUMS_SHA256.txt`](CHECKSUMS_SHA256.txt): SHA-256 manifest for the
  public working tree.

[`scripts/legacy_control_pipeline.py`](scripts/legacy_control_pipeline.py) is
retained antecedent infrastructure because several analysis helpers depend on
its predictor, feature-processing, and trace routines. Its additional
control-oriented branches are not outputs or evaluated methods of the present
paper.

## Verification

From the repository root, run:

```bash
python3 verify_release.py
```

The verifier checks every listed public-file SHA-256, the 144-role/119-state
inventory, and the five headline spatial summaries. The recorded analysis
environment is listed in [`requirements.txt`](requirements.txt).

The frozen PDFs are authoritative. Recompilation under a different TeX
installation may not be byte-identical.

## Public-data boundary

The third-party pooled ASHRAE–China TSV records are not redistributed. Full
EnergyPlus traces, selected EPWs, large synchronized probability arrays, and
upstream weather workbooks are represented by identifiers and checksums where
redistribution is restricted or impractical. The exact upstream six-city
weather batch configuration and its raw CMIP6 and observational inputs were not
preserved; this boundary is stated in the manuscript and weather-provenance
audit.

The primary fitted predictor is available as
[`models/tsv_predictor_bundle.joblib`](models/tsv_predictor_bundle.joblib). The
separately trained no-PMV sensitivity artifact is not included; its recorded
SHA-256, fixed downstream predictions, and manuscript-facing summaries are
documented in the reproducibility manifest and retained outputs.

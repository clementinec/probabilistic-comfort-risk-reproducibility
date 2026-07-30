# Hidden Behind the Mean — R01 Reproducibility Package

This repository contains the compact public reproducibility package for:

**Hidden Behind the Mean: Probabilistic Diagnostics of Discomfort-Relevant
Thermal-Sensation Tails Under Future Weather**

The R01 package replaces the earlier control-oriented repository snapshot. It
contains the frozen clean manuscript and Supplement, final figures, analysis
scripts, processed diagnostic summaries, uncertainty results, model and weather
provenance, and simulation QA records used in the revision.

## Start here

- [`paperA_diagnostic/README.md`](paperA_diagnostic/README.md): package map,
  interpretation boundaries, and reproduction levels.
- [`R01_REPRODUCIBILITY_MANIFEST.md`](R01_REPRODUCIBILITY_MANIFEST.md):
  redistributed, checksum-only, restricted, and unavailable artifacts.
- [`paperA_diagnostic/manuscript_snapshot/`](paperA_diagnostic/manuscript_snapshot/):
  frozen clean manuscript, Supplement, source, and final figures.
- [`paperA_diagnostic/outputs/`](paperA_diagnostic/outputs/): synchronized R01
  summaries and figure/table inputs.
- [`CHECKSUMS_SHA256.txt`](CHECKSUMS_SHA256.txt): SHA-256 manifest for the public
  working tree.

Verify the public packet from the repository root:

```bash
python3 verify_release.py
```

The verifier checks every listed public-file SHA-256 plus the 144-role/119-state
inventory and the five headline spatial summaries.

The recorded revision environment is listed in
[`requirements.txt`](requirements.txt). The frozen PDFs are authoritative;
recompilation under a different TeX installation may not be byte-identical.

## Scope

The study is an information-loss audit conducted on a fixed DOE Medium Office
archetype and fixed occupant assumptions under a structured future-weather
stress test. It does not demonstrate closed-loop control, energy savings,
operational effectiveness, occupant prevalence, or health risk.

The diagnostic quantity is:

```text
p_tail = Pr(TSV <= -2) + Pr(TSV >= +2)
```

It is model-estimated TSV-category mass, not PPD, observed dissatisfaction,
acceptability probability, or a standards threshold.

## Public-data boundary

The third-party pooled ASHRAE–China TSV records are not redistributed. Full
EnergyPlus traces, selected EPWs, large synchronized probability arrays, and
upstream weather workbooks are represented by identifiers and checksums where
redistribution is restricted or impractical. The exact upstream six-city
weather batch configuration and its raw CMIP6/observational inputs were not
preserved; that boundary is stated in the manuscript and provenance audit.

The primary fitted predictor is retained in `models/`. The separately trained
no-PMV sensitivity artifact is not in the compact public packet; its recorded
SHA-256 and its fixed downstream predictions and summaries are preserved in the
R01 outputs and manifest.

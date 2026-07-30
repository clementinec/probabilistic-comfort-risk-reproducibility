# R01 Reproducibility Manifest

Prepared: 30 July 2026

Manuscript: **Hidden Behind the Mean: Probabilistic Diagnostics of
Discomfort-Relevant Thermal-Sensation Tails Under Future Weather**

## Redistributed

- Frozen clean manuscript and Supplement PDFs.
- Sanitized production source and eight final figures.
- Primary fitted predictor:
  - file: `models/control_predictors.joblib`
  - SHA-256:
    `6fbeb06644a36b226b17824a1fca7526bd518e0a9b76a41f878fb1c27efcd619`
- Path-neutral 144-role panel manifest and 17-case warmup-warning manifest.
- Baseline predictor/trace helpers and R01 revision scripts.
- Corrected core case, group, threshold, scalar, model-form, endpoint, and
  legacy-versus-corrected summaries.
- Compact manuscript figure/table inputs, including the fixed plotting sample.
- Corrected panel-conditional uncertainty interval summaries.
- Contributor-clustered bootstrap draws and interval summaries, excluding
  restricted row-level held-out predictions and contributor inventories.
- Held-out, grouped, and overlap-excluded reciprocal source-shift summaries.
- Weather provenance, selector, EPW transform, structural QA, variable-support,
  and descriptive physical-driver outputs.
- Metabolic-profile source summaries and simulation run-QA summaries.

## Path-normalized manifests

The public CSV manifests replace author-local filesystem roots with logical
artifact identifiers. Scientific metadata, filenames, file sizes, state hashes,
and source-artifact hashes are retained.

- Original run-time panel-manifest SHA-256:
  `892b3a5459eb9cd93fbb21cd98a5ce55093d979a3fe746171bc677186ac5a335`
- Public path-normalized panel-manifest SHA-256:
  `76292903883f056cc4c2f5689f19f629b79d92de7e28e1849129903b2685c8b1`

## Checksum-only or omitted for size

- 144 selected EPW role files.
- Full EnergyPlus trace files.
- 119 synchronized state arrays under the logical identifier
  `checksum-only://corrected_zone_npz/` (approximately 183 MB in the working
  archive).
- Large panel-level bootstrap-draw tables; interval summaries and generating
  scripts are redistributed.
- Upstream weather workbooks and hourly forecast series.

Per-artifact hashes, sizes, state reuse, and logical identifiers are retained in
`paperA_diagnostic/outputs/core/` and
`paperA_diagnostic/outputs/weather/provenance/`.

## Third-party restricted

- The pooled field TSV corpus combining the ASHRAE Global Thermal Comfort
  Database II and Chinese Thermal Comfort Dataset.
- Row-level held-out TSV labels and predictions derived from that pooled corpus.
- Named contributor inventories and small-cell contributor summaries.

These are not redistributed. Aggregate validation summaries and contributor-
clustered uncertainty results are included.

## Recorded but not present in the compact packet

- No-PMV fitted predictor:
  - expected SHA-256:
    `8501051a9a3d32f8a8e194de120e4c227dde9d63ffa14a1d9549342c8b6829ad`
  - status: the serialized artifact is not present in the frozen public packet;
    fixed no-PMV predictions and all manuscript-facing aggregate summaries are
    retained.
- Producing version of `run_endpoint_nominal_robustness.py`:
  - run-recorded SHA-256:
    `54955c3190782f3ef75195a3fff7b7c94221460bdc842edef7787998392e0e94`
  - retained final script SHA-256:
    `98efd37c06985def620ab6c938e458fb7583095fa62a3a3008cf93198b103586`
  - status: the retained script was edited after the completed synchronized
    run. The frozen processed outputs are the authoritative intermediates.
- Exact six-city upstream weather batch configuration and raw
  CMIP6/observational bundle:
  - status: not preserved with the archived downstream weather products.

These limitations prevent a byte-identical raw-input-to-final-output rebuild
from the public repository alone. They do not affect inspection of the frozen
R01 outputs, final manuscript values, or disclosed weather-provenance boundary.

## Excluded editorial and superseded material

- Reviewer correspondence, response-letter source, cover letter, and marked
  manuscript.
- Build auxiliaries and editor-generated submission files.
- Legacy mixed-callback probability outputs and smoke-test directories.
- Earlier R2 control-study traces, summaries, and figures, which remain
  recoverable from repository history but are not part of the live R01 packet.

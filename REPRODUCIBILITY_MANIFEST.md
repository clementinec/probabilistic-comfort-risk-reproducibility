# Reproducibility Manifest

Prepared: 30 July 2026

Manuscript: **Hidden Behind the Mean: Probabilistic Diagnostics of
Discomfort-Relevant Thermal-Sensation Tails Under Future Weather**

## Redistributed

- Frozen clean manuscript and Supplement PDFs.
- Sanitized production source and eight final figures.
- Primary fitted predictor:
  - file: `models/tsv_predictor_bundle.joblib`
  - SHA-256:
    `6fbeb06644a36b226b17824a1fca7526bd518e0a9b76a41f878fb1c27efcd619`
- Fixed-building EnergyPlus input:
  - file: `configs/medium_office_fixed_building.idf`
- Path-neutral 144-role panel manifest and 17-case warmup-warning manifest.
- Predictor, trace, validation, uncertainty, weather-audit, and rendering
  scripts.
- Corrected core case, group, threshold, scalar, model-form, endpoint, and
  timing-alignment comparison summaries.
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
`outputs/core/` and `outputs/weather/provenance/`.

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
    `d081499ee2f2b939029f436542b41706954ea78c4790a774f6770d3030997199`
  - status: the retained script was edited after the completed synchronized
    run. The frozen processed outputs are the authoritative intermediates.
- Exact six-city upstream weather batch configuration and raw
  CMIP6/observational bundle:
  - status: not preserved with the archived downstream weather products.

These limitations prevent a byte-identical raw-input-to-final-output rebuild
from the public repository alone. They do not affect inspection of the frozen
outputs, final manuscript values, or disclosed weather-provenance boundary.

## Compatibility identifiers

The strings `paperA_corrected_same_state_v3` and
`PaperA occupied environmental state v1` remain in the synchronized-array
schema and state-hash namespace. They are preserved machine-provenance tokens,
not live package or manuscript labels; changing them would break compatibility
with the recorded state hashes.

## Antecedent infrastructure

`scripts/legacy_control_pipeline.py` is retained because analysis helpers use
its predictor, feature-processing, and trace routines. The file also contains
antecedent control-oriented branches. Those branches are not current manuscript
outputs, were not evaluated as a control study, and do not support claims about
control performance, energy savings, or operational effectiveness.

## Excluded editorial and superseded material

- Reviewer correspondence, response-letter source, cover letter, and marked
  manuscript.
- Build auxiliaries and editor-generated submission files.
- Superseded mixed-callback probability outputs and smoke-test directories.
- Control-oriented traces, summaries, and figures that are not part of the
  diagnostic information-loss audit.

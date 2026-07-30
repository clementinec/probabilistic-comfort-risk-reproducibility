# Callback timing evidence

Source: `paperA_diagnostic/scripts/run_medium_office_diagnostic_panel.py`

Source SHA-256: `4c4f246e5027da1ff1dead202f99a1f031cba3f864d49a623cf46af9b0252b0e`

- The diagnostic probability is calculated from `values` in `apply_control`
  (line 731).
- `apply_control` is registered at the **beginning** of the zone timestep after
  heat-balance initialization (line 810).
- `record` is registered at the **end** of the zone timestep after zone
  reporting (line 811).
- The zone Ta/MRT/RH fields written to the trace row come from the later
  `record` callback (line 803).

Consequently, the stored probability and the environmental values appearing on
the same CSV row were sampled at different callback points. The robustness run
does not overwrite or reinterpret those stored values. It applies the ordinal
and nominal predictors to the identical recorded end-of-step state and labels
that result `corrected same-state inference`; the per-case differences from the
stored callback-timed probabilities are retained in `ordinal_trace_parity.csv`
and the paired threshold outputs.

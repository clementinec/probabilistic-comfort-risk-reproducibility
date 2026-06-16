# Paper A Rebuild Run QA Summary

Date: 2026-06-16

## Main Diagnostic-Reference Panel

- Run directory: `runs/diagnostic_reference`
- Weather panel: 144 EPW cases
- Strategy: `diagnostic_reference`
- Completed runner elapsed time: 137.97 minutes
- Per-case trace files: 144
- Occupied probability rows: 2,405,376
- Diagnostic-control integrity:
  - Maximum `action_delta_c`: 0.000
  - Maximum `setpoint_shift_c`: 0.000
  - Occupied rows with missing zone probabilities: 0
- Combined trace: `runs/diagnostic_reference/traces/medium_office_control_traces.csv`
- Runner summary: `runs/diagnostic_reference/summary/medium_office_trace_summary.csv`

## EnergyPlus Warning QA

- Main run error scan:
  - Cases with severe/fatal/termination signatures: 17
  - Severe warnings: 40
  - Fatal warnings: 0
  - Program terminations: 0
- All severe warnings are `CheckWarmupConvergence` messages.
- A targeted rerun of the 17 warning cases was performed with `Maximum Number of Warmup Days = 50`.
- Warmup50 rerun:
  - Run directory: `runs/diagnostic_reference_warmup50`
  - Per-case trace files: 17
  - Occupied probability rows: 283,968
  - Maximum `action_delta_c`: 0.000
  - Maximum `setpoint_shift_c`: 0.000
  - Occupied rows with missing zone probabilities: 0
  - Cases with severe/fatal/termination signatures: 16
  - Severe warnings: 24
  - Fatal warnings: 0
  - Program terminations: 0

Interpretation: increasing maximum warmup days did not eliminate the convergence warnings. Treat these as EnergyPlus prototype warmup-convergence limitations under selected stress-weather cases, not as failed simulations. The manuscript should report this as a simulation QA limitation if these cases are used.

## Generated Diagnostics

- Trace schema audit: `diagnostics/trace_schema_audit.csv`
- Warmup50 trace schema audit: `diagnostics/warmup50_trace_schema_audit.csv`
- Mean-tail compact diagnostics: `diagnostics/mean_tail_compact`
- Zone-aggregation diagnostics: `diagnostics/zone_aggregation`

## Headline Diagnostic Results

- Full-panel mean `p_tail`: 0.117
- Full-panel 95th percentile `p_tail`: 0.306
- Mean-aggregated high-tail exposure (`p_tail >= 0.20`): 13.4%
- Near-neutral high-tail states (`|mu_TSV| < 0.15` and `p_tail >= 0.20`): 0.0%
- Any-zone high-tail exposure (`max zone p_tail >= 0.20`): 37.0%
- Hidden any-zone high-tail states when mean aggregation is below 0.20: 23.6%
- Max-zone aggregation is more severe than mean aggregation in 31.7% of occupied probability timesteps.

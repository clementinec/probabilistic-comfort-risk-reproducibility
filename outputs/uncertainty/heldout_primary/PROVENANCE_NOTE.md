# Provenance note

This contributor-clustered analysis uses the independent occupant-record
held-out split in `TCN/newin_with_bmr.csv` and the saved fitted predictor
bundles. It does not use EnergyPlus time-series traces.

The begin-timestep/end-timestep alignment defect identified in the legacy
EnergyPlus panel therefore does not affect these held-out validation intervals.
Only the panel bootstrap must be rerun from corrected same-state summaries.

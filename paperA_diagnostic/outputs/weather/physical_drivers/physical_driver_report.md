# Weather and building-physics association audit

> **Corrected same-state inference.** Every probability in this directory was freshly inferred from the Ta/MRT/RH state recorded on the same end-of-step row. The legacy callback-timed probability columns were not used.

## Bottom line

The existing traces support a coherent three-link interpretation without a new
EnergyPlus run:

1. **Outdoor forcing to simulated indoor state.** Across the 144 role-labelled
   cases, the median within-case hourly Spearman association between outdoor
   dry-bulb and mean-zone operative temperature was
   0.824 [IQR 0.753, 0.897].
2. **Indoor state to fitted TSV distribution.** The corresponding association
   between operative temperature and TSV-tail probability was
   0.855 [IQR 0.794, 0.917].
3. **HVAC response accompanies rather than eliminates the pattern.** The
   descriptive cooling-rate/tail association was
   0.574 [IQR 0.411, 0.629].

These are associations in a fixed building and control configuration. They do
not identify causal effects, equipment-capacity failure, or future occupant
outcomes.

The annual within-case correlation is not a temperature-response coefficient:
it mixes seasons while the fitted TSV model also conditions on humidity,
running-mean outdoor temperature, and the other frozen predictors.
10/144
annual cases have a negative
operative-temperature/total-tail rank correlation, all of which should be
retained rather than hidden. Under outdoor dry-bulb at or above 25 C, the
median operative-temperature/warm-tail association is
0.977,
with
0
negative cases. The conditioned results are in
`within_case_conditioned_correlations.csv`.

## Baseline-to-late pathway

Across the 36 matched
city--scenario--selection-role comparisons, annual mean outdoor temperature,
occupied mean-zone operative temperature, mean tail probability, and
high-tail share all increased in
36/36,
36/36,
36/36,
and
36/36
comparisons, respectively. Median changes
were +2.13 C outdoors,
+0.50 C in occupied
mean-zone operative temperature, +0.026
in mean tail probability, and
+10.13 percentage points in the
high-tail share.

An equal-unique-weather-year sensitivity first averages the distinct selected
years within each city--scenario--slice. Outdoor temperature, operative
temperature, mean tail probability and high-tail share increase in
12/12,
12/12,
12/12,
and
12/12
cells, respectively. This
check is in `baseline_to_late_change_unique_weather_cell.csv`.

The association between matched changes in annual mean outdoor temperature and
occupied operative temperature was rho =
0.868; the association between
operative-temperature change and mean-tail change was rho =
0.855. Cooling rate increased in
33/36
comparisons, with a median change of
5.06 kW.

This is a matched descriptive chain through selected annual weather states,
not a variable-attribution model. In particular, solar and humidity covary
with temperature and location; their coefficients must not be read as
independent causal contributions.

## Outdoor-temperature stratification

After collapsing duplicate role labels, the pooled high-tail share was
0.19% for outdoor dry-bulb
20--25 C and
52.69% at or above 35 C. The
associated operative-temperature, RH, cooling-rate, and tail summaries are in
`outdoor_temperature_bin_unique_weather_summary.csv`; the separately retained
role-labelled version documents the originally planned category weighting.
This stratification is descriptive and retains differences among cities,
seasons, and hours.

## Spatial pattern

After collapsing duplicate role labels to 119
unique weather states, the mean high-tail share across the three south
perimeter zones was 25.93%,
versus 11.21% across the
other nine perimeter zones. The exact floor-by-orientation results and the
mean MRT-minus-air-temperature diagnostic are in
`floor_orientation_summary_unique_weather.csv`.

This pattern is consistent with orientation/envelope/HVAC-zoning interaction
in this prototype. It is not a clean estimate of solar causation. Only
horizontal GHI is available; incident facade solar gain and surface heat flux
were not recorded. The horizontal-GHI associations are retained as an
auditable forcing check, not attributed facade gains.

## Humidity and cooling interpretation

Outdoor RH comes from the selected EPW and is aligned to the hourly trace.
Zone RH is the EnergyPlus response used by the TSV predictor. Their
association and the zone-RH/tail association are reported separately so that
outdoor moisture forcing is not conflated with indoor humidity state.

`zone_cooling_rate_w` is the sum of the 15 requested zone sensible cooling
rates. A positive association with outdoor temperature or tail probability
shows coincident system response; it does not by itself establish plant
saturation. No capacity-saturation claim should be made without explicit
unmet-load or coil-capacity evidence.

## Weighting and uncertainty boundary

- Case-level associations use the 119 unique
  city--scenario--year weather states.
- Within-case correlations summarize temporal co-movement and report their
  distribution across cases; 15-minute rows are not treated as independent
  climate realizations.
- The panel uses one MPI-ESM1-2-LR forcing family. It does not quantify GCM,
  stochastic-weather, or urban-microclimate uncertainty.
- Tail probability is conditional on the fitted TSV model and fixed occupant
  assumptions. It is not observed dissatisfaction.

## EnergyPlus decision

No rerun is required for the reviewer-facing physical interpretation. A rerun
would be needed only to make incident-solar, envelope heat-flux, component
capacity, or unmet-load claims. Those stronger claims are unnecessary and
should instead be excluded.

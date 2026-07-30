# Future-weather provenance and QA audit

## Result

The preserved downstream weather chain is internally auditable. The panel has
144 role-labelled EPWs but
119 unique
city--scenario--slice--year states. All 12 paired
source workbooks identify MPI-ESM1-2-LR, `rcm=N/A`, a 1991--2010 baseline, and
daily delta mode. The executable selectors reproduce
144/144 manifest
selections. The source-to-EPW transformation matches exactly for
119/119
unique selected source years.

The source inventory records 12/
12 structural/plausibility passes and
0 logged issues. The EPW audit found
0 files with a row-count,
field-width, broad-range, or dew-point consistency issue.

## Exact preserved chain

1. Twelve scenario-city forecast artifacts are continuous from
   2025-01-01 00:00 through 2100-12-31 00:00. All selected 2025--2089 annual
   windows are complete; the final 23 hours of 2100 are outside the analysis.
2. Their workbooks identify a single MPI-ESM1-2-LR forcing family under
   SSP2-4.5 and SSP5-8.5, with a 1991--2010 baseline and daily climate deltas.
3. `typical` is the year nearest median annual CDH18 in its window; `hot` is
   maximum CDH18; `heatwave_extreme` is maximum hours at or above 35 C, with
   annual maximum temperature and then CDH18 as tie-breakers.
4. The EPW converter copies dry-bulb, relative humidity, pressure, GHI, DNI,
   DHI, and wind speed from the forecast output; computes dew point from
   dry-bulb and RH; converts pressure from hPa to Pa; and writes the selected
   annual records. Other unsupported EPW fields use explicit sentinel/default
   values.

`CDD18_hourly` is a legacy field name. Its value is annual CDH18 in C h:
the hourly sum of `max(T_out - 18 C, 0)`. Dividing by 24 gives conventional
CDD18 in C d and leaves selection ranks unchanged.

## Scope boundary

The broader working archive contains a matching climate-delta generator
implementation (SHA-256 `4f8e1c2fb09b02c074c48214c9ae10e958c09475d64ec4610f84981ddd4d0228`) whose
mixed-resolution daily-delta method, output-sheet schema, metadata fields and
MPI-ESM1-2-LR loader align with the 12 preserved workbooks. However, the exact
study-batch configuration and its raw six-city CMIP6 and observational inputs
are not preserved alongside these outputs. The audit therefore supports the
algorithmic lineage plus exact reproduction from the frozen forecast
artifacts through selection and EPW conversion; it does not support byte-level
end-to-end regeneration of the 2025--2100 forecast CSVs.

The weather-generation method has independent 2011--2020 observational
validation across four North American benchmark cities and multiple CORDEX
and CMIP6 forcing sources in Guo and He,
doi:10.1016/j.enbuild.2026.117508. A companion causal decomposition evaluates
baseline-source, climate-signal resolution and downstream degree-day effects
in doi:10.1016/j.energy.2026.140867. These papers support the method family,
not a claim that the exact six study-city forecast files were externally
validated. The panel contains one GCM family and is not a climate-model
uncertainty ensemble.

## Interpretation safeguards

- `typical` means median-CDH-selected within the stated five- or ten-year
  window; it is not a TMY.
- Selection labels are roles, not independent replicates.
- The workbooks document climate deltas for temperature, pressure, wind speed
  and GHI. A separate exact paired audit shows that relative humidity,
  specific humidity and DNI contain no SSP-specific signal in these frozen
  files; component-specific humidity or direct-beam attribution is therefore
  unsupported.
- Horizontal GHI is not facade-incident solar gain.
- Structural and broad physical-range checks establish data integrity and
  plausibility, not observational predictive validity.

# Future-weather variable-support audit

## Result

The paired forecast workbooks document daily climate-delta transformations
for `GHI`, `pressure`, `temp`, `wind_speed`. At the same
city and timestamp, `relative_humidity`, `specific_humidity`, and `DNI` are
exactly identical between SSP2-4.5 and SSP5-8.5 in every one of the
6 paired city files. They therefore contain no
scenario-specific signal in the preserved product. Temperature, pressure,
wind speed, GHI, and derived/adjusted DHI do differ by scenario.

| variable | exact_equal_pct | mean_ssp585_minus_ssp245 | mean_abs_paired_difference | maximum_abs_paired_difference |
| --- | --- | --- | --- | --- |
| DHI | 52.521 | -1.41115 | 18.408 | 452.791 |
| DNI | 100.000 | 0 | 0 | 0 |
| GHI | 52.468 | -1.40423 | 18.5254 | 452.791 |
| pressure | 0.006 | -0.0207367 | 4.32501 | 40.3781 |
| relative_humidity | 100.000 | 0 | 0 | 0 |
| specific_humidity | 100.000 | 0 | 0 | 0 |
| temp | 0.000 | 0.768969 | 3.29848 | 23.7862 |
| wind_speed | 0.000 | -0.0130337 | 0.999122 | 12.7578 |

## Consequence for interpretation

- The future-weather experiment can describe responses to the **joint selected
  weather files**, but it cannot separately attribute a future change to a
  scenario-specific humidity projection or a scenario-specific direct-normal
  irradiance projection.
- Relative humidity is passed into the EPW. Dew point is recalculated from
  future dry-bulb and that carried RH, so absolute moisture changes
  mechanically with temperature under an effectively fixed-RH weather
  assumption. The source `specific_humidity` column is not written to the EPW.
- GHI and DHI vary across scenarios, while DNI is carried unchanged. Horizontal
  GHI remains usable as a descriptive forcing covariate, but neither it nor
  the radiation partition supports a facade-solar causal attribution.
- These exact paired checks establish what the product represents; they are
  not independent observational validation of the underlying climate method.

## Submission-safe position

Describe the weather data as a single-GCM, daily-delta stress-test product and
state explicitly which variables were delta-adjusted. Frame humidity and solar
results as covarying components of the selected weather files, not independent
climate-driver effects. This limitation can be handled by claim narrowing; an
EnergyPlus rerun is unnecessary unless the revision seeks component-specific
humidity or solar attribution.

# Endpoint-only and nominal-model robustness

## Scope

- 144 frozen EnergyPlus case traces; no building simulation rerun.
- Both saved predictors were applied to identical occupied zone-level states.
- `outermost` means `P(TSV=-3) + P(TSV=+3)`; it is an endpoint-only bounding sensitivity.
- Quantiles in the global continuous table use a deterministic probability histogram with resolution 0.0001.

## Endpoint-only continuous summary: primary ordinal model

| metric | n | mean | p50_approx | p90_approx | p95_approx | p99_approx | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| any_zone | 2405376 | 0.057 | 0.030 | 0.136 | 0.180 | 0.197 | 0.706 |
| area_weighted_mean | 2405376 | 0.025 | 0.019 | 0.044 | 0.059 | 0.099 | 0.440 |
| cold_equal | 2405376 | 0.003 | 0.003 | 0.003 | 0.003 | 0.004 | 0.015 |
| equal_zone_mean | 2405376 | 0.027 | 0.019 | 0.052 | 0.067 | 0.106 | 0.505 |
| warm_equal | 2405376 | 0.024 | 0.017 | 0.050 | 0.066 | 0.106 | 0.505 |
| zone_p90 | 2405376 | 0.044 | 0.026 | 0.107 | 0.135 | 0.190 | 0.699 |

## Endpoint-only spatial screens: primary ordinal model

| threshold | equal_zone_mean_high_pct | area_weighted_mean_high_pct | zone_p90_high_pct | any_zone_high_pct | hidden_any_zone_pct | area_weighted_zone_time_high_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 0.025 | 33.304 | 29.093 | 55.823 | 69.332 | 36.028 | 29.562 |
| 0.050 | 10.987 | 7.269 | 23.631 | 30.338 | 19.351 | 7.240 |
| 0.075 | 3.794 | 2.795 | 16.735 | 23.874 | 20.080 | 4.612 |
| 0.100 | 1.421 | 0.926 | 12.457 | 20.299 | 18.878 | 3.313 |
| 0.150 | 0.121 | 0.061 | 3.721 | 7.654 | 7.533 | 0.596 |
| 0.200 | 0.042 | 0.014 | 0.307 | 0.737 | 0.695 | 0.059 |

## Broad-tail results at the manuscript's 0.20 screen

| model | equal_zone_mean_high_pct | area_weighted_mean_high_pct | zone_p90_high_pct | any_zone_high_pct | hidden_any_zone_pct | area_weighted_zone_time_high_pct |
| --- | --- | --- | --- | --- | --- | --- |
| no_pmv_ordinal | 13.847 | 12.758 | 29.086 | 37.521 | 23.674 | 13.172 |
| nominal | 14.339 | 12.738 | 27.865 | 35.161 | 20.822 | 12.928 |
| ordinal | 13.147 | 12.130 | 27.830 | 36.643 | 23.496 | 12.873 |
| stored_ordinal | 13.448 | 12.446 | 28.194 | 37.032 | 23.584 | 13.172 |

## Paired ordinal-versus-nominal comparison at 0.20

| aggregator | ordinal_high_pct | nominal_high_pct | nominal_minus_ordinal_pp | disagreement_pct | jaccard |
| --- | --- | --- | --- | --- | --- |
| any_zone | 36.643 | 35.161 | -1.482 | 4.752 | 0.876 |
| area_weighted_mean | 12.130 | 12.738 | 0.608 | 1.412 | 0.893 |
| equal_zone_mean | 13.147 | 14.339 | 1.192 | 1.689 | 0.884 |
| zone_p90 | 27.830 | 27.865 | 0.035 | 3.413 | 0.885 |

## Stored-versus-corrected same-state ordinal comparison at 0.20

| aggregator | stored_high_pct | corrected_high_pct | corrected_minus_stored_pp | disagreement_pct | jaccard |
| --- | --- | --- | --- | --- | --- |
| any_zone | 37.032 | 36.643 | -0.389 | 4.117 | 0.894 |
| area_weighted_mean | 12.446 | 12.130 | -0.316 | 1.113 | 0.913 |
| equal_zone_mean | 13.448 | 13.147 | -0.301 | 1.276 | 0.908 |
| zone_p90 | 28.194 | 27.830 | -0.364 | 2.729 | 0.907 |

## Prespecified decision rules

| rule | passed | material_claim_narrowing | detail |
| --- | --- | --- | --- |
| endpoint_spatial_underaggregation_all_prespecified_screens | True | False | Any-zone exceeds equal-zone-mean and hidden-any-zone is nonzero at every endpoint screen. |
| broad_tail_spatial_direction_both_models | True | False | {"nominal": true, "ordinal": true} |
| broad_tail_model_rate_difference_at_0_20 | True | False | Maximum absolute nominal-minus-ordinal difference across aggregators: 1.482 percentage points; disclosure threshold 5.0 pp. |
| stored_vs_corrected_spatial_direction | True | False | {"ordinal": true, "stored_ordinal": true} |
| stored_vs_corrected_rate_difference_at_0_20 | True | False | Maximum corrected-minus-stored absolute difference across headline aggregators: 0.389 pp; disclosure threshold 5.0 pp. |
| broad_tail_mean_probability_model_difference | True | False | Maximum absolute mean-probability difference: 0.01398; disclosure threshold 0.020. |
| ssp585_late_vs_baseline_direction_preserved | True | False | Direction agreement across prespecified screens/aggregators: 100.0% (negligible if both <1.0 pp). |
| stored_vs_corrected_ssp585_future_direction | True | False | Direction agreement across prespecified screens/aggregators: 100.0%. |

## Synchronized PMV scalar audit

| n_steps | corrected_mean_pmv_mean | stored_mean_pmv_mean | corrected_vs_stored_pmv_mae | corrected_vs_stored_pmv_rmse | corrected_vs_stored_pmv_max_abs | corrected_vs_stored_pmv_pearson_r | corrected_abs_pmv_vs_p_tail_pearson_r | corrected_pmv_central_but_tail_020_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2405376 | 0.284 | 0.288 | 0.024 | 0.042 | 0.501 | 0.997 | 0.859 | 0.003 |

## State-alignment audit

- Maximum absolute broad-tail probability difference between corrected same-state ordinal zone values and stored callback-timed values: 4.632e-01.
- Maximum absolute expected-TSV difference: 1.043e+00.
- The mismatch is preserved rather than reconciled away: the callback that generated the stored probability ran at the beginning of the zone timestep, whereas Ta/MRT/RH on the CSV row were recorded at its end. The corrected comparison predicts both models from the recorded state on that row.

## Held-out endpoint support and calibration

| model | component | n_test | support | observed_frequency | mean_predicted_probability | brier | ece_fixed_bins | auroc | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ordinal | outermost | 22223 | 1350 | 0.061 | 0.061 | 0.047 | 0.004 | 0.819 | 0.370 |
| ordinal | cold_endpoint | 22223 | 443 | 0.020 | 0.020 | 0.016 | 0.002 | 0.874 | 0.320 |
| ordinal | warm_endpoint | 22223 | 907 | 0.041 | 0.041 | 0.031 | 0.003 | 0.885 | 0.360 |
| nominal | outermost | 22223 | 1350 | 0.061 | 0.062 | 0.046 | 0.007 | 0.822 | 0.375 |
| nominal | cold_endpoint | 22223 | 443 | 0.020 | 0.021 | 0.016 | 0.003 | 0.872 | 0.326 |
| nominal | warm_endpoint | 22223 | 907 | 0.041 | 0.042 | 0.031 | 0.005 | 0.888 | 0.380 |

The endpoint event is unambiguous but thin: its two labels have substantially less support than the middle classes. Endpoint probability results therefore bound the primary analysis and should not be presented as a more stable or better-validated replacement outcome.

## Interpretation safeguards

- Screens are diagnostic conventions, not dissatisfaction, acceptability, or compliance rates.
- Model agreement in direction does not erase disclosed magnitude or classification disagreements.
- A nonzero endpoint pattern supports only the claim that the spatial diagnostic is not created solely by including TSV ±2.
- Full paths, versions, checksums, state hashes, thresholds, and decision rules are recorded in this directory.

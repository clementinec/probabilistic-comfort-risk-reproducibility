# Zone Aggregation Diagnostic

Source trace: `runs/diagnostic_reference/traces/medium_office_control_traces.csv`

## Key Results

- Across all occupied probability timesteps, mean aggregation gives 13.8% high-tail exposure at `p_disc >= 0.20`, while any-zone aggregation gives 37.5%.
- Mean aggregation hides at least one zone above `p_disc >= 0.20` in 23.7% of occupied probability timesteps.
- A max-zone aggregation would assign a more severe descriptive tail category in 27.6% of occupied probability timesteps; a p90-zone aggregation would do so in 19.6%.
- Aggregate `mu_TSV` and aggregate `d_tail` signs conflict in 1.49% of comparable timesteps (n=1873836), showing where expected sensation and directional tail exposure carry different information.
- Among states above the low reference threshold (`p_tail > 0.065`), the same sign conflict is 1.05% of comparable states.

## City-Level Summary

| city | mean_high_tail_020_pct | max_zone_high_tail_020_pct | hidden_tail_020_pct | max_zone_more_severe_pct | mu_d_tail_sign_conflict_pct_of_comparable |
| --- | --- | --- | --- | --- | --- |
| Ahmedabad | 17.54 | 55.03 | 37.49 | 36.19 | 0.00 |
| Beijing | 5.43 | 16.05 | 10.62 | 12.10 | 5.17 |
| Guangzhou | 12.67 | 32.66 | 19.98 | 30.06 | 1.55 |
| Houston | 12.46 | 29.42 | 16.97 | 31.64 | 2.40 |
| Kolkata | 19.72 | 46.71 | 26.99 | 34.03 | 0.47 |
| Phoenix | 15.26 | 45.26 | 30.00 | 21.76 | 0.01 |

## Tail-Direction Subsets

| subset | city | steps | comparable_steps | share_of_all_occupied_pct | mu_d_tail_sign_conflict_pct_of_comparable | mu_d_tail_sign_conflict_pct_of_subset |
| --- | --- | --- | --- | --- | --- | --- |
| above_low_tail_reference | All | 2061251 | 1854320 | 85.69 | 1.05 | 0.94 |
| above_upper_tail_reference | All | 73511 | 73511 | 3.06 | 0.00 | 0.00 |
| diagnostic_high_tail_p_disc_ge_020 | All | 333078 | 333078 | 13.85 | 0.00 | 0.00 |

### Low-reference subset by city

| subset | city | steps | comparable_steps | share_of_all_occupied_pct | mu_d_tail_sign_conflict_pct_of_comparable | mu_d_tail_sign_conflict_pct_of_subset |
| --- | --- | --- | --- | --- | --- | --- |
| above_low_tail_reference | Ahmedabad | 354523 | 339414 | 14.74 | 0.00 | 0.00 |
| above_low_tail_reference | Beijing | 362541 | 300014 | 15.07 | 5.09 | 4.21 |
| above_low_tail_reference | Guangzhou | 316149 | 278260 | 13.14 | 0.46 | 0.40 |
| above_low_tail_reference | Houston | 313666 | 262559 | 13.04 | 0.96 | 0.81 |
| above_low_tail_reference | Kolkata | 318610 | 303439 | 13.25 | 0.11 | 0.10 |
| above_low_tail_reference | Phoenix | 395762 | 370634 | 16.45 | 0.01 | 0.01 |

## Highest-Risk Zones

| zone | mean_p_disc | p95_p_disc | high_tail_020_pct | above_disc_up_pct |
| --- | --- | --- | --- | --- |
| perimeter_bot_zn_1 | 0.17 | 0.46 | 29.11 | 11.37 |
| perimeter_mid_zn_1 | 0.17 | 0.46 | 27.97 | 11.66 |
| perimeter_top_zn_1 | 0.15 | 0.40 | 23.80 | 9.53 |
| perimeter_mid_zn_3 | 0.13 | 0.38 | 17.20 | 5.72 |
| perimeter_top_zn_3 | 0.12 | 0.38 | 15.24 | 5.25 |
| core_mid | 0.11 | 0.29 | 12.80 | 2.78 |
| perimeter_bot_zn_3 | 0.11 | 0.31 | 12.55 | 3.41 |
| perimeter_top_zn_2 | 0.11 | 0.32 | 12.35 | 3.89 |

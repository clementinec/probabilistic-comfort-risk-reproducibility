# Zone Aggregation Diagnostic

Source trace: `runs/diagnostic_reference/traces/medium_office_control_traces.csv`

## Key Results

- Across all occupied probability timesteps, mean aggregation gives 13.4% high-tail exposure at `p_disc >= 0.20`, while any-zone aggregation gives 37.0%.
- Mean aggregation hides at least one zone above `p_disc >= 0.20` in 23.6% of occupied probability timesteps.
- A max-zone aggregation would assign a more severe descriptive tail category in 31.7% of occupied probability timesteps; a p90-zone aggregation would do so in 20.9%.
- Aggregate `mu_TSV` and aggregate `d_tail` signs conflict in 4.31% of comparable timesteps (n=1642322), showing where expected sensation and directional tail exposure carry different information.
- Among states above the low reference threshold (`p_tail > 0.065`), the same sign conflict is 1.74% of comparable states.

## City-Level Summary

| city | mean_high_tail_020_pct | max_zone_high_tail_020_pct | hidden_tail_020_pct | max_zone_more_severe_pct | mu_d_tail_sign_conflict_pct_of_comparable |
| --- | --- | --- | --- | --- | --- |
| Ahmedabad | 17.11 | 54.48 | 37.37 | 39.13 | 1.05 |
| Beijing | 5.48 | 15.55 | 10.07 | 15.31 | 11.46 |
| Guangzhou | 12.51 | 32.08 | 19.56 | 34.01 | 5.49 |
| Houston | 12.33 | 28.18 | 15.85 | 37.24 | 5.88 |
| Kolkata | 19.90 | 47.27 | 27.36 | 38.92 | 3.66 |
| Phoenix | 13.36 | 44.64 | 31.28 | 25.51 | 0.28 |

## Tail-Direction Subsets

| subset | city | steps | comparable_steps | share_of_all_occupied_pct | mu_d_tail_sign_conflict_pct_of_comparable | mu_d_tail_sign_conflict_pct_of_subset |
| --- | --- | --- | --- | --- | --- | --- |
| above_low_tail_reference | All | 2009297 | 1571422 | 83.53 | 1.74 | 1.36 |
| above_upper_tail_reference | All | 46914 | 46914 | 1.95 | 0.00 | 0.00 |
| diagnostic_high_tail_p_disc_ge_020 | All | 323481 | 323481 | 13.45 | 0.00 | 0.00 |

### Low-reference subset by city

| subset | city | steps | comparable_steps | share_of_all_occupied_pct | mu_d_tail_sign_conflict_pct_of_comparable | mu_d_tail_sign_conflict_pct_of_subset |
| --- | --- | --- | --- | --- | --- | --- |
| above_low_tail_reference | Ahmedabad | 351225 | 284350 | 14.60 | 0.00 | 0.00 |
| above_low_tail_reference | Beijing | 358102 | 258660 | 14.89 | 10.35 | 7.48 |
| above_low_tail_reference | Guangzhou | 302318 | 215885 | 12.57 | 0.01 | 0.01 |
| above_low_tail_reference | Houston | 289577 | 205783 | 12.04 | 0.04 | 0.03 |
| above_low_tail_reference | Kolkata | 313631 | 253937 | 13.04 | 0.01 | 0.01 |
| above_low_tail_reference | Phoenix | 394444 | 352807 | 16.40 | 0.12 | 0.10 |

## Highest-Risk Zones

| zone | mean_p_disc | p95_p_disc | high_tail_020_pct | above_disc_up_pct |
| --- | --- | --- | --- | --- |
| perimeter_bot_zn_1 | 0.16 | 0.40 | 30.51 | 10.98 |
| perimeter_mid_zn_1 | 0.16 | 0.42 | 27.91 | 12.11 |
| perimeter_top_zn_1 | 0.15 | 0.38 | 23.59 | 9.88 |
| perimeter_mid_zn_3 | 0.12 | 0.36 | 16.97 | 5.83 |
| perimeter_top_zn_3 | 0.11 | 0.36 | 15.01 | 5.33 |
| core_mid | 0.11 | 0.31 | 12.91 | 2.98 |
| perimeter_top_zn_2 | 0.11 | 0.31 | 12.86 | 3.88 |
| perimeter_mid_zn_2 | 0.11 | 0.31 | 12.61 | 3.44 |

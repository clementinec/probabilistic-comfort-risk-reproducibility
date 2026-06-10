# Zone Aggregation and Tail-Steering Diagnostic

Source traces: `traces/*_ordinal.csv.gz`

## Key Results

- Across all occupied probability timesteps, mean aggregation gives 9.9% high-tail exposure at `p_disc >= 0.20`, while any-zone aggregation gives 27.1%.
- Mean aggregation hides at least one zone above `p_disc >= 0.20` in 17.2% of occupied probability timesteps.
- A max-zone gate would be more conservative than the submitted mean-zone gate in 35.1% of occupied probability timesteps; a p90-zone gate would be more conservative in 25.6%.
- Aggregate `mu_TSV` and aggregate `d_tail` signs conflict in 4.50% of comparable timesteps (n=90295), supporting the current steering direction as a simple baseline while still motivating tail-based steering for coherence.
- Among gate-active states (`p_disc > 0.065`), the same sign conflict falls to 3.13% of comparable states; it is 0.00% for diagnostic high-tail states (`p_disc >= 0.20`) and for high-gate states (`p_disc >= 0.35`).

## City-Level Summary

| city | mean_high_tail_020_pct | max_zone_high_tail_020_pct | hidden_tail_020_pct | max_gate_more_conservative_pct | mu_d_tail_sign_conflict_pct_of_comparable |
| --- | --- | --- | --- | --- | --- |
| Beijing | 3.83 | 14.98 | 11.15 | 29.73 | 11.37 |
| Houston | 11.18 | 25.73 | 14.55 | 49.70 | 3.88 |
| Phoenix | 14.66 | 40.48 | 25.82 | 25.78 | 0.14 |

## Action-Conditioned Steering Coherence

| subset | city | steps | comparable_steps | share_of_all_occupied_pct | mu_d_tail_sign_conflict_pct_of_comparable | mu_d_tail_sign_conflict_pct_of_subset |
| --- | --- | --- | --- | --- | --- | --- |
| gate_active_p_disc_gt_disc_low | All | 106533 | 76534 | 70.86 | 3.13 | 2.25 |
| high_gate_p_disc_ge_disc_up | All | 2414 | 2414 | 1.61 | 0.00 | 0.00 |
| diagnostic_high_tail_p_disc_ge_020 | All | 14867 | 14867 | 9.89 | 0.00 | 0.00 |

### Gate-active by city

| subset | city | steps | comparable_steps | share_of_all_occupied_pct | mu_d_tail_sign_conflict_pct_of_comparable | mu_d_tail_sign_conflict_pct_of_subset |
| --- | --- | --- | --- | --- | --- | --- |
| gate_active_p_disc_gt_disc_low | Beijing | 34056 | 21449 | 22.65 | 10.99 | 6.92 |
| gate_active_p_disc_gt_disc_low | Houston | 24836 | 19128 | 16.52 | 0.11 | 0.08 |
| gate_active_p_disc_gt_disc_low | Phoenix | 47641 | 35957 | 31.69 | 0.05 | 0.04 |

## Highest-Risk Zones

| zone | mean_p_disc | p95_p_disc | high_tail_020_pct | above_disc_up_pct |
| --- | --- | --- | --- | --- |
| perimeter_bot_zn_1 | 0.13 | 0.38 | 18.97 | 8.34 |
| perimeter_mid_zn_1 | 0.13 | 0.38 | 18.13 | 7.77 |
| perimeter_top_zn_1 | 0.12 | 0.37 | 15.23 | 6.46 |
| perimeter_mid_zn_3 | 0.11 | 0.31 | 12.63 | 4.45 |
| perimeter_top_zn_3 | 0.10 | 0.31 | 11.38 | 4.24 |
| perimeter_top_zn_2 | 0.10 | 0.31 | 10.30 | 3.79 |
| perimeter_mid_zn_2 | 0.10 | 0.29 | 9.42 | 3.18 |
| perimeter_bot_zn_3 | 0.09 | 0.26 | 8.75 | 2.40 |

# Zone-Size Mosaic Diagnostic

Source traces: `/Users/guo/Library/CloudStorage/OneDrive-TheUniversityOfHongKong/Drafts/Probabilities_ENB/paperA_rebuild/runs/diagnostic_reference_zone_raw_full/traces`. High-tail cutoff: `p_tail >= 0.20`.

## Headline

- Mean-zone high-tail exposure averaged 13.45% across cases; any-zone exposure averaged 37.03%.
- Area-weighted zone-time high-tail exposure averaged 13.17%, versus 14.85% when all zones are weighted equally.
- Conditioned floor area represented in the mosaic is 4982.19 m2 across 15 zones.

## City Summary

| city | mean_zone_high_tail_pct | any_zone_high_tail_pct | hidden_any_zone_high_tail_pct | area_weighted_zone_time_high_tail_pct |
| --- | --- | --- | --- | --- |
| Ahmedabad | 17.11 | 54.48 | 37.37 | 17.66 |
| Beijing | 5.48 | 15.55 | 10.07 | 5.86 |
| Guangzhou | 12.51 | 32.08 | 19.56 | 11.92 |
| Houston | 12.33 | 28.18 | 15.85 | 11.06 |
| Kolkata | 19.90 | 47.27 | 27.37 | 20.73 |
| Phoenix | 13.36 | 44.64 | 31.28 | 11.81 |

## Highest-Risk Zones

| zone | zone_label | area_m2 | area_share_pct | mean_high_tail_pct | p95_case_high_tail_pct | mean_p_tail |
| --- | --- | --- | --- | --- | --- | --- |
| perimeter_bot_zn_1 | Bottom P1 | 207.34 | 4.16 | 30.51 | 57.48 | 0.16 |
| perimeter_mid_zn_1 | Middle P1 | 207.34 | 4.16 | 27.91 | 51.08 | 0.16 |
| perimeter_top_zn_1 | Top P1 | 207.34 | 4.16 | 23.59 | 46.00 | 0.15 |
| perimeter_mid_zn_3 | Middle P3 | 207.34 | 4.16 | 16.97 | 32.14 | 0.12 |
| perimeter_top_zn_3 | Top P3 | 207.34 | 4.16 | 15.01 | 30.54 | 0.11 |
| core_mid | Core middle | 983.54 | 19.74 | 12.91 | 29.15 | 0.11 |
| perimeter_top_zn_2 | Top P2 | 131.26 | 2.63 | 12.86 | 27.78 | 0.11 |
| perimeter_mid_zn_2 | Middle P2 | 131.26 | 2.63 | 12.61 | 28.05 | 0.11 |

## Zone Area and Risk Summary

| zone | zone_label | area_m2 | area_share_pct | mean_high_tail_pct | p95_case_high_tail_pct | max_case_high_tail_pct | mean_p_tail | p95_p_tail | area_rank_large_to_small |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| core_bottom | Core bottom | 983.54 | 19.74 | 7.26 | 23.09 | 34.23 | 0.10 | 0.22 | 1 |
| perimeter_bot_zn_1 | Bottom P1 | 207.34 | 4.16 | 30.51 | 57.48 | 67.60 | 0.16 | 0.38 | 4 |
| perimeter_bot_zn_2 | Bottom P2 | 131.26 | 2.63 | 10.37 | 25.90 | 33.36 | 0.10 | 0.26 | 10 |
| perimeter_bot_zn_3 | Bottom P3 | 207.34 | 4.16 | 11.67 | 26.92 | 41.00 | 0.11 | 0.28 | 5 |
| perimeter_bot_zn_4 | Bottom P4 | 131.25 | 2.63 | 8.35 | 23.36 | 30.74 | 0.10 | 0.24 | 13 |
| core_mid | Core middle | 983.54 | 19.74 | 12.91 | 29.15 | 44.12 | 0.11 | 0.28 | 2 |
| perimeter_mid_zn_1 | Middle P1 | 207.34 | 4.16 | 27.91 | 51.08 | 63.02 | 0.16 | 0.40 | 6 |
| perimeter_mid_zn_2 | Middle P2 | 131.26 | 2.63 | 12.61 | 28.05 | 38.01 | 0.11 | 0.29 | 11 |
| perimeter_mid_zn_3 | Middle P3 | 207.34 | 4.16 | 16.97 | 32.14 | 47.94 | 0.12 | 0.33 | 7 |
| perimeter_mid_zn_4 | Middle P4 | 131.25 | 2.63 | 10.99 | 25.14 | 33.44 | 0.10 | 0.28 | 14 |
| core_top | Core top | 983.54 | 19.74 | 11.31 | 27.41 | 38.37 | 0.10 | 0.27 | 3 |
| perimeter_top_zn_1 | Top P1 | 207.34 | 4.16 | 23.59 | 46.00 | 56.35 | 0.15 | 0.37 | 8 |
| perimeter_top_zn_2 | Top P2 | 131.26 | 2.63 | 12.86 | 27.78 | 36.45 | 0.11 | 0.30 | 12 |
| perimeter_top_zn_3 | Top P3 | 207.34 | 4.16 | 15.01 | 30.54 | 42.81 | 0.11 | 0.31 | 9 |
| perimeter_top_zn_4 | Top P4 | 131.25 | 2.63 | 10.45 | 22.88 | 30.04 | 0.10 | 0.28 | 15 |

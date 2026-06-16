# Zone Metabolic Profile Distribution

This diagnostic summarizes case-zone tail exposure across the six metabolic profiles.

## Headline

- Case-zone high-tail exposure spread across profiles has mean 16.85 percentage points and p95 36.15 percentage points.
- The gap between S-real and the highest profile has mean 16.64 percentage points and p95 35.39 percentage points.
- Largest mean profile spreads occur in Bottom P1 (30.4 pp), Middle P1 (28.6 pp), Top P1 (25.0 pp), Middle P3 (18.9 pp).

## Profile Distribution

| scenario | watts_person | met | case_zone_rows | mean_case_zone_high_tail_pct | median_case_zone_high_tail_pct | p75_case_zone_high_tail_pct | p95_case_zone_high_tail_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| COMP-Lo | 90.00 | 0.90 | 2160 | 3.27 | 1.86 | 4.50 | 11.61 |
| S-real | 93.20 | 0.93 | 2160 | 3.26 | 1.78 | 4.43 | 12.56 |
| COMP-Med | 100.00 | 1.00 | 2160 | 8.65 | 6.21 | 11.40 | 26.40 |
| EQ-Max | 108.00 | 1.08 | 2160 | 19.25 | 16.04 | 24.58 | 46.89 |
| COMP-Hi | 110.00 | 1.10 | 2160 | 15.37 | 11.97 | 19.97 | 39.78 |
| LEGACY | 120.00 | 1.20 | 2160 | 18.77 | 14.98 | 24.01 | 45.68 |

## Highest S-real Zone Exposures

| zone_label | floor | position_name | mean_high_tail_pct | p95_case_high_tail_pct | mean_p_tail |
| --- | --- | --- | --- | --- | --- |
| Middle P1 | middle | P1 south | 9.20 | 21.59 | 0.12 |
| Bottom P1 | bottom | P1 south | 7.87 | 22.43 | 0.12 |
| Top P1 | top | P1 south | 7.50 | 18.41 | 0.12 |
| Middle P3 | middle | P3 north | 3.73 | 13.83 | 0.10 |
| Top P3 | top | P3 north | 3.59 | 12.18 | 0.10 |
| Top P2 | top | P2 east | 2.58 | 8.32 | 0.10 |
| Middle P2 | middle | P2 east | 2.25 | 7.83 | 0.10 |
| Bottom P3 | bottom | P3 north | 2.09 | 9.19 | 0.09 |

## Largest Zone Profile Spreads

| zone_label | floor | position_name | mean_high_tail_pct_profile_spread | p95_high_tail_pct_profile_spread | mean_sreal_high_tail_gap_to_max_profile |
| --- | --- | --- | --- | --- | --- |
| Bottom P1 | bottom | P1 south | 30.44 | 46.99 | 29.88 |
| Middle P1 | middle | P1 south | 28.64 | 43.05 | 27.92 |
| Top P1 | top | P1 south | 24.96 | 39.86 | 24.35 |
| Middle P3 | middle | P3 north | 18.89 | 32.60 | 18.74 |
| Core middle | middle | Core | 16.54 | 30.55 | 16.50 |
| Top P3 | top | P3 north | 16.37 | 26.91 | 16.20 |
| Bottom P3 | bottom | P3 north | 16.32 | 29.02 | 16.19 |
| Core top | top | Core | 13.90 | 26.12 | 13.84 |

## Figures

- `zone_metabolic_profile_summary.pdf`
- `zone_metabolic_profile_zone_heatmap.pdf`

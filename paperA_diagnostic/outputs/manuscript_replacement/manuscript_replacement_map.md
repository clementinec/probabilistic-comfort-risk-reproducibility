# Manuscript Replacement Map

All corrected quantities below come from the synchronized end-of-zone-timestep inference. The legacy trace files remain unchanged. `legacy` values are retained for audit, not for reuse.

## Headline quantities

| location_or_object | legacy | corrected | unit | source | difference_corrected_minus_legacy |
| --- | --- | --- | --- | --- | --- |
| Abstract/results/conclusion: mean-zone screened share | 13.4483 | 13.1473 | % | global_threshold_curves.csv | -0.3010 |
| Results/conclusion: any-zone screened share | 37.0322 | 36.6432 | % | global_threshold_curves.csv | -0.3890 |
| Abstract/results/conclusion: mean-hidden any-zone share | 23.5839 | 23.4959 | % | global_threshold_curves.csv | -0.0881 |
| Full-panel mean p_tail | 0.1165 | 0.1156 | probability | corrected_headline_case_summary.csv | -0.0010 |
| Unique-source mean-zone screened share | 12.7289 | 12.4364 | % | corrected_unique_state_global_summary.csv | -0.2925 |
| Unique-source any-zone screened share | 35.7998 | 35.4149 | % | corrected_unique_state_global_summary.csv | -0.3849 |
| Unique-source mean-hidden any-zone share | 23.0708 | 22.9785 | % | corrected_unique_state_global_summary.csv | -0.0923 |
| No-PMV mean-zone screened share | 13.8472 | 13.8472 | % | global_threshold_curves.csv | 0.0000 |
| No-PMV any-zone screened share | 37.5200 | 37.5212 | % | global_threshold_curves.csv | 0.0012 |
| Nominal mean-zone screened share |  | 14.3388 | % | global_threshold_curves.csv |  |
| Nominal any-zone screened share |  | 35.1609 | % | global_threshold_curves.csv |  |
| Nominal mean-hidden any-zone share |  | 20.8220 | % | global_threshold_curves.csv |  |

## Nominal-model robustness at the 0.20 screen

| group_scope | event | threshold | aggregator | n_steps | ordinal_high_count | nominal_high_count | both_high_count | union_high_count | ordinal_only_count | nominal_only_count | disagreement_count | ordinal_high_pct | nominal_high_pct | nominal_minus_ordinal_pp | disagreement_pct | jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global | broad_tail | 0.2000 | any_zone | 2405376.0000 | 881406.0000 | 845751.0000 | 806425.0000 | 920732.0000 | 74981.0000 | 39326.0000 | 114307.0000 | 36.6432 | 35.1609 | -1.4823 | 4.7521 | 0.8759 |
| global | broad_tail | 0.2000 | area_weighted_mean | 2405376.0000 | 291773.0000 | 306403.0000 | 282109.0000 | 316067.0000 | 9664.0000 | 24294.0000 | 33958.0000 | 12.1300 | 12.7383 | 0.6082 | 1.4118 | 0.8926 |
| global | broad_tail | 0.2000 | equal_zone_mean | 2405376.0000 | 316242.0000 | 344903.0000 | 310255.0000 | 350890.0000 | 5987.0000 | 34648.0000 | 40635.0000 | 13.1473 | 14.3388 | 1.1915 | 1.6893 | 0.8842 |
| global | broad_tail | 0.2000 | zone_p90 | 2405376.0000 | 669417.0000 | 670249.0000 | 628781.0000 | 710885.0000 | 40636.0000 | 41468.0000 | 82104.0000 | 27.8300 | 27.8646 | 0.0346 | 3.4134 | 0.8845 |

## Corrected scalar audit

| model | scalar | rows | tail_threshold | pearson_r | spearman_r | standard_threshold | standard_disagreement_pct | standard_false_negative_pct | standard_false_positive_pct | standard_sensitivity_pct | standard_specificity_pct | best_threshold | best_disagreement_pct | best_false_negative_pct | best_false_positive_pct | best_sensitivity_pct | best_specificity_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ordinal | abs_mean_pmv | 2405376.0000 | 0.2000 | 0.8593 | 0.7516 | 0.5000 | 18.9264 | 0.0034 | 18.9230 | 99.9741 | 78.2126 | 0.9127 | 3.2657 | 1.5013 | 1.7644 | 88.5806 | 97.9685 |
| ordinal | abs_mu_tsv | 2405376.0000 | 0.2000 | 0.9755 | 0.8947 | 0.5000 | 3.4353 | 0.0414 | 3.3939 | 99.6854 | 96.0924 | 0.6084 | 1.0684 | 0.6078 | 0.4606 | 95.3766 | 99.4697 |

## Corrected no-PMV audit

| model | scalar | rows | tail_threshold | pearson_r | spearman_r | standard_threshold | standard_disagreement_pct | standard_false_negative_pct | standard_false_positive_pct | standard_sensitivity_pct | standard_specificity_pct | best_threshold | best_disagreement_pct | best_false_negative_pct | best_false_positive_pct | best_sensitivity_pct | best_specificity_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_pmv_ordinal | abs_mean_pmv | 2405376.0000 | 0.2000 | 0.8574 | 0.7379 | 0.5000 | 18.2427 | 0.0115 | 18.2312 | 99.9168 | 78.8386 | 0.9013 | 3.8988 | 1.9703 | 1.9286 | 85.7712 | 97.7615 |
| no_pmv_ordinal | abs_mu_tsv | 2405376.0000 | 0.2000 | 0.9761 | 0.8891 | 0.5000 | 4.3104 | 0.0289 | 4.2815 | 99.7913 | 95.0303 | 0.6271 | 1.0331 | 0.6401 | 0.3930 | 95.3771 | 99.5439 |

## Unique-environmental-state sensitivity

| scope | unique_states | equal_zone_mean_mean | area_weighted_mean_mean | zone_p90_mean | any_zone_mean | equal_zone_mean_high_pct | area_weighted_mean_high_pct | zone_p90_high_pct | any_zone_high_pct | hidden_any_zone_pct | area_weighted_zone_time_high_pct | unweighted_zone_time_high_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 119_unique_environmental_states_equal_weight | 119.0000 | 0.1132 | 0.1086 | 0.1598 | 0.1907 | 12.4364 | 11.4603 | 26.6574 | 35.4149 | 22.9785 | 12.1850 | 13.8261 |

## Figure disposition

| object | status | reason | replacement |
| --- | --- | --- | --- |
| Figure 1: expected TSV versus p_tail | REGENERATE | Both axes are recomputed on the synchronized end-of-step environmental state. | figures/mean_vs_tail_scatter_sample_corrected.pdf |
| Figure 2: scalar-tail comparison | REGENERATE | PMV, expected TSV, and p_tail are now all evaluated on the same recorded state. | figures/scalar_tail_comparison_hexbin_corrected.pdf |
| Figure 3: PMV-feature robustness | REGENERATE | Corrected PMV and both ordinal outputs are synchronized and labels are explicit. | figures/pmv_feature_robustness_hexbin_corrected.pdf |
| Threshold-sensitivity figure | REGENERATE | Use synchronized global_threshold_curves.csv; legacy directions can be checked but legacy coordinates should not be reused. | ../robustness_threshold_curves.pdf |
| Zone-size mosaic figure | REGENERATE | Every cell is recomputed from synchronized zone probabilities. | figures/zone_size_mosaic_heatmap_area.pdf |
| Zone/floor position figure | REGENERATE | Zone, floor, warm-tail, and cold-tail summaries use synchronized probabilities. | figures/zone_floor_position_comparison.pdf |
| Held-out model calibration/validation figures | REMAINS VALID | These use the held-out occupant dataset, not the callback-timed simulation trace. | None |
| Metabolic sensitivity figures | VERIFY/REGENERATE SEPARATELY | Valid only if their inference was recomputed from recorded environmental fields; do not infer validity from the corrected reference-state run alone. | See metabolic-analysis provenance. |

## Complete value ledger

The full old-to-corrected ledger is `manuscript_replacement_values.csv`; it includes scenario, time-slice, city, severity, zone, floor, scalar, nominal, and endpoint entries.

## Integration guardrails

- Use corrected quantities consistently in the abstract, Results, Discussion, Conclusions, tables, captions, and response letter.
- Do not mix stored callback-timed PMV/probabilities with corrected end-state quantities.
- Describe the nominal and endpoint analyses as robustness/sensitivity checks, not as external validation.
- Endpoint-only results are support-limited because TSV -3 and +3 are sparse in held-out data.
- Keep the 144 role-weighted panel and the 119 unique-state sensitivity distinct.

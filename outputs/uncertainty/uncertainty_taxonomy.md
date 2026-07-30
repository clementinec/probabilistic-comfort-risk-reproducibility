# Uncertainty taxonomy

## Provenance status

The first panel-bootstrap run used legacy case summaries whose stored TSV
probabilities and environmental columns referred to different positions within
the EnergyPlus timestep. Those panel intervals remain available in repository
history for auditability but are not part of the reported results.

The corrected same-state robustness artifacts subsequently passed the 144-role,
119-unique-state schema, hash, finite-array, and count checks. The reported
panel bootstraps are in `outputs/uncertainty/panel_primary/` and
`outputs/uncertainty/panel_endpoint/`. The independent contributor-clustered
held-out analyses are unaffected by trace timing.

## Purpose

The study does not collapse all uncertainty into one confidence interval. The
available artifacts support conditional resampling of cases and contributors,
but several larger uncertainties are structural and cannot be assigned
empirical coverage from this experiment. The table below separates them.

| Uncertainty source | What varies | Status in this study | What the result can support | What it cannot support |
|---|---|---|---|---|
| Selection-role case variation | Which city×scenario×time-slice blocks receive weight, with typical/hot/heatwave roles kept together | Quantified on corrected same-state results with 10,000 draws over 48 blocks | Conditional resampling intervals for global and spatial panel summaries | A confidence interval for regional building stock or future climate |
| Paired time-slice variation | Which of the 12 city×scenario blocks receive weight, with all four slices and three selector roles retained | Quantified on corrected same-state results with paired block bootstrap | Whether the baseline-to-future contrast is stable across the tested blocks | Attribution of the contrast solely to anthropogenic warming |
| Paired pathway variation | Which of six paired cities receive weight, with SSP2-4.5 and SSP5-8.5 retained together | Quantified, but only six resampling units | A descriptive across-test-city robustness interval | Regional or global SSP uncertainty |
| Selector overlap | Different severity selectors choosing the same physical source year | Preserved in role-weighted blocks and checked with 119 equal-weight source years | Whether role duplication materially changes the panel-wide pattern | Frequency or probability of typical, hot, or heatwave years |
| Contributor composition in pooled held-out validation | Which field-study contributors receive weight | Quantified with 5,000 contributor-cluster bootstrap draws over 63 contributors | Conditional intervals for Brier, fixed-bin ECE, AUROC, average precision, prevalence, and mean prediction | Independent-contributor transport, because the original train/test split was record-level |
| Predictor fitting and calibration estimation | Alternative training/calibration samples, fitted trees, and isotonic maps | Not propagated by the fixed-prediction bootstrap | Conditional performance of the saved fitted models | Full parameter or retraining uncertainty |
| Predictor specification | Ordinal versus nominal formulation, feature definitions, hyperparameters, and calibration method | Addressed by separate model-form sensitivities, not by this bootstrap | Robustness across explicitly tested model forms | Model-form completeness |
| Dataset transport | Climate, culture, building type, HVAC mode, contributor, and source shifts | Diagnosed by contributor-grouped and reciprocal cross-corpus validation | Evidence that transport is weaker than pooled random-split performance | External validation of the final pooled predictor on a wholly unused third corpus |
| Future-weather construction | Bias correction/downscaling, EPW generation, selected climate model, and source-data QA | Requires provenance and QA documentation; not sampled here | A conditional stress test under the selected weather construction | Climate-model ensemble uncertainty or probabilistic future-year claims |
| Annual weather selection | Deterministic typical, hot, and heatwave-extreme selectors and unequal candidate-window lengths | Structurally acknowledged; only selected years are simulated | Comparisons among the selected stress-test files | Event frequency, return period, or an unbiased estimate of decade means |
| Building-model structure | One DOE Medium Office, retained Denver design sizing, well-mixed zones, schedules, and equipment assumptions | Fixed by design | Isolation of weather/diagnostic behavior for that archetype | Generalization to other buildings, local code designs, retrofits, or equipment |
| EnergyPlus model-form and numerical uncertainty | Heat-balance assumptions, input accuracy, convergence warnings, and unresolved sub-zone physics | QA audited but not probabilistically quantified | Reproducibility within the stated simulation setup | A calibrated uncertainty band for real building temperatures or loads |
| Zone-to-exposure mapping | Occupant location, duration, density, solar patch, local air movement, and sub-zone gradients | Structural limitation | Modeled zone-state screening and aggregation loss | Occupant prevalence or measured personal exposure |
| Occupant-response structure | Clothing, activity, demographics, culture, expectations, preference, adaptation, and behavior | Mostly fixed; metabolic rate examined as one-factor response sensitivity | Conditional TSV-probability sensitivity to stated profiles | Future dissatisfaction, acceptability, vulnerability, or population prevalence |
| Diagnostic construct | Choice of TSV tail classes and probability screen | Threshold sweeps and endpoint-only sensitivity can bound it | Stability of conclusions across declared diagnostic definitions | A universal comfort or compliance threshold |
| Control and energy response | Trigger logic, equipment action, dwell time, stability, energy use, and rebound | Not simulated | Identification of possible control inputs for future testing | Comfort improvement, energy savings, or validated control effectiveness |

## What is quantitatively established now

All intervals below are conditional percentile ranges from resampling the existing
test design. They are not climate-projection confidence intervals.

- Against the calibrated PMV-only tail baseline, the primary TSV probability
  model has a paired Brier difference of -0.0235
  (-0.0293 to -0.0180), AUROC difference of +0.1767
  (+0.1375 to +0.2034), and average-precision difference of +0.2271
  (+0.1863 to +0.2597).
- Primary-versus-no-PMV paired differences include zero for Brier, AUROC, and
  average precision. The defensible description is therefore that their
  held-out tail performance is similar, not that either feature set is
  demonstrably superior.
- The endpoint-only event (`TSV in {-3,+3}`) has 1,350 observations in the
  22,223-record test split. The primary endpoint probability has Brier 0.0465
  (conditional contributor-bootstrap range 0.034--0.057), AUROC 0.819
  (0.773--0.858), and average precision 0.370 (0.250--0.481).
- Endpoint discrimination also remains stronger than the calibrated PMV-only
  endpoint baseline in paired draws: AUROC difference +0.181
  (+0.131--+0.221) and average-precision difference +0.260
  (+0.171--+0.328). This makes the endpoint test a viable support-limited
  bounding sensitivity, although it remains less supported than the primary
  outer-category event and does not validate dissatisfaction.
- On corrected same-state panel inference, mean-zone high-tail exceedance is
  13.15% (11.02--15.45), any-zone exceedance is 36.64% (32.39--40.87), and their
  paired gap is 23.50 percentage points (20.79--26.12).
- The corrected reference-to-near-2030s contrast remains indistinguishable from
  zero: -0.02 points (-2.00--+1.70). Mid-2050s and late-2080s contrasts remain
  positive: +5.05 points (+1.81--+9.03) and +11.93 points
  (+8.10--+16.31).
- Equal weighting of the 119 unique source years preserves the spatial pattern:
  mean-zone 12.44% (10.60--14.47), any-zone 35.41% (31.66--39.31), and hidden
  any-zone 22.98% (20.55--25.52).
- Endpoint-only spatial under-aggregation remains nonzero at every prespecified
  screen from 0.025 to 0.20. The role-weighted any-minus-mean gap ranges from
  36.03 points (34.58--37.48) at the 0.025 screen to 0.70 points
  (0.41--1.04) at the 0.20 screen.
- Endpoint mean-zone late-century contrasts remain positive through the 0.15
  screen but include zero at 0.20; any-zone late-century contrasts remain
  positive at every screen, including +1.24 points (+0.32--+2.31) at 0.20.
- Correcting the timestep alignment changes absolute broad-tail estimates by
  less than 0.4 percentage points and does not alter an emphasized inference.
  Legacy numbers remain available only for provenance and are not part of the
  reported results.

## ECE interpretation

Fixed-bin ECE is a descriptive, bin-dependent calibration summary. The PMV-only
baseline has a very small pooled point ECE (0.003), but its contributor-bootstrap
distribution is strongly right-skewed (2.5th--97.5th percentile range
0.004--0.044). The original point falling just below that percentile range is
possible because ECE is nonlinear and contributor resampling changes both
prevalence and bin composition. This should be reported as composition
sensitivity, not converted into a claim that one model is universally better
calibrated. Brier score and discrimination provide the cleaner model comparison
in this analysis.

## Methods summary

> Uncertainty summaries were computed at the level of independent design units
> rather than by treating 15-minute timesteps as independent. For panel-wide and
> spatial summaries, we used 10,000 fixed-seed bootstrap draws over 48
> city-by-scenario-by-time-slice blocks, retaining the three weather-selection
> roles within each sampled block so coincident source years remained paired.
> Time-slice contrasts resampled 12 city-by-scenario blocks with all four slices
> retained. We separately repeated global and paired time-slice summaries after
> giving each of the 119 distinct source-year trajectories equal weight while
> retaining those years within their 48 parent design blocks. All panel
> probabilities were recomputed from environmental values representing the same
> recorded timestep. For the pooled held-out TSV analysis, 5,000 fixed-seed
> draws resampled 63 contributors and assigned a common multiplicity to all test
> records from each selected contributor. Reported 2.5th--97.5th percentile
> ranges quantify conditional case or contributor composition only; they do not
> include climate-model, weather-construction, model-refitting, building, or
> future-occupant structural uncertainty.

## Results summary

> The main spatial result was stable under corrected same-state design-block
> resampling: mean-zone exceedance was 13.15% (conditional 2.5th--97.5th
> percentile range, 11.02--15.45), any-zone exceedance was 36.64%
> (32.39--40.87), and their paired gap was 23.50 percentage points
> (20.79--26.12). The near-2030s minus reference contrast included zero
> (-0.02 points, -2.00--+1.70), whereas the mid-2050s and late-2080s contrasts
> remained positive (+5.05 points, +1.81--+9.03; and +11.93 points,
> +8.10--+16.31). Equal weighting of the 119 distinct source years preserved the
> spatial and temporal ordering.

> The endpoint-only sensitivity preserved a nonzero any-zone-minus-mean-zone gap
> at every prespecified probability screen from 0.025 to 0.20. Its magnitude
> narrowed from 36.03 percentage points (34.58--37.48) at 0.025 to 0.70 points
> (0.41--1.04) at 0.20. This supports spatial under-aggregation as a
> definition-robust pattern while confirming that endpoint-only absolute rates
> depend strongly on the selected reporting screen.

> Contributor-clustered held-out resampling preserved the primary model's
> advantage over the calibrated PMV-only baseline in Brier score and
> discrimination. In contrast, paired differences between the primary and
> no-PMV TSV predictors included zero, supporting their interpretation as
> closely performing robustness variants rather than a demonstrated ranking.

## Limitations summary

> The resampling intervals are conditional on the selected six cities, two
> pathways, deterministic annual selectors, one climate-model/weather
> construction, one fixed building, and one fitted TSV predictor. Resampling
> existing case blocks does not quantify the larger structural uncertainties
> associated with climate-model ensembles, weather-file generation, building
> stock, HVAC adaptation, occupant behavior, model refitting, or transfer to a
> wholly unused occupant corpus. Accordingly, the future-weather results remain
> structured stress-test contrasts rather than probabilistic forecasts.

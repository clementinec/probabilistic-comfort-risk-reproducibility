# Overlap-excluded reciprocal source-holdout validation

Each retained target corpus was held out in full after the prespecified overlap exclusions. Preprocessing statistics, feature scaling, LightGBM fitting, and isotonic calibration used only the named source corpus. There was no target-source recalibration or fine-tuning.

| Direction | Features | Model | Exact | Within ±1 | MAE | Log loss | Tail prev. | Mean p_tail | Brier | ECE | AUROC | AP | F1@0.20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ASHRAE->China | full | ordinal | 42.9% | 81.9% | 0.800 | 2.148 | 16.5% | 0.225 | 0.151 | 0.096 | 0.592 | 0.259 | 31.3% |
| ASHRAE->China | full | nominal | 45.6% | 82.8% | 0.766 | 1.772 | 16.5% | 0.208 | 0.143 | 0.077 | 0.596 | 0.264 | 31.0% |
| ASHRAE->China | no_pmv | ordinal | 42.7% | 81.3% | 0.809 | 2.191 | 16.5% | 0.227 | 0.152 | 0.100 | 0.590 | 0.254 | 31.6% |
| ASHRAE->China | no_pmv | nominal | 44.8% | 82.0% | 0.787 | 1.903 | 16.5% | 0.206 | 0.144 | 0.078 | 0.589 | 0.256 | 30.7% |
| China->ASHRAE | full | ordinal | 39.8% | 77.8% | 0.886 | 2.184 | 22.3% | 0.121 | 0.189 | 0.124 | 0.553 | 0.269 | 25.2% |
| China->ASHRAE | full | nominal | 40.6% | 77.4% | 0.886 | 1.949 | 22.3% | 0.126 | 0.187 | 0.116 | 0.560 | 0.274 | 25.5% |
| China->ASHRAE | no_pmv | ordinal | 39.8% | 77.7% | 0.887 | 2.145 | 22.3% | 0.123 | 0.190 | 0.125 | 0.553 | 0.267 | 23.3% |
| China->ASHRAE | no_pmv | nominal | 40.5% | 77.5% | 0.886 | 1.848 | 22.3% | 0.124 | 0.187 | 0.117 | 0.560 | 0.274 | 25.2% |

## Split inventory

| Direction | Source total | Train | Calibration | Target test | Neutral exact | Neutral within ±1 | Neutral MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASHRAE->China | 104,354 | 85,938 | 18,416 | 40,421 | 49.7% | 83.5% | 0.730 |
| China->ASHRAE | 40,421 | 33,287 | 7,134 | 104,354 | 41.8% | 77.7% | 0.866 |

The neutral baseline predicts TSV = 0 for every target record. It is reported because Within ±1 can be high under a neutral-heavy class distribution without demonstrating useful transport.

## Excluded overlap strata

| Normalized city-year | Source label | Excluded records |
|---|---|---:|
| changsha:2007 | ASHRAE | 47 |
| changsha:2007 | China | 47 |
| harbin:2001 | ASHRAE | 240 |
| harbin:2001 | China | 119 |
| harbin:2009 | ASHRAE | 755 |
| harbin:2009 | China | 637 |
| harbin:2011 | ASHRAE | 482 |
| harbin:2011 | China | 384 |
| nanyang:2006 | ASHRAE | 175 |
| nanyang:2006 | China | 251 |
| yueyang:2007 | ASHRAE | 118 |
| yueyang:2007 | China | 118 |

## Specification

- Source split: TSV-stratified, random state 42; requested train/calibration fractions 0.823529/0.176471.
- LightGBM trees per model/head: 400; learning rate 0.05; unlimited depth; column fraction 0.9; no effective row bagging (`subsample=0.9`, `subsample_freq=0`); minimum child samples 20; L1/L2 regularization 0.
- Ordinal model: six cumulative binary heads, source-only isotonic calibration, cumulative monotonic repair, and seven-class reconstruction.
- Tail event: observed |TSV| >= 2; diagnostic screen: predicted p_tail >= 0.20.
- Tail ECE uses fixed probability bins [0,.05,.10,.15,.20,.30,.40,.60,1.00],
  matching the study's held-out validation.
- The nominal rows are a model-family sensitivity; the ordinal/full row is the exact primary model specification.

The six potentially overlapping city-year strata were removed from both source labels before splitting or evaluation.

Cross-corpus calibration is intentionally evaluated without access to the target distribution. Accordingly, mean-probability/prevalence gaps and ECE measure transport under dataset shift, not an in-domain recalibration result.

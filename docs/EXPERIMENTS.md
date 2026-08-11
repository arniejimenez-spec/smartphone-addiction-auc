# Experiment ledger

This file is the durable record of experiments that influence a submission.
Exploratory runs that do not change a modeling decision can remain in local
artifact logs.

## v1.0.0 — LightGBM baseline

Date: 2026-08-06

### Validation

- Split: stratified random 80/20 holdout
- Seed: 42
- Training rows: 553,095
- Validation rows: 138,274
- Metric: ROC AUC
- Validation AUC: **0.9633798201**
- Public leaderboard AUC: **0.96524**

### Model

- LightGBM binary classifier
- 1,800 boosting iterations
- Learning rate: 0.035
- Leaves: 47
- Minimum child samples: 40
- Row subsampling: 0.85
- Column subsampling: 0.90
- L1/L2 regularization: 0.15 / 2.0
- ID excluded

### Feature notes

The strongest features by gain were daily screen time, social-media hours,
weekend screen time, notifications per day, and app opens per day. Derived
features included leisure time, tracked/untracked screen time, weekend delta,
screen-to-sleep ratios, notifications per app open, and a missing-value count.

### Data observations

- Training target prevalence: 0.7094243.
- Target prevalence was stable across ID deciles, so ID was excluded.
- Numeric feature means were essentially unchanged between train and test.
- Missing-value frequencies shifted between train and test, motivating explicit
  missingness analysis in v2.

### Submission checks

- 296,302 test rows
- Exact sample-submission ID order
- All predictions finite and within `[0, 1]`
- Prediction range: 0.0001282 to 0.9999997
- Prediction mean: 0.7095798

## v2.0.0 - Fixed-fold model selection

Date: 2026-08-06

### Validation

- Split: five-fold stratified cross-validation
- Fold seed: 42
- Training rows: 691,369
- Test rows: 296,302
- LightGBM ceiling: 1,200 iterations per fold
- Metric: ROC AUC on complete out-of-fold predictions

| Candidate | OOF AUC | Decision |
|---|---:|---|
| `lgbm_a` | 0.9616050 | Reject |
| `lgbm_b` | 0.9615171 | Reject |
| `lgbm_c` | **0.9624332** | Select |
| Equal rank blend | 0.9621142 | Reject |
| Diversity-weighted rank blend | 0.9620934 | Reject |

The best coarse weight-grid result was 95% `lgbm_c` plus 5% `lgbm_b` at
0.9624352. Its 0.0000020 improvement was treated as noise rather than evidence
and was not selected.

### Selected model

`lgbm_c` uses 79 leaves, learning rate 0.03, minimum child size 110, row
subsampling 0.82, column subsampling 0.78, and stronger L1/L2 regularization.
The submission averages predictions from the five fold models. Fold AUCs were
0.9615732, 0.9622956, 0.9626279, 0.9634170, and 0.9622574.

### Missingness and distribution shift

Adversarial validation achieved 0.563091 AUC. The shift is mild and is driven
primarily by the missing-value pattern; numeric feature centers remain nearly
identical. V2 includes per-column missing indicators and a categorical missing
pattern while retaining native numeric missing-value handling.

### CatBoost decision

CatBoost fold 1 achieved 0.9582927 AUC with all 800 iterations and took 959
seconds. `lgbm_c` achieved 0.9615732 on the identical fold in 369 seconds.
The remaining CatBoost folds were stopped because the first-fold evidence did
not justify approximately another hour of computation.

### Submission checks

- Selected candidate: five-fold `lgbm_c`
- OOF AUC: **0.9624332**
- Prediction range: 0.0005036 to 0.9999961
- Prediction mean: 0.7094520
- V1/V2 test rank correlation: 0.9969586
- SHA-256: `c18cb72657da0d5e8fb8412526001cbdb12a7bae1a802384bb77b94fb16bf90e`
- Public leaderboard AUC: pending

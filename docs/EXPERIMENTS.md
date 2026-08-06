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


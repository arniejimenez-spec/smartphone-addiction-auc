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
- Public leaderboard AUC: **0.96524**
- Outcome: tied v1; the v1/v2 rank correlation of 0.9969586 correctly
  indicated that the new fold ensemble did not materially change test ranking.

## v3.0.0 - Label-free missing-feature reconstruction

Date: 2026-08-21

### Motivation

V2 OOF AUC was 0.972067 on rows with no missing fields, 0.967321 with one
missing field, and 0.945264 with two or more. Approximately 61% of training
rows contain at least one missing value. V1 and v2 test ranks correlated at
0.996959, so another near-identical tree ensemble was unlikely to help.

### Reconstruction validation

Feature reconstruction models use combined train/test covariates without
`addicted_label`. Each reported score is measured on 50,000 held-out observed
values.

| Feature | R-squared | MAE |
|---|---:|---:|
| Daily screen time | 0.814114 | 0.80119 |
| Weekend screen time | 0.632529 | 1.36521 |
| Social-media hours | 0.547022 | 0.60278 |
| Work/study hours | 0.453311 | 0.65384 |
| Gaming hours | 0.420733 | 0.50369 |

Sleep, notifications, and app opens were excluded from reconstruction after
diagnostic R-squared values near 0.02.

### Target-model validation

- Split: the same five stratified folds and seed 42 used by v2
- Maximum iterations: 2,000
- Learning rate: 0.03
- Leaves: 79
- Fold AUCs: 0.9624780, 0.9633357, 0.9635387, 0.9644580, 0.9632812
- Reconstructed model OOF AUC: **0.9634131**
- V2 baseline OOF AUC: 0.9624332
- Reconstructed-model gain: **+0.0009799**

All three missingness slices improved:

| Original missing count | Rows | V2 | Reconstructed | Gain |
|---|---:|---:|---:|---:|
| 0 | 269,185 | 0.972067 | 0.972743 | +0.000676 |
| 1 | 180,459 | 0.967321 | 0.968440 | +0.001118 |
| 2+ | 241,725 | 0.945264 | 0.946647 | +0.001383 |

### Specialist ablation

Fold-1 specialists trained only on their matching missingness bucket. Compared
with the reconstructed full model on the same validation rows:

| Bucket | Full model | Specialist | Specialist delta |
|---|---:|---:|---:|
| 0 missing | 0.971870 | 0.970429 | -0.001440 |
| 1 missing | 0.968473 | 0.965381 | -0.003091 |
| 2+ missing | 0.944863 | 0.941428 | -0.003434 |

The remaining specialist folds were stopped because every slice showed clear
underperformance. Completed checkpoints and logs remain in local artifacts.

### Blend selection

The coarse OOF grid selected an 80% reconstructed / 20% v2 percentile-rank
blend at **0.9634574**. This improves the reconstructed model by 0.0000443 and
exceeds the predefined 0.00002 minimum blend-gain threshold.

### Submission checks

- Selected candidate: 80% reconstructed / 20% v2 rank blend
- OOF AUC: **0.9634574**
- Prediction range: 0.0000034 to 0.9999939
- Prediction mean: 0.5000000 (expected for percentile-rank blending)
- Unique predictions: 275,135 of 296,302
- V2/V3 test rank correlation: 0.9982310
- SHA-256: `3f574728b61abfb3b5a03ab16d65be97635e69395c166b3aca9a19c9172de998`
- Public leaderboard AUC: **0.96459**
- Leaderboard delta versus v2: **-0.00065**

### Outcome and interpretation

The 0.001024 OOF improvement did not transfer to the public leaderboard. V3
therefore does not replace v2 as the champion. The disagreement is evidence
that the current IID folds do not fully represent the public evaluation set.
The most plausible contributors are the known train/test missing-pattern shift
and transductive reconstruction models learning feature relationships whose
utility varies under that shift. Public-leaderboard sampling noise may also
contribute, but the result is treated as a genuine warning rather than noise.

Future experiments should use validation that deliberately mirrors test
missingness, including adversarially weighted or missing-pattern-stratified
fold scoring. Feature changes should report both ordinary OOF AUC and a
test-likeness-weighted AUC before submission.

## v4.0.0 - Test-density-weighted conservative blend

Date: 2026-08-21

### Shift validation

A five-fold LightGBM domain classifier was trained on combined train/test
covariates without `addicted_label`. Its complete out-of-fold adversarial AUC
was **0.5640617**. The resulting normalized density weights ranged from
0.346728 to 3.398659, with an effective training size of 664,909 rows.

Exact missing-pattern weighting and full adversarial density weighting both
continued to prefer v3 in local scoring, despite v3's lower leaderboard score.
The v3 mismatch is therefore not explained by observable covariate shift alone,
and reconstructed features were excluded from the v4 target model.

### Target model and blend gate

The challenger uses v2's `lgbm_c` configuration, raw covariates, derived
features, and explicit missingness features. It changes only the target-model
training weights. The same five stratified folds and seed 42 are retained.

| Candidate | Ordinary OOF | Density-weighted OOF | Decision |
|---|---:|---:|---|
| V2 baseline | 0.9624332 | 0.9631260 | Incumbent |
| Weighted challenger | 0.9623248 | 0.9629730 | Reject alone |
| 75% v2 / 25% challenger | **0.9624633** | **0.9631460** | Select challenger |

The selected blend gains 0.0000302 ordinary AUC and 0.00001996 weighted AUC.
The weighted gain is within 3.95e-8 of the predefined 0.00002 gate and passes
only under the documented 1e-7 numerical comparison tolerance. No larger
challenger weight passed both gates.

### Submission checks

- Rows: 296,302 in exact sample-submission ID order
- Selected candidate: 75% v2 / 25% density-weighted LightGBM rank blend
- Prediction range: 0.000003375 to 0.999996625
- Prediction mean: 0.5000000
- Unique predictions: 270,391 of 296,302
- V2/V4 test rank correlation: 0.9999847
- V3/V4 test rank correlation: 0.9982480
- SHA-256: `120f12c9cdf085d1f97b9d4f68600bd87ad6deb41c1ff0d0d40500fa76990593`
- Public leaderboard AUC: **0.96357**
- Leaderboard delta versus v2: **-0.00167**
- Leaderboard delta versus v3: **-0.00102**

### Decision

V4 does not replace v2. Its very small local gains failed to transfer and the
submission scored below both v2 and v3. Test-density weighting is therefore
retained as a diagnostic but rejected as a model-selection gate for this
competition. The 0.9999847 V2/V4 test rank correlation also confirms that
closely related LightGBM blends are not producing useful new leaderboard
signal. V2 remains the champion at 0.96524.

## v5.0.0 - Robust standalone XGBoost challenger

Date: 2026-08-21

### Model-family diagnostic

A monotonic LightGBM challenger was rejected after scoring 0.952011 versus
0.961573 for v2 on the identical first fold. Hard marginal constraints removed
important conditional interactions. Histogram XGBoost with one-hot categorical
handling scored 0.963502 at 1,600 trees on that fold, a +0.001929 gain, and was
promoted to the robustness gate.

### Three-seed gate

Each candidate uses the first fold from a five-fold stratified splitter, giving
an 80/20 train/validation split. The v2 comparison is label-strict OOF scoring
on the same held-out rows.

| Split seed | XGBoost AUC | V2 OOF AUC | Gain |
|---:|---:|---:|---:|
| 42 | 0.9637986 | 0.9615732 | +0.0022254 |
| 17 | 0.9642194 | 0.9622253 | +0.0019941 |
| 83 | 0.9648232 | 0.9625589 | +0.0022644 |
| Mean | — | — | **+0.0021613** |

All three gains exceeded the 0.0005 per-seed threshold and the mean exceeded
the 0.0010 threshold, so the candidate advanced to full OOF training.

### Tree-ceiling ablation

The first five-fold run used 2,000 trees and scored 0.9645037 OOF, but every
fold's best iteration was at the ceiling. A fold-1 extension was required to
gain at least 0.0001; 3,000 trees improved fold 1 from 0.9637986 to 0.9640659,
a +0.0002673 gain. The longer ceiling was therefore applied to all folds.

### Final five-fold validation

| Fold | V2 AUC | XGBoost AUC | Gain | Best iteration |
|---:|---:|---:|---:|---:|
| 1 | 0.9615732 | 0.9640659 | +0.0024927 | 2,808 |
| 2 | 0.9622956 | 0.9646269 | +0.0023312 | 2,875 |
| 3 | 0.9626279 | 0.9647387 | +0.0021109 | 2,893 |
| 4 | 0.9634170 | 0.9656778 | +0.0022609 | 2,987 |
| 5 | 0.9622574 | 0.9644602 | +0.0022028 | 2,906 |
| OOF | 0.9624332 | **0.9647124** | **+0.0022792** | — |

The candidate improved every fold. V2/XGBoost OOF rank correlation is 0.99335.
The best blend used 90% XGBoost and scored 0.9647268, only +0.0000145 over
standalone XGBoost, so it failed the 0.0001 blend-gain threshold.

### Submission checks

- Rows: 296,302 in exact sample-submission ID order
- Selected candidate: standalone five-fold XGBoost rank average
- Prediction range: 0.000003375 to 0.999996625
- Prediction mean: 0.5000000
- Unique predictions: 244,379 of 296,302
- V2/V5 test rank correlation: 0.9948432
- V3/V5 test rank correlation: 0.9945099
- V4/V5 test rank correlation: 0.9948090
- SHA-256: `11058c10bd4f160029689cea8a72327a2fa67bd01efebce936573828dded02d3`
- Public leaderboard AUC: **0.96623**
- Leaderboard delta versus v2: **+0.00099**
- Leaderboard delta versus v3: **+0.00164**
- Leaderboard delta versus v4: **+0.00266**

### Decision

V5 replaces v2 as the champion. Its materially different model family,
three-seed gains, improvement on every full fold, and +0.002279 complete-OOF
gain transferred to a +0.00099 leaderboard improvement. The decision to use
standalone XGBoost rather than force an immaterial blend is retained.

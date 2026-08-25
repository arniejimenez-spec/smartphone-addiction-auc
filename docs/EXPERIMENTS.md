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

## v7.0.0 - Synthetic-value-encoded LightGBM

Date: 2026-08-22

### Source and scope

V7 adapts the selected non-pseudo-label pipeline from Naji's public
[Single LGBM Model LB 0.96990 CV 0.96862](https://www.kaggle.com/code/najiama/single-lgbm-model-lb-0-96990-cv-0-96862?scriptVersionId=344072919)
notebook, released under Apache-2.0. The source reports 0.96863 CV and 0.96988
public-LB AUC before pseudo-labeling. Its pseudo-label update lowers CV to
0.96861 and adds only 0.00002 public-LB AUC, so pseudo-labeling is excluded from
this candidate.

### Features and leakage control

The final 44 features contain:

- Nine raw numeric fields and two selected categorical fields
- Fifteen ratio, intensity, aggregate, and conditional screen-slack features
- Nine exact-value frequency encodings computed label-free on train plus test
- Nine exact-value target encodings fitted only on each model's training fold

The target-encoding smoothing prior uses the training fold's label mean, not
the global label mean used in the source notebook. Validation and test values
are mapped from the training fold only. Combined train/test frequency counts
are transductive but do not use `addicted_label`.

### Model

- Five-fold shuffled stratified CV, seed 42
- LightGBM GBDT with 127 leaves and 1,023 bins
- Learning rate 0.01 and 10,000-tree ceiling
- Feature fraction 0.34; bagging fraction 0.75 every five rounds
- Minimum child samples 200; L1 0.1; L2 1.0
- Deterministic column-wise training and 500-round early stopping

### Fold-1 gate

The candidate had to improve exact v5 fold-1 OOF AUC by at least 0.001.

| Model | Fold-1 AUC | Gain vs v5 | Best iteration | Decision |
|---|---:|---:|---:|---|
| V5 XGBoost | 0.9640659 | - | 2,808 | Baseline |
| V7 LightGBM | **0.9679357** | **+0.0038698** | 4,505 | Advance |

The v7 checkpoint matches the notebook's reported 0.96794 fold-1 result within
rounding.

### Complete five-fold validation

| Fold | V5 AUC | V7 AUC | Gain | Best iteration |
|---:|---:|---:|---:|---:|
| 1 | 0.9640659 | 0.9679357 | +0.0038698 | 4,505 |
| 2 | 0.9646269 | 0.9687011 | +0.0040742 | 3,902 |
| 3 | 0.9647387 | 0.9687474 | +0.0040087 | 4,118 |
| 4 | 0.9656778 | 0.9692309 | +0.0035531 | 4,758 |
| 5 | 0.9644602 | 0.9684003 | +0.0039401 | 4,547 |
| OOF | 0.9647124 | **0.9686010** | **+0.0038887** | - |

V5/V7 OOF rank correlation is 0.98672. A 90% v7 / 10% v5 rank blend scored
0.9686663, a +0.0000652 improvement over standalone v7. Because that gain is
below the pre-declared 0.0001 blend threshold, standalone v7 is selected.

### Submission checks

- Rows: 296,302 in exact sample-submission ID order
- Selected candidate: standalone five-fold v7 probability average
- Prediction range: 0.000220141 to 0.999999684
- Prediction mean: 0.7101952
- Unique predictions: 296,302 of 296,302
- SHA-256: `3ff72b4e2a830fa6bde86b143adab772ae4cb527cb343c82c6f434a5e3a3323e`
- Public leaderboard AUC: **0.96983**
- Leaderboard delta versus v5: **+0.00360**
- Distance from 0.97000: **0.00017**

### Decision

V7 clears the full advancement gate by a wide margin and scores 0.96983 on the
public leaderboard. It replaces v5 as champion after transferring a +0.003889
OOF gain into a +0.00360 leaderboard gain. The standalone selection is retained
because the locally stronger blend did not clear the reliability threshold.

## v8.0.0 - Cross-fitted public OOF meta-stack

Date: 2026-08-24

### Source and member audit

V8 reproduces the base meta-stack from Byer's Apache-2.0
[S6E8 Rank-Logit-Regime Fusion](https://www.kaggle.com/code/hboyang/s6e8-rank-logit-regime-fusion-lb0-97125)
notebook. The local audit loaded exactly 205 unique members from nine public
OOF libraries. All 691,369 OOF rows and 296,302 test rows were finite and in
the original competition order. ID-bearing CSV and Parquet sources were also
checked against `train.csv` and `test.csv` IDs.

Every member is converted to an average-tie percentile rank. The cached pool
has shapes `(691369, 205)` and `(296302, 205)` in float32. Competition data,
external libraries, and generated matrices remain excluded from Git.

### Meta-model

- Five-fold shuffled stratified CV, seed 42
- Standardization fitted separately inside each fold
- Logistic regression with `C=0.1`, `lbfgs`, tolerance `1e-5`
- Maximum 1,200 optimizer iterations
- Test probabilities averaged equally across the five fold models

The reference notebook reports 0.97022124036 for this base stack. The local
implementation scores 0.97021865083, a difference of only -0.00000258953.

### Complete five-fold validation

| Fold | V7 AUC | V8 AUC | Gain |
|---:|---:|---:|---:|
| 1 | 0.967935745 | 0.969592606 | +0.001656861 |
| 2 | 0.968701062 | 0.970332932 | +0.001631871 |
| 3 | 0.968747420 | 0.970298686 | +0.001551266 |
| 4 | 0.969230883 | 0.970901152 | +0.001670269 |
| 5 | 0.968400303 | 0.970003820 | +0.001603516 |
| OOF | 0.968601035 | **0.970218651** | **+0.001617615** |

The advancement gate requires at least +0.0005 pooled OOF AUC and a positive
gain on every fold. V8 passes both requirements comfortably.

### V7-member ablation

Adding percentile-ranked v7 as member 206 scores 0.97021752355, decreasing AUC
by 0.00000112729. The predeclared member-retention threshold is +0.00002, so
the exact 205-member public stack is selected.

### Submission checks

- Rows: 296,302 in exact v7/test ID order
- Selected candidate: standalone 205-member cross-fitted logistic stack
- Prediction range: 0.007310552 to 0.999993458
- Prediction mean: 0.709246347
- Unique predictions: 244,738 of 296,302
- V7/V8 test rank correlation: 0.994467920
- SHA-256: `17a5e30d2fbb1de656344c53df548a4ea4e15c6817a2395647ee6d79f957907c`
- Public leaderboard AUC: **0.97124**
- Leaderboard delta versus v7: **+0.00141**

### Scope of the 0.97125 reference result

The source notebook's published 0.97125 submission adds a 64-bit GPU
rank-logit model and a roughly 1,600-feature missingness/disagreement regime
model, then rank-blends that fusion with the cross-fitted base using alpha 0.7.
Its published run used Kaggle T4 GPUs and did not recompute fusion OOF
validation. V8 records only the base stack that was independently reproduced
and honestly validated locally; the public GPU output is not mislabeled as a
locally trained artifact.

### Decision

V8 scores **0.97124**, improving on v7 by **0.00141** and becoming the new
champion. Its large, consistent OOF gain and near-exact reproduction of the
source base metric transferred closely to the leaderboard. The decision to
exclude v7 from the final stack is retained because the 206th member decreased
honest OOF AUC.

## v9.0.0 - Honest rank-logit/regime fusion and ablations

Date: 2026-08-24

### Protocol

V9 uses the frozen 205-member ranked public OOF/test cache and the cross-fitted
v8 base prediction as member 206. Every result uses five shuffled stratified
outer folds with seed 42. Candidate-to-v8 blend weights are chosen only from
the other four outer folds, then applied to the untouched fold. Promotion
requires at least +0.0001 pooled OOF AUC over v8 and a positive gain on every
outer fold.

Three candidates were evaluated:

- **Fusion:** all 206 rank and clipped-logit features, plus a separate regime
  model using source-family rank/logit summaries interacted with completeness,
  heavy missingness, and member disagreement. The two predictions are rank
  mixed 55/45.
- **Stability:** 96 public members selected separately for each outer fold from
  three inner models trained only on that outer fold's fit rows.
- **Hierarchical:** source-family mean, spread, range, minimum, and maximum
  summaries with full regime interactions.

### Complete five-fold validation

| Fold | V8 AUC | Fusion AUC | Gain |
|---:|---:|---:|---:|
| 1 | 0.969592606 | 0.969841836 | +0.000249230 |
| 2 | 0.970332932 | 0.970579215 | +0.000246283 |
| 3 | 0.970298686 | 0.970507808 | +0.000209122 |
| 4 | 0.970901152 | 0.971032138 | +0.000130986 |
| 5 | 0.970003820 | 0.970150035 | +0.000146216 |
| OOF | 0.970218651 | **0.970422207** | **+0.000203556** |

Fusion passes both advancement requirements. Both of its logistic components
reached the fixed 1,000-iteration `lbfgs` cap on every fold. The scores and
predictions are valid deterministic bounded-solver results, but v9 does not
misstate them as fully converged optima.

### Ablations and nested blends

| Candidate | OOF AUC | Delta vs v8 | Decision |
|---|---:|---:|---|
| Hierarchical | 0.970201035 | -0.000017616 | Reject |
| V8 + hierarchical | 0.970226790 | +0.000008139 | Reject |
| Stability-pruned | 0.970238714 | +0.000020063 | Reject |
| V8 + stability | 0.970250634 | +0.000031983 | Reject |
| V8 + fusion | 0.970366102 | +0.000147451 | Reject; below raw fusion |
| Raw fusion | **0.970422207** | **+0.000203556** | Select |

The pruning result is positive on every fold but too small to justify replacing
or blending the full fusion. Source-family hierarchy is useful inside the
regime branch but loses signal as a standalone replacement for individual
members.

### Submission checks

- Rows: 296,302 in exact `test.csv` ID order
- Columns: `id`, `addicted_label`
- Prediction range: 0.000018225 to 0.999983802
- Prediction mean: 0.500001687
- Unique predictions: 289,565 of 296,302
- V8/V9 test rank correlation: 0.998052360
- SHA-256: `60c7769953ad5af1c05fdcedefb638b5531289886789e840932319d16f181cb9`

### Decision

Raw fusion is selected and `submission_v9.csv` is promoted to the repository
root. It is an honestly validated challenger; v8 remains the confirmed public
leaderboard champion until v9 receives a competition score.

### Leaderboard outcome

V9 scored **0.97123**, 0.00001 below v8's 0.97124. The +0.00020356 local OOF
gain did not transfer to a public-score improvement. The result is effectively
a tie at public leaderboard precision, but the strict version policy retains
v8 as champion and records v9 as non-promoting.

## v10.0.0 - Exact converged GPU fusion

Date started: 2026-08-25

### Hypothesis

V9 compressed the reference regime interactions to fit the local CPU machine
and both logistic models reached their 1,000-iteration ceilings. V10 tests the
remaining clean hypothesis: retain the exact 206-member, 412-column rank/logit
matrix and full 1,653-column regime matrix, then optimize both in float64 until
a numerical convergence condition is met on a Kaggle GPU.

### Exact construction

- 205 audited public members plus the five-fold v8 base prediction
- 206 source columns independently half-ranked for OOF and test
- 206 clipped-logit columns, producing 412 dual features
- Dual features repeated for complete rows, rows missing at least four fields,
  and standardized member-disagreement regimes
- Five aggregate columns: mean, standard deviation, range, complete indicator,
  and heavy-missingness indicator
- Total regime width: **1,653**
- Raw fusion: 55% ranked dual logits, 45% ranked regime logits
- Selected reference-style output: 70% ranked fusion, 30% v8 base prediction

### Convergence and reproducibility gate

The optimizer uses full-batch float64 PyTorch LBFGS with strong-Wolfe line
search. Training is divided into 250-iteration blocks with checkpoints and a
4,000-iteration initial budget. Each block records the objective, maximum
absolute gradient, maximum parameter step, directional derivative, closure
count, and elapsed time. A submission is written only after both fits meet the
gradient, parameter-change, or objective-change tolerance.

The selected closure differentiates the declared L2 term, yielding a consistent
`C=3.5` objective. A smoke test showed why this matters: the source closure,
which returns the penalty without differentiating it, can produce a zero
strong-Wolfe step while its gradient is still too large. An explicit
`--source-closure` mode remains available for reproduction, but it is not the
selected convergence-gated run.

### Input audit

- Training rows: 691,369
- Test rows: 296,302
- Audited members: 205 unique names
- OOF cache shape: `(691369, 205)`
- Test cache shape: `(296302, 205)`
- All cached values finite

GPU execution, convergence diagnostics, submission checksum, and leaderboard
score remain pending.

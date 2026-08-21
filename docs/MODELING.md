# Modeling decisions

## Metric implications

ROC AUC measures ordering between positive and negative examples. Calibration,
fixed classification thresholds, and prevalence matching do not directly
improve the competition metric. Ensembles therefore combine percentile ranks,
which put differently calibrated models on a common scale.

## Validation

The v1 score used one stratified holdout for speed. Starting with v2, model
changes are evaluated with fixed stratified folds and out-of-fold predictions.
Leaderboard movements are supporting evidence, not the main selection signal.

## ID handling

Train IDs are contiguous from 0 through 691,368 and test IDs continue from
691,369. Training target prevalence is flat across ID deciles and the feature
distributions show no meaningful numeric drift, so ID is excluded. This avoids
learning a sequence artifact that is unlikely to generalize.

## Missing values

Numeric missing values are handled natively by boosted trees. Categorical
missing values receive an explicit `__MISSING__` level. V2 also evaluates
per-column missing indicators because the missingness mix differs between the
training and test sets.

## Ensemble policy

Models enter the final blend only when they improve out-of-fold AUC or add
meaningful error diversity. Test predictions are averaged within each model
family across folds, converted to percentile ranks, and then blended using
weights fixed from out-of-fold results.

V2 confirmed why this rule matters: the strongest individual model family
scored 0.962433 OOF, while equal and preset diversity blends scored 0.962114
and 0.962093. The single family was selected rather than forcing a blend.

## Label-free feature reconstruction

V3 addresses the dominant error source: 61% of training rows have at least one
missing field, and v2 AUC declined sharply as missing count increased. Five
LightGBM regressors learn predictable feature-to-feature relationships using
combined train/test covariates without access to `addicted_label`. Actual
observed values are preserved; only missing values are reconstructed. Both the
original native-missing columns and reconstructed columns enter the target
model, allowing it to account for imputation uncertainty through the existing
missing indicators.

This transductive step uses test covariates but never test labels or target
proxies. Model selection remains based on target-strict out-of-fold predictions.

## Specialist rejection

Separate models for zero, one, and two-or-more missing fields were tested on
the exact first fold. They underperformed the full reconstructed model within
every corresponding slice by 0.00144, 0.00309, and 0.00343 AUC. The full model
benefits from substantially more training rows and was retained.

## OOF-to-leaderboard mismatch

V3 improved ordinary five-fold OOF AUC by 0.001024 but scored 0.00065 below v2
on the public leaderboard. Ordinary IID folds are therefore necessary but not
sufficient for this competition. New feature pipelines—especially those that
interact with missingness—must also be evaluated under a test-like validation
view derived from missing-pattern distributions or adversarial weights.

## Test-like validation and v4 release gate

V4 fits a separate five-fold domain classifier to distinguish training from
test covariates without using `addicted_label`. Its adversarial AUC is 0.564062,
confirming mild but learnable shift. Cross-fitted domain probabilities are
converted to normalized test/train density ratios; clipping limits the weights
to 0.347 through 3.399 and preserves an effective sample size of about 664,909.

Candidate predictions are scored twice: ordinary OOF AUC and density-weighted
OOF AUC. A blend must gain at least 0.00002 on both views, allowing only a
1e-7 numerical comparison tolerance. This is intentionally stricter than
selecting the best point on one OOF curve after the v3 leaderboard mismatch.

Density weighting did not make the standalone target model stronger. It did
change enough pairwise rankings to complement v2: the 25% challenger blend
gained 0.0000302 ordinary AUC and 0.00001996 density-weighted AUC. V4 is
therefore a cautious leaderboard challenger, while v2 remains the incumbent.

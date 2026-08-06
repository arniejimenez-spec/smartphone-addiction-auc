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

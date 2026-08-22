# Changelog

All notable modeling releases are documented here. Leaderboard scores are
added only after the corresponding submission is evaluated.

## v7.0.0 candidate - 2026-08-22

- Adapted the selected 44-feature, non-pseudo-label LightGBM design from
  Naji's Apache-2.0 Kaggle notebook.
- Added combined train/test exact-value frequency encodings for nine numeric
  fields and fold-local smoothed target encodings for the same values.
- Added ratio, intensity, and screen-time consistency/slack features that
  expose repeated structure in the synthetic data.
- Reproduced the notebook's fold-1 result at 0.9679357 and cleared the v5 gate
  by +0.0038698 AUC.
- Improved complete five-fold OOF AUC from 0.9647124 to **0.9686010**, with
  gains between +0.0035531 and +0.0040742 on every fold.
- Rejected the 90% v7 / 10% v5 blend because its +0.0000652 OOF gain was below
  the pre-declared +0.0001 blend threshold.
- Generated and validated standalone `submission_v7.csv` in the repository
  root and artifact directory; leaderboard score is pending.

## v5.0.0 - 2026-08-21

- Added XGBoost as a genuinely different model family with aligned one-hot
  categorical features and native numeric missing-value handling.
- Added a three-seed robustness gate requiring at least +0.0005 AUC on every
  split and +0.0010 mean gain over label-strict v2 OOF predictions.
- Passed the seed gate with gains of +0.002225, +0.001994, and +0.002264.
- Increased the final ceiling from 2,000 to 3,000 trees after a pre-declared
  fold-1 ablation gained 0.000267 AUC.
- Improved complete five-fold OOF AUC from 0.9624332 to **0.9647124**.
- Rejected a 90-95% XGBoost blend because its maximum 0.0000145 improvement
  over standalone XGBoost was below the 0.0001 blend threshold.
- Generated and validated the standalone `submission_v5.csv` candidate.
- Leaderboard score: **0.96623**.
- Outcome: improved on the v2 champion by **0.00099** and became the new
  champion model. The multi-seed, different-model-family validation transferred
  successfully to the leaderboard.

## v4.0.0 - 2026-08-21

- Added reproducible five-fold adversarial validation and cross-fitted
  train-to-test density weights.
- Added ordinary and density-weighted OOF AUC as independent release gates.
- Returned to v2's raw and explicit-missingness features; excluded v3 feature
  reconstruction after its leaderboard regression.
- Trained a density-weighted `lgbm_c` challenger on the same folds and model
  configuration as v2.
- Selected a conservative 75% v2 / 25% density-weighted percentile-rank blend.
- Improved ordinary OOF AUC from 0.9624332 to 0.9624633 and test-like weighted
  OOF AUC from 0.9631260 to 0.9631460.
- Added v4 unit tests, resumable fold checkpoints, and submission validation.
- Leaderboard score: **0.96357**.
- Outcome: underperformed v2 by 0.00167 and v3 by 0.00102; adversarial density
  weighting did not make local validation predictive of leaderboard movement.
  V2 remains the champion model.

## v3.0.0 - 2026-08-21

- Added label-free reconstruction for five predictable missing fields.
- Cached reconstructed covariates and all target-model fold predictions.
- Increased the target-model ceiling from 1,200 to 2,000 iterations.
- Improved complete five-fold OOF AUC from 0.962433 to 0.963413.
- Evaluated missingness specialists and rejected them after consistent
  first-fold underperformance across all three missingness buckets.
- Selected an 80% reconstructed / 20% v2 percentile-rank blend at 0.963457
  OOF AUC.
- Added v3 unit tests and reproducibility documentation.
- Leaderboard score: **0.96459**.
- Outcome: underperformed v2 by 0.00065 despite a 0.001024 OOF improvement;
  v2 remains the champion model.

## v2.0.0 - 2026-08-06

- Added fixed five-fold stratified out-of-fold validation.
- Added three LightGBM configurations and optional CatBoost evaluation.
- Added resumable fold-level validation and test-prediction checkpoints.
- Added explicit per-feature missingness and missing-pattern features.
- Added adversarial train-vs-test validation; measured AUC was 0.563091.
- Added rank-based blend evaluation with single-model candidate comparison.
- Selected the five-fold `lgbm_c` ensemble at 0.962433 OOF AUC.
- Rejected LightGBM family blends and CatBoost based on validation evidence.
- Added unit tests and a GitHub Actions workflow.
- Leaderboard score: **0.96524**.

## v1.0.0 - 2026-08-06

- Added the first reproducible LightGBM training and submission pipeline.
- Added domain-derived screen-time, engagement, and ratio features.
- Achieved 0.963380 on the fixed 80/20 validation holdout.
- Achieved **0.96524** on the competition leaderboard.

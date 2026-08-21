# Changelog

All notable modeling releases are documented here. Leaderboard scores are
added only after the corresponding submission is evaluated.

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

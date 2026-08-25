# Changelog

All notable modeling releases are documented here. Leaderboard scores are
added only after the corresponding submission is evaluated.

## v10.0.0 - 2026-08-25

- Added the exact 206-member rank-logit construction and full 1,653-column
  completeness, heavy-missingness, and disagreement regime feature space.
- Added a float64 PyTorch LBFGS GPU solver that trains in resumable blocks,
  checkpoints every block, and records objective, gradient, step, and
  convergence diagnostics.
- Corrected the reference closure so the declared L2 penalty contributes to
  both the gradient and objective; retained the defective source closure only
  as an explicit, non-default reproduction mode.
- Added a strict output gate: `submission_v10.csv` is not written unless both
  the dual and regime fits meet a numerical convergence criterion.
- Added a self-contained Kaggle GPU notebook, an input/run guide, six data-free
  tests, and a real audit of the 205-member local cache.
- Raised the adaptive per-fit ceiling from 4,000 to 20,000 iterations after the
  first T4 run reached `3.249e-7` maximum absolute gradient at iteration 4,000.
- Calibrated the first-order stopping gate to `5e-7` after the second T4 run
  showed that continuing the dual fit from iteration 3,500 (`4.342e-7`) through
  iteration 20,000 improved the mean objective by only `2.416e-8`. This remains
  over 37 times stricter than the reference's iteration-1,000 gradient.
- Completed the convergence-gated Kaggle GPU run and verified the selected
  296,302-row submission against the test IDs. All predictions are finite and
  inside `[0, 1]`; the submission SHA-256 is
  `5eb622b4e766badc90fbe5cb62541679df87f0ab93a1e9ca1df50bc9fca04fd9`.
- Leaderboard score: **0.97125**.
- Outcome: matched the published reference score, improved on v8 by 0.00001,
  and became the final project champion.

## v9.0.0 - 2026-08-24

- Added five-fold honest validation for full 206-member rank-logit fusion plus
  compressed completeness, missingness, and disagreement regime interactions.
- Added stability pruning that selects 96 members inside three inner folds of
  each outer-training split, preventing validation-fold selection leakage.
- Added a source-family hierarchical ablation and leakage-free nested blends
  against the v8 base prediction.
- Selected raw fusion at **0.97042221 OOF AUC**, improving on v8 by
  **0.00020356**, with positive fold gains from +0.00013099 to +0.00024923.
- Rejected standalone hierarchical compression (0.97020103), stability pruning
  (0.97023871), and all v8 blends because none beat raw fusion.
- Recorded that both fusion logistic fits reached the fixed 1,000-iteration
  limit on every fold; results are valid bounded-solver ablations rather than a
  claim of full numerical convergence.
- Generated and validated `submission_v9.csv` in the repository root. Public
  leaderboard score: **0.97123**.
- Outcome: effectively tied v8 but decreased by 0.00001, so v8 remains the
  champion and v9 is retained as a non-promoting ablation.

## v8.0.0 - 2026-08-24

- Added an exact loader for 205 aligned public OOF/test prediction members from
  nine downloaded libraries, with strict row, ID, shape, finiteness, member
  count, and member-order checks.
- Reproduced the reference public base meta-stack at **0.97021865 OOF AUC**, only
  0.00000259 below its published 0.97022124 result.
- Improved on v7 by **0.00161762 OOF AUC**, with gains from +0.00155127 to
  +0.00167027 across all five frozen folds.
- Evaluated v7 as a 206th meta member and rejected it because OOF AUC decreased
  slightly to 0.97021752.
- Added cached float32 member matrices, fold coefficients, complete OOF
  predictions, metrics, an advancement gate, and five data-free v8 tests.
- Generated and validated `submission_v8.csv` in the repository root and
  artifact directory.
- Leaderboard score: **0.97124**.
- Outcome: improved on v7 by **0.00141** and became the new champion. The
  +0.00161762 OOF gain transferred closely to the leaderboard.

## v7.0.0 - 2026-08-22

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
  root and artifact directory.
- Leaderboard score: **0.96983**.
- Outcome: improved on v5 by **0.00360** and became the new champion.

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

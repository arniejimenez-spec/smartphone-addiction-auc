# Smartphone Addiction Prediction

Reproducible tabular-modeling project for predicting `addicted_label`. The
competition metric is area under the ROC curve (ROC AUC), so model selection
focuses on ranking quality rather than probability calibration.

## Results

| Version | Model | Local validation | Leaderboard | Submission |
|---|---|---:|---:|---|
| v1.0.0 | LightGBM, 1,800 trees | 0.963380 | **0.96524** | `submission_lgbm.csv` |
| v2.0.0 | Five-fold LightGBM ensemble | 0.962433 OOF | **0.96524** | `submission_v2.csv` |
| v3.0.0 | Reconstructed features + v2 rank blend | **0.963457 OOF** | 0.96459 | `submission_v3.csv` |
| v4.0.0 | V2 + test-density-weighted rank blend | 0.962463 OOF | 0.96357 | `submission_v4.csv` |
| v5.0.0 | Five-fold standalone XGBoost | **0.964712 OOF** | **0.96623** | `submission_v5.csv` |
| v7.0.0 | Synthetic-value-encoded LightGBM | **0.968601 OOF** | **0.96983** | `submission_v7.csv` |
| v8.0.0 | 205-member cross-fitted OOF meta-stack | **0.970219 OOF** | **0.97124** | `submission_v8.csv` |
| v9.0.0 | Honestly validated rank-logit/regime fusion | **0.970422 OOF** | Pending | `submission_v9.csv` |

V8 is the confirmed leaderboard champion at **0.97124**, improving on v7 by
0.00141. Its +0.001618 OOF gain was positive on all five folds and transferred
closely to the public leaderboard.

The v1 validation score uses a fixed 80/20 stratified holdout with seed 42.
The competition data has 691,369 training rows, 296,302 test rows, nine
numeric features, three categorical features, and substantial missingness.

## Repository layout

```text
.
|-- train_model.py       # CatBoost baseline and shared v1 feature builder
|-- train_lgbm.py        # LightGBM holdout validation
|-- train_final.py       # Full-data v1 model and submission generation
|-- train_v2.py          # Resumable five-fold validation and ensembling
|-- train_v3.py          # Feature reconstruction, specialists, and v3 blend
|-- validate_v4.py       # Cross-fitted train-vs-test density weights
|-- train_v4.py          # Dual-gated density-weighted v4 challenger
|-- train_v5.py          # Multi-seed-gated XGBoost pipeline
|-- train_v7.py          # Synthetic-value encodings and deep LightGBM
|-- train_v8.py          # Aligned public OOF library and logistic meta-stack
|-- train_v9.py          # Rank-logit/regime fusion and honest ablations
|-- analyze_shift.py     # Initial adversarial train-vs-test validation
|-- docs/
|   |-- EXPERIMENTS.md   # Experiment ledger and leaderboard results
|   `-- MODELING.md      # Validation and modeling decisions
|-- tests/test_v2.py     # Data-free feature and ensemble unit tests
|-- requirements.txt
`-- CHANGELOG.md
```

Competition CSVs, trained models, local dependencies, and submissions are
deliberately excluded from Git. Place `train.csv`, `test.csv`, and
`sample_submission.csv` in the repository root before running the pipeline.

## Reproduce v7

V7 adapts the non-pseudo-label feature and LightGBM design from Naji's public
[Kaggle notebook](https://www.kaggle.com/code/najiama/single-lgbm-model-lb-0-96990-cv-0-96862?scriptVersionId=344072919),
released under Apache-2.0. It uses fold-local target smoothing and a gated,
resumable two-stage run:

```powershell
python train_v7.py --mode gate --run-name gate
python train_v7.py --mode full --run-name full --gate-run-name gate --resume
```

The selected submission is written to both
`artifacts/v7/full/submission_v7.csv` and root `submission_v7.csv`.

## Reproduce v8

V8 reconstructs the base stack from Byer's Apache-2.0
[Rank-Logit-Regime Fusion notebook](https://www.kaggle.com/code/hboyang/s6e8-rank-logit-regime-fusion-lb0-97125).
The required aligned public OOF libraries live under `external/v8` and are
excluded from Git. Audit every member before fitting:

```powershell
python train_v8.py --mode audit
python train_v8.py --mode train
```

The script rank-normalizes exactly 205 OOF/test members, fits a frozen
five-fold standardized logistic stack, and compares adding local v7 as a 206th
member. V7 is retained only for an OOF gain of at least 0.00002; it was rejected
because AUC decreased from 0.97021865 to 0.97021752. The selected base stack
cleared the release gate against v7 on every fold and writes
`artifacts/v8/full/submission_v8.csv` plus root `submission_v8.csv`.

The source notebook's final 0.97125 leaderboard file adds a much larger
rank-logit and missingness-regime fusion fit designed for a Kaggle T4 GPU. That
GPU-only layer is not represented as locally reproduced evidence here; v8's
tracked validation result is the independently reproduced base stack.

## Reproduce v9

V9 honestly validates the reference notebook's fusion idea and two independent
ablations on the same frozen seed-42 outer folds:

```powershell
python train_v9.py --candidates hierarchical,fusion,stability --resume
```

The selected fusion retains all 205 public members plus the cross-fitted v8
base prediction in rank and logit space. Its regime model uses source-family
summaries interacted with completeness, heavy-missingness, and disagreement
indicators. Stability pruning selects 96 members using three inner folds of the
outer-training rows only; hierarchical compression is evaluated separately.

Raw fusion scores **0.97042221 OOF**, a +0.00020356 gain over v8, with positive
gains on every fold. It passes the predeclared +0.0001/every-fold release gate
and writes `artifacts/v9/full/submission_fusion.csv` plus root
`submission_v9.csv`. Both fusion fits reached the fixed 1,000-iteration local
solver cap on every fold, so that bounded-optimization limitation is retained
in the experiment record.

## Reproduce v1

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_lgbm.py --iterations 1800 --holdout random --validation-only
python train_final.py --iterations 1800
```

The generated submission is written to `artifacts/submission_lgbm.csv`.

## Reproduce v2

The full experiment uses fixed five-fold out-of-fold validation and is
resumable at the model/fold level:

```powershell
python analyze_shift.py
python train_v2.py --folds 5 `
  --models lgbm_a,lgbm_b,lgbm_c,catboost `
  --lgbm-iterations 1200 --catboost-iterations 800 `
  --run-name full --resume
```

CatBoost was stopped after fold 1 because it scored 0.958293 versus 0.961573
for `lgbm_c` on the same fold and required roughly four times more runtime.
The final v2 artifact was rebuilt from the 15 completed LightGBM fold
checkpoints. The strongest validated candidate was the five-fold `lgbm_c`
ensemble; cross-family blends were rejected because they reduced OOF AUC. The
selected file is written to `artifacts/v2/full/submission_v2.csv`.

## Reproduce v3

V3 reconstructs five predictable missing numeric fields using label-free
feature models trained on the combined train/test covariates. Target models
remain strictly out-of-fold. Reconstruction and fold predictions are cached:

```powershell
python train_v3.py --folds 5 `
  --target-iterations 2000 `
  --reconstruction-iterations 500 `
  --run-name full --resume --skip-specialists
```

Missingness specialists were evaluated on fold 1 and rejected because they
underperformed the reconstructed full model in every missingness bucket. The
selected v3 submission is an 80% reconstructed / 20% v2 percentile-rank blend,
written to `artifacts/v3/full/submission_v3.csv`.

## Reproduce v4

V4 returns to v2's raw and missingness features. It builds out-of-fold
train-versus-test probabilities, converts them to density ratios, and trains a
weighted `lgbm_c` challenger. A rank blend is eligible only if it improves both
ordinary and density-weighted OOF AUC:

```powershell
python validate_v4.py --run-name full
python train_v4.py --run-name full --iterations 1200 --resume
```

The sole dual-gate candidate is 75% v2 plus 25% density-weighted LightGBM. Its
selected file is written to `artifacts/v4/full/submission_v4.csv`.

## Reproduce v5

V5 changes model family to histogram-based XGBoost with one-hot categorical
features. It must first beat label-strict v2 OOF predictions on three different
80/20 splits before full training is allowed:

```powershell
python train_v5.py --mode gate --run-name gate `
  --validation-seeds 42,17,83 --iterations 2000
python train_v5.py --mode full --run-name full3000 `
  --gate-run-name gate --iterations 3000 --resume
```

The final model is standalone XGBoost. A v2 blend was rejected because its
0.000014 OOF gain was below the 0.0001 blend threshold. The selected submission
is written to `artifacts/v5/full3000/submission_v5.csv`.

## Versioning policy

- Git tags mark submitted model generations (`v1.0.0`, `v2.0.0`, and so on).
- `docs/EXPERIMENTS.md` records validation design, AUC, leaderboard score, and
  the exact model configuration.
- Generated artifacts remain local; reproducible code and metrics are tracked.

See [CHANGELOG.md](CHANGELOG.md) and [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
for the release history and detailed evidence.

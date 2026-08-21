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

V2 remains the current leaderboard champion at **0.96524**. V3 is retained as
a documented negative result: its improved OOF score did not transfer to the
leaderboard. V4 is also a documented negative result: adversarially weighted
validation did not repair the local-to-leaderboard mismatch.

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

## Versioning policy

- Git tags mark submitted model generations (`v1.0.0`, `v2.0.0`, and so on).
- `docs/EXPERIMENTS.md` records validation design, AUC, leaderboard score, and
  the exact model configuration.
- Generated artifacts remain local; reproducible code and metrics are tracked.

See [CHANGELOG.md](CHANGELOG.md) and [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
for the release history and detailed evidence.

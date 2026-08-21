# Smartphone Addiction Prediction

Reproducible tabular-modeling project for predicting `addicted_label`. The
competition metric is area under the ROC curve (ROC AUC), so model selection
focuses on ranking quality rather than probability calibration.

## Results

| Version | Model | Local validation | Leaderboard | Submission |
|---|---|---:|---:|---|
| v1.0.0 | LightGBM, 1,800 trees | 0.963380 | **0.96524** | `submission_lgbm.csv` |
| v2.0.0 | Five-fold LightGBM ensemble | 0.962433 OOF | **0.96524** | `submission_v2.csv` |

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
|-- analyze_shift.py     # Adversarial train-vs-test validation
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

## Versioning policy

- Git tags mark submitted model generations (`v1.0.0`, `v2.0.0`, and so on).
- `docs/EXPERIMENTS.md` records validation design, AUC, leaderboard score, and
  the exact model configuration.
- Generated artifacts remain local; reproducible code and metrics are tracked.

See [CHANGELOG.md](CHANGELOG.md) and [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
for the release history and detailed evidence.

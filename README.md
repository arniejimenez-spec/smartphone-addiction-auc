# Smartphone Addiction Prediction

Reproducible tabular-modeling project for predicting `addicted_label`. The
competition metric is area under the ROC curve (ROC AUC), so model selection
focuses on ranking quality rather than probability calibration.

## Results

| Version | Model | Local validation | Leaderboard | Submission |
|---|---|---:|---:|---|
| v1.0.0 | LightGBM, 1,800 trees | 0.963380 | **0.96524** | `submission_lgbm.csv` |

The v1 validation score uses a fixed 80/20 stratified holdout with seed 42.
The competition data has 691,369 training rows, 296,302 test rows, nine
numeric features, three categorical features, and substantial missingness.

## Repository layout

```text
.
|-- train_model.py       # CatBoost baseline and shared v1 feature builder
|-- train_lgbm.py        # LightGBM holdout validation
|-- train_final.py       # Full-data v1 model and submission generation
|-- docs/
|   |-- EXPERIMENTS.md   # Experiment ledger and leaderboard results
|   `-- MODELING.md      # Validation and modeling decisions
`-- requirements.txt
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

## Versioning policy

- Git tags mark submitted model generations (`v1.0.0`, `v2.0.0`, and so on).
- `docs/EXPERIMENTS.md` records validation design, AUC, leaderboard score, and
  the exact model configuration.
- Generated artifacts remain local; reproducible code and metrics are tracked.


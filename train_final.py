"""Fit the selected full-data LightGBM model and create a Kaggle submission."""

from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from train_lgbm import categorical_dtypes, lgbm_params
from train_model import CAT_COLS, ID_COL, ROOT, TARGET, add_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    x_train = categorical_dtypes(add_features(train, include_id=False))
    x_test = categorical_dtypes(add_features(test, include_id=False))
    y = train[TARGET].astype(int)

    model = lgb.LGBMClassifier(**lgbm_params(args.seed, args.iterations))
    model.fit(
        x_train,
        y,
        categorical_feature=CAT_COLS,
        callbacks=[lgb.log_evaluation(100)],
    )
    pred = model.predict_proba(x_test)[:, 1]
    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: pred})

    expected = pd.read_csv(ROOT / "sample_submission.csv")
    assert list(submission.columns) == list(expected.columns)
    assert len(submission) == len(expected)
    assert submission[ID_COL].equals(expected[ID_COL])
    assert np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()

    submission_path = out_dir / "submission_lgbm.csv"
    model_path = out_dir / "lightgbm_model.txt"
    submission.to_csv(submission_path, index=False)
    model.booster_.save_model(str(model_path))
    summary = {
        "model": "LightGBM",
        "iterations": args.iterations,
        "seed": args.seed,
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "validation_auc": 0.9633798200527028,
        "prediction_min": float(pred.min()),
        "prediction_mean": float(pred.mean()),
        "prediction_max": float(pred.max()),
        "submission": str(submission_path),
        "saved_model": str(model_path),
    }
    (out_dir / "final_run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Build test-likeness weights for conservative v4 model selection.

The domain classifier is trained without ``addicted_label``. Its out-of-fold
probabilities are converted to clipped train-to-test density ratios that can
be used for both validation scoring and target-model fitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train_model import ID_COL, ROOT, TARGET
from train_v2 import V2_CAT_COLS, add_v2_features, as_lgbm_categories


def density_ratio(
    probability: np.ndarray,
    train_rows: int,
    test_rows: int,
    lower: float = 0.1,
    upper: float = 10.0,
) -> np.ndarray:
    """Convert P(test | x) to a normalized, clipped test/train ratio."""
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    ratio = probability / (1.0 - probability) * (train_rows / test_rows)
    ratio = np.clip(ratio, lower, upper)
    return ratio / ratio.mean()


def adversarial_config(seed: int, iterations: int) -> dict:
    return {
        "objective": "binary",
        "n_estimators": iterations,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 100,
        "colsample_bytree": 0.90,
        "reg_lambda": 5.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--run-name", default="full")
    args = parser.parse_args()

    run_dir = ROOT / "artifacts" / "v4" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    combined = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
    domain_y = np.r_[np.zeros(len(train), dtype=np.int8), np.ones(len(test), dtype=np.int8)]

    domain_x = add_v2_features(combined)
    domain_x, _ = as_lgbm_categories(domain_x, domain_x.iloc[:0].copy())
    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    domain_oof = np.zeros(len(combined), dtype=float)
    fold_records: list[dict] = []
    for fold, (fit_idx, valid_idx) in enumerate(splitter.split(domain_x, domain_y), start=1):
        model = lgb.LGBMClassifier(**adversarial_config(args.seed + fold, args.iterations))
        model.fit(
            domain_x.iloc[fit_idx],
            domain_y[fit_idx],
            eval_set=[(domain_x.iloc[valid_idx], domain_y[valid_idx])],
            eval_metric="auc",
            categorical_feature=V2_CAT_COLS,
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        prediction = model.predict_proba(
            domain_x.iloc[valid_idx], num_iteration=model.best_iteration_
        )[:, 1]
        domain_oof[valid_idx] = prediction
        fold_records.append({
            "fold": fold,
            "auc": float(roc_auc_score(domain_y[valid_idx], prediction)),
            "best_iteration": int(model.best_iteration_),
        })

    train_probability = domain_oof[: len(train)]
    weights = density_ratio(train_probability, len(train), len(test))
    np.save(run_dir / "adversarial_oof.npy", domain_oof)
    pd.DataFrame({
        ID_COL: train[ID_COL],
        "adversarial_probability": train_probability,
        "density_weight": weights,
    }).to_csv(run_dir / "train_density_weights.csv", index=False)

    metrics = {
        "version": "v4.0.0",
        "folds": args.folds,
        "seed": args.seed,
        "adversarial_auc": float(roc_auc_score(domain_y, domain_oof)),
        "effective_training_rows": float(weights.sum() ** 2 / np.square(weights).sum()),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "folds_detail": fold_records,
    }
    (run_dir / "adversarial_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

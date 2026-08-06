"""Fast LightGBM training pipeline for the smartphone addiction competition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from train_model import CAT_COLS, ID_COL, ROOT, TARGET, add_features


def lgbm_params(seed: int, n_estimators: int) -> dict:
    return {
        "objective": "binary",
        "n_estimators": n_estimators,
        "learning_rate": 0.035,
        "num_leaves": 47,
        "max_depth": -1,
        "min_child_samples": 40,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.15,
        "reg_lambda": 2.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }


def categorical_dtypes(x: pd.DataFrame) -> pd.DataFrame:
    for col in CAT_COLS:
        x[col] = x[col].astype("category")
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout", choices=["random", "forward"], default="random")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--include-id", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()

    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    train = pd.read_csv(ROOT / "train.csv")
    x = categorical_dtypes(add_features(train, include_id=args.include_id))
    y = train[TARGET].astype(int)

    if args.holdout == "random":
        train_idx, valid_idx = train_test_split(
            np.arange(len(train)),
            test_size=args.validation_size,
            random_state=args.seed,
            stratify=y,
        )
    else:
        split = int(len(train) * (1.0 - args.validation_size))
        train_idx = np.arange(split)
        valid_idx = np.arange(split, len(train))

    model = lgb.LGBMClassifier(**lgbm_params(args.seed, args.iterations))
    model.fit(
        x.iloc[train_idx],
        y.iloc[train_idx],
        eval_set=[(x.iloc[valid_idx], y.iloc[valid_idx])],
        eval_metric="auc",
        categorical_feature=CAT_COLS,
        callbacks=[lgb.early_stopping(120), lgb.log_evaluation(100)],
    )
    valid_pred = model.predict_proba(x.iloc[valid_idx], num_iteration=model.best_iteration_)[:, 1]
    auc = float(roc_auc_score(y.iloc[valid_idx], valid_pred))
    best_iteration = int(model.best_iteration_)
    print(f"VALIDATION_AUC={auc:.8f}")
    print(f"BEST_ITERATION={best_iteration}")

    tag = f"lgbm_{args.holdout}_seed{args.seed}" + ("_id" if args.include_id else "")
    metrics = {
        "holdout": args.holdout,
        "seed": args.seed,
        "validation_size": args.validation_size,
        "include_id": args.include_id,
        "validation_auc": auc,
        "best_iteration": best_iteration,
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(valid_idx)),
    }
    (out_dir / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({
        "feature": x.columns,
        "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False).to_csv(
        out_dir / f"feature_importance_{tag}.csv", index=False
    )

    if args.validation_only:
        return

    test = pd.read_csv(ROOT / "test.csv")
    x_test = categorical_dtypes(add_features(test, include_id=args.include_id))
    final_model = lgb.LGBMClassifier(**lgbm_params(args.seed, best_iteration))
    final_model.fit(x, y, categorical_feature=CAT_COLS)
    test_pred = final_model.predict_proba(x_test)[:, 1]

    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred})
    submission_path = out_dir / "submission_lgbm.csv"
    submission.to_csv(submission_path, index=False)
    final_model.booster_.save_model(str(out_dir / "lightgbm_model.txt"))
    print(f"SUBMISSION={submission_path}")


if __name__ == "__main__":
    main()

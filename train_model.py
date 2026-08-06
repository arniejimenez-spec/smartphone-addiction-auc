"""Train and validate CatBoost models for the smartphone addiction competition.

Run with the bundled Python runtime after adding ``python_packages`` to
``PYTHONPATH``. The script writes validation metrics, feature importances, a
saved model, and a submission file into ``artifacts/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
TARGET = "addicted_label"
ID_COL = "id"
CAT_COLS = ["gender", "stress_level", "academic_work_impact"]


def add_features(frame: pd.DataFrame, include_id: bool = False) -> pd.DataFrame:
    """Return model features with safe, domain-relevant interactions."""
    x = frame.drop(columns=[TARGET], errors="ignore").copy()
    if not include_id:
        x = x.drop(columns=[ID_COL])

    numeric_original = [
        c for c in x.columns if c not in CAT_COLS and pd.api.types.is_numeric_dtype(x[c])
    ]
    x["missing_count"] = frame.drop(columns=[TARGET, ID_COL], errors="ignore").isna().sum(axis=1)
    x["leisure_hours"] = x["social_media_hours"] + x["gaming_hours"]
    x["tracked_hours"] = x["leisure_hours"] + x["work_study_hours"]
    x["untracked_screen_hours"] = x["daily_screen_time_hours"] - x["tracked_hours"]
    x["weekend_screen_delta"] = x["weekend_screen_time"] - x["daily_screen_time_hours"]
    x["weekend_screen_ratio"] = x["weekend_screen_time"] / (x["daily_screen_time_hours"] + 0.25)
    x["screen_sleep_ratio"] = x["daily_screen_time_hours"] / (x["sleep_hours"] + 0.25)
    x["awake_screen_ratio"] = x["daily_screen_time_hours"] / (24.0 - x["sleep_hours"] + 0.25)
    x["notifications_per_open"] = x["notifications_per_day"] / (x["app_opens_per_day"] + 1.0)
    x["engagement_index"] = (
        x["daily_screen_time_hours"]
        + x["social_media_hours"]
        + x["gaming_hours"]
        + x["weekend_screen_time"]
    )

    # CatBoost requires categorical missing values to be represented explicitly.
    for col in CAT_COLS:
        x[col] = x[col].fillna("__MISSING__").astype(str)

    # Guard against infinities from derived ratios while retaining native NaNs.
    num_cols = list(dict.fromkeys(numeric_original + [
        "missing_count", "leisure_hours", "tracked_hours", "untracked_screen_hours",
        "weekend_screen_delta", "weekend_screen_ratio", "screen_sleep_ratio",
        "awake_screen_ratio", "notifications_per_open", "engagement_index",
    ]))
    x[num_cols] = x[num_cols].replace([np.inf, -np.inf], np.nan)
    return x


def model_params(seed: int, iterations: int) -> dict:
    return {
        "iterations": iterations,
        "learning_rate": 0.07,
        "depth": 8,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "l2_leaf_reg": 5.0,
        "random_strength": 0.5,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
        "random_seed": seed,
        "thread_count": -1,
        "allow_writing_files": False,
        "verbose": 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout", choices=["random", "forward"], default="random")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--include-id", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()

    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    train = pd.read_csv(ROOT / "train.csv")
    x = add_features(train, include_id=args.include_id)
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

    model = CatBoostClassifier(**model_params(args.seed, args.iterations))
    model.fit(
        x.iloc[train_idx],
        y.iloc[train_idx],
        cat_features=CAT_COLS,
        eval_set=(x.iloc[valid_idx], y.iloc[valid_idx]),
        early_stopping_rounds=120,
        use_best_model=True,
    )
    valid_pred = model.predict_proba(x.iloc[valid_idx])[:, 1]
    auc = float(roc_auc_score(y.iloc[valid_idx], valid_pred))
    best_iteration = int(model.get_best_iteration())
    print(f"VALIDATION_AUC={auc:.8f}")
    print(f"BEST_ITERATION={best_iteration}")

    tag = f"{args.holdout}_seed{args.seed}"
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
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False).to_csv(
        out_dir / f"feature_importance_{tag}.csv", index=False
    )

    if args.validation_only:
        return

    test = pd.read_csv(ROOT / "test.csv")
    x_test = add_features(test, include_id=args.include_id)
    final_iterations = max(50, best_iteration + 1)
    final_model = CatBoostClassifier(**model_params(args.seed, final_iterations))
    final_model.fit(x, y, cat_features=CAT_COLS)
    test_pred = final_model.predict_proba(x_test)[:, 1]

    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred})
    submission_path = out_dir / "submission_catboost.csv"
    submission.to_csv(submission_path, index=False)
    final_model.save_model(out_dir / "catboost_model.cbm")
    print(f"SUBMISSION={submission_path}")


if __name__ == "__main__":
    main()

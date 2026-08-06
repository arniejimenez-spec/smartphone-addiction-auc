"""Adversarial validation: quantify how distinguishable test is from train."""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from train_model import ROOT
from train_v2 import V2_CAT_COLS, add_v2_features, as_lgbm_categories


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    train_x = add_v2_features(train)
    test_x = add_v2_features(test)
    train_x, test_x = as_lgbm_categories(train_x, test_x)
    x = pd.concat([train_x, test_x], ignore_index=True)
    y = pd.Series(np.r_[np.zeros(len(train_x)), np.ones(len(test_x))].astype("int8"))
    tr_idx, va_idx = train_test_split(
        np.arange(len(x)), test_size=0.25, random_state=42, stratify=y
    )
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=500, learning_rate=0.05,
        num_leaves=31, min_child_samples=100, reg_lambda=5.0,
        colsample_bytree=0.9, random_state=42, n_jobs=-1, verbosity=-1,
    )
    model.fit(
        x.iloc[tr_idx], y.iloc[tr_idx],
        eval_set=[(x.iloc[va_idx], y.iloc[va_idx])], eval_metric="auc",
        categorical_feature=V2_CAT_COLS,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    pred = model.predict_proba(x.iloc[va_idx], num_iteration=model.best_iteration_)[:, 1]
    auc = float(roc_auc_score(y.iloc[va_idx], pred))
    importance = pd.DataFrame({
        "feature": x.columns,
        "gain": model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    output = ROOT / "artifacts" / "v2" / "adversarial_validation"
    output.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output / "feature_importance.csv", index=False)
    summary = {
        "auc": auc,
        "best_iteration": int(model.best_iteration_),
        "interpretation": "0.5 is indistinguishable; higher values indicate train/test shift",
        "top_features": importance.head(15).to_dict(orient="records"),
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

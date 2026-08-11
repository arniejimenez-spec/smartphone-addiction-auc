"""V2: fixed-fold out-of-fold validation and rank-averaged model ensemble.

The pipeline is resumable at the fold/model level. Generated predictions,
models, and submissions live under ``artifacts/v2`` and are ignored by Git.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train_model import CAT_COLS, ID_COL, ROOT, TARGET, add_features


V2_CAT_COLS = CAT_COLS + ["missing_pattern"]
BASE_FEATURES = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
    *CAT_COLS,
]


def add_v2_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add explicit missingness signals to the shared v1 features."""
    x = add_features(frame, include_id=False)
    missing = frame[BASE_FEATURES].isna()
    for col in BASE_FEATURES:
        x[f"{col}__missing"] = missing[col].astype("int8")
    bit_weights = (1 << np.arange(len(BASE_FEATURES), dtype=np.int64))
    x["missing_pattern"] = (
        missing.to_numpy(dtype=np.int64).dot(bit_weights).astype(str)
    )
    for col in V2_CAT_COLS:
        x[col] = x[col].fillna("__MISSING__").astype(str)
    return x


def as_lgbm_categories(
    train_x: pd.DataFrame, test_x: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign identical categorical levels across train and test."""
    train_x = train_x.copy()
    test_x = test_x.copy()
    for col in V2_CAT_COLS:
        levels = sorted(set(train_x[col].unique()) | set(test_x[col].unique()))
        train_x[col] = pd.Categorical(train_x[col], categories=levels)
        test_x[col] = pd.Categorical(test_x[col], categories=levels)
    return train_x, test_x


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Return stable average percentile ranks in the open interval (0, 1)."""
    return pd.Series(values).rank(method="average").to_numpy() / (len(values) + 1.0)


def lgbm_config(name: str, iterations: int, seed: int) -> dict:
    common = {
        "objective": "binary",
        "n_estimators": iterations,
        "n_jobs": -1,
        "verbosity": -1,
    }
    configs = {
        "lgbm_a": {
            "learning_rate": 0.035,
            "num_leaves": 47,
            "min_child_samples": 40,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.90,
            "reg_alpha": 0.15,
            "reg_lambda": 2.0,
            "random_state": seed,
        },
        "lgbm_b": {
            "learning_rate": 0.040,
            "num_leaves": 31,
            "min_child_samples": 80,
            "subsample": 0.90,
            "subsample_freq": 1,
            "colsample_bytree": 0.82,
            "reg_alpha": 0.30,
            "reg_lambda": 4.0,
            "random_state": seed + 95,
        },
        "lgbm_c": {
            "learning_rate": 0.030,
            "num_leaves": 79,
            "min_child_samples": 110,
            "subsample": 0.82,
            "subsample_freq": 1,
            "colsample_bytree": 0.78,
            "reg_alpha": 0.50,
            "reg_lambda": 6.0,
            "random_state": seed + 1_984,
        },
    }
    if name not in configs:
        raise ValueError(f"Unknown LightGBM model: {name}")
    return common | configs[name]


def catboost_config(iterations: int, seed: int) -> dict:
    return {
        "iterations": iterations,
        "learning_rate": 0.06,
        "depth": 7,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "l2_leaf_reg": 7.0,
        "random_strength": 0.7,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
        "random_seed": seed + 7,
        "thread_count": -1,
        "allow_writing_files": False,
        "verbose": 200,
    }


def train_lgbm_fold(
    name: str,
    train_x: pd.DataFrame,
    y: pd.Series,
    test_x: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    model = lgb.LGBMClassifier(**lgbm_config(name, iterations, seed))
    model.fit(
        train_x.iloc[train_idx],
        y.iloc[train_idx],
        eval_set=[(train_x.iloc[valid_idx], y.iloc[valid_idx])],
        eval_metric="auc",
        categorical_feature=V2_CAT_COLS,
        callbacks=[lgb.early_stopping(120), lgb.log_evaluation(200)],
    )
    best = int(model.best_iteration_)
    valid_pred = model.predict_proba(train_x.iloc[valid_idx], num_iteration=best)[:, 1]
    test_pred = model.predict_proba(test_x, num_iteration=best)[:, 1]
    return valid_pred, test_pred, best


def train_catboost_fold(
    train_x: pd.DataFrame,
    y: pd.Series,
    test_x: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    model = CatBoostClassifier(**catboost_config(iterations, seed))
    model.fit(
        train_x.iloc[train_idx],
        y.iloc[train_idx],
        cat_features=V2_CAT_COLS,
        eval_set=(train_x.iloc[valid_idx], y.iloc[valid_idx]),
        early_stopping_rounds=100,
        use_best_model=True,
    )
    best = int(model.get_best_iteration()) + 1
    valid_pred = model.predict_proba(train_x.iloc[valid_idx])[:, 1]
    test_pred = model.predict_proba(test_x)[:, 1]
    return valid_pred, test_pred, best


def save_submission(path: Path, ids: pd.Series, predictions: np.ndarray) -> None:
    if not np.isfinite(predictions).all():
        raise ValueError("Submission contains non-finite predictions")
    if not ((predictions >= 0) & (predictions <= 1)).all():
        raise ValueError("Submission predictions must be in [0, 1]")
    pd.DataFrame({ID_COL: ids, TARGET: predictions}).to_csv(path, index=False)


def normalized_weights(model_names: list[str], preset: str) -> dict[str, float]:
    if preset == "equal":
        raw = {name: 1.0 for name in model_names}
    else:
        defaults = {"lgbm_a": 0.30, "lgbm_b": 0.25, "lgbm_c": 0.25, "catboost": 0.20}
        raw = {name: defaults[name] for name in model_names}
    total = sum(raw.values())
    return {name: value / total for name, value in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models", default="lgbm_a,lgbm_b,lgbm_c,catboost",
        help="Comma-separated subset of lgbm_a,lgbm_b,lgbm_c,catboost",
    )
    parser.add_argument("--lgbm-iterations", type=int, default=1800)
    parser.add_argument("--catboost-iterations", type=int, default=1000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--run-name", default="full")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    allowed = {"lgbm_a", "lgbm_b", "lgbm_c", "catboost"}
    unknown = set(model_names) - allowed
    if unknown:
        raise ValueError(f"Unknown model(s): {sorted(unknown)}")

    run_dir = ROOT / "artifacts" / "v2" / args.run_name
    pred_dir = run_dir / "fold_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    if args.max_rows and args.max_rows < len(train):
        train = train.sample(args.max_rows, random_state=args.seed).sort_index().reset_index(drop=True)
    y = train[TARGET].astype(int)
    train_raw = add_v2_features(train)
    test_raw = add_v2_features(test)
    train_lgbm, test_lgbm = as_lgbm_categories(train_raw, test_raw)

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(splitter.split(train_raw, y))
    model_oof = {name: np.zeros(len(train), dtype=np.float64) for name in model_names}
    model_test = {name: np.zeros(len(test), dtype=np.float64) for name in model_names}
    fold_records: list[dict] = []

    for name in model_names:
        print(f"\n=== MODEL {name} ===", flush=True)
        for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
            valid_path = pred_dir / f"{name}_fold{fold}_valid.npy"
            test_path = pred_dir / f"{name}_fold{fold}_test.npy"
            meta_path = pred_dir / f"{name}_fold{fold}.json"
            started = time.time()
            if args.resume and valid_path.exists() and test_path.exists() and meta_path.exists():
                valid_pred = np.load(valid_path)
                test_pred = np.load(test_path)
                best_iteration = int(json.loads(meta_path.read_text())["best_iteration"])
                resumed = True
            else:
                print(f"Training {name}, fold {fold}/{args.folds}", flush=True)
                if name.startswith("lgbm"):
                    valid_pred, test_pred, best_iteration = train_lgbm_fold(
                        name, train_lgbm, y, test_lgbm, train_idx, valid_idx,
                        args.lgbm_iterations, args.seed + fold,
                    )
                else:
                    valid_pred, test_pred, best_iteration = train_catboost_fold(
                        train_raw, y, test_raw, train_idx, valid_idx,
                        args.catboost_iterations, args.seed + fold,
                    )
                np.save(valid_path, valid_pred)
                np.save(test_path, test_pred)
                resumed = False
            fold_auc = float(roc_auc_score(y.iloc[valid_idx], valid_pred))
            record = {
                "model": name,
                "fold": fold,
                "auc": fold_auc,
                "best_iteration": best_iteration,
                "seconds": round(time.time() - started, 2),
                "resumed": resumed,
            }
            meta_path.write_text(json.dumps(record, indent=2))
            fold_records.append(record)
            model_oof[name][valid_idx] = valid_pred
            model_test[name] += test_pred / args.folds
            print(json.dumps(record), flush=True)

    model_scores = {
        name: float(roc_auc_score(y, model_oof[name])) for name in model_names
    }
    oof_ranks = {name: percentile_rank(model_oof[name]) for name in model_names}
    test_ranks = {name: percentile_rank(model_test[name]) for name in model_names}

    blend_candidates: dict[str, tuple[dict[str, float], np.ndarray, np.ndarray]] = {
        f"single_{name}": ({name: 1.0}, model_oof[name], model_test[name])
        for name in model_names
    }
    for preset in ("equal", "diversity"):
        weights = normalized_weights(model_names, preset)
        blend_oof = sum(weights[name] * oof_ranks[name] for name in model_names)
        blend_test = sum(weights[name] * test_ranks[name] for name in model_names)
        blend_candidates[preset] = (weights, blend_oof, blend_test)

    blend_scores = {
        preset: float(roc_auc_score(y, values[1]))
        for preset, values in blend_candidates.items()
    }
    selected = max(blend_scores, key=blend_scores.get)
    selected_weights, selected_oof, selected_test = blend_candidates[selected]

    pd.DataFrame({
        ID_COL: train[ID_COL],
        TARGET: y,
        **{f"pred_{name}": model_oof[name] for name in model_names},
        "pred_blend": selected_oof,
    }).to_csv(run_dir / "oof_predictions.csv", index=False)
    for name in model_names:
        save_submission(run_dir / f"submission_{name}.csv", test[ID_COL], model_test[name])
    final_path = run_dir / "submission_v2.csv"
    save_submission(final_path, test[ID_COL], selected_test)

    summary = {
        "version": "v2.0.0",
        "run_name": args.run_name,
        "folds": args.folds,
        "seed": args.seed,
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "models": model_names,
        "model_oof_auc": model_scores,
        "blend_oof_auc": blend_scores,
        "selected_candidate": selected,
        "selected_weights": selected_weights,
        "selected_oof_auc": blend_scores[selected],
        "prediction_min": float(selected_test.min()),
        "prediction_mean": float(selected_test.mean()),
        "prediction_max": float(selected_test.max()),
        "submission": str(final_path),
        "folds_detail": fold_records,
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("\n=== V2 SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""V3: reconstruct predictable missing fields, then validate target specialists.

The reconstruction stage is label-free and transductive: it learns feature-to-
feature relationships from the combined train and test covariates without ever
using ``addicted_label``. Target models remain strictly out-of-fold.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from train_model import ID_COL, ROOT, TARGET
from train_v2 import (
    BASE_FEATURES,
    V2_CAT_COLS,
    add_v2_features,
    as_lgbm_categories,
    percentile_rank,
    save_submission,
)


RECONSTRUCTED_FEATURES = [
    "daily_screen_time_hours",
    "weekend_screen_time",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
]
RAW_CAT_COLS = ["gender", "stress_level", "academic_work_impact"]


def reconstruction_params(seed: int, iterations: int) -> dict:
    return {
        "objective": "regression",
        "n_estimators": iterations,
        "learning_rate": 0.05,
        "num_leaves": 47,
        "min_child_samples": 80,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.90,
        "reg_lambda": 3.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }


def target_params(seed: int, iterations: int, specialist: bool = False) -> dict:
    return {
        "objective": "binary",
        "n_estimators": iterations,
        "learning_rate": 0.03,
        "num_leaves": 63 if specialist else 79,
        "min_child_samples": 90 if specialist else 110,
        "subsample": 0.84 if specialist else 0.82,
        "subsample_freq": 1,
        "colsample_bytree": 0.82 if specialist else 0.78,
        "reg_alpha": 0.40 if specialist else 0.50,
        "reg_lambda": 5.0 if specialist else 6.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }


def prepare_raw_covariates(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    raw = pd.concat(
        [
            train.drop(columns=[ID_COL, TARGET]),
            test.drop(columns=[ID_COL]),
        ],
        ignore_index=True,
    )
    for col in RAW_CAT_COLS:
        raw[col] = raw[col].fillna("__MISSING__").astype("category")
    return raw


def validate_reconstruction(
    raw: pd.DataFrame,
    target: str,
    seed: int,
    iterations: int,
    max_rows: int = 250_000,
) -> dict:
    observed = np.flatnonzero(raw[target].notna().to_numpy())
    if len(observed) > max_rows:
        observed = np.random.default_rng(seed).choice(observed, max_rows, replace=False)
    train_idx, valid_idx = train_test_split(observed, test_size=0.2, random_state=seed)
    columns = [col for col in raw.columns if col != target]
    model = lgb.LGBMRegressor(**reconstruction_params(seed, iterations))
    model.fit(raw.iloc[train_idx][columns], raw.iloc[train_idx][target], categorical_feature=RAW_CAT_COLS)
    predictions = model.predict(raw.iloc[valid_idx][columns])
    return {
        "r2": float(r2_score(raw.iloc[valid_idx][target], predictions)),
        "mae": float(mean_absolute_error(raw.iloc[valid_idx][target], predictions)),
        "validation_rows": int(len(valid_idx)),
    }


def reconstruct_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cache_dir: Path,
    seed: int,
    iterations: int,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw = prepare_raw_covariates(train, test)
    train_x = add_v2_features(train)
    test_x = add_v2_features(test)
    metrics: dict[str, dict] = {}

    for offset, target in enumerate(RECONSTRUCTED_FEATURES):
        values_path = cache_dir / f"{target}.npy"
        meta_path = cache_dir / f"{target}.json"
        if resume and values_path.exists() and meta_path.exists():
            filled = np.load(values_path)
            meta = json.loads(meta_path.read_text())
            if len(filled) != len(raw):
                raise ValueError(f"Stale reconstruction cache for {target}")
        else:
            started = time.time()
            observed = raw[target].notna().to_numpy()
            columns = [col for col in raw.columns if col != target]
            validation = validate_reconstruction(
                raw, target, seed + offset, iterations
            )
            model = lgb.LGBMRegressor(
                **reconstruction_params(seed + offset, iterations)
            )
            model.fit(
                raw.loc[observed, columns],
                raw.loc[observed, target],
                categorical_feature=RAW_CAT_COLS,
            )
            filled = raw[target].to_numpy(dtype=np.float64, copy=True)
            missing = ~observed
            predicted = model.predict(raw.loc[missing, columns])
            lower = float(raw.loc[observed, target].min())
            upper = float(raw.loc[observed, target].max())
            filled[missing] = np.clip(predicted, lower, upper)
            meta = {
                "feature": target,
                "missing_rows": int(missing.sum()),
                "observed_rows": int(observed.sum()),
                "iterations": iterations,
                "seconds": round(time.time() - started, 2),
                **validation,
            }
            np.save(values_path, filled)
            meta_path.write_text(json.dumps(meta, indent=2))
        metrics[target] = meta
        new_col = f"{target}__reconstructed"
        train_x[new_col] = filled[: len(train)]
        test_x[new_col] = filled[len(train) :]
        print(f"Reconstruction ready: {target} {json.dumps(meta)}", flush=True)

    train_x, test_x = as_lgbm_categories(train_x, test_x)
    return train_x, test_x, metrics


def missing_bucket(frame: pd.DataFrame) -> np.ndarray:
    counts = frame[BASE_FEATURES].isna().sum(axis=1).to_numpy()
    return np.where(counts == 0, 0, np.where(counts == 1, 1, 2)).astype("int8")


def fit_target_model(
    train_x: pd.DataFrame,
    y: pd.Series,
    test_x: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    seed: int,
    iterations: int,
    specialist: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    model = lgb.LGBMClassifier(**target_params(seed, iterations, specialist))
    model.fit(
        train_x.iloc[train_idx],
        y.iloc[train_idx],
        eval_set=[(train_x.iloc[valid_idx], y.iloc[valid_idx])],
        eval_metric="auc",
        categorical_feature=V2_CAT_COLS,
        callbacks=[lgb.early_stopping(180), lgb.log_evaluation(250)],
    )
    best = int(model.best_iteration_)
    valid_pred = model.predict_proba(train_x.iloc[valid_idx], num_iteration=best)[:, 1]
    test_pred = model.predict_proba(test_x, num_iteration=best)[:, 1]
    return valid_pred, test_pred, best


def load_fold_checkpoint(
    pred_dir: Path, name: str, fold: int
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    valid_path = pred_dir / f"{name}_fold{fold}_valid.npy"
    test_path = pred_dir / f"{name}_fold{fold}_test.npy"
    meta_path = pred_dir / f"{name}_fold{fold}.json"
    if valid_path.exists() and test_path.exists() and meta_path.exists():
        return np.load(valid_path), np.load(test_path), json.loads(meta_path.read_text())
    return None


def save_fold_checkpoint(
    pred_dir: Path,
    name: str,
    fold: int,
    valid_pred: np.ndarray,
    test_pred: np.ndarray,
    meta: dict,
) -> None:
    np.save(pred_dir / f"{name}_fold{fold}_valid.npy", valid_pred)
    np.save(pred_dir / f"{name}_fold{fold}_test.npy", test_pred)
    (pred_dir / f"{name}_fold{fold}.json").write_text(json.dumps(meta, indent=2))


def grid_blends(
    y: pd.Series,
    oof: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    step: int = 10,
) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    names = list(oof)
    oof_rank = {name: percentile_rank(oof[name]) for name in names}
    test_rank = {name: percentile_rank(test[name]) for name in names}
    results: list[dict] = []
    blend_oof: dict[str, np.ndarray] = {}
    blend_test: dict[str, np.ndarray] = {}

    # Singles remain candidates in their original probability scale.
    for name in names:
        blend_oof[name] = oof[name]
        blend_test[name] = test[name]
        results.append({
            "name": name,
            "weights": {name: 1.0},
            "auc": float(roc_auc_score(y, oof[name])),
        })

    if len(names) == 2:
        for first in range(0, 101, step):
            weights = np.array([first, 100 - first], dtype=float) / 100.0
            if np.count_nonzero(weights) < 2:
                continue
            key = "blend_" + "_".join(
                f"{name}-{int(weight * 100):03d}"
                for name, weight in zip(names, weights)
            )
            blend_oof[key] = sum(
                weights[i] * oof_rank[name] for i, name in enumerate(names)
            )
            blend_test[key] = sum(
                weights[i] * test_rank[name] for i, name in enumerate(names)
            )
            results.append({
                "name": key,
                "weights": {
                    name: float(weights[i]) for i, name in enumerate(names)
                },
                "auc": float(roc_auc_score(y, blend_oof[key])),
            })
    elif len(names) == 3:
        for first in range(0, 101, step):
            for second in range(0, 101 - first, step):
                third = 100 - first - second
                weights = np.array([first, second, third], dtype=float) / 100.0
                if np.count_nonzero(weights) < 2:
                    continue
                key = "blend_" + "_".join(
                    f"{name}-{int(weight * 100):03d}"
                    for name, weight in zip(names, weights)
                )
                blend_oof[key] = sum(weights[i] * oof_rank[name] for i, name in enumerate(names))
                blend_test[key] = sum(weights[i] * test_rank[name] for i, name in enumerate(names))
                results.append({
                    "name": key,
                    "weights": {name: float(weights[i]) for i, name in enumerate(names)},
                    "auc": float(roc_auc_score(y, blend_oof[key])),
                })

    results.sort(key=lambda item: item["auc"], reverse=True)
    best_single = max(
        (item for item in results if item["name"] in names),
        key=lambda item: item["auc"],
    )
    best = results[0]
    # Require a material OOF gain before accepting tuned blend weights.
    if best["name"] not in names and best["auc"] - best_single["auc"] < 0.00002:
        best = best_single
    selection = {
        "selected": best,
        "best_grid_result": results[0],
        "best_single": best_single,
        "top_results": results[:15],
        "minimum_blend_gain": 0.00002,
    }
    return selection, blend_oof, blend_test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-iterations", type=int, default=2000)
    parser.add_argument("--reconstruction-iterations", type=int, default=500)
    parser.add_argument("--run-name", default="full")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-specialists", action="store_true")
    args = parser.parse_args()

    run_dir = ROOT / "artifacts" / "v3" / args.run_name
    pred_dir = run_dir / "fold_predictions"
    recon_dir = ROOT / "artifacts" / "v3" / "reconstructed_features"
    pred_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    y = train[TARGET].astype(int)
    train_x, test_x, reconstruction_metrics = reconstruct_features(
        train,
        test,
        recon_dir,
        args.seed,
        args.reconstruction_iterations,
        args.resume,
    )
    train_bucket = missing_bucket(train)
    test_bucket = missing_bucket(test)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(splitter.split(train_x, y))

    candidate_oof = {"reconstructed": np.zeros(len(train), dtype=np.float64)}
    candidate_test = {"reconstructed": np.zeros(len(test), dtype=np.float64)}
    if not args.skip_specialists:
        candidate_oof["specialists"] = np.zeros(len(train), dtype=np.float64)
        candidate_test["specialists"] = np.zeros(len(test), dtype=np.float64)
    fold_records: list[dict] = []

    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        print(f"\n=== RECONSTRUCTED FOLD {fold}/{args.folds} ===", flush=True)
        cached = load_fold_checkpoint(pred_dir, "reconstructed", fold) if args.resume else None
        if cached:
            valid_pred, test_pred, record = cached
            record["resumed"] = True
        else:
            started = time.time()
            valid_pred, test_pred, best = fit_target_model(
                train_x, y, test_x, train_idx, valid_idx,
                args.seed + fold, args.target_iterations, False,
            )
            record = {
                "candidate": "reconstructed",
                "fold": fold,
                "auc": float(roc_auc_score(y.iloc[valid_idx], valid_pred)),
                "best_iteration": best,
                "seconds": round(time.time() - started, 2),
                "resumed": False,
            }
            save_fold_checkpoint(pred_dir, "reconstructed", fold, valid_pred, test_pred, record)
        candidate_oof["reconstructed"][valid_idx] = valid_pred
        candidate_test["reconstructed"] += test_pred / args.folds
        fold_records.append(record)
        print(json.dumps(record), flush=True)

        if args.skip_specialists:
            continue
        print(f"=== SPECIALISTS FOLD {fold}/{args.folds} ===", flush=True)
        for bucket in (0, 1, 2):
            name = f"specialist_b{bucket}"
            fold_train_idx = train_idx[train_bucket[train_idx] == bucket]
            fold_valid_idx = valid_idx[train_bucket[valid_idx] == bucket]
            fold_test_idx = np.flatnonzero(test_bucket == bucket)
            cached = load_fold_checkpoint(pred_dir, name, fold) if args.resume else None
            if cached:
                bucket_valid, bucket_test, record = cached
                record["resumed"] = True
            else:
                started = time.time()
                bucket_valid, bucket_test, best = fit_target_model(
                    train_x,
                    y,
                    test_x.iloc[fold_test_idx],
                    fold_train_idx,
                    fold_valid_idx,
                    args.seed + 100 + 10 * bucket + fold,
                    args.target_iterations,
                    True,
                )
                record = {
                    "candidate": name,
                    "bucket": bucket,
                    "fold": fold,
                    "train_rows": int(len(fold_train_idx)),
                    "validation_rows": int(len(fold_valid_idx)),
                    "test_rows": int(len(fold_test_idx)),
                    "auc": float(roc_auc_score(y.iloc[fold_valid_idx], bucket_valid)),
                    "best_iteration": best,
                    "seconds": round(time.time() - started, 2),
                    "resumed": False,
                }
                save_fold_checkpoint(pred_dir, name, fold, bucket_valid, bucket_test, record)
            candidate_oof["specialists"][fold_valid_idx] = bucket_valid
            candidate_test["specialists"][fold_test_idx] += bucket_test / args.folds
            fold_records.append(record)
            print(json.dumps(record), flush=True)

    v2_oof_path = ROOT / "artifacts" / "v2" / "full" / "oof_predictions.csv"
    v2_test_path = ROOT / "artifacts" / "v2" / "full" / "submission_lgbm_c.csv"
    if v2_oof_path.exists() and v2_test_path.exists():
        v2_oof = pd.read_csv(v2_oof_path)
        v2_test = pd.read_csv(v2_test_path)
        if not v2_oof[ID_COL].equals(train[ID_COL]) or not v2_test[ID_COL].equals(test[ID_COL]):
            raise ValueError("V2 prediction artifacts do not align with current data")
        candidate_oof["v2_baseline"] = v2_oof["pred_lgbm_c"].to_numpy()
        candidate_test["v2_baseline"] = v2_test[TARGET].to_numpy()

    selection, all_oof, all_test = grid_blends(y, candidate_oof, candidate_test)
    selected_name = selection["selected"]["name"]
    selected_oof = all_oof[selected_name]
    selected_test = all_test[selected_name]

    oof_output = pd.DataFrame({ID_COL: train[ID_COL], TARGET: y})
    for name, predictions in candidate_oof.items():
        oof_output[f"pred_{name}"] = predictions
    oof_output["pred_selected"] = selected_oof
    oof_output.to_csv(run_dir / "oof_predictions.csv", index=False)
    for name, predictions in candidate_test.items():
        save_submission(run_dir / f"submission_{name}.csv", test[ID_COL], predictions)
    submission_path = run_dir / "submission_v3.csv"
    save_submission(submission_path, test[ID_COL], selected_test)

    slice_metrics = {}
    for bucket in (0, 1, 2):
        mask = train_bucket == bucket
        slice_metrics[str(bucket)] = {
            "rows": int(mask.sum()),
            "target_rate": float(y.to_numpy()[mask].mean()),
            **{
                name: float(roc_auc_score(y.to_numpy()[mask], predictions[mask]))
                for name, predictions in candidate_oof.items()
            },
        }
    summary = {
        "version": "v3.0.0",
        "run_name": args.run_name,
        "folds": args.folds,
        "seed": args.seed,
        "target_iterations": args.target_iterations,
        "reconstruction_iterations": args.reconstruction_iterations,
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "reconstruction": reconstruction_metrics,
        "candidate_oof_auc": {
            name: float(roc_auc_score(y, predictions))
            for name, predictions in candidate_oof.items()
        },
        "missing_bucket_oof_auc": slice_metrics,
        "selection": selection,
        "selected_oof_auc": float(roc_auc_score(y, selected_oof)),
        "prediction_min": float(selected_test.min()),
        "prediction_mean": float(selected_test.mean()),
        "prediction_max": float(selected_test.max()),
        "submission": str(submission_path),
        "folds_detail": fold_records,
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("\n=== V3 SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

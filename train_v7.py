"""V7: notebook-derived LightGBM with synthetic-value encodings.

The feature and model design is adapted from Naji's public Kaggle notebook,
"Single LGBM Model LB 0.96990 CV 0.96862" (Apache-2.0).  V7 deliberately
starts from the stronger non-pseudo-label version of that notebook:

* label-free exact-value frequencies are learned from train and test together;
* exact numeric values receive fold-local smoothed target encodings;
* ratio and screen-time consistency features expose the synthetic generator;
* a low-learning-rate LightGBM is trained with five fixed folds.

``gate`` evaluates fold 1 against the exact v5 OOF rows and saves both
validation and test predictions. ``full`` reuses that fold, trains the other
four, evaluates standalone and v5/v7 rank blends, and writes submission_v7.csv
to both the artifact directory and repository root when the gate passes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train_model import ID_COL, ROOT, TARGET
from train_v2 import percentile_rank, save_submission


NUM_COLS = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]
CAT_COLS = ["stress_level", "academic_work_impact"]
TARGET_ENCODED_COLS = [f"{col}_target_enc" for col in NUM_COLS]
GATE_MIN_GAIN = 0.0010
FULL_MIN_GAIN = 0.0010
BLEND_MIN_GAIN = 0.0001


def feature_engineering(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the notebook's selected ratio and screen-consistency features."""
    x = frame.drop(columns=[TARGET], errors="ignore").copy()
    active = ["social_media_hours", "gaming_hours", "work_study_hours"]
    daily = x["daily_screen_time_hours"]
    total = x[active].sum(axis=1, min_count=1)
    eps = 1e-9

    x["total_active_hours"] = total
    x["passive_screen_time"] = daily - total
    x["social_ratio"] = x["social_media_hours"] / (daily + eps)
    x["gaming_ratio"] = x["gaming_hours"] / (daily + eps)
    x["work_ratio"] = x["work_study_hours"] / (daily + eps)
    x["screen_to_sleep"] = daily / (x["sleep_hours"] + eps)
    x["weekend_vs_daily"] = x["weekend_screen_time"] / (daily + eps)
    x["app_opens_per_hour"] = x["app_opens_per_day"] / (daily + eps)
    x["notif_per_hour"] = x["notifications_per_day"] / (daily + eps)
    x["notif_per_app_open"] = (
        x["notifications_per_day"] / (x["app_opens_per_day"] + eps)
    )

    x["total_breakdown_hours"] = total
    missing_components = x[active].isna().sum(axis=1)
    daily_missing_with_components = daily.isna() & (missing_components == 0)
    one_component_missing = daily.notna() & (missing_components == 1)
    all_screen_known = daily.notna() & (missing_components == 0)
    slack = daily - total
    x["ci_daily_lb"] = np.where(daily_missing_with_components, total, np.nan)
    x["ci_missing_comp_ub"] = np.where(one_component_missing, slack, np.nan)
    x["ci_missing_comp_mid"] = np.where(
        one_component_missing, np.clip(slack / 2.0, 0, None), np.nan
    )
    x["ci_slack_exact"] = np.where(all_screen_known, slack, np.nan)
    return x


def prepare_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build aligned 44-column train/test frames before target encoding."""
    train_x = feature_engineering(train)
    test_x = feature_engineering(test)

    stress_map = {"Low": 0, "Medium": 1, "High": 2}
    train_x["stress_level"] = train_x["stress_level"].map(stress_map)
    test_x["stress_level"] = test_x["stress_level"].map(stress_map)

    academic = pd.concat(
        [train_x["academic_work_impact"], test_x["academic_work_impact"]],
        ignore_index=True,
    ).fillna("Unknown").astype(str)
    academic_levels = list(dict.fromkeys(academic.tolist()))
    academic_codes = {value: index for index, value in enumerate(academic_levels)}
    for frame in (train_x, test_x):
        frame["academic_work_impact"] = (
            frame["academic_work_impact"]
            .fillna("Unknown")
            .astype(str)
            .map(academic_codes)
        )

    for col in NUM_COLS:
        combined = pd.concat(
            [train_x[col].astype(str), test_x[col].astype(str)],
            ignore_index=True,
        )
        counts = combined.value_counts()
        train_x[f"{col}_freq"] = (
            train_x[col].astype(str).map(counts).fillna(0).astype("int32")
        )
        test_x[f"{col}_freq"] = (
            test_x[col].astype(str).map(counts).fillna(0).astype("int32")
        )

    for col in TARGET_ENCODED_COLS:
        train_x[col] = np.nan
        test_x[col] = np.nan

    selected = [
        *NUM_COLS,
        *CAT_COLS,
        "total_active_hours",
        "passive_screen_time",
        "social_ratio",
        "gaming_ratio",
        "work_ratio",
        "screen_to_sleep",
        "weekend_vs_daily",
        "app_opens_per_hour",
        "notif_per_hour",
        "notif_per_app_open",
        "total_breakdown_hours",
        "ci_daily_lb",
        "ci_missing_comp_ub",
        "ci_missing_comp_mid",
        "ci_slack_exact",
        *[f"{col}_freq" for col in NUM_COLS],
        *TARGET_ENCODED_COLS,
    ]
    train_x = train_x[selected].copy()
    test_x = test_x[selected].copy()

    for col in CAT_COLS:
        levels = sorted(
            set(train_x[col].dropna().unique()) | set(test_x[col].dropna().unique())
        )
        train_x[col] = pd.Categorical(train_x[col], categories=levels)
        test_x[col] = pd.Categorical(test_x[col], categories=levels)

    numeric = [col for col in selected if col not in CAT_COLS]
    train_x[numeric] = train_x[numeric].replace([np.inf, -np.inf], np.nan)
    test_x[numeric] = test_x[numeric].replace([np.inf, -np.inf], np.nan)
    for col in numeric:
        if not col.endswith("_freq"):
            train_x[col] = train_x[col].astype("float32")
            test_x[col] = test_x[col].astype("float32")
    return train_x, test_x


def apply_fold_target_encoding(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame | None,
    y: pd.Series,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    smoothing: float = 30.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Fit numeric-value target maps on one training fold only."""
    fold_train = train_x.iloc[train_idx].copy()
    fold_valid = train_x.iloc[valid_idx].copy()
    fold_test = None if test_x is None else test_x.copy()
    y_train = y.iloc[train_idx].to_numpy(dtype=float)
    fold_mean = float(y_train.mean())

    for col in NUM_COLS:
        train_keys = fold_train[col].astype(str)
        stats = (
            pd.DataFrame({"key": train_keys.to_numpy(), "target": y_train})
            .groupby("key", sort=False)["target"]
            .agg(["mean", "count"])
        )
        encoded = (
            stats["mean"] * stats["count"] + fold_mean * smoothing
        ) / (stats["count"] + smoothing)
        target_col = f"{col}_target_enc"
        fold_train[target_col] = train_keys.map(encoded).fillna(fold_mean).to_numpy(
            dtype="float32"
        )
        fold_valid[target_col] = (
            fold_valid[col].astype(str).map(encoded).fillna(fold_mean).to_numpy(dtype="float32")
        )
        if fold_test is not None:
            fold_test[target_col] = (
                fold_test[col].astype(str).map(encoded).fillna(fold_mean).to_numpy(dtype="float32")
            )
    return fold_train, fold_valid, fold_test


def lgbm_config(seed: int, iterations: int) -> dict:
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": iterations,
        "learning_rate": 0.01,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 200,
        "feature_fraction": 0.34,
        "bagging_fraction": 0.75,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
        "max_bin": 1023,
        "deterministic": True,
        "force_col_wise": True,
    }


def train_fold(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame | None,
    y: pd.Series,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    seed: int,
    iterations: int,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    fold_train, fold_valid, fold_test = apply_fold_target_encoding(
        train_x, test_x, y, train_idx, valid_idx, smoothing
    )
    model = lgb.LGBMClassifier(**lgbm_config(seed, iterations))
    model.fit(
        fold_train,
        y.iloc[train_idx],
        eval_set=[(fold_valid, y.iloc[valid_idx])],
        categorical_feature=CAT_COLS,
        callbacks=[lgb.early_stopping(500), lgb.log_evaluation(300)],
    )
    best = int(model.best_iteration_)
    valid_pred = model.predict_proba(fold_valid, num_iteration=best)[:, 1]
    test_pred = (
        None
        if fold_test is None
        else model.predict_proba(fold_test, num_iteration=best)[:, 1]
    )
    return valid_pred, test_pred, best


def load_v5_predictions(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    oof = pd.read_csv(ROOT / "artifacts" / "v5" / "full3000" / "oof_predictions.csv")
    submission = pd.read_csv(
        ROOT / "artifacts" / "v5" / "full3000" / "submission_v5.csv"
    )
    if not oof[ID_COL].equals(train[ID_COL]):
        raise ValueError("V5 OOF predictions are not aligned to train.csv")
    if not submission[ID_COL].equals(test[ID_COL]):
        raise ValueError("V5 submission predictions are not aligned to test.csv")
    return (
        oof["pred_xgboost"].to_numpy(dtype=float),
        submission[TARGET].to_numpy(dtype=float),
    )


def blend_grid(
    y: pd.Series, v5: np.ndarray, v7: np.ndarray, step: float = 0.05
) -> list[dict]:
    v5_rank = percentile_rank(v5)
    v7_rank = percentile_rank(v7)
    records: list[dict] = []
    for weight in np.arange(0.0, 1.0 + step / 2.0, step):
        prediction = (1.0 - weight) * v5_rank + weight * v7_rank
        records.append({
            "v7_weight": round(float(weight), 4),
            "auc": float(roc_auc_score(y, prediction)),
        })
    return records


def _fold_paths(run_dir: Path, fold: int) -> tuple[Path, Path, Path]:
    pred_dir = run_dir / "fold_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    return (
        pred_dir / f"lgbm_fold{fold}_valid.npy",
        pred_dir / f"lgbm_fold{fold}_test.npy",
        pred_dir / f"lgbm_fold{fold}.json",
    )


def fit_or_resume_fold(
    args: argparse.Namespace,
    run_dir: Path,
    fold: int,
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    valid_path, test_path, meta_path = _fold_paths(run_dir, fold)
    started = time.time()
    if args.resume and valid_path.exists() and test_path.exists() and meta_path.exists():
        valid_pred = np.load(valid_path)
        test_pred = np.load(test_path)
        record = json.loads(meta_path.read_text())
        record["resumed"] = True
    else:
        valid_pred, test_pred, best = train_fold(
            train_x,
            test_x,
            y,
            train_idx,
            valid_idx,
            args.seed,
            args.iterations,
            args.smoothing,
        )
        assert test_pred is not None
        np.save(valid_path, valid_pred)
        np.save(test_path, test_pred)
        record = {
            "fold": fold,
            "auc": float(roc_auc_score(y.iloc[valid_idx], valid_pred)),
            "best_iteration": best,
            "seconds": round(time.time() - started, 2),
            "resumed": False,
        }
        meta_path.write_text(json.dumps(record, indent=2))
    if len(valid_pred) != len(valid_idx) or len(test_pred) != len(test_x):
        raise ValueError(f"Cached fold {fold} predictions have incorrect length")
    print(json.dumps(record), flush=True)
    return valid_pred, test_pred, record


def run_gate(
    args: argparse.Namespace,
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
) -> None:
    run_dir = ROOT / "artifacts" / "v7" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    y = train[TARGET].astype(int)
    train_idx, valid_idx = next(
        iter(StratifiedKFold(args.folds, shuffle=True, random_state=args.seed).split(train_x, y))
    )
    valid_pred, _, record = fit_or_resume_fold(
        args, run_dir, 1, train_x, test_x, y, train_idx, valid_idx
    )
    v5_oof, _ = load_v5_predictions(train, test)
    v5_auc = float(roc_auc_score(y.iloc[valid_idx], v5_oof[valid_idx]))
    v7_auc = float(roc_auc_score(y.iloc[valid_idx], valid_pred))
    summary = {
        "version": "v7-gate",
        "fold": 1,
        "seed": args.seed,
        "features": int(train_x.shape[1]),
        "iterations": args.iterations,
        "smoothing": args.smoothing,
        "v5_auc": v5_auc,
        "v7_auc": v7_auc,
        "gain": v7_auc - v5_auc,
        "minimum_gain": GATE_MIN_GAIN,
        "gate_passed": v7_auc - v5_auc >= GATE_MIN_GAIN,
        "fold_detail": record,
    }
    (run_dir / "gate_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def run_full(
    args: argparse.Namespace,
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
) -> None:
    gate_dir = ROOT / "artifacts" / "v7" / args.gate_run_name
    gate_path = gate_dir / "gate_metrics.json"
    if not gate_path.exists() or not json.loads(gate_path.read_text())["gate_passed"]:
        raise ValueError("The v7 fold-1 gate has not passed")

    run_dir = ROOT / "artifacts" / "v7" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    y = train[TARGET].astype(int)
    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    v7_oof = np.zeros(len(train), dtype=float)
    v7_test = np.zeros(len(test), dtype=float)
    fold_records: list[dict] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train_x, y), start=1):
        # Reuse the gate's identical first fold without retraining it.
        source_dir = gate_dir if fold == 1 else run_dir
        fold_args = argparse.Namespace(**vars(args))
        fold_args.resume = args.resume or fold == 1
        valid_pred, test_pred, record = fit_or_resume_fold(
            fold_args, source_dir, fold, train_x, test_x, y, train_idx, valid_idx
        )
        v7_oof[valid_idx] = valid_pred
        v7_test += test_pred / args.folds
        fold_records.append(record | {"fold": fold})

    v5_oof, v5_test = load_v5_predictions(train, test)
    v5_auc = float(roc_auc_score(y, v5_oof))
    v7_auc = float(roc_auc_score(y, v7_oof))
    grid = blend_grid(y, v5_oof, v7_oof)
    best_blend = max(grid, key=lambda record: record["auc"])
    selected_weight = 1.0
    selected_auc = v7_auc
    selection = "standalone_v7"
    if float(best_blend["auc"]) - v7_auc >= BLEND_MIN_GAIN:
        selected_weight = float(best_blend["v7_weight"])
        selected_auc = float(best_blend["auc"])
        selection = "v5_v7_rank_blend"

    gain = v7_auc - v5_auc
    full_gate_passed = gain >= FULL_MIN_GAIN
    summary = {
        "version": "v7.0.0",
        "source": "Naji public Kaggle notebook version 5, Apache-2.0",
        "folds": args.folds,
        "seed": args.seed,
        "features": int(train_x.shape[1]),
        "iterations": args.iterations,
        "smoothing": args.smoothing,
        "v5_oof_auc": v5_auc,
        "v7_oof_auc": v7_auc,
        "v7_gain": gain,
        "minimum_full_gain": FULL_MIN_GAIN,
        "full_gate_passed": full_gate_passed,
        "rank_correlation": float(
            pd.Series(v5_oof).corr(pd.Series(v7_oof), method="spearman")
        ),
        "blend_grid": grid,
        "minimum_blend_gain": BLEND_MIN_GAIN,
        "selection": selection,
        "selected_v7_weight": selected_weight,
        "selected_oof_auc": selected_auc,
        "folds_detail": fold_records,
        "submission": None,
    }
    pd.DataFrame({
        ID_COL: train[ID_COL],
        TARGET: y,
        "pred_v5": v5_oof,
        "pred_v7": v7_oof,
    }).to_csv(run_dir / "oof_predictions.csv", index=False)

    if full_gate_passed:
        if selected_weight == 1.0:
            selected_test = v7_test
        else:
            selected_test = (
                (1.0 - selected_weight) * percentile_rank(v5_test)
                + selected_weight * percentile_rank(v7_test)
            )
        artifact_path = run_dir / "submission_v7.csv"
        root_path = ROOT / "submission_v7.csv"
        save_submission(artifact_path, test[ID_COL], selected_test)
        shutil.copyfile(artifact_path, root_path)
        summary["submission"] = str(root_path)
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gate", "full"), default="gate")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--smoothing", type=float, default=30.0)
    parser.add_argument("--run-name", default="gate")
    parser.add_argument("--gate-run-name", default="gate")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    train_x, test_x = prepare_features(train, test)
    if args.mode == "gate":
        run_gate(args, train, test, train_x, test_x)
    else:
        run_full(args, train, test, train_x, test_x)


if __name__ == "__main__":
    main()

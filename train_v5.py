"""V5: robustly validated XGBoost challenger.

V5 changes model family rather than making another small LightGBM variation.
The release process has two stages:

1. ``gate`` trains one 80/20 fold for each requested split seed and requires
   XGBoost to beat label-strict v2 OOF predictions on every validation subset.
2. ``full`` trains fixed five-fold XGBoost OOF/test predictions and selects a
   v2/XGBoost rank blend only when it materially beats the stronger standalone
   model. If XGBoost does not beat v2 by the full-OOF gate, no v5 submission is
   produced.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from train_model import CAT_COLS, ID_COL, ROOT, TARGET
from train_v2 import add_v2_features, percentile_rank, save_submission


GATE_MIN_GAIN_EACH = 0.0005
GATE_MIN_MEAN_GAIN = 0.0010
GATE_MIN_SEEDS = 3
FULL_MIN_GAIN = 0.0005
BLEND_MIN_GAIN = 0.0001


def add_v5_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aligned numeric/one-hot features for XGBoost.

    V2's high-cardinality categorical ``missing_pattern`` is omitted because
    the individual missing indicators preserve the same information without a
    model-family-specific category treatment.
    """
    train_x = add_v2_features(train).drop(columns=["missing_pattern"])
    test_x = add_v2_features(test).drop(columns=["missing_pattern"])
    combined = pd.concat([train_x, test_x], ignore_index=True)
    combined = pd.get_dummies(combined, columns=CAT_COLS, dtype="int8")
    return (
        combined.iloc[: len(train)].reset_index(drop=True),
        combined.iloc[len(train) :].reset_index(drop=True),
    )


def xgb_config(seed: int, iterations: int) -> dict:
    return {
        "n_estimators": iterations,
        "learning_rate": 0.035,
        "max_depth": 7,
        "min_child_weight": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.2,
        "reg_lambda": 5.0,
        "max_bin": 256,
        "tree_method": "hist",
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": seed,
        "n_jobs": -1,
        "early_stopping_rounds": 120,
    }


def train_fold(
    train_x: pd.DataFrame,
    y: pd.Series,
    test_x: pd.DataFrame | None,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    seed: int,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    model = XGBClassifier(**xgb_config(seed, iterations))
    model.fit(
        train_x.iloc[train_idx],
        y.iloc[train_idx],
        eval_set=[(train_x.iloc[valid_idx], y.iloc[valid_idx])],
        verbose=200,
    )
    valid_pred = model.predict_proba(train_x.iloc[valid_idx])[:, 1]
    test_pred = None if test_x is None else model.predict_proba(test_x)[:, 1]
    return valid_pred, test_pred, int(model.best_iteration)


def gate_passes(records: list[dict]) -> bool:
    gains = [float(record["gain"]) for record in records]
    return bool(
        len(gains) >= GATE_MIN_SEEDS
        and min(gains) >= GATE_MIN_GAIN_EACH
        and float(np.mean(gains)) >= GATE_MIN_MEAN_GAIN
    )


def rank_blend_grid(
    y: pd.Series, baseline: np.ndarray, challenger: np.ndarray, step: float = 0.05
) -> list[dict]:
    baseline_rank = percentile_rank(baseline)
    challenger_rank = percentile_rank(challenger)
    records: list[dict] = []
    for weight in np.arange(0.0, 1.0 + step / 2.0, step):
        prediction = (1.0 - weight) * baseline_rank + weight * challenger_rank
        records.append({
            "xgboost_weight": round(float(weight), 4),
            "auc": float(roc_auc_score(y, prediction)),
        })
    return records


def load_v2_oof(train: pd.DataFrame) -> np.ndarray:
    path = ROOT / "artifacts" / "v2" / "full" / "oof_predictions.csv"
    frame = pd.read_csv(path)
    if not frame[ID_COL].equals(train[ID_COL]):
        raise ValueError("V2 OOF artifact is not aligned to the training data")
    return frame["pred_blend"].to_numpy(dtype=float)


def run_gate(args: argparse.Namespace, train: pd.DataFrame, train_x: pd.DataFrame) -> None:
    run_dir = ROOT / "artifacts" / "v5" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    y = train[TARGET].astype(int)
    baseline = load_v2_oof(train)
    seeds = [int(value.strip()) for value in args.validation_seeds.split(",") if value.strip()]
    records: list[dict] = []

    for seed in seeds:
        valid_path = run_dir / f"seed{seed}_fold1_valid.npy"
        meta_path = run_dir / f"seed{seed}_fold1.json"
        train_idx, valid_idx = next(
            iter(StratifiedKFold(args.folds, shuffle=True, random_state=seed).split(train_x, y))
        )
        started = time.time()
        if args.resume and valid_path.exists():
            valid_pred = np.load(valid_path)
            best_iteration = (
                int(json.loads(meta_path.read_text())["best_iteration"])
                if meta_path.exists()
                else args.iterations - 1
            )
            resumed = True
        else:
            valid_pred, _, best_iteration = train_fold(
                train_x, y, None, train_idx, valid_idx, seed + 1, args.iterations
            )
            np.save(valid_path, valid_pred)
            resumed = False
        candidate_auc = float(roc_auc_score(y.iloc[valid_idx], valid_pred))
        baseline_auc = float(roc_auc_score(y.iloc[valid_idx], baseline[valid_idx]))
        record = {
            "seed": seed,
            "validation_rows": int(len(valid_idx)),
            "xgboost_auc": candidate_auc,
            "v2_oof_auc": baseline_auc,
            "gain": candidate_auc - baseline_auc,
            "best_iteration": best_iteration,
            "seconds": round(time.time() - started, 2),
            "resumed": resumed,
        }
        meta_path.write_text(json.dumps(record, indent=2))
        records.append(record)
        print(json.dumps(record), flush=True)

    summary = {
        "version": "v5.0.0-gate",
        "seeds": seeds,
        "minimum_gain_each_seed": GATE_MIN_GAIN_EACH,
        "minimum_mean_gain": GATE_MIN_MEAN_GAIN,
        "minimum_seed_count": GATE_MIN_SEEDS,
        "iterations": args.iterations,
        "mean_gain": float(np.mean([record["gain"] for record in records])),
        "minimum_observed_gain": float(min(record["gain"] for record in records)),
        "gate_passed": gate_passes(records),
        "records": records,
    }
    (run_dir / "gate_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def run_full(
    args: argparse.Namespace,
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
) -> None:
    gate_path = ROOT / "artifacts" / "v5" / args.gate_run_name / "gate_metrics.json"
    if not gate_path.exists() or not json.loads(gate_path.read_text())["gate_passed"]:
        raise ValueError("The multi-seed v5 gate has not passed")

    run_dir = ROOT / "artifacts" / "v5" / args.run_name
    pred_dir = run_dir / "fold_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    y = train[TARGET].astype(int)
    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    challenger_oof = np.zeros(len(train), dtype=float)
    challenger_test = np.zeros(len(test), dtype=float)
    fold_records: list[dict] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train_x, y), start=1):
        valid_path = pred_dir / f"xgboost_fold{fold}_valid.npy"
        test_path = pred_dir / f"xgboost_fold{fold}_test.npy"
        meta_path = pred_dir / f"xgboost_fold{fold}.json"
        started = time.time()
        if args.resume and valid_path.exists() and test_path.exists() and meta_path.exists():
            valid_pred = np.load(valid_path)
            test_pred = np.load(test_path)
            best_iteration = int(json.loads(meta_path.read_text())["best_iteration"])
            resumed = True
        else:
            valid_pred, test_pred, best_iteration = train_fold(
                train_x, y, test_x, train_idx, valid_idx,
                args.seed + fold, args.iterations,
            )
            assert test_pred is not None
            np.save(valid_path, valid_pred)
            np.save(test_path, test_pred)
            resumed = False
        record = {
            "fold": fold,
            "auc": float(roc_auc_score(y.iloc[valid_idx], valid_pred)),
            "best_iteration": best_iteration,
            "seconds": round(time.time() - started, 2),
            "resumed": resumed,
        }
        meta_path.write_text(json.dumps(record, indent=2))
        challenger_oof[valid_idx] = valid_pred
        challenger_test += test_pred / args.folds
        fold_records.append(record)
        print(json.dumps(record), flush=True)

    baseline_oof = load_v2_oof(train)
    baseline_submission = pd.read_csv(
        ROOT / "artifacts" / "v2" / "full" / "submission_v2.csv"
    )
    if not baseline_submission[ID_COL].equals(test[ID_COL]):
        raise ValueError("V2 submission artifact is not aligned to the test data")
    baseline_test = baseline_submission[TARGET].to_numpy(dtype=float)
    baseline_auc = float(roc_auc_score(y, baseline_oof))
    challenger_auc = float(roc_auc_score(y, challenger_oof))
    full_gate_passed = challenger_auc - baseline_auc >= FULL_MIN_GAIN
    grid = rank_blend_grid(y, baseline_oof, challenger_oof)
    best_grid = max(grid, key=lambda record: record["auc"])

    # Prefer the distinct standalone model unless blending adds material value.
    selected_weight = 1.0
    selected_auc = challenger_auc
    if best_grid["auc"] - challenger_auc >= BLEND_MIN_GAIN:
        selected_weight = float(best_grid["xgboost_weight"])
        selected_auc = float(best_grid["auc"])

    summary = {
        "version": "v5.0.0",
        "folds": args.folds,
        "seed": args.seed,
        "iterations": args.iterations,
        "v2_oof_auc": baseline_auc,
        "xgboost_oof_auc": challenger_auc,
        "xgboost_gain": challenger_auc - baseline_auc,
        "minimum_full_oof_gain": FULL_MIN_GAIN,
        "full_gate_passed": full_gate_passed,
        "rank_correlation": float(
            pd.Series(challenger_oof).corr(pd.Series(baseline_oof), method="spearman")
        ),
        "blend_grid": grid,
        "minimum_blend_gain": BLEND_MIN_GAIN,
        "selected_xgboost_weight": selected_weight,
        "selected_oof_auc": selected_auc,
        "folds_detail": fold_records,
    }

    pd.DataFrame({
        ID_COL: train[ID_COL],
        TARGET: y,
        "pred_v2_baseline": baseline_oof,
        "pred_xgboost": challenger_oof,
    }).to_csv(run_dir / "oof_predictions.csv", index=False)

    if full_gate_passed:
        selected_test = (
            (1.0 - selected_weight) * percentile_rank(baseline_test)
            + selected_weight * percentile_rank(challenger_test)
        )
        submission_path = run_dir / "submission_v5.csv"
        save_submission(submission_path, test[ID_COL], selected_test)
        summary["submission"] = str(submission_path)
    else:
        summary["submission"] = None
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gate", "full"), default="gate")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-seeds", default="42,17,83")
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--run-name", default="gate")
    parser.add_argument("--gate-run-name", default="gate")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    train_x, test_x = add_v5_features(train, test)
    if args.mode == "gate":
        run_gate(args, train, train_x)
    else:
        run_full(args, train, test, train_x, test_x)


if __name__ == "__main__":
    main()

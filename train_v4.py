"""Train a conservative test-density-weighted challenger to v2.

V4 deliberately returns to v2's raw/missingness features. A second lgbm_c
model is fitted with label-free train-to-test density weights. It is eligible
for submission only when a rank blend improves both ordinary OOF AUC and
density-weighted OOF AUC over the v2 champion.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train_model import ID_COL, ROOT, TARGET
from train_v2 import (
    V2_CAT_COLS,
    add_v2_features,
    as_lgbm_categories,
    lgbm_config,
    percentile_rank,
    save_submission,
)


def load_density_weights(path: Path, train_ids: pd.Series) -> np.ndarray:
    """Load weights and enforce exact one-to-one ID alignment."""
    frame = pd.read_csv(path)
    required = {ID_COL, "density_weight"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Weight file must contain {sorted(required)}")
    if frame[ID_COL].duplicated().any():
        raise ValueError("Weight file contains duplicate IDs")
    aligned = frame.set_index(ID_COL)["density_weight"].reindex(train_ids)
    if aligned.isna().any():
        raise ValueError("Weight file does not cover every training ID")
    weights = aligned.to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Density weights must be finite and positive")
    return weights


def weighted_auc(y: pd.Series | np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    return float(roc_auc_score(y, pred, sample_weight=weights))


def eligible_blends(
    y: pd.Series,
    baseline: np.ndarray,
    challenger: np.ndarray,
    density_weights: np.ndarray,
    step: float = 0.05,
    minimum_gain: float = 0.00002,
    comparison_tolerance: float = 0.0000001,
) -> list[dict]:
    """Evaluate blends; only dual-metric improvements are eligible."""
    baseline_rank = percentile_rank(baseline)
    challenger_rank = percentile_rank(challenger)
    baseline_auc = float(roc_auc_score(y, baseline_rank))
    baseline_weighted_auc = weighted_auc(y, baseline_rank, density_weights)
    records: list[dict] = []
    for challenger_weight in np.arange(0.0, 1.0 + step / 2.0, step):
        prediction = (
            (1.0 - challenger_weight) * baseline_rank
            + challenger_weight * challenger_rank
        )
        ordinary = float(roc_auc_score(y, prediction))
        shifted = weighted_auc(y, prediction, density_weights)
        records.append({
            "challenger_weight": round(float(challenger_weight), 4),
            "ordinary_auc": ordinary,
            "density_weighted_auc": shifted,
            "ordinary_gain": ordinary - baseline_auc,
            "density_weighted_gain": shifted - baseline_weighted_auc,
            "eligible": (
                ordinary - baseline_auc + comparison_tolerance >= minimum_gain
                and shifted - baseline_weighted_auc + comparison_tolerance >= minimum_gain
            ),
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--run-name", default="full")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_dir = ROOT / "artifacts" / "v4" / args.run_name
    pred_dir = run_dir / "fold_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.weights or ROOT / "artifacts" / "v4" / "full" / "train_density_weights.csv"

    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    y = train[TARGET].astype(int)
    density_weights = load_density_weights(weights_path, train[ID_COL])
    train_x, test_x = as_lgbm_categories(add_v2_features(train), add_v2_features(test))
    splits = list(StratifiedKFold(args.folds, shuffle=True, random_state=args.seed).split(train_x, y))
    folds_to_run = args.folds if args.max_folds is None else min(args.max_folds, args.folds)
    challenger_oof = np.full(len(train), np.nan, dtype=float)
    challenger_test = np.zeros(len(test), dtype=float)
    fold_records: list[dict] = []

    for fold, (train_idx, valid_idx) in enumerate(splits[:folds_to_run], start=1):
        valid_path = pred_dir / f"density_lgbm_c_fold{fold}_valid.npy"
        test_path = pred_dir / f"density_lgbm_c_fold{fold}_test.npy"
        meta_path = pred_dir / f"density_lgbm_c_fold{fold}.json"
        started = time.time()
        if args.resume and valid_path.exists() and test_path.exists() and meta_path.exists():
            valid_pred = np.load(valid_path)
            test_pred = np.load(test_path)
            best_iteration = int(json.loads(meta_path.read_text())["best_iteration"])
            resumed = True
        else:
            model = lgb.LGBMClassifier(**lgbm_config("lgbm_c", args.iterations, args.seed + fold))
            model.fit(
                train_x.iloc[train_idx],
                y.iloc[train_idx],
                sample_weight=density_weights[train_idx],
                eval_set=[(train_x.iloc[valid_idx], y.iloc[valid_idx])],
                eval_metric="auc",
                categorical_feature=V2_CAT_COLS,
                callbacks=[lgb.early_stopping(120), lgb.log_evaluation(200)],
            )
            best_iteration = int(model.best_iteration_)
            valid_pred = model.predict_proba(
                train_x.iloc[valid_idx], num_iteration=best_iteration
            )[:, 1]
            test_pred = model.predict_proba(test_x, num_iteration=best_iteration)[:, 1]
            np.save(valid_path, valid_pred)
            np.save(test_path, test_pred)
            resumed = False

        record = {
            "fold": fold,
            "ordinary_auc": float(roc_auc_score(y.iloc[valid_idx], valid_pred)),
            "density_weighted_auc": weighted_auc(
                y.iloc[valid_idx], valid_pred, density_weights[valid_idx]
            ),
            "best_iteration": best_iteration,
            "seconds": round(time.time() - started, 2),
            "resumed": resumed,
        }
        meta_path.write_text(json.dumps(record, indent=2))
        challenger_oof[valid_idx] = valid_pred
        challenger_test += test_pred / folds_to_run
        fold_records.append(record)
        print(json.dumps(record), flush=True)

    if folds_to_run < args.folds:
        summary = {
            "version": "v4.0.0-diagnostic",
            "completed_folds": folds_to_run,
            "folds_detail": fold_records,
        }
        (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return

    v2_oof = pd.read_csv(ROOT / "artifacts" / "v2" / "full" / "oof_predictions.csv")
    v2_submission = pd.read_csv(ROOT / "artifacts" / "v2" / "full" / "submission_v2.csv")
    if not v2_oof[ID_COL].equals(train[ID_COL]) or not v2_submission[ID_COL].equals(test[ID_COL]):
        raise ValueError("V2 artifacts are not aligned to the current data")
    baseline_oof = v2_oof["pred_blend"].to_numpy()
    baseline_test = v2_submission[TARGET].to_numpy()
    blends = eligible_blends(y, baseline_oof, challenger_oof, density_weights)
    eligible = [record for record in blends if record["eligible"]]
    selected = max(eligible, key=lambda record: record["ordinary_auc"]) if eligible else blends[0]
    alpha = selected["challenger_weight"]
    selected_oof = (
        (1.0 - alpha) * percentile_rank(baseline_oof)
        + alpha * percentile_rank(challenger_oof)
    )
    selected_test = (
        (1.0 - alpha) * percentile_rank(baseline_test)
        + alpha * percentile_rank(challenger_test)
    )

    pd.DataFrame({
        ID_COL: train[ID_COL],
        TARGET: y,
        "density_weight": density_weights,
        "pred_v2_baseline": baseline_oof,
        "pred_density_challenger": challenger_oof,
        "pred_selected": selected_oof,
    }).to_csv(run_dir / "oof_predictions.csv", index=False)
    save_submission(run_dir / "submission_v4.csv", test[ID_COL], selected_test)
    summary = {
        "version": "v4.0.0",
        "minimum_gain_per_validation_gate": 0.00002,
        "comparison_tolerance": 0.0000001,
        "selected_challenger_weight": alpha,
        "selected_is_new_candidate": bool(alpha > 0),
        "selected_ordinary_auc": selected["ordinary_auc"],
        "selected_density_weighted_auc": selected["density_weighted_auc"],
        "blend_candidates": blends,
        "folds_detail": fold_records,
        "submission": str(run_dir / "submission_v4.csv"),
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

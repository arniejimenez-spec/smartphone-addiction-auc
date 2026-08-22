"""V6: fold-safe masked-data augmentation for missing-value robustness.

Artificially masked copies are created only from each target-model training
fold. Validation rows are never copied into training. Mask templates come from
real test rows with two or more missing fields and use covariates only.
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
from train_v5 import add_v5_features, xgb_config


RAW_FEATURES = [
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

ABLATION_GLOBAL_GAIN = 0.0003
GATE_GLOBAL_GAIN_EACH = 0.0005
GATE_GLOBAL_MEAN_GAIN = 0.0010
HARD_BUCKET_GAIN_EACH = 0.0010
MAX_EASY_BUCKET_LOSS = 0.0002


def missing_bucket(frame: pd.DataFrame) -> np.ndarray:
    count = frame[RAW_FEATURES].isna().sum(axis=1).to_numpy()
    return np.where(count == 0, 0, np.where(count == 1, 1, 2)).astype(np.int8)


def hard_test_masks(test: pd.DataFrame) -> np.ndarray:
    masks = test[RAW_FEATURES].isna().to_numpy(dtype=bool)
    hard = masks.sum(axis=1) >= 2
    if not hard.any():
        raise ValueError("Test data contains no 2+ missing-field mask patterns")
    return masks[hard]


def make_masked_copies(
    fold_raw: pd.DataFrame,
    mask_templates: np.ndarray,
    ratio: float,
    seed: int,
) -> pd.DataFrame:
    """Create labeled training copies with real test-like hard masks."""
    if not 0 < ratio <= 1:
        raise ValueError("augmentation ratio must be in (0, 1]")
    existing = fold_raw[RAW_FEATURES].isna().sum(axis=1).to_numpy()
    eligible = np.flatnonzero(existing <= 1)
    count = min(int(round(len(fold_raw) * ratio)), len(eligible))
    if count == 0:
        raise ValueError("No rows are eligible for masked augmentation")
    rng = np.random.default_rng(seed)
    source = rng.choice(eligible, size=count, replace=False)
    template_idx = rng.integers(0, len(mask_templates), size=count)
    selected_masks = mask_templates[template_idx]
    copies = fold_raw.iloc[source].copy().reset_index(drop=True)
    for offset, column in enumerate(RAW_FEATURES):
        copies.loc[selected_masks[:, offset], column] = np.nan
    if (copies[RAW_FEATURES].isna().sum(axis=1) < 2).any():
        raise AssertionError("Every augmented copy must have at least two missing fields")
    return copies


def encode_masked_copies(copies: pd.DataFrame, columns: pd.Index) -> pd.DataFrame:
    encoded = add_v2_features(copies).drop(columns=["missing_pattern"])
    encoded = pd.get_dummies(encoded, columns=CAT_COLS, dtype="int8")
    return encoded.reindex(columns=columns, fill_value=0)


def train_augmented_fold(
    train: pd.DataFrame,
    train_x: pd.DataFrame,
    y: pd.Series,
    test_x: pd.DataFrame | None,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    mask_templates: np.ndarray,
    ratio: float,
    augmentation_weight: float,
    seed: int,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray | None, int, int]:
    fold_raw = train.iloc[train_idx]
    copies = make_masked_copies(fold_raw, mask_templates, ratio, seed + 10_000)
    copy_x = encode_masked_copies(copies, train_x.columns)
    fit_x = pd.concat([train_x.iloc[train_idx], copy_x], ignore_index=True)
    fit_y = pd.concat([y.iloc[train_idx].reset_index(drop=True), copies[TARGET]], ignore_index=True)
    weights = np.r_[
        np.ones(len(train_idx), dtype=np.float32),
        np.full(len(copies), augmentation_weight, dtype=np.float32),
    ]
    model = XGBClassifier(**xgb_config(seed, iterations))
    model.fit(
        fit_x,
        fit_y,
        sample_weight=weights,
        eval_set=[(train_x.iloc[valid_idx], y.iloc[valid_idx])],
        verbose=200,
    )
    valid_pred = model.predict_proba(train_x.iloc[valid_idx])[:, 1]
    test_pred = None if test_x is None else model.predict_proba(test_x)[:, 1]
    return valid_pred, test_pred, int(model.best_iteration), len(copies)


def load_v5_oof(train: pd.DataFrame) -> np.ndarray:
    frame = pd.read_csv(ROOT / "artifacts" / "v5" / "full3000" / "oof_predictions.csv")
    if not frame[ID_COL].equals(train[ID_COL]):
        raise ValueError("V5 OOF artifact is not aligned to training data")
    return frame["pred_xgboost"].to_numpy(dtype=float)


def score_record(
    train: pd.DataFrame,
    y: pd.Series,
    valid_idx: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
) -> dict:
    bucket = missing_bucket(train.iloc[valid_idx])
    candidate_auc = float(roc_auc_score(y.iloc[valid_idx], candidate))
    baseline_auc = float(roc_auc_score(y.iloc[valid_idx], baseline[valid_idx]))
    slices: dict[str, dict] = {}
    for value, label in ((0, "0"), (1, "1"), (2, "2+")):
        mask = bucket == value
        candidate_slice = float(roc_auc_score(y.iloc[valid_idx].to_numpy()[mask], candidate[mask]))
        baseline_slice = float(
            roc_auc_score(y.iloc[valid_idx].to_numpy()[mask], baseline[valid_idx][mask])
        )
        slices[label] = {
            "rows": int(mask.sum()),
            "candidate_auc": candidate_slice,
            "v5_auc": baseline_slice,
            "gain": candidate_slice - baseline_slice,
        }
    return {
        "candidate_auc": candidate_auc,
        "v5_auc": baseline_auc,
        "global_gain": candidate_auc - baseline_auc,
        "slices": slices,
    }


def ablation_passes(record: dict) -> bool:
    return bool(
        record["global_gain"] >= ABLATION_GLOBAL_GAIN
        and record["slices"]["2+"]["gain"] >= HARD_BUCKET_GAIN_EACH
        and record["slices"]["0"]["gain"] >= -MAX_EASY_BUCKET_LOSS
        and record["slices"]["1"]["gain"] >= -MAX_EASY_BUCKET_LOSS
    )


def gate_passes(records: list[dict]) -> bool:
    if len(records) < 3:
        return False
    global_gains = [record["global_gain"] for record in records]
    return bool(
        min(global_gains) >= GATE_GLOBAL_GAIN_EACH
        and float(np.mean(global_gains)) >= GATE_GLOBAL_MEAN_GAIN
        and min(record["slices"]["2+"]["gain"] for record in records)
        >= HARD_BUCKET_GAIN_EACH
        and min(record["slices"]["0"]["gain"] for record in records)
        >= -MAX_EASY_BUCKET_LOSS
        and min(record["slices"]["1"]["gain"] for record in records)
        >= -MAX_EASY_BUCKET_LOSS
    )


def run_validation(
    args: argparse.Namespace,
    train: pd.DataFrame,
    train_x: pd.DataFrame,
    masks: np.ndarray,
) -> None:
    run_dir = ROOT / "artifacts" / "v6" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    y = train[TARGET].astype(int)
    baseline = load_v5_oof(train)
    seeds = [int(value) for value in args.validation_seeds.split(",") if value.strip()]
    records: list[dict] = []
    for seed in seeds:
        valid_path = run_dir / f"seed{seed}_fold1_valid.npy"
        meta_path = run_dir / f"seed{seed}_fold1.json"
        train_idx, valid_idx = next(
            iter(StratifiedKFold(5, shuffle=True, random_state=seed).split(train_x, y))
        )
        started = time.time()
        if args.resume and valid_path.exists() and meta_path.exists():
            prediction = np.load(valid_path)
            cached = json.loads(meta_path.read_text())
            best_iteration = int(cached["best_iteration"])
            augmented_rows = int(cached["augmented_rows"])
            resumed = True
        else:
            prediction, _, best_iteration, augmented_rows = train_augmented_fold(
                train, train_x, y, None, train_idx, valid_idx, masks,
                args.augmentation_ratio, args.augmentation_weight,
                seed + 1, args.iterations,
            )
            np.save(valid_path, prediction)
            resumed = False
        record = {
            "seed": seed,
            **score_record(train, y, valid_idx, prediction, baseline),
            "best_iteration": best_iteration,
            "augmented_rows": augmented_rows,
            "seconds": round(time.time() - started, 2),
            "resumed": resumed,
        }
        meta_path.write_text(json.dumps(record, indent=2))
        records.append(record)
        print(json.dumps(record), flush=True)
    summary = {
        "version": "v6.0.0-validation",
        "mode": args.mode,
        "seeds": seeds,
        "iterations": args.iterations,
        "augmentation_ratio": args.augmentation_ratio,
        "augmentation_weight": args.augmentation_weight,
        "ablation_passed": len(records) == 1 and ablation_passes(records[0]),
        "gate_passed": gate_passes(records),
        "records": records,
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def run_full(
    args: argparse.Namespace,
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    masks: np.ndarray,
) -> None:
    gate_path = ROOT / "artifacts" / "v6" / args.gate_run_name / "metrics.json"
    if not gate_path.exists() or not json.loads(gate_path.read_text())["gate_passed"]:
        raise ValueError("The v6 three-seed gate has not passed")
    run_dir = ROOT / "artifacts" / "v6" / args.run_name
    pred_dir = run_dir / "fold_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    y = train[TARGET].astype(int)
    baseline = load_v5_oof(train)
    oof = np.zeros(len(train), dtype=float)
    test_pred = np.zeros(len(test), dtype=float)
    fold_records: list[dict] = []
    splitter = StratifiedKFold(5, shuffle=True, random_state=args.seed)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train_x, y), start=1):
        valid_path = pred_dir / f"masked_xgboost_fold{fold}_valid.npy"
        test_path = pred_dir / f"masked_xgboost_fold{fold}_test.npy"
        meta_path = pred_dir / f"masked_xgboost_fold{fold}.json"
        started = time.time()
        if args.resume and valid_path.exists() and test_path.exists() and meta_path.exists():
            valid_pred = np.load(valid_path)
            fold_test = np.load(test_path)
            cached = json.loads(meta_path.read_text())
            best_iteration = int(cached["best_iteration"])
            augmented_rows = int(cached["augmented_rows"])
            resumed = True
        else:
            valid_pred, fold_test, best_iteration, augmented_rows = train_augmented_fold(
                train, train_x, y, test_x, train_idx, valid_idx, masks,
                args.augmentation_ratio, args.augmentation_weight,
                args.seed + fold, args.iterations,
            )
            assert fold_test is not None
            np.save(valid_path, valid_pred)
            np.save(test_path, fold_test)
            resumed = False
        record = {
            "fold": fold,
            **score_record(train, y, valid_idx, valid_pred, baseline),
            "best_iteration": best_iteration,
            "augmented_rows": augmented_rows,
            "seconds": round(time.time() - started, 2),
            "resumed": resumed,
        }
        meta_path.write_text(json.dumps(record, indent=2))
        oof[valid_idx] = valid_pred
        test_pred += fold_test / 5
        fold_records.append(record)
        print(json.dumps(record), flush=True)

    global_record = score_record(train, y, np.arange(len(train)), oof, baseline)
    full_passed = ablation_passes(global_record) and global_record["global_gain"] >= 0.0010
    summary = {
        "version": "v6.0.0",
        "iterations": args.iterations,
        "augmentation_ratio": args.augmentation_ratio,
        "augmentation_weight": args.augmentation_weight,
        **global_record,
        "full_gate_passed": full_passed,
        "folds_detail": fold_records,
        "submission": None,
    }
    pd.DataFrame({
        ID_COL: train[ID_COL], TARGET: y,
        "pred_v5": baseline, "pred_v6_masked": oof,
    }).to_csv(run_dir / "oof_predictions.csv", index=False)
    if full_passed:
        submission_path = run_dir / "submission_v6.csv"
        save_submission(submission_path, test[ID_COL], percentile_rank(test_pred))
        summary["submission"] = str(submission_path)
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("ablation", "gate", "full"), default="ablation")
    parser.add_argument("--run-name", default="ablation")
    parser.add_argument("--gate-run-name", default="gate")
    parser.add_argument("--validation-seeds", default="42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--augmentation-ratio", type=float, default=0.50)
    parser.add_argument("--augmentation-weight", type=float, default=0.25)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    train_x, test_x = add_v5_features(train, test)
    masks = hard_test_masks(test)
    if args.mode in {"ablation", "gate"}:
        run_validation(args, train, train_x, masks)
    else:
        run_full(args, train, test, train_x, test_x, masks)


if __name__ == "__main__":
    main()

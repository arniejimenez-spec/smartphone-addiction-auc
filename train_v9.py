"""V9: honestly validated rank-logit/regime fusion ablations.

V9 starts from the frozen 205-member v8 prediction pool and its cross-fitted
base prediction.  It evaluates three predeclared meta-models on the same outer
five folds:

* ``fusion``: the full 206-member rank+logit model blended with a compressed
  missingness/disagreement regime model;
* ``stability``: a rank+logit model whose members are selected using three
  inner folds from the outer training rows only;
* ``hierarchical``: family-level rank/logit summaries with full regime
  interactions.

Every candidate is predicted out of fold.  Candidate/base blend weights are
selected on the outer-fit OOF rows and applied to the untouched outer-valid
rows.  A root ``submission_v9.csv`` is written only for a gain of at least
0.0001 OOF AUC over v8 and a positive gain on all five folds.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from train_model import ID_COL, ROOT, TARGET
from train_v8 import N_SPLITS, SEED, percentile_rank, public_member_names


V8_ARTIFACT = ROOT / "artifacts" / "v8" / "full"
ARTIFACT_DIR = ROOT / "artifacts" / "v9" / "full"
FUSION_C = 3.5
BASE_C = 0.1
MAX_ITER = 1000
FUSION_WEIGHT = 0.55
MIN_GAIN = 0.0001
STABILITY_KEEP = 96
INNER_SPLITS = 3
ALPHA_GRID = np.array([0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 1 / 3, 0.4, 0.5])
ALL_CANDIDATES = ("fusion", "stability", "hierarchical")


def member_family(name: str) -> str:
    """Map every public member to a stable source-family label."""
    if name.startswith("bolt_"):
        return "bolt"
    if name.startswith("sz_"):
        return "szymon"
    if name.startswith("weak50_"):
        return "weak50"
    if name.startswith("naji"):
        return "naji"
    if name.startswith("fm_"):
        return "fm"
    if name.startswith("golem_"):
        return "golem"
    if name.startswith("a_"):
        return "adarsh"
    if name.startswith("fresh_"):
        return "fresh"
    if name.startswith("candidate_"):
        return "candidate"
    if name.startswith("local_"):
        return "local"
    return "extra"


def family_index(names: list[str]) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        groups.setdefault(member_family(name), []).append(index)
    return {name: np.asarray(indices, dtype=int) for name, indices in sorted(groups.items())}


def clipped_logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype="float32"), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values)).astype("float32", copy=False)


def base_rank(values: np.ndarray) -> np.ndarray:
    return percentile_rank(np.asarray(values)).reshape(-1, 1)


def build_dual_features(
    pool: np.ndarray,
    base_values: np.ndarray,
    selected: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return rank+logit features and their rank half.

    The cached public pool already contains the reference notebook's global
    percentile ranks.  Only the appended base prediction needs ranking here.
    """
    ranks = np.asarray(pool if selected is None else pool[:, selected], dtype="float32")
    rank_half = np.column_stack([ranks, base_rank(base_values)]).astype("float32", copy=False)
    logits = np.column_stack([clipped_logit(ranks), clipped_logit(base_values)])
    return np.column_stack([rank_half, logits]).astype("float32", copy=False), rank_half


def family_core_features(
    pool: np.ndarray,
    base_values: np.ndarray,
    groups: dict[str, np.ndarray],
) -> np.ndarray:
    """Return two features per family (mean rank and mean logit) plus base."""
    parts: list[np.ndarray] = []
    for indices in groups.values():
        values = np.asarray(pool[:, indices], dtype="float32")
        parts.append(values.mean(axis=1, dtype="float32"))
        parts.append(clipped_logit(values).mean(axis=1, dtype="float32"))
    parts.append(percentile_rank(base_values))
    parts.append(clipped_logit(base_values))
    return np.column_stack(parts).astype("float32", copy=False)


def compressed_regime_features(
    dual: np.ndarray,
    rank_half: np.ndarray,
    family_core: np.ndarray,
    complete: np.ndarray,
    missing_many: np.ndarray,
    disagreement_mean: float | None = None,
    disagreement_scale: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Full dual vocabulary plus family-level regime interactions."""
    disagreement = rank_half.std(axis=1, dtype="float32")
    if disagreement_mean is None:
        disagreement_mean = float(disagreement.mean())
    if disagreement_scale is None:
        disagreement_scale = float(disagreement.std()) + 1e-12
    d = ((disagreement - disagreement_mean) / disagreement_scale).astype("float32")
    complete = np.asarray(complete, dtype="float32")
    missing_many = np.asarray(missing_many, dtype="float32")
    aggregate = np.column_stack([
        rank_half.mean(axis=1, dtype="float32"),
        disagreement,
        rank_half.max(axis=1) - rank_half.min(axis=1),
        complete,
        missing_many,
    ])
    features = np.column_stack([
        dual,
        family_core * complete[:, None],
        family_core * missing_many[:, None],
        family_core * d[:, None],
        aggregate,
    ]).astype("float32", copy=False)
    return features, disagreement_mean, disagreement_scale


def hierarchical_base_features(
    pool: np.ndarray,
    base_values: np.ndarray,
    groups: dict[str, np.ndarray],
) -> np.ndarray:
    """Five label-free summaries per source family plus ranked/raw base."""
    parts: list[np.ndarray] = []
    for indices in groups.values():
        values = np.asarray(pool[:, indices], dtype="float32")
        logits = clipped_logit(values)
        parts.extend([
            values.mean(axis=1, dtype="float32"),
            values.std(axis=1, dtype="float32"),
            values.max(axis=1) - values.min(axis=1),
            logits.mean(axis=1, dtype="float32"),
            logits.std(axis=1, dtype="float32"),
        ])
    parts.extend([percentile_rank(base_values), clipped_logit(base_values)])
    return np.column_stack(parts).astype("float32", copy=False)


def full_regime_features(
    base_features: np.ndarray,
    complete: np.ndarray,
    missing_many: np.ndarray,
    disagreement_mean: float | None = None,
    disagreement_scale: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Apply the source notebook's four regimes to a compact hierarchy."""
    disagreement = base_features.std(axis=1, dtype="float32")
    if disagreement_mean is None:
        disagreement_mean = float(disagreement.mean())
    if disagreement_scale is None:
        disagreement_scale = float(disagreement.std()) + 1e-12
    d = ((disagreement - disagreement_mean) / disagreement_scale).astype("float32")
    complete = np.asarray(complete, dtype="float32")
    missing_many = np.asarray(missing_many, dtype="float32")
    aggregate = np.column_stack([
        base_features.mean(axis=1, dtype="float32"),
        disagreement,
        base_features.max(axis=1) - base_features.min(axis=1),
        complete,
        missing_many,
    ])
    features = np.column_stack([
        base_features,
        base_features * complete[:, None],
        base_features * missing_many[:, None],
        base_features * d[:, None],
        aggregate,
    ]).astype("float32", copy=False)
    return features, disagreement_mean, disagreement_scale


@dataclass
class FitResult:
    valid: np.ndarray
    test: np.ndarray
    iterations: int


def fit_logistic(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    test_x: np.ndarray,
    c: float,
    standardize: bool,
) -> FitResult:
    """Fit one float32 logistic model and immediately release large inputs."""
    train_x = np.asarray(train_x, dtype="float32", order="C")
    valid_x = np.asarray(valid_x, dtype="float32", order="C")
    test_x = np.asarray(test_x, dtype="float32", order="C")
    scaler: StandardScaler | None = None
    if standardize:
        scaler = StandardScaler(copy=False)
        scaler.fit_transform(train_x)
        scaler.transform(valid_x)
        scaler.transform(test_x)
    model = LogisticRegression(C=c, max_iter=MAX_ITER, solver="lbfgs", tol=1e-5)
    model.fit(train_x, train_y)
    result = FitResult(
        valid=model.predict_proba(valid_x)[:, 1],
        test=model.predict_proba(test_x)[:, 1],
        iterations=int(model.n_iter_[0]),
    )
    del model, scaler
    return result


def select_stable_members(
    pool: np.ndarray,
    y: np.ndarray,
    outer_train: np.ndarray,
    outer_fold: int,
    keep: int = STABILITY_KEEP,
) -> tuple[np.ndarray, dict]:
    """Select members using only three inner folds of the outer-fit rows."""
    inner = StratifiedKFold(
        n_splits=INNER_SPLITS, shuffle=True, random_state=SEED + 100 + outer_fold
    )
    outer_x = np.asarray(pool[outer_train], dtype="float32")
    outer_y = y[outer_train]
    coefficients: list[np.ndarray] = []
    for inner_train, _ in inner.split(outer_x, outer_y):
        fit_x = np.array(outer_x[inner_train], dtype="float32", order="C", copy=True)
        scaler = StandardScaler(copy=False)
        scaler.fit_transform(fit_x)
        model = LogisticRegression(
            C=BASE_C, max_iter=MAX_ITER, solver="lbfgs", tol=1e-5
        )
        model.fit(fit_x, outer_y[inner_train])
        coefficients.append(model.coef_[0].astype("float64"))
        del fit_x, scaler, model
        gc.collect()
    coef = np.vstack(coefficients)
    signs = np.sign(coef)
    stable = np.all(signs == signs[0], axis=0) & np.all(signs != 0, axis=0)
    mean_abs = np.abs(coef).mean(axis=0)
    variability = coef.std(axis=0)
    score = mean_abs / (variability + 0.02)
    stable_indices = np.flatnonzero(stable)
    ranked_stable = stable_indices[np.argsort(score[stable_indices])[::-1]]
    if len(ranked_stable) < keep:
        remaining = np.setdiff1d(np.argsort(score)[::-1], ranked_stable, assume_unique=False)
        selected = np.concatenate([ranked_stable, remaining[: keep - len(ranked_stable)]])
    else:
        selected = ranked_stable[:keep]
    details = {
        "stable_count": int(stable.sum()),
        "selected_count": int(len(selected)),
        "selected_indices": selected.tolist(),
    }
    del outer_x, coefficients, coef
    gc.collect()
    return selected.astype(int), details


def mix_predictions(first: np.ndarray, second: np.ndarray, first_weight: float) -> np.ndarray:
    return percentile_rank(
        first_weight * percentile_rank(first) + (1.0 - first_weight) * percentile_rank(second)
    )


@dataclass
class CandidateResult:
    name: str
    oof: np.ndarray
    test: np.ndarray
    auc: float
    fold_auc: list[float]
    details: list[dict]


def candidate_fold_with_test(
    candidate: str,
    pool_oof: np.ndarray,
    pool_test: np.ndarray,
    base_oof: np.ndarray,
    base_test: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    complete: np.ndarray,
    missing_many: np.ndarray,
    test_complete: np.ndarray,
    test_missing_many: np.ndarray,
    groups: dict[str, np.ndarray],
    fold: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    started = time.time()
    details: dict = {"fold": fold, "candidate": candidate}

    if candidate == "fusion":
        train_dual, train_ranks = build_dual_features(pool_oof[train_idx], base_oof[train_idx])
        valid_dual, valid_ranks = build_dual_features(pool_oof[valid_idx], base_oof[valid_idx])
        test_dual, test_ranks = build_dual_features(pool_test, base_test)
        dual = fit_logistic(train_dual, y[train_idx], valid_dual, test_dual, FUSION_C, False)
        details["dual_iterations"] = dual.iterations

        train_core = family_core_features(pool_oof[train_idx], base_oof[train_idx], groups)
        valid_core = family_core_features(pool_oof[valid_idx], base_oof[valid_idx], groups)
        test_core = family_core_features(pool_test, base_test, groups)
        train_regime, d_mean, d_scale = compressed_regime_features(
            train_dual, train_ranks, train_core, complete[train_idx], missing_many[train_idx]
        )
        valid_regime, _, _ = compressed_regime_features(
            valid_dual, valid_ranks, valid_core, complete[valid_idx], missing_many[valid_idx], d_mean, d_scale
        )
        test_regime, _, _ = compressed_regime_features(
            test_dual, test_ranks, test_core, test_complete, test_missing_many, d_mean, d_scale
        )
        del train_dual, valid_dual, test_dual, train_ranks, valid_ranks, test_ranks
        del train_core, valid_core, test_core
        gc.collect()
        regime = fit_logistic(
            train_regime, y[train_idx], valid_regime, test_regime, FUSION_C, True
        )
        details["regime_iterations"] = regime.iterations
        details["dual_auc"] = float(roc_auc_score(y[valid_idx], dual.valid))
        details["regime_auc"] = float(roc_auc_score(y[valid_idx], regime.valid))
        valid_pred = mix_predictions(dual.valid, regime.valid, FUSION_WEIGHT)
        test_pred = mix_predictions(dual.test, regime.test, FUSION_WEIGHT)

    elif candidate == "stability":
        selected, selection = select_stable_members(pool_oof, y, train_idx, fold)
        details.update(selection)
        train_x, _ = build_dual_features(pool_oof[train_idx], base_oof[train_idx], selected)
        valid_x, _ = build_dual_features(pool_oof[valid_idx], base_oof[valid_idx], selected)
        test_x, _ = build_dual_features(pool_test, base_test, selected)
        fitted = fit_logistic(train_x, y[train_idx], valid_x, test_x, FUSION_C, True)
        details["iterations"] = fitted.iterations
        valid_pred, test_pred = fitted.valid, fitted.test

    elif candidate == "hierarchical":
        train_base = hierarchical_base_features(pool_oof[train_idx], base_oof[train_idx], groups)
        valid_base = hierarchical_base_features(pool_oof[valid_idx], base_oof[valid_idx], groups)
        test_base = hierarchical_base_features(pool_test, base_test, groups)
        train_x, d_mean, d_scale = full_regime_features(
            train_base, complete[train_idx], missing_many[train_idx]
        )
        valid_x, _, _ = full_regime_features(
            valid_base, complete[valid_idx], missing_many[valid_idx], d_mean, d_scale
        )
        test_x, _, _ = full_regime_features(
            test_base, test_complete, test_missing_many, d_mean, d_scale
        )
        del train_base, valid_base, test_base
        gc.collect()
        fitted = fit_logistic(train_x, y[train_idx], valid_x, test_x, FUSION_C, True)
        details["iterations"] = fitted.iterations
        valid_pred, test_pred = fitted.valid, fitted.test
    else:
        raise ValueError(f"Unknown candidate: {candidate}")

    details["valid_auc"] = float(roc_auc_score(y[valid_idx], valid_pred))
    details["seconds"] = time.time() - started
    return valid_pred, test_pred, details


def run_candidate(
    candidate: str,
    pool_oof: np.ndarray,
    pool_test: np.ndarray,
    names: list[str],
    base_oof: np.ndarray,
    base_test: np.ndarray,
    y: np.ndarray,
    complete: np.ndarray,
    missing_many: np.ndarray,
    test_complete: np.ndarray,
    test_missing_many: np.ndarray,
    artifact_dir: Path,
    resume: bool,
) -> CandidateResult:
    candidate_dir = artifact_dir / candidate
    candidate_dir.mkdir(parents=True, exist_ok=True)
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype="float64")
    test = np.zeros(len(pool_test), dtype="float64")
    fold_scores: list[float] = []
    details: list[dict] = []
    groups = family_index(names)

    for fold, (train_idx, valid_idx) in enumerate(folds.split(pool_oof, y), start=1):
        path = candidate_dir / f"fold_{fold}.npz"
        meta_path = candidate_dir / f"fold_{fold}.json"
        if resume and path.exists() and meta_path.exists():
            cached = np.load(path)
            if np.array_equal(cached["valid_idx"], valid_idx):
                valid_pred = cached["valid_pred"]
                test_pred = cached["test_pred"]
                detail = json.loads(meta_path.read_text(encoding="utf-8"))
            else:
                raise ValueError(f"Cached {candidate} fold {fold} indices do not align")
        else:
            valid_pred, test_pred, detail = candidate_fold_with_test(
                candidate, pool_oof, pool_test, base_oof, base_test, y,
                train_idx, valid_idx, complete, missing_many,
                test_complete, test_missing_many, groups, fold,
            )
            np.savez_compressed(
                path, valid_idx=valid_idx, valid_pred=valid_pred, test_pred=test_pred
            )
            meta_path.write_text(json.dumps(detail, indent=2), encoding="utf-8")
        oof[valid_idx] = valid_pred
        test += test_pred / N_SPLITS
        score = float(roc_auc_score(y[valid_idx], valid_pred))
        fold_scores.append(score)
        details.append(detail)
        print(f"candidate={candidate} fold={fold} auc={score:.9f}", flush=True)
        gc.collect()

    return CandidateResult(
        name=candidate,
        oof=oof,
        test=test,
        auc=float(roc_auc_score(y, oof)),
        fold_auc=fold_scores,
        details=details,
    )


def empirical_rank(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference))
    return np.searchsorted(ordered, values, side="right") / len(ordered)


def nested_blend(
    base_oof: np.ndarray,
    base_test: np.ndarray,
    candidate: CandidateResult,
    y: np.ndarray,
) -> CandidateResult:
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    nested = np.zeros(len(y), dtype="float64")
    selected_alpha: list[float] = []
    fold_scores: list[float] = []
    for train_idx, valid_idx in folds.split(base_oof, y):
        base_train = percentile_rank(base_oof[train_idx])
        candidate_train = percentile_rank(candidate.oof[train_idx])
        alpha = max(
            ALPHA_GRID,
            key=lambda value: roc_auc_score(
                y[train_idx], (1.0 - value) * base_train + value * candidate_train
            ),
        )
        selected_alpha.append(float(alpha))
        base_valid = empirical_rank(base_oof[train_idx], base_oof[valid_idx])
        candidate_valid = empirical_rank(candidate.oof[train_idx], candidate.oof[valid_idx])
        nested[valid_idx] = (1.0 - alpha) * base_valid + alpha * candidate_valid
        fold_scores.append(float(roc_auc_score(y[valid_idx], nested[valid_idx])))
    mean_alpha = float(np.mean(selected_alpha))
    test = percentile_rank(
        (1.0 - mean_alpha) * percentile_rank(base_test)
        + mean_alpha * percentile_rank(candidate.test)
    )
    return CandidateResult(
        name=f"base_plus_{candidate.name}",
        oof=nested,
        test=test,
        auc=float(roc_auc_score(y, nested)),
        fold_auc=fold_scores,
        details=[{"selected_alpha": selected_alpha, "mean_alpha": mean_alpha}],
    )


def advancement_gate(candidate: CandidateResult, base_oof: np.ndarray, y: np.ndarray) -> dict:
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    base_folds = [
        float(roc_auc_score(y[valid_idx], base_oof[valid_idx]))
        for _, valid_idx in folds.split(base_oof, y)
    ]
    gains = [new - old for new, old in zip(candidate.fold_auc, base_folds, strict=True)]
    base_auc = float(roc_auc_score(y, base_oof))
    return {
        "base_auc": base_auc,
        "candidate_auc": candidate.auc,
        "gain": candidate.auc - base_auc,
        "base_fold_auc": base_folds,
        "candidate_fold_auc": candidate.fold_auc,
        "fold_gains": gains,
        "overall_pass": candidate.auc >= base_auc + MIN_GAIN,
        "all_folds_pass": all(gain > 0 for gain in gains),
        "passed": candidate.auc >= base_auc + MIN_GAIN and all(gain > 0 for gain in gains),
    }


def load_inputs() -> tuple:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    y = train[TARGET].to_numpy(dtype="int8")
    manifest = json.loads(
        (V8_ARTIFACT / "cache" / "public_pool_manifest.json").read_text(encoding="utf-8")
    )
    names = manifest["members"]
    if names != public_member_names():
        raise ValueError("V8 pool manifest does not match the frozen 205-member registry")
    pool_oof = np.load(V8_ARTIFACT / "cache" / "public_pool_oof.npy", mmap_mode="r")
    pool_test = np.load(V8_ARTIFACT / "cache" / "public_pool_test.npy", mmap_mode="r")
    v8_oof = pd.read_csv(V8_ARTIFACT / "oof_predictions.csv")
    v8_test = pd.read_csv(ROOT / "submission_v8.csv")
    if not np.array_equal(v8_oof[ID_COL].to_numpy(), train[ID_COL].to_numpy()):
        raise ValueError("V8 OOF IDs do not align with train.csv")
    if not np.array_equal(v8_test[ID_COL].to_numpy(), test[ID_COL].to_numpy()):
        raise ValueError("V8 test IDs do not align with test.csv")
    feature_cols = [column for column in train if column not in (ID_COL, TARGET)]
    missing = train[feature_cols].isna().sum(axis=1).to_numpy()
    test_missing = test[feature_cols].isna().sum(axis=1).to_numpy()
    return (
        train, test, y, pool_oof, pool_test, names,
        v8_oof["pred_v8"].to_numpy(), v8_test[TARGET].to_numpy(),
        (missing == 0).astype("float32"), (missing >= 4).astype("float32"),
        (test_missing == 0).astype("float32"), (test_missing >= 4).astype("float32"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument(
        "--candidates", default=",".join(ALL_CANDIDATES),
        help="Comma-separated subset of fusion,stability,hierarchical",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    requested = tuple(value.strip() for value in args.candidates.split(",") if value.strip())
    unknown = set(requested) - set(ALL_CANDIDATES)
    if unknown:
        raise ValueError(f"Unknown candidates: {sorted(unknown)}")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    (
        train, test, y, pool_oof, pool_test, names, base_oof, base_test,
        complete, missing_many, test_complete, test_missing_many,
    ) = load_inputs()
    results: list[CandidateResult] = []
    for candidate in requested:
        result = run_candidate(
            candidate, pool_oof, pool_test, names, base_oof, base_test, y,
            complete, missing_many, test_complete, test_missing_many,
            args.artifact_dir, args.resume,
        )
        results.append(result)
        results.append(nested_blend(base_oof, base_test, result, y))
        print(f"{result.name}_oof_auc={result.auc:.9f}", flush=True)

    for result in results:
        pd.DataFrame({ID_COL: train[ID_COL], TARGET: y, "prediction": result.oof}).to_csv(
            args.artifact_dir / f"oof_{result.name}.csv", index=False
        )
        pd.DataFrame({ID_COL: test[ID_COL], TARGET: result.test}).to_csv(
            args.artifact_dir / f"submission_{result.name}.csv", index=False
        )

    selected = max(results, key=lambda result: result.auc)
    gate = advancement_gate(selected, base_oof, y)
    metrics = {
        "selected": selected.name,
        "candidates": {
            result.name: {
                "oof_auc": result.auc,
                "fold_auc": result.fold_auc,
                "details": result.details,
            }
            for result in results
        },
        "gate": gate,
    }
    (args.artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if gate["passed"]:
        pd.DataFrame({ID_COL: test[ID_COL], TARGET: selected.test}).to_csv(
            ROOT / "submission_v9.csv", index=False
        )
        print(f"PASS: wrote {ROOT / 'submission_v9.csv'}", flush=True)
    else:
        print("FAIL: no v9 candidate cleared the advancement gate", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()

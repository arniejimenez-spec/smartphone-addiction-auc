"""V8: cross-fitted OOF meta-stack over the strongest public model library.

V8 reproduces the base layer of the public ``Rank + Logit + Regime Fusion``
notebook (LB 0.97125) using only members that provide aligned out-of-fold and
test predictions.  Every member is converted to a percentile rank, then a
five-fold logistic regression is fit out of fold.  The frozen v7 predictions
can be evaluated as one additional member, but are retained only when they
improve honest OOF AUC.

The external prediction libraries are intentionally excluded from Git.  Put
them under ``external/v8`` and run::

    python train_v8.py --mode audit
    python train_v8.py --mode train

The training run writes diagnostics to ``artifacts/v8/full`` and, only when
the OOF advancement gate passes, writes ``submission_v8.csv`` at the project
root.
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


EXTERNAL_DIR = ROOT / "external" / "v8"
ARTIFACT_DIR = ROOT / "artifacts" / "v8" / "full"
N_SPLITS = 5
SEED = 42
META_C = 0.1
META_MAX_ITER = 1200
MIN_V7_GAIN = 0.0005
MIN_MEMBER_GAIN = 0.00002

NAJI_MEMBERS = ["07", "08", "09", "10", "12", "13", "14", "16", "18", "19"]

LOCAL_CANDIDATES = [
    "candidate_naji16_boltwide185_lookupv2_165_xgb10_01_rank",
    "candidate_naji16_boltwide1877_lookup1649_x0047_fm003_rank",
    "candidate_naji16_bolt_lookupv3v2_coord_rank",
    "candidate_naji16_bolt_deepfm_coord_rank",
    "candidate_naji16_bolt_final_coord_rank",
    "candidate_naji16_bolt_lookup256l8_rank",
]

BOLT_MEMBERS = [
    "xgb_hpo_d7", "xgb_te_5fold", "xgb_te_4fold", "xgb_d7_alt1",
    "xgb_d7_alt2", "xgb_dd_d4", "xgb_dd_d5", "xgb_dd_d6",
    "cat_nested_te", "cat_dual_view", "cat_dual_seed81", "cat_cpu5",
    "cat_pair_evidence", "lgb_te_5fold", "lgb_pair_lattice",
    "lgb_driver_recon", "lgb_missing_global", "lgb_raw_d6",
    "histgb_5fold", "xgb_raw_bag", "repr_lgb_global", "lookup_v1",
    "lookup_v2_s03", "lookup_v2_s81", "lookup_v2_s1037", "lookup_v2_s42",
    "lookup_v2_s959", "lookup_v3_evidence", "realmlp_lattice",
    "deepfm_exact", "fttransformer", "dcnv2_cross", "gandalf_gflu",
    "tabr_retrieval", "ebm_exact", "foldsafe_te_xgb",
    "foldsafe_te_xgb_10f", "foldsafe_te_cat", "foldsafe_te_multi",
    "foldsafe_te_wide", "lookup_v2_s20260901",
]

SZYMON_MEMBERS = [
    "altview", "cat", "cat_tuned", "digit_cat", "digit_lgbm", "digit_xgb",
    "hgb", "imp_cat", "imp_lgbm", "imp_lgbm_tuned", "imp_xgb",
    "imp_xgb_tuned", "lat_cat", "lat_lgbm", "lat_lgbm_s5", "lat_xgb",
    "latmax_lgbm", "latr1_lgbm", "latr1_xgb", "lattri_lgbm",
    "lattri_xgb", "latwide_cat", "latwide_lgbm", "latwide_xgb", "lgbm",
    "lgbm_tuned", "lookup", "naji01", "naji02", "naji03", "naji04",
    "naji05", "pub_cat", "pub_donlgbm", "pub_evg", "pub_ravi",
    "pub_resnet", "pub_rmlp", "pub_ryota", "pub_tabm", "pub_tabnet",
    "pubfe_cat", "pubfe_lgb", "pubfe_xgb", "pubmk_cat", "pubmk_nn",
    "rmlp_lat", "rmlp_lat3", "tabm_bounds", "tabm_deep", "tabm_deeper",
    "tabm_div", "tabm_imp", "tabm_seed3", "tabm_wide", "tabm_x12",
    "view_bounds_cat", "view_bounds_lgbm", "view_nolattice_lgbm",
    "view_rank_cat", "view_rank_lgbm", "view_resid_lgbm", "xgb", "xgb_tuned",
]

FM_MEMBERS = ["fmdeep", "fmnum", "fmplr", "fmpure", "fmwide"]
GOLEM_MEMBERS = list("abcdefg")
FRESH_MEMBERS = [
    "fresh_tabm_fresh_rich_s2026", "fresh_realmlp_fresh_s2026",
    "fresh_lookup_fresh_d256_l8_s5150", "fresh_lookup_fresh_d384_l6_s2718",
    "fresh_cat_fresh_d9_s606", "fresh_xgb_fresh_d6_s606",
    "fresh_xgb_fresh_d7_s314159",
]
ADARSH_MEMBERS = ["logregte", "catnative", "gxgbnote", "glgbnote2", "gcatnote"]
LOCAL_FINAL_MEMBERS = [
    "local_tabm_rich_seed3", "local_tabm_rich_alt", "local_tabm_rich_seed909",
    "local_lookup_d384_l4",
]
EXTRA_MEMBERS = [
    "kirill_o1", "koda_exact_te", "stringify_str3_d6",
    "stringify_strall_d6", "stringify_str3derived_d7", "cat_strall_d8",
]


def public_member_names() -> list[str]:
    """Return the exact 205 public members used by the reference base stack."""
    names = [f"naji{x}" for x in NAJI_MEMBERS[:8]]
    names += LOCAL_CANDIDATES
    names += [f"bolt_{x}" for x in BOLT_MEMBERS]
    names += [f"sz_{x}" for x in SZYMON_MEMBERS]
    names += [f"fm_{x}" for x in FM_MEMBERS]
    names += [f"golem_{x}" for x in GOLEM_MEMBERS]
    names += FRESH_MEMBERS
    names += [f"naji{x}" for x in NAJI_MEMBERS[8:]]
    names += [f"a_{x}" for x in ADARSH_MEMBERS]
    names += LOCAL_FINAL_MEMBERS
    names += [f"weak50_m{i:02d}" for i in range(1, 51)]
    names += EXTRA_MEMBERS
    if len(names) != 205 or len(set(names)) != len(names):
        raise AssertionError(f"Expected 205 unique public members, got {len(names)}")
    return names


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Average-tie percentile ranks as float32, matching the source notebook."""
    values = np.asarray(values).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Predictions contain NaN or infinite values")
    return (rankdata(values, method="average") / len(values)).astype("float32")


def validate_prediction(values: np.ndarray, rows: int, name: str) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    if len(values) != rows:
        raise ValueError(f"{name}: expected {rows} rows, found {len(values)}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: predictions contain NaN or infinity")
    return values


def _npy_pair(folder: Path, stem: str, n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    oof_path = folder / f"oof_{stem}.npy"
    test_path = folder / f"test_{stem}.npy"
    if not oof_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing prediction pair for {stem} in {folder}")
    return (
        validate_prediction(np.load(oof_path, mmap_mode="r"), n_train, f"oof_{stem}"),
        validate_prediction(np.load(test_path, mmap_mode="r"), n_test, f"test_{stem}"),
    )


def _naji_pair(folder: Path, number: str, train_ids: np.ndarray, test_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prefix = f"{number}_blend" if number not in {"01", "02", "03", "04", "05"} else number
    oof_path = folder / f"{prefix}_oof_predictions.csv"
    candidates = [folder / f"{prefix}_submission.csv", folder / f"{prefix}_submission.csv.csv"]
    test_path = next((p for p in candidates if p.exists()), candidates[0])
    if not oof_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing Naji {number} OOF/submission pair")
    oof = pd.read_csv(oof_path, usecols=[ID_COL, TARGET])
    test = pd.read_csv(test_path, usecols=[ID_COL, TARGET])
    if not np.array_equal(oof[ID_COL].to_numpy(), train_ids):
        raise ValueError(f"Naji {number} OOF IDs are not aligned")
    if not np.array_equal(test[ID_COL].to_numpy(), test_ids):
        raise ValueError(f"Naji {number} test IDs are not aligned")
    return oof[TARGET].to_numpy(), test[TARGET].to_numpy()


@dataclass
class PredictionPool:
    train: np.ndarray
    test: np.ndarray
    names: list[str]


def load_public_pool(
    external_dir: Path,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    cache_dir: Path | None = None,
    rebuild_cache: bool = False,
) -> PredictionPool:
    """Load, align, rank, and optionally cache the exact public member pool."""
    expected = public_member_names()
    n_train, n_test = len(train_ids), len(test_ids)
    if cache_dir is not None:
        manifest_path = cache_dir / "public_pool_manifest.json"
        oof_cache = cache_dir / "public_pool_oof.npy"
        test_cache = cache_dir / "public_pool_test.npy"
        if not rebuild_cache and manifest_path.exists() and oof_cache.exists() and test_cache.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("members") == expected and manifest.get("shape") == [n_train, n_test]:
                return PredictionPool(
                    np.load(oof_cache, mmap_mode="r"),
                    np.load(test_cache, mmap_mode="r"),
                    expected,
                )

    x_oof = np.empty((n_train, len(expected)), dtype="float32")
    x_test = np.empty((n_test, len(expected)), dtype="float32")
    loaded: list[str] = []

    def add(name: str, oof: np.ndarray, test: np.ndarray) -> None:
        if name != expected[len(loaded)]:
            raise AssertionError(f"Member order mismatch: expected {expected[len(loaded)]}, got {name}")
        oof = validate_prediction(oof, n_train, f"{name} OOF")
        test = validate_prediction(test, n_test, f"{name} test")
        x_oof[:, len(loaded)] = percentile_rank(oof)
        x_test[:, len(loaded)] = percentile_rank(test)
        loaded.append(name)

    for number in NAJI_MEMBERS[:8]:
        add(f"naji{number}", *_naji_pair(external_dir, number, train_ids, test_ids))

    local_dir = external_dir / "6" / "openx_our_members"
    for stem in LOCAL_CANDIDATES:
        add(stem, *_npy_pair(local_dir, stem, n_train, n_test))

    bolt_oof = pd.read_parquet(
        external_dir / "oof_predictions.parquet", columns=[ID_COL, *BOLT_MEMBERS]
    )
    bolt_test = pd.read_parquet(
        external_dir / "test_predictions.parquet", columns=[ID_COL, *BOLT_MEMBERS]
    )
    if not np.array_equal(bolt_oof[ID_COL].to_numpy(), train_ids):
        raise ValueError("Bolt OOF IDs are not aligned")
    if not np.array_equal(bolt_test[ID_COL].to_numpy(), test_ids):
        raise ValueError("Bolt test IDs are not aligned")
    for stem in BOLT_MEMBERS:
        add(f"bolt_{stem}", bolt_oof[stem].to_numpy(), bolt_test[stem].to_numpy())
    del bolt_oof, bolt_test
    gc.collect()

    for stem in SZYMON_MEMBERS:
        add(f"sz_{stem}", *_npy_pair(external_dir / "oof", stem, n_train, n_test))
    for stem in FM_MEMBERS:
        add(f"fm_{stem}", *_npy_pair(external_dir, stem, n_train, n_test))
    for stem in GOLEM_MEMBERS:
        add(f"golem_{stem}", *_npy_pair(external_dir / "4", stem, n_train, n_test))
    for stem in FRESH_MEMBERS:
        add(stem, *_npy_pair(local_dir, stem, n_train, n_test))
    for number in NAJI_MEMBERS[8:]:
        add(f"naji{number}", *_naji_pair(external_dir, number, train_ids, test_ids))
    for stem in ADARSH_MEMBERS:
        add(f"a_{stem}", *_npy_pair(external_dir / "5", stem, n_train, n_test))
    for stem in LOCAL_FINAL_MEMBERS:
        add(stem, *_npy_pair(local_dir, stem, n_train, n_test))

    weak_oof = np.load(external_dir / "7" / "oof.npy", mmap_mode="r")
    weak_test = np.load(external_dir / "7" / "test.npy", mmap_mode="r")
    if weak_oof.shape != (n_train, 50) or weak_test.shape != (n_test, 50):
        raise ValueError(f"Weak-50 matrices have unexpected shapes {weak_oof.shape}/{weak_test.shape}")
    for index in range(50):
        add(f"weak50_m{index + 1:02d}", weak_oof[:, index], weak_test[:, index])
    for stem in EXTRA_MEMBERS:
        add(stem, *_npy_pair(external_dir / "8", stem, n_train, n_test))

    if loaded != expected:
        raise AssertionError("Loaded member registry does not match the reference registry")

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / "public_pool_oof.npy", x_oof)
        np.save(cache_dir / "public_pool_test.npy", x_test)
        (cache_dir / "public_pool_manifest.json").write_text(
            json.dumps({"members": loaded, "shape": [n_train, n_test]}, indent=2),
            encoding="utf-8",
        )
    return PredictionPool(x_oof, x_test, loaded)


def load_v7(train_ids: np.ndarray, test_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    oof_path = ROOT / "artifacts" / "v7" / "full" / "oof_predictions.csv"
    test_path = ROOT / "submission_v7.csv"
    oof = pd.read_csv(oof_path, usecols=[ID_COL, "pred_v7"])
    test = pd.read_csv(test_path, usecols=[ID_COL, TARGET])
    if not np.array_equal(oof[ID_COL].to_numpy(), train_ids):
        raise ValueError("v7 OOF IDs are not aligned")
    if not np.array_equal(test[ID_COL].to_numpy(), test_ids):
        raise ValueError("v7 test IDs are not aligned")
    return percentile_rank(oof["pred_v7"].to_numpy()), percentile_rank(test[TARGET].to_numpy())


def append_member(pool: PredictionPool, name: str, oof: np.ndarray, test: np.ndarray) -> PredictionPool:
    return PredictionPool(
        np.column_stack([pool.train, oof]).astype("float32", copy=False),
        np.column_stack([pool.test, test]).astype("float32", copy=False),
        [*pool.names, name],
    )


@dataclass
class StackResult:
    oof: np.ndarray
    test: np.ndarray
    auc: float
    fold_auc: list[float]
    coefficients: pd.DataFrame


def fit_meta_stack(x_oof: np.ndarray, x_test: np.ndarray, y: np.ndarray, names: list[str]) -> StackResult:
    """Fit the frozen five-fold standardized logistic meta-model."""
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_pred = np.zeros(len(y), dtype="float64")
    test_pred = np.zeros(len(x_test), dtype="float64")
    fold_scores: list[float] = []
    coefficient_rows: list[dict] = []

    for fold, (train_idx, valid_idx) in enumerate(folds.split(x_oof, y), start=1):
        started = time.time()
        fold_train = np.array(x_oof[train_idx], dtype="float32", order="C", copy=True)
        fold_valid = np.array(x_oof[valid_idx], dtype="float32", order="C", copy=True)
        fold_test = np.array(x_test, dtype="float32", order="C", copy=True)
        scaler = StandardScaler(copy=False)
        scaler.fit_transform(fold_train)
        scaler.transform(fold_valid)
        scaler.transform(fold_test)
        model = LogisticRegression(
            C=META_C, max_iter=META_MAX_ITER, solver="lbfgs", tol=1e-5,
            random_state=SEED + fold,
        )
        model.fit(fold_train, y[train_idx])
        oof_pred[valid_idx] = model.predict_proba(fold_valid)[:, 1]
        test_pred += model.predict_proba(fold_test)[:, 1] / N_SPLITS
        score = float(roc_auc_score(y[valid_idx], oof_pred[valid_idx]))
        fold_scores.append(score)
        for member, coefficient in zip(names, model.coef_[0], strict=True):
            coefficient_rows.append({"fold": fold, "member": member, "coefficient": float(coefficient)})
        print(
            f"fold={fold} auc={score:.9f} iterations={model.n_iter_[0]} "
            f"seconds={time.time() - started:.1f}",
            flush=True,
        )
        del fold_train, fold_valid, fold_test, scaler, model
        gc.collect()

    return StackResult(
        oof=oof_pred,
        test=test_pred,
        auc=float(roc_auc_score(y, oof_pred)),
        fold_auc=fold_scores,
        coefficients=pd.DataFrame(coefficient_rows),
    )


def fold_auc(values: np.ndarray, y: np.ndarray) -> list[float]:
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return [float(roc_auc_score(y[valid], values[valid])) for _, valid in folds.split(values, y)]


def select_stack(public: StackResult, with_v7: StackResult | None) -> tuple[str, StackResult]:
    if with_v7 is not None and with_v7.auc >= public.auc + MIN_MEMBER_GAIN:
        return "public_plus_v7", with_v7
    return "public", public


def advancement_gate(result: StackResult, v7_oof: np.ndarray, y: np.ndarray) -> dict:
    v7_auc = float(roc_auc_score(y, v7_oof))
    v7_folds = fold_auc(v7_oof, y)
    deltas = [new - old for new, old in zip(result.fold_auc, v7_folds, strict=True)]
    return {
        "v7_auc": v7_auc,
        "v8_auc": result.auc,
        "gain": result.auc - v7_auc,
        "v7_fold_auc": v7_folds,
        "v8_fold_auc": result.fold_auc,
        "fold_gains": deltas,
        "overall_pass": result.auc >= v7_auc + MIN_V7_GAIN,
        "all_folds_pass": all(delta > 0 for delta in deltas),
        "passed": result.auc >= v7_auc + MIN_V7_GAIN and all(delta > 0 for delta in deltas),
    }


def audit_sources(external_dir: Path, train_ids: np.ndarray, test_ids: np.ndarray) -> dict:
    """Perform a full exact-member load; successful return proves alignment."""
    started = time.time()
    pool = load_public_pool(external_dir, train_ids, test_ids, cache_dir=None)
    return {
        "members": len(pool.names),
        "unique_members": len(set(pool.names)),
        "oof_shape": list(pool.train.shape),
        "test_shape": list(pool.test.shape),
        "finite": bool(np.isfinite(pool.train).all() and np.isfinite(pool.test).all()),
        "seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["audit", "train"], default="train")
    parser.add_argument("--external-dir", type=Path, default=EXTERNAL_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--compare-v7", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    train = pd.read_csv(ROOT / "train.csv", usecols=[ID_COL, TARGET])
    test = pd.read_csv(ROOT / "test.csv", usecols=[ID_COL])
    train_ids = train[ID_COL].to_numpy()
    test_ids = test[ID_COL].to_numpy()
    y = train[TARGET].to_numpy(dtype="int8")

    if args.mode == "audit":
        report = audit_sources(args.external_dir, train_ids, test_ids)
        print(json.dumps(report, indent=2))
        return

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    pool = load_public_pool(
        args.external_dir, train_ids, test_ids,
        cache_dir=args.artifact_dir / "cache", rebuild_cache=args.rebuild_cache,
    )
    print(f"Loaded {len(pool.names)} public members in {time.time() - started:.1f}s", flush=True)
    public = fit_meta_stack(pool.train, pool.test, y, pool.names)
    print(f"public_stack_oof_auc={public.auc:.9f}", flush=True)

    v7_oof, v7_test = load_v7(train_ids, test_ids)
    with_v7: StackResult | None = None
    if args.compare_v7:
        pool_v7 = append_member(pool, "local_v7", v7_oof, v7_test)
        with_v7 = fit_meta_stack(pool_v7.train, pool_v7.test, y, pool_v7.names)
        print(f"public_plus_v7_oof_auc={with_v7.auc:.9f}", flush=True)
        del pool_v7
        gc.collect()

    selected_name, selected = select_stack(public, with_v7)
    gate = advancement_gate(selected, v7_oof, y)
    metrics = {
        "selected": selected_name,
        "public_member_count": len(pool.names),
        "selected_member_count": len(pool.names) + int(selected_name == "public_plus_v7"),
        "public_oof_auc": public.auc,
        "public_fold_auc": public.fold_auc,
        "public_plus_v7_oof_auc": None if with_v7 is None else with_v7.auc,
        "public_plus_v7_fold_auc": None if with_v7 is None else with_v7.fold_auc,
        "reference_public_oof_auc": 0.97022124036,
        "gate": gate,
        "elapsed_seconds": time.time() - started,
    }
    (args.artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.artifact_dir / "members.json").write_text(
        json.dumps({"selected": selected_name, "members": pool.names + (["local_v7"] if selected_name == "public_plus_v7" else [])}, indent=2),
        encoding="utf-8",
    )
    selected.coefficients.to_csv(args.artifact_dir / "coefficients.csv", index=False)
    pd.DataFrame({
        ID_COL: train_ids, TARGET: y, "pred_v7": v7_oof, "pred_v8": selected.oof,
    }).to_csv(args.artifact_dir / "oof_predictions.csv", index=False)
    submission = pd.DataFrame({ID_COL: test_ids, TARGET: selected.test})
    submission.to_csv(args.artifact_dir / "submission_v8.csv", index=False)

    if gate["passed"]:
        submission.to_csv(ROOT / "submission_v8.csv", index=False)
        print(f"PASS: wrote {ROOT / 'submission_v8.csv'}")
    else:
        print("FAIL: v8 did not clear the advancement gate; root submission was not written")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

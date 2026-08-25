"""V10 exact full rank-logit/regime fusion for a Kaggle GPU.

This is the full 206-member feature construction from Byer's Apache-2.0
``S6E8 Rank-Logit-Regime Fusion`` notebook, adapted to consume the audited v8
pool cache.  Unlike the reference's single capped optimizer call, each GPU
fit runs in resumable LBFGS blocks, records convergence diagnostics, and will
not publish ``submission_v10.csv`` unless both models converge.

Expected inputs
---------------
* competition ``train.csv`` and ``test.csv``;
* ``public_pool_oof.npy``, ``public_pool_test.npy``, and
  ``public_pool_manifest.json`` from ``artifacts/v8/full/cache``.

On Kaggle, attach the competition and a private dataset containing the three
cache files. Paths are auto-discovered under ``/kaggle/input`` or may be set
explicitly with ``--data-root`` and ``--pool-root``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ID_COL = "id"
TARGET = "addicted_label"
N_SPLITS = 5
SEED = 42
EXPECTED_MEMBERS = 205
EXPECTED_TRAIN_ROWS = 691_369
EXPECTED_TEST_ROWS = 296_302
BASE_C = 0.1
FUSION_C = 3.5
MIX_WEIGHT = 0.55
NESTED_ALPHA = 0.70
REFERENCE_BASE_AUC = 0.9702186508338726


@dataclass
class FitDiagnostics:
    name: str
    features: int
    rows: int
    device: str
    dtype: str
    penalty_gradient: bool
    completed_iterations: int
    closure_evaluations: int
    blocks: int
    objective: float
    gradient_max_abs: float
    gradient_tolerance: float
    parameter_change_max_abs: float
    directional_derivative: float
    converged: bool
    stop_reason: str
    seconds: float


def rank01(values: np.ndarray) -> np.ndarray:
    """Match the reference notebook's deterministic half-rank transform."""
    values = np.asarray(values).reshape(-1)
    return (np.argsort(np.argsort(values)) + 0.5) / values.size


def clipped_logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def build_regime_features(
    rank_half: np.ndarray,
    dual: np.ndarray,
    complete: np.ndarray,
    missing_many: np.ndarray,
) -> np.ndarray:
    """Build the exact ``[b, b*c, b*m, b*d, agg]`` regime matrix.

    The output is filled in place to avoid the three multi-gigabyte temporary
    interaction arrays created by a direct ``column_stack`` implementation.
    """
    rank_half = np.asarray(rank_half, dtype=np.float64, order="C")
    dual = np.asarray(dual, dtype=np.float64, order="C")
    complete = np.asarray(complete, dtype=np.float64).reshape(-1)
    missing_many = np.asarray(missing_many, dtype=np.float64).reshape(-1)
    if rank_half.shape[0] != dual.shape[0]:
        raise ValueError("Rank and dual row counts differ")
    if len(complete) != dual.shape[0] or len(missing_many) != dual.shape[0]:
        raise ValueError("Regime indicators do not align with feature rows")

    rows, width = dual.shape
    result = np.empty((rows, 4 * width + 5), dtype=np.float64)
    result[:, :width] = dual
    np.multiply(dual, complete[:, None], out=result[:, width : 2 * width])
    np.multiply(dual, missing_many[:, None], out=result[:, 2 * width : 3 * width])
    disagreement = rank_half.std(axis=1)
    disagreement = (disagreement - disagreement.mean()) / (disagreement.std() + 1e-12)
    np.multiply(dual, disagreement[:, None], out=result[:, 3 * width : 4 * width])
    result[:, 4 * width] = rank_half.mean(axis=1)
    result[:, 4 * width + 1] = rank_half.std(axis=1)
    result[:, 4 * width + 2] = rank_half.max(axis=1) - rank_half.min(axis=1)
    result[:, 4 * width + 3] = complete
    result[:, 4 * width + 4] = missing_many
    return result


def _find_under_kaggle(filename: str) -> list[Path]:
    base = Path("/kaggle/input")
    return sorted(base.glob(f"**/{filename}")) if base.exists() else []


def resolve_file(filename: str, explicit_root: Path | None, local_candidates: list[Path]) -> Path:
    candidates: list[Path] = []
    if explicit_root is not None:
        candidates.extend([explicit_root / filename, explicit_root / "cache" / filename])
    candidates.extend(path / filename for path in local_candidates)
    candidates.extend(_find_under_kaggle(filename))
    existing: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved.exists() and resolved not in seen:
            existing.append(resolved)
            seen.add(resolved)
    if not existing:
        roots = [str(explicit_root)] if explicit_root else [str(p) for p in local_candidates]
        raise FileNotFoundError(f"Could not locate {filename}; searched roots {roots} and /kaggle/input")
    if explicit_root is None and len(existing) > 1:
        print(f"warning: multiple {filename} files found; using {existing[0]}", flush=True)
    return existing[0]


def load_inputs(data_root: Path | None, pool_root: Path | None) -> tuple[Any, ...]:
    project = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    train_path = resolve_file("train.csv", data_root, [project])
    test_path = resolve_file("test.csv", data_root, [project])
    pool_candidates = [project / "artifacts" / "v8" / "full" / "cache"]
    oof_path = resolve_file("public_pool_oof.npy", pool_root, pool_candidates)
    test_pool_path = resolve_file("public_pool_test.npy", pool_root, pool_candidates)
    manifest_path = resolve_file("public_pool_manifest.json", pool_root, pool_candidates)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    pool_oof = np.load(oof_path, mmap_mode="r")
    pool_test = np.load(test_pool_path, mmap_mode="r")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    members = manifest.get("members", [])

    if len(train) != EXPECTED_TRAIN_ROWS or len(test) != EXPECTED_TEST_ROWS:
        raise ValueError(f"Unexpected train/test rows: {len(train)} / {len(test)}")
    if pool_oof.shape != (len(train), EXPECTED_MEMBERS):
        raise ValueError(f"Unexpected OOF pool shape: {pool_oof.shape}")
    if pool_test.shape != (len(test), EXPECTED_MEMBERS):
        raise ValueError(f"Unexpected test pool shape: {pool_test.shape}")
    if len(members) != EXPECTED_MEMBERS or len(set(members)) != EXPECTED_MEMBERS:
        raise ValueError("Manifest must contain 205 unique members")
    if not np.isfinite(pool_oof).all() or not np.isfinite(pool_test).all():
        raise ValueError("Prediction pool contains non-finite values")
    if list(train.columns).count(TARGET) != 1 or TARGET in test:
        raise ValueError("Competition target columns are not as expected")

    paths = {
        "train": str(train_path),
        "test": str(test_path),
        "pool_oof": str(oof_path),
        "pool_test": str(test_pool_path),
        "manifest": str(manifest_path),
    }
    return train, test, pool_oof, pool_test, members, paths


def refit_base(
    pool_oof: np.ndarray,
    pool_test: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    folds = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    base_oof = np.empty(len(y), dtype=np.float64)
    base_test = np.zeros(pool_test.shape[0], dtype=np.float64)
    details: list[dict[str, Any]] = []
    for fold, (fit_idx, valid_idx) in enumerate(folds.split(np.zeros(len(y)), y), start=1):
        started = time.perf_counter()
        scaler = StandardScaler().fit(pool_oof[fit_idx])
        model = LogisticRegression(C=BASE_C, max_iter=1200, solver="lbfgs", tol=1e-5)
        model.fit(scaler.transform(pool_oof[fit_idx]), y[fit_idx])
        base_oof[valid_idx] = model.predict_proba(scaler.transform(pool_oof[valid_idx]))[:, 1]
        base_test += model.predict_proba(scaler.transform(pool_test))[:, 1] / N_SPLITS
        auc = float(roc_auc_score(y[valid_idx], base_oof[valid_idx]))
        details.append({
            "fold": fold,
            "auc": auc,
            "iterations": int(model.n_iter_[0]),
            "seconds": time.perf_counter() - started,
        })
        print(f"base fold={fold} auc={auc:.9f}", flush=True)
    pooled = float(roc_auc_score(y, base_oof))
    if abs(pooled - REFERENCE_BASE_AUC) > 5e-5:
        raise RuntimeError(
            f"Base OOF AUC {pooled:.9f} is not aligned with expected {REFERENCE_BASE_AUC:.9f}"
        )
    print(f"base pooled OOF={pooled:.9f}", flush=True)
    return base_oof, base_test, details


def build_dual(pool: np.ndarray, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    augmented = np.column_stack([np.asarray(pool), np.asarray(base).reshape(-1)])
    ranks = np.column_stack([rank01(augmented[:, j]) for j in range(augmented.shape[1])])
    dual = np.hstack([ranks, clipped_logit(augmented)])
    return np.asarray(dual, dtype=np.float64, order="C"), ranks


def standardize_in_place(
    train_features: np.ndarray,
    test_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale == 0] = 1.0
    train_features -= mean
    train_features /= scale
    test_features -= mean
    test_features /= scale
    return mean, scale


def _optimizer_state(optimizer: Any) -> dict[str, Any]:
    first_parameter = optimizer.param_groups[0]["params"][0]
    return optimizer.state[first_parameter]


def convergence_reason(
    gradient_max: float,
    gradient_tolerance: float,
    previous_objective: float | None,
    objective: float,
    parameter_change: float,
    differentiate_penalty: bool,
) -> str | None:
    """Return the audited reason a fit may stop, or ``None`` to continue."""
    if gradient_max <= gradient_tolerance:
        return f"first-order gradient tolerance <= {gradient_tolerance:.3e}"
    if (
        differentiate_penalty
        and previous_objective is not None
        and abs(previous_objective - objective) <= 1e-14
        and parameter_change <= 1e-12
    ):
        return "certified objective-and-step tolerance"
    return None


def fit_gpu_logistic(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    output_dir: Path,
    max_total_iter: int,
    block_iter: int,
    chunk_rows: int,
    allow_cpu: bool,
    resume: bool,
    differentiate_penalty: bool,
    gradient_tolerance: float,
) -> tuple[np.ndarray, FitDiagnostics]:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("PyTorch is required; run v10 in a Kaggle GPU notebook") from exc

    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("CUDA GPU not available; v10 requires Kaggle GPU acceleration")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_train = np.asarray(x_train, dtype=np.float64, order="C")
    x_test = np.asarray(x_test, dtype=np.float64, order="C")
    y_train = np.asarray(y_train, dtype=np.float64)
    if x_train.shape[0] != len(y_train) or x_test.shape[1] != x_train.shape[1]:
        raise ValueError(f"Misaligned {name} matrices")

    started = time.perf_counter()
    train_tensor = torch.as_tensor(x_train, device=device, dtype=torch.float64)
    target_tensor = torch.as_tensor(y_train, device=device, dtype=torch.float64)
    test_tensor = torch.as_tensor(x_test, device=device, dtype=torch.float64)
    model = torch.nn.Linear(x_train.shape[1], 1, bias=True, device=device, dtype=torch.float64)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=block_iter,
        max_eval=math.ceil(2.0 * block_iter), tolerance_grad=1e-10,
        tolerance_change=1e-15, history_size=50, line_search_fn="strong_wolfe",
    )
    regularization = 1.0 / (2.0 * FUSION_C * len(y_train))
    checkpoint = output_dir / f"{name}_checkpoint.pt"
    completed_iterations = 0
    blocks = 0
    closure_evaluations = 0
    previous_objective: float | None = None
    converged = False
    stop_reason = "maximum iterations reached"
    objective = float("nan")
    gradient_max = float("inf")
    parameter_change = float("inf")
    directional_derivative = float("-inf")
    if resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved.get("features") != x_train.shape[1]:
            raise ValueError(f"{name} checkpoint feature count does not match")
        if bool(saved.get("differentiate_penalty", False)) != differentiate_penalty:
            raise ValueError(f"{name} checkpoint penalty-gradient mode does not match")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        completed_iterations = int(saved.get("completed_iterations", 0))
        blocks = int(saved.get("blocks", 0))
        closure_evaluations = int(saved.get("closure_evaluations", 0))
        previous_objective = saved.get("objective")
        converged = bool(saved.get("converged", False))
        stop_reason = str(saved.get("stop_reason", "checkpoint convergence"))
        objective = float(saved.get("objective", float("nan")))
        gradient_max = float(saved.get("gradient_max_abs", float("inf")))
        parameter_change = float(saved.get("parameter_change_max_abs", float("inf")))
        directional_derivative = float(saved.get("directional_derivative", float("-inf")))
        print(f"resumed {name} at iteration {completed_iterations}", flush=True)

    def closure() -> Any:
        nonlocal closure_evaluations
        closure_evaluations += 1
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device=device, dtype=torch.float64)
        for start in range(0, len(y_train), chunk_rows):
            logits = model(train_tensor[start : start + chunk_rows]).squeeze(1)
            loss = functional.binary_cross_entropy_with_logits(
                logits, target_tensor[start : start + chunk_rows], reduction="sum"
            ) / len(y_train)
            loss.backward()
            total += loss.detach()
        penalty = regularization * model.weight.square().sum()
        if differentiate_penalty:
            penalty.backward()
        return total + penalty.detach()

    while not converged and completed_iterations < max_total_iter:
        state = _optimizer_state(optimizer)
        before = int(state.get("n_iter", 0))
        optimizer.step(closure)
        state = _optimizer_state(optimizer)
        after = int(state.get("n_iter", 0))
        used = after - before
        completed_iterations += used
        blocks += 1
        objective_tensor = closure()
        objective = float(objective_tensor.detach().cpu())
        gradient_max = max(
            float(parameter.grad.detach().abs().max().cpu())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        direction = state.get("d")
        step_size = state.get("t")
        if direction is not None and step_size is not None:
            parameter_change = float((direction * step_size).abs().max().detach().cpu())
            flat_gradient = torch.cat(
                [parameter.grad.detach().reshape(-1) for parameter in model.parameters()]
            )
            directional_derivative = float(flat_gradient.dot(direction).detach().cpu())
        else:
            parameter_change = 0.0
            directional_derivative = 0.0
        relative_change = (
            float("inf") if previous_objective is None else
            abs(previous_objective - objective) / max(1.0, abs(previous_objective))
        )
        print(
            f"{name} block={blocks} iterations={completed_iterations} "
            f"objective={objective:.15g} grad_max={gradient_max:.3e} "
            f"step_max={parameter_change:.3e} gtd={directional_derivative:.3e} "
            f"relative_change={relative_change:.3e}",
            flush=True,
        )
        reason = convergence_reason(
            gradient_max,
            gradient_tolerance,
            previous_objective,
            objective,
            parameter_change,
            differentiate_penalty,
        )
        if reason is not None:
            converged = True
            stop_reason = reason

        torch.save(
            {
                "features": x_train.shape[1],
                "differentiate_penalty": differentiate_penalty,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "completed_iterations": completed_iterations,
                "blocks": blocks,
                "closure_evaluations": closure_evaluations,
                "converged": converged,
                "stop_reason": stop_reason,
                "objective": objective,
                "gradient_max_abs": gradient_max,
                "gradient_tolerance": gradient_tolerance,
                "parameter_change_max_abs": parameter_change,
                "directional_derivative": directional_derivative,
            },
            checkpoint,
        )
        if converged:
            break
        previous_objective = objective

    if not converged:
        raise RuntimeError(
            f"{name} did not converge after {completed_iterations} iterations; "
            f"checkpoint retained at {checkpoint}. Re-run with a larger --max-total-iter."
        )

    with torch.no_grad():
        prediction = np.empty(x_test.shape[0], dtype=np.float64)
        for start in range(0, x_test.shape[0], chunk_rows):
            prediction[start : start + chunk_rows] = (
                model(test_tensor[start : start + chunk_rows]).squeeze(1).cpu().numpy()
            )
    state = _optimizer_state(optimizer)
    diagnostics = FitDiagnostics(
        name=name,
        features=x_train.shape[1],
        rows=x_train.shape[0],
        device=str(device),
        dtype="float64",
        penalty_gradient=differentiate_penalty,
        completed_iterations=completed_iterations,
        closure_evaluations=closure_evaluations,
        blocks=blocks,
        objective=objective,
        gradient_max_abs=gradient_max,
        gradient_tolerance=gradient_tolerance,
        parameter_change_max_abs=parameter_change,
        directional_derivative=directional_derivative,
        converged=converged,
        stop_reason=stop_reason,
        seconds=time.perf_counter() - started,
    )
    del train_tensor, target_tensor, test_tensor, model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return prediction, diagnostics


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train, test, pool_oof, pool_test, members, paths = load_inputs(
        args.data_root, args.pool_root
    )
    if args.audit_only:
        result = {
            "mode": "audit",
            "train_rows": len(train),
            "test_rows": len(test),
            "pool_oof_shape": list(pool_oof.shape),
            "pool_test_shape": list(pool_test.shape),
            "members": len(members),
            "paths": paths,
        }
        (output_dir / "audit_v10.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2), flush=True)
        return result
    y = train[TARGET].to_numpy(dtype=np.int8)
    feature_columns = [column for column in train if column not in (ID_COL, TARGET)]
    missing = train[feature_columns].isna().sum(axis=1).to_numpy()
    test_missing = test[feature_columns].isna().sum(axis=1).to_numpy()
    complete = (missing == 0).astype(np.float64)
    missing_many = (missing >= 4).astype(np.float64)
    test_complete = (test_missing == 0).astype(np.float64)
    test_missing_many = (test_missing >= 4).astype(np.float64)

    print(f"[{dt.datetime.now():%H:%M:%S}] refitting required v8 base member", flush=True)
    base_oof, base_test, base_details = refit_base(pool_oof, pool_test, y)
    base_auc = float(roc_auc_score(y, base_oof))

    print(f"[{dt.datetime.now():%H:%M:%S}] building exact dual matrices", flush=True)
    train_dual, train_ranks = build_dual(pool_oof, base_oof)
    test_dual, test_ranks = build_dual(pool_test, base_test)
    if train_dual.shape[1] != 412:
        raise RuntimeError(f"Expected 412 dual columns, got {train_dual.shape[1]}")
    dual_prediction, dual_diagnostics = fit_gpu_logistic(
        "dual", train_dual, y, test_dual, output_dir,
        args.max_total_iter, args.block_iter, args.chunk_rows,
        args.allow_cpu, args.resume, not args.source_closure,
        args.gradient_tolerance,
    )

    print(f"[{dt.datetime.now():%H:%M:%S}] building exact 1,653-column regime matrices", flush=True)
    train_regime = build_regime_features(
        train_ranks, train_dual, complete, missing_many
    )
    test_regime = build_regime_features(
        test_ranks, test_dual, test_complete, test_missing_many
    )
    if train_regime.shape[1] != 1653:
        raise RuntimeError(f"Expected 1,653 regime columns, got {train_regime.shape[1]}")
    standardize_in_place(train_regime, test_regime)
    del train_dual, test_dual, train_ranks, test_ranks
    gc.collect()
    regime_prediction, regime_diagnostics = fit_gpu_logistic(
        "regime", train_regime, y, test_regime, output_dir,
        args.max_total_iter, args.block_iter, args.chunk_rows,
        args.allow_cpu, args.resume, not args.source_closure,
        args.gradient_tolerance,
    )
    del train_regime, test_regime
    gc.collect()

    test_mix = rank01(
        MIX_WEIGHT * rank01(dual_prediction)
        + (1.0 - MIX_WEIGHT) * rank01(regime_prediction)
    )
    test_nested = rank01((1.0 - NESTED_ALPHA) * base_test + NESTED_ALPHA * test_mix)
    np.save(output_dir / "test_mix.npy", test_mix)
    np.save(output_dir / "test_nested.npy", test_nested)
    pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_mix}).to_csv(
        output_dir / "submission_v10_mix.csv", index=False
    )
    pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_nested}).to_csv(
        output_dir / "submission_v10.csv", index=False
    )

    result = {
        "version": "v10.0.0",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": "exact 206-member rank-logit/regime feature construction",
        "member_count": len(members) + 1,
        "dual_features": 412,
        "regime_features": 1653,
        "base_oof_auc": base_auc,
        "mix_weight": MIX_WEIGHT,
        "nested_alpha": NESTED_ALPHA,
        "source_closure": args.source_closure,
        "paths": paths,
        "base_folds": base_details,
        "dual_fit": asdict(dual_diagnostics),
        "regime_fit": asdict(regime_diagnostics),
        "selected_submission": "submission_v10.csv",
        "alternative_submission": "submission_v10_mix.csv",
    }
    (output_dir / "result_v10.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    print(f"DONE: {output_dir / 'submission_v10.csv'}", flush=True)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--pool-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=project / "artifacts" / "v10")
    parser.add_argument("--max-total-iter", type=int, default=20000)
    parser.add_argument("--block-iter", type=int, default=250)
    parser.add_argument("--chunk-rows", type=int, default=131072)
    parser.add_argument("--gradient-tolerance", type=float, default=5e-7)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--source-closure",
        action="store_true",
        help="Reproduce the source's non-differentiated L2 term; not selected by default",
    )
    parser.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (
        args.max_total_iter <= 0
        or args.block_iter <= 0
        or args.chunk_rows <= 0
        or args.gradient_tolerance <= 0
    ):
        parser.error("iteration, chunk, and tolerance values must be positive")
    if args.block_iter > args.max_total_iter:
        parser.error("--block-iter cannot exceed --max-total-iter")
    return args


def main() -> None:
    try:
        run(parse_args())
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()

# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under dual license: AGPL-3.0 (open-source) or commercial. See LICENSE.

"""K-fold cross-validation engine for discrete choice models."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch

from .config import FullModelSpec, ModelSpec
from .pipeline import PipelineResult, estimate_dataframe, estimator_options, prepare_from_spec

logger = logging.getLogger(__name__)


@dataclass
class CVFoldResult:
    fold: int
    train_ll: float
    test_ll: float
    train_n_obs: int
    test_n_obs: int
    train_n_ind: int
    test_n_ind: int
    hit_rate: float
    converged: bool
    runtime: float
    message: str = ""


@dataclass
class CrossValidationResult:
    k: int
    fold_results: list[CVFoldResult]
    mean_test_ll: float
    mean_test_ll_per_obs: float
    mean_hit_rate: float
    total_runtime: float
    model_type: str
    seed: int


def cross_validate(
    df: pd.DataFrame,
    spec: FullModelSpec,
    k: int = 5,
    seed: int = 123,
    device: torch.device | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> CrossValidationResult:
    """Run K-fold cross-validation on a discrete choice model.

    Splits data by individual (panel-level), trains on K-1 folds, evaluates
    on the held-out fold. Returns aggregated out-of-sample performance metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format choice data.
    spec : FullModelSpec
        Full model specification.
    k : int
        Number of folds.
    seed : int
        Random seed for fold assignment.
    device : torch.device or None
        Compute device.
    progress_callback : callable or None
        Called with (fold_idx, k, status_msg) after each fold.

    Returns
    -------
    CrossValidationResult
    """
    total_start = time.perf_counter()

    if k < 2:
        raise ValueError("Cross-validation requires at least two folds.")
    expanded_df, expanded_spec = prepare_from_spec(df, spec)

    # Get unique individual IDs and create fold assignments
    if expanded_df[spec.id_col].isna().any():
        raise ValueError("Respondent IDs cannot be missing.")
    unique_ids = np.asarray(expanded_df[spec.id_col].unique())
    n_individuals = len(unique_ids)
    if k > n_individuals:
        raise ValueError(
            f"k={k} exceeds the number of individuals ({n_individuals}). "
            f"Set k <= {n_individuals}."
        )
    rng = np.random.default_rng(seed)
    fold_assignments = np.array_split(rng.permutation(unique_ids), k)

    fold_results: list[CVFoldResult] = []

    for fold_idx in range(k):
        fold_start = time.perf_counter()
        test_ids = set(fold_assignments[fold_idx])
        train_ids = set(unique_ids) - test_ids

        train_df = expanded_df[expanded_df[spec.id_col].isin(train_ids)].copy()
        test_df = expanded_df[expanded_df[spec.id_col].isin(test_ids)].copy()

        train_n_ind = len(train_ids)
        test_n_ind = len(test_ids)
        # n_obs = number of choice tasks (not rows)
        train_n_obs = train_df.groupby([spec.id_col, spec.task_col]).ngroups
        test_n_obs = test_df.groupby([spec.id_col, spec.task_col]).ngroups

        try:
            # Estimate on training data using expanded spec (no re-expansion)
            train_result = _estimate_fold(
                train_df, expanded_spec, spec, device=device,
            )
            estimation = train_result.estimation
            theta_hat = estimation.raw_theta
            train_ll = estimation.log_likelihood
            converged = estimation.success

            if theta_hat is None:
                raise ValueError("Estimation did not produce raw_theta")

            test_ll = hit_rate = float("nan")
            if converged:
                test_ll, hit_rate = _score_fold(test_df, expanded_spec, spec, theta_hat, device)

            fold_runtime = time.perf_counter() - fold_start
            fold_results.append(CVFoldResult(
                fold=fold_idx,
                train_ll=train_ll,
                test_ll=test_ll,
                train_n_obs=train_n_obs,
                test_n_obs=test_n_obs,
                train_n_ind=train_n_ind,
                test_n_ind=test_n_ind,
                hit_rate=hit_rate,
                converged=converged,
                runtime=fold_runtime,
                message=estimation.message,
            ))

        except Exception as exc:
            logger.warning("Cross-validation fold %d failed.", fold_idx + 1)
            fold_runtime = time.perf_counter() - fold_start
            fold_results.append(CVFoldResult(
                fold=fold_idx,
                train_ll=float("nan"),
                test_ll=float("nan"),
                train_n_obs=train_n_obs,
                test_n_obs=test_n_obs,
                train_n_ind=train_n_ind,
                test_n_ind=test_n_ind,
                hit_rate=float("nan"),
                converged=False,
                runtime=fold_runtime,
                message=str(exc),
            ))

        if progress_callback is not None:
            status = f"Fold {fold_idx + 1}/{k} done"
            if np.isfinite(fold_results[-1].test_ll):
                status += f" (test LL={fold_results[-1].test_ll:.2f})"
            progress_callback(fold_idx, k, status)

    total_runtime = time.perf_counter() - total_start

    # Compute aggregated metrics (ignoring NaN folds)
    valid_folds = [f for f in fold_results if f.converged and np.isfinite(f.test_ll)]
    if valid_folds:
        mean_test_ll = float(np.mean([f.test_ll for f in valid_folds]))
        total_test_obs = sum(f.test_n_obs for f in valid_folds)
        total_test_ll = sum(f.test_ll for f in valid_folds)
        mean_test_ll_per_obs = total_test_ll / total_test_obs if total_test_obs > 0 else float("nan")
        mean_hit_rate = float(np.average(
            [f.hit_rate for f in valid_folds], weights=[f.test_n_obs for f in valid_folds],
        ))
    else:
        mean_test_ll = float("nan")
        mean_test_ll_per_obs = float("nan")
        mean_hit_rate = float("nan")

    return CrossValidationResult(
        k=k,
        fold_results=fold_results,
        mean_test_ll=mean_test_ll,
        mean_test_ll_per_obs=mean_test_ll_per_obs,
        mean_hit_rate=mean_hit_rate,
        total_runtime=total_runtime,
        model_type=spec.model_type,
        seed=seed,
    )


def _estimate_fold(
    train_df: pd.DataFrame,
    expanded_spec: ModelSpec,
    full_spec: FullModelSpec,
    device: torch.device | None = None,
) -> PipelineResult:
    """Estimate model on a training fold using already-expanded data."""
    columns = list(dict.fromkeys(var.column for var in expanded_spec.variables))
    varying = train_df.groupby(
        [expanded_spec.id_col, expanded_spec.task_col], sort=False,
    )[columns].nunique().gt(1).any()
    if not varying.all():
        unavailable = ", ".join(varying.index[~varying])
        raise ValueError(f"No within-task training variation for: {unavailable}.")
    return estimate_dataframe(
        df=train_df, spec=expanded_spec, model_type=full_spec.model_type,
        maxiter=full_spec.maxiter, seed=full_spec.seed, device=device,
        **estimator_options(full_spec),
    )


def _score_fold(test_df, expanded_spec, full_spec, theta_hat, device):
    """Score the held-out panel likelihood and marginal best-choice predictions."""
    from .pipeline import build_estimator
    from .prediction import choice_probabilities

    estimator, tensors = build_estimator(
        test_df, expanded_spec, model_type=full_spec.model_type,
        seed=full_spec.seed, device=device, **estimator_options(full_spec, for_fit=False),
    )
    with torch.no_grad():
        theta = torch.as_tensor(theta_hat, dtype=tensors.X.dtype, device=tensors.X.device)
        test_ll = -float(estimator._neg_log_likelihood_tensor(theta).cpu())
        probabilities = choice_probabilities(estimator, theta)
        hit_rate = float((probabilities.argmax(dim=-1) == tensors.y).float().mean().cpu())
    if not np.isfinite(test_ll) or not np.isfinite(hit_rate):
        raise ValueError("Held-out likelihood or prediction was non-finite.")
    return test_ll, hit_rate

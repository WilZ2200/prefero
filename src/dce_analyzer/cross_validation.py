"""K-fold cross-validation engine for discrete choice models."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import torch

from .config import FullModelSpec, ModelSpec
from .data import ChoiceTensors, prepare_choice_tensors
from .latent_class import LatentClassEstimator
from .model import (
    ConditionalLogitEstimator,
    GmnlEstimator,
    MixedLogitEstimator,
)
from .pipeline import PipelineResult, estimate_dataframe, estimate_from_spec

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

    # Run estimate_from_spec once on full data to get expanded spec/df
    # This handles dummy coding and interaction expansion
    full_result = estimate_from_spec(df, spec, device=device)
    expanded_spec = full_result.expanded_spec
    expanded_df = full_result.expanded_df

    if expanded_spec is None or expanded_df is None:
        # Fallback: no dummy coding was needed
        expanded_spec = spec.to_model_spec()
        expanded_df = df.copy()

    # Get unique individual IDs and create fold assignments
    unique_ids = expanded_df[spec.id_col].unique()
    n_individuals = len(unique_ids)
    if k > n_individuals:
        raise ValueError(
            f"k={k} exceeds the number of individuals ({n_individuals}). "
            f"Set k <= {n_individuals}."
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_ids)
    fold_assignments = np.array_split(unique_ids, k)

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

            # Compute out-of-sample log-likelihood
            test_ll = _compute_test_ll(
                test_df, expanded_spec, spec, theta_hat, device=device,
            )

            # Compute hit rate
            hit_rate = _compute_hit_rate(
                test_df, expanded_spec, spec, estimation,
            )

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
            ))

        except Exception as exc:
            logger.warning("Fold %d failed: %s", fold_idx, exc)
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
            ))

        if progress_callback is not None:
            status = f"Fold {fold_idx + 1}/{k} done"
            if fold_results[-1].test_ll != float("nan"):
                status += f" (test LL={fold_results[-1].test_ll:.2f})"
            progress_callback(fold_idx, k, status)

    total_runtime = time.perf_counter() - total_start

    # Compute aggregated metrics (ignoring NaN folds)
    valid_folds = [f for f in fold_results if not np.isnan(f.test_ll)]
    if valid_folds:
        mean_test_ll = float(np.mean([f.test_ll for f in valid_folds]))
        total_test_obs = sum(f.test_n_obs for f in valid_folds)
        total_test_ll = sum(f.test_ll for f in valid_folds)
        mean_test_ll_per_obs = total_test_ll / total_test_obs if total_test_obs > 0 else float("nan")
        mean_hit_rate = float(np.mean([f.hit_rate for f in valid_folds]))
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
    return estimate_dataframe(
        df=train_df,
        spec=expanded_spec,
        model_type=full_spec.model_type,
        maxiter=full_spec.maxiter,
        seed=full_spec.seed,
        device=device,
        n_classes=full_spec.n_classes,
        n_starts=full_spec.n_starts,
        correlated=full_spec.correlated,
        membership_cols=full_spec.membership_cols,
        correlation_groups=full_spec.correlation_groups,
        bws_worst_col=full_spec.bws_worst_col,
        estimate_lambda_w=full_spec.estimate_lambda_w,
        lc_method=full_spec.lc_method,
        custom_start=full_spec.custom_start,
    )


def _compute_test_ll(
    test_df: pd.DataFrame,
    expanded_spec: ModelSpec,
    full_spec: FullModelSpec,
    theta_hat: np.ndarray,
    device: torch.device | None = None,
) -> float:
    """Compute out-of-sample log-likelihood on test fold."""
    test_tensors = prepare_choice_tensors(test_df, expanded_spec, device=device)

    # Prepare BWS data if needed
    bws_data = None
    if full_spec.bws_worst_col:
        from .bws import prepare_bws_data, validate_bws
        validate_bws(test_df, expanded_spec, full_spec.bws_worst_col)
        bws_data = prepare_bws_data(
            test_df, expanded_spec, full_spec.bws_worst_col,
            test_tensors.n_obs, test_tensors.n_alts,
            test_tensors.X.device,
            estimate_lambda_w=full_spec.estimate_lambda_w,
        )

    dev = test_tensors.X.device

    if full_spec.model_type in ("mixed", "conditional", "gmnl"):
        if full_spec.model_type == "mixed":
            test_estimator = MixedLogitEstimator(
                test_tensors, expanded_spec.variables,
                n_draws=full_spec.n_draws, device=dev, seed=full_spec.seed,
                correlated=full_spec.correlated,
                correlation_groups=full_spec.correlation_groups,
                bws_data=bws_data,
            )
        elif full_spec.model_type == "conditional":
            test_estimator = ConditionalLogitEstimator(
                test_tensors, expanded_spec.variables,
                device=dev, seed=full_spec.seed,
                bws_data=bws_data,
            )
        else:  # gmnl
            test_estimator = GmnlEstimator(
                test_tensors, expanded_spec.variables,
                n_draws=full_spec.n_draws, device=dev, seed=full_spec.seed,
                correlated=full_spec.correlated,
                correlation_groups=full_spec.correlation_groups,
                bws_data=bws_data,
            )

        with torch.no_grad():
            theta_tensor = torch.tensor(theta_hat, dtype=torch.float32, device=dev)
            test_nll = float(test_estimator._neg_log_likelihood_tensor(theta_tensor).cpu().item())
        return -test_nll

    elif full_spec.model_type == "latent_class":
        test_estimator = LatentClassEstimator(
            test_tensors, expanded_spec.variables,
            n_classes=full_spec.n_classes, device=dev, seed=full_spec.seed,
            membership_cols=full_spec.membership_cols,
            df=test_df, id_col=full_spec.id_col,
            bws_data=bws_data,
        )

        with torch.no_grad():
            theta_tensor = torch.tensor(theta_hat, dtype=torch.float32, device=dev)
            test_nll = float(test_estimator._neg_log_likelihood_tensor(theta_tensor).cpu().item())
        return -test_nll

    else:
        raise ValueError(f"Unsupported model_type: {full_spec.model_type}")


def _compute_hit_rate(
    test_df: pd.DataFrame,
    expanded_spec: ModelSpec,
    full_spec: FullModelSpec,
    estimation,
) -> float:
    """Compute prediction accuracy (hit rate) on test data.

    Uses mean beta parameters to compute deterministic utility,
    predicts argmax per task, compares with actual choices.
    """
    est_df = estimation.estimates

    # Extract mean beta vector from estimates
    beta_vec = _extract_mean_betas(est_df, expanded_spec, full_spec)
    if beta_vec is None:
        return float("nan")

    # Build feature matrix and actual choices for test data
    sort_cols = [full_spec.id_col, full_spec.task_col, full_spec.alt_col]
    work = test_df.sort_values(sort_cols).reset_index(drop=True)
    n_obs = work.groupby([full_spec.id_col, full_spec.task_col]).ngroups
    n_alts = work.groupby([full_spec.id_col, full_spec.task_col]).size().iloc[0]

    feature_cols = [v.column for v in expanded_spec.variables]
    X_flat = work[feature_cols].astype(float).to_numpy(dtype=np.float32)
    X = X_flat.reshape(n_obs, n_alts, len(feature_cols))

    # Compute deterministic utility V = X @ beta
    V = X @ beta_vec  # (n_obs, n_alts)
    predicted = np.argmax(V, axis=1)  # (n_obs,)

    # Get actual choices
    choice_mat = (
        work[full_spec.choice_col]
        .to_numpy(dtype=work[full_spec.choice_col].dtype)
        .reshape(n_obs, n_alts)
    )
    alt_mat = work[full_spec.alt_col].to_numpy().reshape(n_obs, n_alts)
    from .data import _choice_indices
    actual = _choice_indices(choice_mat, alt_mat)

    return float(np.mean(predicted == actual))


def _extract_mean_betas(
    est_df: pd.DataFrame,
    expanded_spec: ModelSpec,
    full_spec: FullModelSpec,
) -> np.ndarray | None:
    """Extract a mean beta vector (one per variable) from the estimates table.

    Works for all model types:
    - CL/MXL/GMNL: use beta_ (fixed) and mu_ (random) rows
    - LC: compute class-probability-weighted average of class betas
    """
    n_vars = len(expanded_spec.variables)
    var_names = [v.name for v in expanded_spec.variables]

    if full_spec.model_type == "latent_class":
        return _extract_lc_mean_betas(est_df, var_names, n_vars)

    # CL / MXL / GMNL
    beta_vec = np.zeros(n_vars, dtype=np.float32)
    for i, name in enumerate(var_names):
        # Look for beta_{name} (fixed) or mu_{name} (random)
        beta_row = est_df[est_df["parameter"] == f"beta_{name}"]
        if len(beta_row) > 0:
            beta_vec[i] = float(beta_row["estimate"].iloc[0])
            continue
        mu_row = est_df[est_df["parameter"] == f"mu_{name}"]
        if len(mu_row) > 0:
            beta_vec[i] = float(mu_row["estimate"].iloc[0])
            continue
        # Could not find parameter
        logger.warning("Could not find beta for variable '%s' in estimates", name)
        return None

    return beta_vec


def _extract_lc_mean_betas(
    est_df: pd.DataFrame,
    var_names: list[str],
    n_vars: int,
) -> np.ndarray | None:
    """Extract class-probability-weighted average betas for latent class."""
    # Get class probabilities
    pi_rows = est_df[est_df["parameter"].str.startswith("pi_class")]
    if len(pi_rows) == 0:
        return None

    n_classes = len(pi_rows)
    class_probs = np.array([float(pi_rows[pi_rows["parameter"] == f"pi_class{q+1}"]["estimate"].iloc[0])
                            for q in range(n_classes)])

    # Get class-specific betas
    beta_vec = np.zeros(n_vars, dtype=np.float32)
    for q in range(n_classes):
        for i, name in enumerate(var_names):
            param_name = f"beta_{name}_class{q+1}"
            row = est_df[est_df["parameter"] == param_name]
            if len(row) > 0:
                beta_vec[i] += class_probs[q] * float(row["estimate"].iloc[0])
            else:
                logger.warning("Could not find '%s' in LC estimates", param_name)
                return None

    return beta_vec

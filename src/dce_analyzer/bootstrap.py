# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under dual license: AGPL-3.0 (open-source) or commercial. See LICENSE.

"""Non-parametric bootstrap for discrete-choice model inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .config import FullModelSpec, ModelSpec
from .latent_class import LatentClassResult
from .model import EstimationResult
from .pipeline import estimate_dataframe, estimator_options, prepare_from_spec


@dataclass
class BootstrapResult:
    """Stores the results of a bootstrap procedure."""

    n_replications: int
    n_successful: int
    param_names: list[str]
    estimates_matrix: np.ndarray  # (n_successful, n_params) — each row is one replication
    bootstrap_se: dict[str, float]
    percentile_ci: dict[str, tuple[float, float]]  # 95% CI per parameter
    original_estimates: dict[str, float]

    def summary_dataframe(self) -> pd.DataFrame:
        rows = []
        for index, name in enumerate(self.param_names):
            rows.append(
                {
                    "parameter": name,
                    "original": self.original_estimates.get(name, float("nan")),
                    "bootstrap_mean": float(np.mean(self.estimates_matrix[:, index])),
                    "bootstrap_se": self.bootstrap_se.get(name, float("nan")),
                    "ci_lower": self.percentile_ci[name][0] if name in self.percentile_ci else float("nan"),
                    "ci_upper": self.percentile_ci[name][1] if name in self.percentile_ci else float("nan"),
                }
            )
        return pd.DataFrame(rows)


def _resample_individuals(df: pd.DataFrame, id_col: str, rng: np.random.Generator) -> pd.DataFrame:
    """Resample individuals with replacement, keeping all tasks per individual."""
    groups = {key: rows for key, rows in df.groupby(id_col, sort=False)}
    unique_ids = df[id_col].unique()
    sampled_ids = rng.choice(unique_ids, size=len(unique_ids), replace=True)

    parts = []
    for new_idx, orig_id in enumerate(sampled_ids):
        chunk = groups[orig_id].copy()
        chunk[id_col] = new_idx
        parts.append(chunk)

    return pd.concat(parts, ignore_index=True)


def _class_order(reference: LatentClassResult, fit: LatentClassResult) -> np.ndarray:
    reference_betas = reference.class_estimates.pivot(index="class_id", columns="parameter", values="estimate")
    fit_betas = fit.class_estimates.pivot(index="class_id", columns="parameter", values="estimate")
    expected = reference_betas.to_numpy()
    observed = fit_betas.reindex_like(reference_betas).to_numpy()
    scale = np.maximum(np.abs(expected).max(axis=0), np.abs(observed).max(axis=0))
    differences = (expected[:, None, :] - observed[None, :, :]) / np.where(scale > 0, scale, 1.0)
    _, order = linear_sum_assignment(np.sum(differences ** 2, axis=2))
    return order + 1


def _estimates_in_class_order(
    fit: LatentClassResult, class_order: np.ndarray,
) -> pd.Series:
    estimates = fit.estimates.set_index("parameter")["estimate"].to_dict()
    aligned = {
        name: value for name, value in estimates.items()
        if not name.rsplit("_class", 1)[-1].isdigit()
    }
    for new_class, old_class in enumerate(class_order, start=1):
        suffix = f"_class{old_class}"
        for name, value in estimates.items():
            if name.endswith(suffix):
                aligned[f"{name.removesuffix(suffix)}_class{new_class}"] = value

    if fit.membership_estimates is not None and not fit.membership_estimates.empty:
        membership = fit.membership_estimates.pivot(index="class_id", columns="variable", values="estimate")
        order = membership.reindex(class_order, fill_value=0.0)
        rebased = order - order.iloc[0]
        for new_class, (_, coefficients) in enumerate(rebased.iloc[1:].iterrows(), start=2):
            for column, value in coefficients.items():
                name = "gamma" if column == "_intercept" else f"delta_{column}"
                aligned[f"{name}_class{new_class}"] = value
    return pd.Series(aligned, dtype=float)


def run_bootstrap(
    df: pd.DataFrame,
    spec: FullModelSpec | ModelSpec,
    model_type: str = "mixed",
    n_replications: int = 100,
    maxiter: int | None = None,
    seed: int = 42,
    progress_callback: Callable[[int, int], None] | None = None,
    *,
    correlated: bool = False,
    correlation_groups: list[list[int]] | None = None,
    n_classes: int | None = None,
    n_starts: int = 10,
    membership_cols: list[str] | None = None,
    bws_worst_col: str | None = None,
    estimate_lambda_w: bool = True,
    lc_method: str = "em",
    fixed_gamma: float | None = None,
    fitted_result: EstimationResult | LatentClassResult | None = None,
) -> BootstrapResult:
    """
    Run non-parametric bootstrap by resampling individuals with replacement.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format choice data.
    spec : FullModelSpec or ModelSpec
        Complete specification, or a legacy specification with keyword options.
    model_type : str
        "mixed", "conditional", "gmnl", or "latent_class".
    n_replications : int
        Number of bootstrap replications.
    maxiter : int
        Override the optimizer iteration limit for the replications.
    seed : int
        Base seed for reproducibility.
    progress_callback : callable, optional
        Called with (current_replication, n_replications) after each replication.
    correlated : bool
        Enable full correlation (Cholesky) for random parameters.
    correlation_groups : list[list[int]], optional
        Selective correlation groups (block-diagonal Cholesky).
    n_classes : int, optional
        Number of latent classes (for latent_class model type).
    n_starts : int
        Number of random starts (for latent_class).
    membership_cols : list[str], optional
        Membership covariates (for latent_class).
    bws_worst_col : str, optional
        Column name for BWS worst choices.
    estimate_lambda_w : bool
        Whether to estimate lambda_w for BWS.
    lc_method : str
        Latent class optimizer for the legacy specification.
    fixed_gamma : float, optional
        Fixed GMNL mixing parameter for the legacy specification.
    fitted_result : EstimationResult or LatentClassResult, optional
        Original fitted result for the same data and specification.

    Returns
    -------
    BootstrapResult
    """
    if n_replications < 2:
        raise ValueError("Bootstrap needs at least two replications.")
    rng = np.random.default_rng(seed)

    estimation_seed = None
    if isinstance(spec, FullModelSpec):
        full_spec = spec
        df, spec = prepare_from_spec(df, full_spec)
        model_type = full_spec.model_type
        maxiter = full_spec.maxiter if maxiter is None else maxiter
        estimation_seed = full_spec.seed
        extra_kwargs = estimator_options(full_spec)
    else:
        maxiter = 200 if maxiter is None else maxiter
        extra_kwargs = dict(
            correlated=correlated, correlation_groups=correlation_groups,
            n_classes=n_classes, n_starts=n_starts, membership_cols=membership_cols,
            bws_worst_col=bws_worst_col, estimate_lambda_w=estimate_lambda_w,
            lc_method=lc_method, fixed_gamma=fixed_gamma,
        )

    original_est = fitted_result
    if original_est is None:
        original_est = estimate_dataframe(
            df, spec, model_type=model_type, maxiter=maxiter,
            seed=seed if estimation_seed is None else estimation_seed, **extra_kwargs,
        ).estimation
    if not original_est.success or not np.isfinite(original_est.log_likelihood):
        raise ValueError("Bootstrap requires a converged original fit with a finite likelihood.")
    if isinstance(original_est, LatentClassResult):
        original = _estimates_in_class_order(original_est, np.arange(1, original_est.n_classes + 1))
    else:
        original = original_est.estimates.set_index("parameter")["estimate"]
    if original.empty or not np.isfinite(original.to_numpy()).all():
        raise ValueError("Bootstrap requires finite original parameter estimates.")
    param_names = original.index.tolist()
    original_values = original.to_dict()

    all_estimates: list[np.ndarray] = []
    first_error = None

    for b in range(n_replications):
        rep_seed = int(rng.integers(0, 2**31))
        resampled = _resample_individuals(df, spec.id_col, np.random.default_rng(rep_seed))

        try:
            result = estimate_dataframe(
                resampled, spec, model_type=model_type, maxiter=maxiter,
                seed=rep_seed if estimation_seed is None else estimation_seed, **extra_kwargs,
            )
            fit = result.estimation
            if fit.success and np.isfinite(fit.log_likelihood):
                if isinstance(fit, LatentClassResult) and isinstance(original_est, LatentClassResult):
                    estimates = _estimates_in_class_order(fit, _class_order(original_est, fit))
                else:
                    estimates = fit.estimates.set_index("parameter")["estimate"]
                values = estimates.reindex(param_names).to_numpy()
                if np.isfinite(values).all():
                    all_estimates.append(values)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            if first_error is None:
                first_error = exc

        if progress_callback is not None:
            progress_callback(b + 1, n_replications)

    n_successful = len(all_estimates)
    if n_successful < 2:
        raise RuntimeError(
            f"Only {n_successful} of {n_replications} bootstrap replications succeeded. "
            "Cannot compute bootstrap statistics."
        ) from first_error

    estimates_matrix = np.array(all_estimates)  # (n_successful, n_params)

    bootstrap_se = {}
    percentile_ci = {}
    for i, name in enumerate(param_names):
        col = estimates_matrix[:, i]
        bootstrap_se[name] = float(np.std(col, ddof=1))
        percentile_ci[name] = (float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5)))

    return BootstrapResult(
        n_replications=n_replications,
        n_successful=n_successful,
        param_names=param_names,
        estimates_matrix=estimates_matrix,
        bootstrap_se=bootstrap_se,
        percentile_ci=percentile_ci,
        original_estimates=original_values,
    )

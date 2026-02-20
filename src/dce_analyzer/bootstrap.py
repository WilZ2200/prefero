# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under dual license: AGPL-3.0 (open-source) or commercial. See LICENSE.

"""Non-parametric bootstrap for discrete-choice model inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import ModelSpec
from .pipeline import estimate_dataframe


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
        for name in self.param_names:
            rows.append(
                {
                    "parameter": name,
                    "original": self.original_estimates.get(name, float("nan")),
                    "bootstrap_se": self.bootstrap_se.get(name, float("nan")),
                    "ci_lower": self.percentile_ci[name][0] if name in self.percentile_ci else float("nan"),
                    "ci_upper": self.percentile_ci[name][1] if name in self.percentile_ci else float("nan"),
                }
            )
        return pd.DataFrame(rows)


def _resample_individuals(df: pd.DataFrame, id_col: str, rng: np.random.Generator) -> pd.DataFrame:
    """Resample individuals with replacement, keeping all tasks per individual."""
    unique_ids = df[id_col].unique()
    sampled_ids = rng.choice(unique_ids, size=len(unique_ids), replace=True)

    parts = []
    for new_idx, orig_id in enumerate(sampled_ids):
        chunk = df[df[id_col] == orig_id].copy()
        chunk[id_col] = new_idx
        parts.append(chunk)

    return pd.concat(parts, ignore_index=True)


def run_bootstrap(
    df: pd.DataFrame,
    spec: ModelSpec,
    model_type: str = "mixed",
    n_replications: int = 100,
    maxiter: int = 200,
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
) -> BootstrapResult:
    """
    Run non-parametric bootstrap by resampling individuals with replacement.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format choice data.
    spec : ModelSpec
        Model specification.
    model_type : str
        "mixed", "conditional", "gmnl", or "latent_class".
    n_replications : int
        Number of bootstrap replications.
    maxiter : int
        Max optimizer iterations per replication.
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

    Returns
    -------
    BootstrapResult
    """
    rng = np.random.default_rng(seed)

    # Build extra kwargs for estimate_dataframe
    extra_kwargs: dict[str, Any] = {}
    if correlated:
        extra_kwargs["correlated"] = True
    if correlation_groups is not None:
        extra_kwargs["correlation_groups"] = correlation_groups
    if n_classes is not None:
        extra_kwargs["n_classes"] = n_classes
    if n_starts != 10:
        extra_kwargs["n_starts"] = n_starts
    if membership_cols:
        extra_kwargs["membership_cols"] = membership_cols
    if bws_worst_col:
        extra_kwargs["bws_worst_col"] = bws_worst_col
        extra_kwargs["estimate_lambda_w"] = estimate_lambda_w

    # run original estimation for reference
    original = estimate_dataframe(
        df, spec, model_type=model_type, maxiter=maxiter, seed=seed, **extra_kwargs,
    )
    original_est = original.estimation
    param_names = original_est.estimates["parameter"].tolist()
    original_values = dict(
        zip(original_est.estimates["parameter"], original_est.estimates["estimate"])
    )

    all_estimates: list[np.ndarray] = []
    n_successful = 0

    for b in range(n_replications):
        rep_seed = int(rng.integers(0, 2**31))
        resampled = _resample_individuals(df, spec.id_col, np.random.default_rng(rep_seed))

        try:
            result = estimate_dataframe(
                resampled, spec, model_type=model_type, maxiter=maxiter,
                seed=rep_seed, **extra_kwargs,
            )
            est_values = result.estimation.estimates["estimate"].to_numpy()
            all_estimates.append(est_values)
            n_successful += 1
        except Exception:
            pass  # skip failed replications

        if progress_callback is not None:
            progress_callback(b + 1, n_replications)

    if n_successful < 2:
        raise RuntimeError(
            f"Only {n_successful} of {n_replications} bootstrap replications succeeded. "
            "Cannot compute bootstrap statistics."
        )

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

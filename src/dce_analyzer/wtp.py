# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under dual license: AGPL-3.0 (open-source) or commercial. See LICENSE.

"""Willingness-to-pay (WTP) from population-mean utility coefficients."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import EstimationResult

WTP_ESTIMAND = (
    "WTP is the negative ratio of each attribute's population-mean utility coefficient "
    "to the population-mean cost coefficient. When cost is random, this measures the "
    "trade-off between mean coefficients and can differ from the mean of individual WTP."
)


def _add_derivative(gradient: np.ndarray, row: pd.Series, value: float) -> bool:
    index = row.get("theta_index")
    if pd.isna(index) or not 0 <= index < len(gradient):
        return False
    gradient[int(index)] += value
    return True


def _utility_mean(
    row: pd.Series, table: pd.DataFrame, result: EstimationResult, n_theta: int,
) -> tuple[float, np.ndarray | None]:
    """Return the mean utility coefficient and its derivative in raw theta space."""
    coefficient = float(row["estimate"])
    gradient = np.zeros(n_theta)
    gradient_available = _add_derivative(gradient, row, 1.0)
    distribution = row.get("distribution")
    if row.name.startswith("mu_") and distribution not in {"normal", "lognormal"}:
        raise ValueError(f"The distribution for '{row.name}' is unavailable. Refit the model.")
    if distribution != "lognormal":
        return coefficient, gradient if gradient_available else None

    name = row.name.split("_", 1)[1]
    sd_name = f"sd_{name}"
    if sd_name not in table.index:
        raise ValueError(f"Missing lognormal standard deviation for '{name}'.")
    sd_row = table.loc[sd_name]
    sd = float(sd_row["estimate"])
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(np.exp(coefficient + 0.5 * sd**2))
    if not np.isfinite(mean):
        raise ValueError(f"The mean utility coefficient for '{name}' is non-finite.")
    if not gradient_available:
        return mean, None

    if _add_derivative(gradient, sd_row, sd * -np.expm1(-sd)):
        return mean, gradient * mean

    # For correlated tastes, the log variance is the sum of squares of this
    # Cholesky row. A positive diagonal is reported on the softplus scale.
    names = result.random_param_names or []
    chol_gradient = np.zeros(n_theta)
    sum_squares = 0.0
    for other in names:
        label = f"chol_{name}_{other}"
        if label not in table.index:
            continue
        entry = table.loc[label]
        loading = float(entry["estimate"])
        derivative = loading * -np.expm1(-loading) if name == other else loading
        if not _add_derivative(chol_gradient, entry, derivative):
            return mean, None
        sum_squares += loading**2
    if not np.isclose(sum_squares, sd**2):
        return mean, None
    return mean, (gradient + chol_gradient) * mean


def compute_wtp(result: EstimationResult, cost_variable: str) -> pd.DataFrame:
    """Return trade-offs between mean utility coefficients and delta-method intervals."""
    if not result.success:
        raise ValueError("WTP requires a converged model.")

    table = result.estimates.set_index("parameter")
    if not table.index.is_unique:
        raise ValueError("WTP requires distinct parameter names.")
    rows = table[table.index.str.startswith(("beta_", "mu_"))]
    cost_name = next(
        (p for p in (f"beta_{cost_variable}", f"mu_{cost_variable}") if p in rows.index),
        None,
    )
    if cost_name is None:
        raise ValueError(f"Could not find a utility coefficient for cost variable '{cost_variable}'.")

    vcov = result.vcov_matrix
    if vcov is not None and "theta_index" not in table:
        if len(table) == len(vcov) and table.index.str.startswith("beta_").all():
            table["theta_index"] = np.arange(len(table))
            rows = table
        else:
            vcov = None
    if vcov is not None and (
        vcov.ndim != 2 or vcov.shape[0] != vcov.shape[1] or not np.isfinite(vcov).all()
    ):
        vcov = None
    n_theta = len(vcov) if vcov is not None else 0
    cost, cost_gradient = _utility_mean(rows.loc[cost_name], table, result, n_theta)
    if not np.isfinite(cost) or abs(cost) < 1e-10:
        raise ValueError("The mean cost coefficient is zero, near zero or non-finite.")

    output = []
    for _, row in rows.drop(index=cost_name).iterrows():
        attribute, attribute_gradient = _utility_mean(row, table, result, n_theta)
        wtp = -attribute / cost
        se = np.nan
        if vcov is not None and cost_gradient is not None and attribute_gradient is not None:
            gradient = (-attribute_gradient - wtp * cost_gradient) / cost
            variance = float(gradient @ vcov @ gradient)
            if variance >= 0 and np.isfinite(variance):
                se = np.sqrt(variance)
        output.append({
            "attribute": row.name.split("_", 1)[1],
            "wtp_estimate": wtp,
            "wtp_std_error": se,
            "wtp_ci_lower": wtp - 1.96 * se,
            "wtp_ci_upper": wtp + 1.96 * se,
        })
    return pd.DataFrame(output, columns=[
        "attribute", "wtp_estimate", "wtp_std_error", "wtp_ci_lower", "wtp_ci_upper",
    ])


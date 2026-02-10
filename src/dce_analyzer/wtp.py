"""Willingness-to-pay (WTP) computation from estimation results."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from .model import EstimationResult


def compute_wtp(
    result: EstimationResult,
    cost_variable: str,
) -> pd.DataFrame:
    """Compute WTP for each non-cost attribute relative to the cost variable.

    For fixed coefficients:
        WTP_attr = -beta_attr / beta_cost

    Standard errors are computed via the delta method when the
    variance-covariance matrix is available.

    Parameters
    ----------
    result : EstimationResult
        Fitted model output (must contain estimates DataFrame).
    cost_variable : str
        Name of the cost/price variable (e.g. "price").  The function looks
        for ``beta_<cost_variable>`` (fixed) or ``mu_<cost_variable>`` (random)
        in the estimates table.

    Returns
    -------
    pd.DataFrame
        Columns: attribute, wtp_estimate, wtp_std_error, wtp_ci_lower, wtp_ci_upper
    """
    est = result.estimates
    vcov = result.vcov_matrix

    # --- locate cost parameter ---
    cost_param_name = f"beta_{cost_variable}"
    cost_row = est.loc[est["parameter"] == cost_param_name]
    if cost_row.empty:
        cost_param_name = f"mu_{cost_variable}"
        cost_row = est.loc[est["parameter"] == cost_param_name]
    if cost_row.empty:
        raise ValueError(
            f"Could not find cost parameter for variable '{cost_variable}'. "
            f"Looked for 'beta_{cost_variable}' and 'mu_{cost_variable}' in estimates."
        )

    beta_cost = float(cost_row["estimate"].iloc[0])
    if abs(beta_cost) < 1e-10:
        warnings.warn(
            f"Cost coefficient is near zero ({beta_cost:.2e}); WTP values will be unreliable."
        )

    cost_idx = int(cost_row.index[0])

    # Build a mapping from parameter name to its row index in estimates
    param_to_idx = {row["parameter"]: i for i, row in est.iterrows()}

    # --- identify attribute parameters ---
    # We only compute WTP for mean / fixed coefficients (beta_ or mu_), excluding
    # the cost variable itself and any sd_ rows.
    attr_rows = est[
        est["parameter"].str.startswith(("beta_", "mu_"))
        & (est["parameter"] != cost_param_name)
    ]

    rows = []
    for _, arow in attr_rows.iterrows():
        param_name: str = arow["parameter"]
        beta_attr = float(arow["estimate"])
        attr_name = param_name.split("_", 1)[1]  # strip beta_/mu_ prefix

        wtp_est = -beta_attr / beta_cost if abs(beta_cost) > 1e-10 else float("nan")

        se = float("nan")
        ci_lo = float("nan")
        ci_hi = float("nan")

        if vcov is not None and abs(beta_cost) > 1e-10:
            attr_idx = param_to_idx[param_name]
            # Delta method: WTP = -b_attr / b_cost = g(b_attr, b_cost)
            # dg/d(b_attr) = -1 / b_cost
            # dg/d(b_cost) = b_attr / b_cost^2
            grad = np.zeros(len(est))
            grad[attr_idx] = -1.0 / beta_cost
            grad[cost_idx] = beta_attr / (beta_cost ** 2)

            # The vcov matrix is in *theta* space (raw parameters).
            # For fixed-coeff rows, theta index == estimates row index
            # (since _parameter_table produces rows in the same order as theta elements).
            # We need to map estimates-row indices to theta indices.
            theta_grad = _map_grad_to_theta(grad, est, result)
            if theta_grad is not None:
                var_wtp = float(theta_grad @ vcov @ theta_grad)
                se = float(np.sqrt(max(var_wtp, 0.0)))
                ci_lo = wtp_est - 1.96 * se
                ci_hi = wtp_est + 1.96 * se

        rows.append(
            {
                "attribute": attr_name,
                "wtp_estimate": wtp_est,
                "wtp_std_error": se,
                "wtp_ci_lower": ci_lo,
                "wtp_ci_upper": ci_hi,
            }
        )

    if not rows:
        warnings.warn("No non-cost attribute parameters found for WTP computation.")

    return pd.DataFrame(rows)


def _map_grad_to_theta(
    grad_est: np.ndarray,
    estimates: pd.DataFrame,
    result: EstimationResult,
) -> np.ndarray | None:
    """Map a gradient w.r.t. estimates-table rows to a gradient w.r.t. raw theta.

    Uses the ``theta_index`` column in the estimates table (populated by
    ``_parameter_table``) to correctly map each row to its position in the
    raw theta vector.  This handles interleaved mu/sd rows in independent
    mixed logit as well as correlated models where sd rows are derived.
    """
    if result.vcov_matrix is None:
        return None

    n_theta = result.vcov_matrix.shape[0]
    theta_grad = np.zeros(n_theta)

    has_theta_index = "theta_index" in estimates.columns

    for row_idx in range(len(estimates)):
        if abs(grad_est[row_idx]) < 1e-30:
            continue

        if has_theta_index:
            tidx = int(estimates.iloc[row_idx]["theta_index"])
        else:
            tidx = row_idx  # legacy fallback

        if tidx < 0 or tidx >= n_theta:
            continue  # derived param (e.g. sd from Cholesky) or out of range

        param_name = estimates.iloc[row_idx]["parameter"]
        if param_name.startswith("sd_"):
            raw_val = _inverse_softplus(estimates.iloc[row_idx]["estimate"])
            sigmoid = 1.0 / (1.0 + np.exp(-raw_val))
            theta_grad[tidx] = grad_est[row_idx] * sigmoid
        else:
            theta_grad[tidx] = grad_est[row_idx]

    return theta_grad


def _inverse_softplus(y: float) -> float:
    """Inverse of softplus(x) = log(1 + exp(x)), ignoring the tiny 1e-6 offset."""
    y = max(y, 1e-10)
    if y > 20:
        return y
    return float(np.log(np.expm1(y)))

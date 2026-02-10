"""BWS (Best-Worst Scaling) data preparation, validation, and likelihood."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import ModelSpec
from .data import _choice_indices


@dataclass
class BwsData:
    """BWS-specific tensors computed alongside standard ChoiceTensors."""

    y_worst: torch.Tensor  # (n_obs,) worst choice index (0-based within alternatives)
    estimate_lambda_w: bool  # whether lambda_w is a free parameter


# ── Composable log-probability functions ────────────────────────────

def standard_log_prob(
    utility: torch.Tensor,
    y: torch.Tensor,
    alt_dim: int = -1,
) -> torch.Tensor:
    """Standard MNL log-probability (softmax over alternatives).

    Parameters
    ----------
    utility : Tensor with alternatives along *alt_dim*.
              e.g. (n_obs, n_draws, n_alts) or (n_obs, n_alts, Q)
    y       : (n_obs,) chosen alternative index.
    alt_dim : which dimension holds alternatives (default -1, i.e. last).

    Returns
    -------
    log_prob : Tensor with the alt_dim dimension removed.
    """
    log_denom = torch.logsumexp(utility, dim=alt_dim)  # sum over alts
    # Build index to gather the chosen utility along alt_dim
    idx = _expand_index(y, utility, alt_dim)
    chosen_u = torch.gather(utility, alt_dim, idx).squeeze(alt_dim)
    return chosen_u - log_denom


def bws_log_prob(
    utility: torch.Tensor,
    y_best: torch.Tensor,
    y_worst: torch.Tensor,
    lambda_w: float | torch.Tensor,
    alt_dim: int = -1,
) -> torch.Tensor:
    """Composable BWS log-probability: P(best) x P(worst|best removed).

    Parameters
    ----------
    utility  : Tensor with alternatives along *alt_dim*.
    y_best   : (n_obs,) best choice indices.
    y_worst  : (n_obs,) worst choice indices.
    lambda_w : worst scale parameter (scalar or tensor).
    alt_dim  : which dimension holds alternatives.

    Returns
    -------
    log_prob : log_prob_best + log_prob_worst, with alt_dim removed.
    """
    # Best choice: standard softmax over all alts
    log_prob_best = standard_log_prob(utility, y_best, alt_dim=alt_dim)

    # Worst choice: negated & scaled utilities, mask out best alt
    worst_utility = -lambda_w * utility
    best_idx = _expand_index(y_best, utility, alt_dim)
    mask = torch.zeros_like(utility)
    mask.scatter_(alt_dim, best_idx, float('-inf'))
    worst_utility = worst_utility + mask

    log_prob_worst = standard_log_prob(worst_utility, y_worst, alt_dim=alt_dim)

    return log_prob_best + log_prob_worst


def _expand_index(
    y: torch.Tensor, utility: torch.Tensor, alt_dim: int,
) -> torch.Tensor:
    """Expand a 1-D index (n_obs,) so it can be used with torch.gather on *alt_dim*.

    The returned index has the same number of dims as *utility*, with size 1
    along *alt_dim* and broadcast-expanded along all other dims.
    """
    ndim = utility.dim()
    # Normalise alt_dim to positive
    if alt_dim < 0:
        alt_dim = ndim + alt_dim

    # Start with y as shape (n_obs,)
    idx = y
    # We need to reshape y to have ndim dimensions, with size 1 on alt_dim
    # and matching utility on all other dims.
    shape = [1] * ndim
    shape[0] = y.shape[0]  # n_obs is always dim 0
    idx = idx.view(shape)

    # Expand to match utility on every dimension except alt_dim (keep size 1 there)
    expand_shape = list(utility.shape)
    expand_shape[alt_dim] = 1
    return idx.expand(expand_shape)


def validate_bws(
    df,
    spec: ModelSpec,
    worst_col: str,
) -> None:
    """Validate BWS requirements: J>=3, worst_col exists, worst!=best per task."""
    if worst_col not in df.columns:
        raise ValueError(f"Worst choice column '{worst_col}' not in data.")
    # Check J >= 3
    group_sizes = df.groupby([spec.id_col, spec.task_col]).size()
    if group_sizes.min() < 3:
        raise ValueError("BWS requires at least 3 alternatives per task (J >= 3).")


def prepare_bws_data(
    df,
    spec: ModelSpec,
    worst_col: str,
    n_obs: int,
    n_alts: int,
    device: torch.device | None = None,
    estimate_lambda_w: bool = True,
) -> BwsData:
    """Compute worst-choice indices from the data."""
    sort_cols = [spec.id_col, spec.task_col, spec.alt_col]
    work = df.sort_values(sort_cols).reset_index(drop=True)

    worst_mat = (
        work[worst_col]
        .to_numpy(dtype=work[worst_col].dtype)
        .reshape(n_obs, n_alts)
    )
    alt_mat = work[spec.alt_col].to_numpy().reshape(n_obs, n_alts)
    y_worst_np = _choice_indices(worst_mat, alt_mat)

    # Also compute best-choice indices to validate worst != best
    choice_mat = (
        work[spec.choice_col]
        .to_numpy(dtype=work[spec.choice_col].dtype)
        .reshape(n_obs, n_alts)
    )
    y_best_np = _choice_indices(choice_mat, alt_mat)

    conflicts = np.where(y_worst_np == y_best_np)[0]
    if len(conflicts) > 0:
        raise ValueError(
            f"Worst choice equals best choice in {len(conflicts)} observation(s). "
            f"First conflict at observation index {conflicts[0]}."
        )

    y_worst = torch.tensor(y_worst_np, dtype=torch.long, device=device)
    return BwsData(y_worst=y_worst, estimate_lambda_w=estimate_lambda_w)

# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under dual license: AGPL-3.0 (open-source) or commercial. See LICENSE.

"""Marginal probabilities for fitted discrete choice models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .latent_class import LatentClassEstimator
from .pipeline import build_estimator, estimator_options, prepare_from_spec


@torch.no_grad()
def choice_probabilities(estimator, theta: torch.Tensor) -> torch.Tensor:
    """Compute each task's marginal best-choice probabilities for a new respondent."""
    if theta.ndim != 1 or theta.numel() != estimator.n_params or not torch.isfinite(theta).all():
        raise ValueError("Fit parameters do not match this specification. Refit the model.")

    if isinstance(estimator, LatentClassEstimator):
        betas, weights = estimator._unpack_theta(theta)
        utilities = torch.einsum("nav,qv->nqa", estimator.X, betas)
        if weights.ndim == 1:
            weights = weights.expand(estimator.n_obs, -1)
        else:
            weights = weights[estimator.panel_idx]
        probabilities = (utilities.softmax(dim=-1) * weights.unsqueeze(-1)).sum(dim=1)
    else:
        betas = estimator._betas_from_theta(theta)
        utilities = torch.einsum("nav,ndv->nda", estimator.X, betas[estimator.panel_idx])
        probabilities = utilities.softmax(dim=-1).mean(dim=1)

    if not torch.isfinite(probabilities).all():
        raise ValueError("The fitted model returned non-finite choice probabilities.")
    return probabilities


def predict_entry(df: pd.DataFrame, entry: dict) -> pd.DataFrame:
    """Summarise observed and model-predicted best shares by alternative label."""
    estimation = entry["estimation"]
    if not estimation.success or estimation.raw_theta is None:
        raise ValueError("Predictions require a converged model with fitted parameters.")
    spec = entry.get("expanded_spec") or entry.get("spec")
    full_spec = entry.get("full_spec")
    design = entry.get("expanded_df")
    if full_spec is not None:
        if design is None:
            design, spec = prepare_from_spec(df, full_spec)
        model_type = full_spec.model_type
        options = estimator_options(full_spec, for_fit=False)
        seed = full_spec.seed
    else:
        model_type = entry.get("model_type", "conditional")
        if model_type != "conditional":
            raise ValueError("The saved run has no complete model specification. Refit it first.")
        if design is None:
            design = df
        seed, options = 123, {}
    if spec is None:
        raise ValueError("The model input specification is unavailable.")
    estimator, tensors = build_estimator(
        design, spec, model_type=model_type, seed=seed, device=torch.device("cpu"), **options,
    )
    theta = torch.as_tensor(estimation.raw_theta, dtype=tensors.X.dtype)
    probabilities = choice_probabilities(estimator, theta).numpy()
    labels = design.sort_values([spec.id_col, spec.task_col, spec.alt_col])[spec.alt_col].to_numpy()
    actual = (tensors.y.numpy()[:, None] == np.arange(tensors.n_alts)).ravel()
    shares = pd.DataFrame({
        "Alternative": labels, "Actual Share": actual, "Predicted Share": probabilities.ravel(),
    })
    return shares.groupby("Alternative", sort=False, as_index=True).sum().div(tensors.n_obs).reset_index()

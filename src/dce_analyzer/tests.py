# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under dual license: AGPL-3.0 (open-source) or commercial. See LICENSE.

"""Statistical tests for discrete choice model comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linprog, minimize
from scipy.special import logsumexp
from scipy.stats import chi2

from .config import ModelSpec
from .data import prepare_choice_tensors


@dataclass
class SwaitLouviereResult:
    """Result of the Swait-Louviere pooling test."""

    chi2_stat: float
    df: int
    p_value: float
    ll_pooled: float
    ll_group1: float
    ll_group2: float
    n_parameters: int
    group1_label: str
    group2_label: str
    reject_null: bool  # at 5% level
    relative_scale: float = 1.0
    ll_equal_scale: float = float("nan")
    scale_p_value: float = float("nan")

    @property
    def conclusion(self) -> str:
        if self.reject_null:
            return (
                f"Reject H0 at 5% level (p={self.p_value:.4f}). "
                "The two groups have significantly different preference structures."
            )
        return (
            f"Fail to reject H0 at 5% level (p={self.p_value:.4f}). "
            "No significant difference in preferences between the two groups."
        )


def swait_louviere_test(
    df: pd.DataFrame,
    spec: ModelSpec,
    grouping_col: str,
    model_type: str = "conditional",
    maxiter: int = 200,
    seed: int = 123,
    n_classes: int = 2,
    n_starts: int = 10,
) -> SwaitLouviereResult:
    """Compare two conditional logit groups allowing different utility scales.

    The preference test compares separate fits to common preferences with
    a free relative scale (k-1 df). The equal-scale test adds one restriction
    to the common-preferences model. Missing groups are excluded by respondent.
    """
    if model_type != "conditional":
        raise ValueError("Swait–Louviere testing is available for conditional logit.")
    if grouping_col not in df:
        raise ValueError(f"Grouping column {grouping_col!r} was not found.")
    memberships = df.groupby(spec.id_col)[grouping_col].nunique()
    if (memberships > 1).any():
        raise ValueError("The grouping variable must be constant within each respondent.")
    missing_ids = df.loc[df[grouping_col].isna(), spec.id_col]
    sample = df.loc[~df[spec.id_col].isin(missing_ids)]
    groups = sample[grouping_col].unique()
    if len(groups) != 2:
        raise ValueError("The complete respondents must form exactly two groups.")
    if len(spec.variables) < 2:
        raise ValueError("The preference test requires at least two utility coefficients.")

    samples = []
    for group in groups:
        tensors = prepare_choice_tensors(
            sample.loc[sample[grouping_col] == group], spec, device=torch.device("cpu"),
        )
        X = tensors.X.numpy().astype(np.float64)
        samples.append((X - X.mean(axis=1, keepdims=True), tensors.y.numpy()))
    return _fit_scale_test(samples, groups, maxiter)
def _choice_loss(beta, X, y):
    utility = X @ beta
    normalizer = logsumexp(utility, axis=1)
    rows = np.arange(len(y))
    probability = np.exp(utility - normalizer[:, None])
    probability[rows, y] -= 1
    return (normalizer - utility[rows, y]).sum(), np.einsum("ij,ijk->k", probability, X)


def _fit_scale_test(samples, groups, maxiter):
    X_all = np.concatenate([X for X, _ in samples])
    scale = np.sqrt(np.mean(X_all ** 2, axis=(0, 1)))
    if (scale == 0).any():
        raise ValueError("Each utility coefficient must vary within a choice task.")
    samples = [(X / scale, y) for X, y in samples]
    width = len(scale)
    for X, y in samples:
        if np.linalg.matrix_rank(X.reshape(-1, width)) < width:
            raise ValueError("The utility coefficients must be identifiable in each group.")
        if _separated_choices(X, y):
            raise ValueError("A group has separation, so it has no finite likelihood maximum.")

    def pooled(theta, estimate_scale):
        beta = theta[:width]
        relative = np.exp(theta[width]) if estimate_scale else 1.0
        loss1, grad1 = _choice_loss(beta, *samples[0])
        loss2, grad2 = _choice_loss(relative * beta, *samples[1])
        gradient = grad1 + relative * grad2
        if estimate_scale:
            gradient = np.append(gradient, grad2 @ (relative * beta))
        return loss1 + loss2, gradient

    def fit(objective, start, *, bounds=None):
        return minimize(objective, start, jac=True, method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": maxiter, "gtol": 1e-7, "ftol": 1e-12})

    separate = [fit(lambda beta: _choice_loss(beta, X, y), np.zeros(width)) for X, y in samples]
    equal = fit(lambda theta: pooled(theta, False), np.zeros(width))
    bounds = [(None, None)] * width + [(-8, 8)]
    candidates = [fit(lambda theta: pooled(theta, True), np.append(equal.x, log_scale),
                      bounds=bounds) for log_scale in (-np.log(2), 0.0, np.log(2))]
    valid = [result for result in candidates if result.success and np.isfinite(result.fun)]
    if not valid or not all(result.success for result in [*separate, equal]):
        raise ValueError("At least one pooling fit failed to converge.")
    free = min(valid, key=lambda result: result.fun)
    return _scale_test_summary(free, equal, separate, groups, width)


def _scale_test_summary(free, equal, separate, groups, width):
    if abs(free.x[-1]) >= 7.99 or np.linalg.norm(free.x[:-1]) <= 1e-8:
        raise ValueError("The relative scale is not identified in this sample.")
    separate_loss = sum(result.fun for result in separate)
    lr_preferences = 2 * (free.fun - separate_loss)
    lr_scale = 2 * (equal.fun - free.fun)
    if min(lr_preferences, lr_scale) < -1e-5:
        raise ValueError("The pooling fits violate the nested likelihood ordering.")
    lr_preferences = max(0.0, lr_preferences)
    p_value = float(chi2.sf(lr_preferences, width - 1))
    return SwaitLouviereResult(
        chi2_stat=lr_preferences, df=width - 1, p_value=p_value,
        ll_pooled=-free.fun, ll_group1=-separate[0].fun, ll_group2=-separate[1].fun,
        n_parameters=width, group1_label=str(groups[0]), group2_label=str(groups[1]),
        reject_null=p_value < 0.05, relative_scale=float(np.exp(free.x[-1])),
        ll_equal_scale=-equal.fun, scale_p_value=float(chi2.sf(max(0.0, lr_scale), 1)),
    )


def _separated_choices(X, y):
    chosen = X[np.arange(len(y)), y]
    differences = (chosen[:, None, :] - X).reshape(-1, X.shape[-1])
    # A separating direction weakly improves every choice margin and
    # strictly improves at least one. A unit box bounds the search.
    result = linprog(
        -differences.mean(axis=0),
        A_ub=-differences, b_ub=np.zeros(len(differences)),
        bounds=(-1, 1), method="highs",
    )
    if not result.success:
        raise ValueError("Could not check the group for choice separation.")
    return result.fun < -1e-8


def comparison_outcome(entry):
    """Key for comparing runs on the active dataset in the app."""
    spec = entry.get("spec")
    if spec is None:
        return None
    full_spec = entry.get("full_spec")
    estimation = entry["estimation"]
    return (
        spec.id_col, spec.task_col, spec.alt_col, spec.choice_col,
        getattr(full_spec, "bws_worst_col", None),
        estimation.n_observations, estimation.n_individuals,
    )


def conditional_lr(restricted, unrestricted, data=None):
    """Ordinary LR inference for nested, finite, best-choice CL fits."""
    if any(e["model_type"] != "conditional" for e in (restricted, unrestricted)):
        raise ValueError("The chi-square reference is available for nested conditional logit models.")
    outcome = comparison_outcome(restricted)
    if outcome is None or outcome != comparison_outcome(unrestricted) or outcome[4] is not None:
        raise ValueError("Use the same estimation sample and the same single-choice outcome.")
    reduced, complete = (e["estimation"] for e in (restricted, unrestricted))
    if not all(e.success and np.isfinite(e.log_likelihood) for e in (reduced, complete)):
        raise ValueError("Both fits must have converged with finite likelihoods.")

    reduced_terms = {(v.name, v.column) for v in restricted["spec"].variables}
    complete_terms = {(v.name, v.column) for v in unrestricted["spec"].variables}
    dof = complete.n_parameters - reduced.n_parameters
    if not reduced_terms < complete_terms or dof != len(complete_terms) - len(reduced_terms):
        raise ValueError("The larger model must retain each term of the smaller model.")
    if data is not None:
        from .pipeline import prepare_from_spec

        designs = []
        for entry in (restricted, unrestricted):
            design = entry.get("expanded_df")
            if design is None:
                spec = entry.get("full_spec")
                design = data if spec is None else prepare_from_spec(data, spec)[0]
            designs.append(design)
        columns = list(outcome[:4]) + sorted({col for _, col in reduced_terms})
        if not designs[0][columns].equals(designs[1][columns]):
            raise ValueError("Both models must use the same rows, outcomes and shared feature coding.")
    statistic = 2 * (complete.log_likelihood - reduced.log_likelihood)
    if statistic < -1e-5:
        raise ValueError("The larger model has lower likelihood. Refit both models first.")
    return max(0.0, statistic), dof, float(chi2.sf(max(0.0, statistic), dof))

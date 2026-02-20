# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under dual license: AGPL-3.0 (open-source) or commercial. See LICENSE.

"""Statistical tests for discrete choice model comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .config import ModelSpec, VariableSpec
from .pipeline import estimate_dataframe


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
    """
    Swait-Louviere pooling test for preference heterogeneity across groups.

    Tests whether two groups of respondents share the same preference parameters
    by comparing the pooled log-likelihood to the sum of group-specific
    log-likelihoods.

    Test statistic: -2 * [LL_pooled - (LL_group1 + LL_group2)]
    Distributed as chi-squared with K degrees of freedom (K = number of parameters).

    Parameters
    ----------
    df : pd.DataFrame
        Long-format choice data.
    spec : ModelSpec
        Model specification (column roles and variables).
    grouping_col : str
        Column used to split data into two groups. Must have exactly two unique values.
    model_type : str
        Type of model to estimate ('conditional', 'mixed', or 'latent_class').
    maxiter : int
        Maximum optimizer iterations.
    seed : int
        Random seed for estimation.

    Returns
    -------
    SwaitLouviereResult
    """
    if grouping_col not in df.columns:
        raise ValueError(f"Grouping column '{grouping_col}' not found in data.")

    groups = df[grouping_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError(
            f"Grouping column must have exactly 2 unique values, found {len(groups)}: {groups}"
        )

    group1_label, group2_label = str(groups[0]), str(groups[1])
    df_g1 = df[df[grouping_col] == groups[0]].copy()
    df_g2 = df[df[grouping_col] == groups[1]].copy()

    est_kwargs = dict(spec=spec, model_type=model_type, maxiter=maxiter, seed=seed)
    if model_type == "latent_class":
        est_kwargs["n_classes"] = n_classes
        est_kwargs["n_starts"] = n_starts

    # Estimate on each group and pooled
    result_pooled = estimate_dataframe(df=df, **est_kwargs)
    result_g1 = estimate_dataframe(df=df_g1, **est_kwargs)
    result_g2 = estimate_dataframe(df=df_g2, **est_kwargs)

    ll_pooled = result_pooled.estimation.log_likelihood
    ll_g1 = result_g1.estimation.log_likelihood
    ll_g2 = result_g2.estimation.log_likelihood
    k = result_pooled.estimation.n_parameters

    chi2_stat = -2.0 * (ll_pooled - (ll_g1 + ll_g2))
    p_value = float(1.0 - chi2.cdf(chi2_stat, k))

    return SwaitLouviereResult(
        chi2_stat=chi2_stat,
        df=k,
        p_value=p_value,
        ll_pooled=ll_pooled,
        ll_group1=ll_g1,
        ll_group2=ll_g2,
        n_parameters=k,
        group1_label=group1_label,
        group2_label=group2_label,
        reject_null=p_value < 0.05,
    )

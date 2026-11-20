# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under dual license: AGPL-3.0 (open-source) or commercial. See LICENSE.

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd
import torch

from .config import FullModelSpec, ModelSpec, VariableSpec
from .data import ChoiceTensors, prepare_choice_tensors
from .latent_class import LatentClassEstimator, LatentClassResult
from .model import ConditionalLogitEstimator, EstimationResult, GmnlEstimator, MixedLogitEstimator


@dataclass
class PipelineResult:
    tensors: ChoiceTensors
    estimation: EstimationResult | LatentClassResult
    wtp: pd.DataFrame | None = field(default=None)
    expanded_spec: ModelSpec | None = field(default=None, repr=False)
    expanded_df: pd.DataFrame | None = field(default=None, repr=False)


def build_estimator(
    df: pd.DataFrame,
    spec: ModelSpec,
    model_type: str = "mixed",
    seed: int = 123,
    device: torch.device | None = None,
    n_classes: int | None = None,
    correlated: bool = False,
    membership_cols: list[str] | None = None,
    correlation_groups: list[list[int]] | None = None,
    bws_worst_col: str | None = None,
    estimate_lambda_w: bool = True,
    fixed_gamma: float | None = None,
) -> tuple[MixedLogitEstimator | LatentClassEstimator, ChoiceTensors]:
    """Build the configured estimator and its tensors without fitting it."""
    tensors = prepare_choice_tensors(df, spec, device=device)

    # Prepare BWS data if worst column specified
    bws_data = None
    if bws_worst_col:
        if bws_worst_col in (membership_cols or spec.membership_cols or []):
            raise ValueError("The worst-choice outcome cannot be used as a predictor.")
        from .bws import prepare_bws_data, validate_bws

        validate_bws(df, spec, bws_worst_col)
        bws_data = prepare_bws_data(
            df,
            spec,
            bws_worst_col,
            tensors.n_obs,
            tensors.n_alts,
            tensors.X.device,
            estimate_lambda_w=estimate_lambda_w,
        )

    if model_type == "mixed":
        estimator = MixedLogitEstimator(
            tensors=tensors,
            variables=spec.variables,
            n_draws=spec.n_draws,
            device=tensors.X.device,
            seed=seed,
            correlated=correlated,
            correlation_groups=correlation_groups,
            bws_data=bws_data,
        )
    elif model_type == "conditional":
        estimator = ConditionalLogitEstimator(
            tensors=tensors,
            variables=spec.variables,
            device=tensors.X.device,
            seed=seed,
            bws_data=bws_data,
        )
    elif model_type == "gmnl":
        estimator = GmnlEstimator(
            tensors=tensors,
            variables=spec.variables,
            n_draws=spec.n_draws,
            device=tensors.X.device,
            seed=seed,
            bws_data=bws_data,
            correlated=correlated,
            correlation_groups=correlation_groups,
            fixed_gamma=fixed_gamma,
        )
    elif model_type == "latent_class":
        q = n_classes if n_classes is not None else spec.n_classes
        mc = membership_cols or spec.membership_cols
        estimator = LatentClassEstimator(
            tensors=tensors,
            variables=spec.variables,
            n_classes=q,
            device=tensors.X.device,
            seed=seed,
            membership_cols=mc,
            df=df,
            id_col=spec.id_col,
            bws_data=bws_data,
        )
    else:
        raise ValueError(
            "model_type must be 'mixed', 'conditional', 'gmnl', or 'latent_class'."
        )
    return estimator, tensors


def estimate_dataframe(
    df: pd.DataFrame,
    spec: ModelSpec,
    model_type: str = "mixed",
    maxiter: int = 300,
    seed: int = 123,
    device: torch.device | None = None,
    n_classes: int | None = None,
    n_starts: int = 10,
    correlated: bool = False,
    membership_cols: list[str] | None = None,
    correlation_groups: list[list[int]] | None = None,
    bws_worst_col: str | None = None,
    estimate_lambda_w: bool = True,
    lc_method: str = "em",
    custom_start: list[float] | None = None,
    fixed_gamma: float | None = None,
) -> PipelineResult:
    estimator, tensors = build_estimator(
        df=df, spec=spec, model_type=model_type, seed=seed, device=device,
        n_classes=n_classes, correlated=correlated, membership_cols=membership_cols,
        correlation_groups=correlation_groups, bws_worst_col=bws_worst_col,
        estimate_lambda_w=estimate_lambda_w, fixed_gamma=fixed_gamma,
    )
    if isinstance(estimator, LatentClassEstimator):
        estimation = estimator.fit(
            maxiter=maxiter, n_starts=n_starts, method=lc_method, initial_theta=custom_start,
        )
    else:
        estimation = estimator.fit(maxiter=maxiter, initial_theta=custom_start)
    return PipelineResult(tensors=tensors, estimation=estimation)


def prepare_from_spec(df: pd.DataFrame, spec: FullModelSpec) -> tuple[pd.DataFrame, ModelSpec]:
    """Materialise the utility columns in their original parameter order."""
    df = df.copy()

    # ── Expand dummy-coded variables ──────────────────────────────────
    dummy_cols = {dc.column for dc in spec.dummy_codings}
    # Build mapping: original column -> list of expanded VariableSpecs
    _dummy_expansions: dict[str, list[VariableSpec]] = {}

    for dc in spec.dummy_codings:
        matched = [v for v in spec.variables if v.column == dc.column]
        if not matched:
            raise ValueError(
                f"Dummy coding column '{dc.column}' not found in variables."
            )
        if dc.column not in df or df[dc.column].isna().any():
            raise ValueError(f"Dummy-coded column '{dc.column}' must exist and contain no missing values.")
        dummy_names, mapping = dc.expand(df)
        if len(mapping) != len(dummy_names):
            raise ValueError(f"Dummy levels for '{dc.column}' produce duplicate column names.")
        for dname, level in mapping.items():
            if dname in df.columns:
                raise ValueError(f"Dummy column '{dname}' would overwrite an existing column.")
            df[dname] = (df[dc.column] == level).astype(int)

        base_var = matched[0]
        _dummy_expansions[dc.column] = [
            VariableSpec(name=dname, column=dname, distribution=base_var.distribution)
            for dname in dummy_names
        ]

    # Build final variable list: replace each dummy placeholder in-place
    # to preserve the UI's variable ordering (critical for correlation_groups)
    all_variables: list[VariableSpec] = []
    for v in spec.variables:
        if v.column in dummy_cols:
            all_variables.extend(_dummy_expansions[v.column])
        else:
            all_variables.append(v)

    extra_vars: list[VariableSpec] = []

    for inter in spec.interactions:
        col_name = inter.name
        if col_name in df.columns:
            raise ValueError(f"Interaction '{col_name}' would overwrite an existing column.")
        for col in inter.columns:
            if col not in df.columns:
                raise ValueError(
                    f"Interaction column '{col}' not found in data."
                )
        product = df[inter.columns[0]].astype(float)
        for col in inter.columns[1:]:
            product = product * df[col].astype(float)
        df[col_name] = product
        extra_vars.append(VariableSpec(name=col_name, column=col_name, distribution="fixed"))

    all_variables = all_variables + extra_vars

    if spec.pure_scale:
        all_variables = [replace(v, distribution="fixed") for v in all_variables]
    return df, replace(spec.to_model_spec(), variables=all_variables)

def estimator_options(spec: FullModelSpec, *, for_fit: bool = True) -> dict:
    """Keep all modelling choices when refitting prepared data."""
    options = dict(
        n_classes=spec.n_classes,
        correlated=spec.correlated and not spec.pure_scale,
        membership_cols=spec.membership_cols,
        correlation_groups=None if spec.pure_scale else spec.correlation_groups,
        bws_worst_col=spec.bws_worst_col,
        estimate_lambda_w=spec.estimate_lambda_w,
        fixed_gamma=spec.fixed_gamma,
    )
    if for_fit:
        options.update(n_starts=spec.n_starts, lc_method=spec.lc_method, custom_start=spec.custom_start)
    return options


def estimate_from_spec(
    df: pd.DataFrame, spec: FullModelSpec, device: torch.device | None = None,
) -> PipelineResult:
    """Estimate a complete specification on raw long-format choice data."""
    expanded_df, expanded_spec = prepare_from_spec(df, spec)
    result = estimate_dataframe(
        df=expanded_df, spec=expanded_spec, model_type=spec.model_type,
        maxiter=spec.maxiter, seed=spec.seed, device=device,
        **estimator_options(spec),
    )
    result.expanded_spec = expanded_spec
    result.expanded_df = expanded_df
    return result


def save_estimation_outputs(estimation: EstimationResult | LatentClassResult, output_prefix: str | Path) -> None:
    output_prefix = Path(output_prefix)
    if output_prefix.suffix:
        output_prefix = output_prefix.with_suffix("")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    estimates_path = output_prefix.parent / f"{output_prefix.name}_estimates.csv"
    summary_path = output_prefix.parent / f"{output_prefix.name}_summary.json"

    estimation.estimates.to_csv(estimates_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(estimation.summary_dict(), handle, indent=2, default=str)

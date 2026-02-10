from __future__ import annotations

import json
from dataclasses import dataclass, field
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
) -> PipelineResult:
    tensors = prepare_choice_tensors(df, spec, device=device)

    # Prepare BWS data if worst column specified
    bws_data = None
    if bws_worst_col:
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
        return PipelineResult(
            tensors=tensors,
            estimation=estimator.fit(maxiter=maxiter, initial_theta=custom_start),
        )
    elif model_type == "conditional":
        estimator = ConditionalLogitEstimator(
            tensors=tensors,
            variables=spec.variables,
            device=tensors.X.device,
            seed=seed,
            bws_data=bws_data,
        )
        return PipelineResult(
            tensors=tensors,
            estimation=estimator.fit(maxiter=maxiter, initial_theta=custom_start),
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
        )
        return PipelineResult(
            tensors=tensors,
            estimation=estimator.fit(maxiter=maxiter, initial_theta=custom_start),
        )
    elif model_type == "latent_class":
        q = n_classes if n_classes is not None else spec.n_classes
        mc = membership_cols or spec.membership_cols
        lc_estimator = LatentClassEstimator(
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
        return PipelineResult(
            tensors=tensors,
            estimation=lc_estimator.fit(
                maxiter=maxiter, n_starts=n_starts, method=lc_method,
                initial_theta=custom_start,
            ),
        )
    else:
        raise ValueError(
            "model_type must be 'mixed', 'conditional', 'gmnl', or 'latent_class'."
        )


def estimate_from_spec(
    df: pd.DataFrame,
    spec: FullModelSpec,
    device: torch.device | None = None,
) -> PipelineResult:
    """Single entry-point: all configuration comes from *spec*.

    1. Dummy-coded columns are materialised from *spec.dummy_codings*.
    2. Heterogeneity interactions are materialised as interaction columns.
    Both are appended as fixed VariableSpecs before estimation.
    """
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
        dummy_names, mapping = dc.expand(df)
        for dname, level in mapping.items():
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

    model_spec = ModelSpec(
        id_col=spec.id_col,
        task_col=spec.task_col,
        alt_col=spec.alt_col,
        choice_col=spec.choice_col,
        variables=all_variables,
        n_draws=spec.n_draws,
        n_classes=spec.n_classes,
        membership_cols=spec.membership_cols,
    )

    result = estimate_dataframe(
        df=df,
        spec=model_spec,
        model_type=spec.model_type,
        maxiter=spec.maxiter,
        seed=spec.seed,
        device=device,
        n_classes=spec.n_classes,
        n_starts=spec.n_starts,
        correlated=spec.correlated,
        membership_cols=spec.membership_cols,
        correlation_groups=spec.correlation_groups,
        bws_worst_col=spec.bws_worst_col,
        estimate_lambda_w=spec.estimate_lambda_w,
        lc_method=spec.lc_method,
        custom_start=spec.custom_start,
    )
    result.expanded_spec = model_spec
    result.expanded_df = df
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

# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under dual license: AGPL-3.0 (open-source) or commercial. See LICENSE.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SimulationOutput:
    data: pd.DataFrame
    true_parameters: dict[str, float]


def generate_simulated_dce(
    n_individuals: int = 300,
    n_tasks: int = 8,
    n_alts: int = 3,
    seed: int = 42,
    attribute_names: list[str] | None = None,
    covariate_names: list[str] | None = None,
    bws: bool = False,
) -> SimulationOutput:
    """
    Generate a panel choice dataset in long format.

    Parameters
    ----------
    attribute_names : list[str] or None
        Custom attribute column names. When None, uses the hardcoded defaults
        (price, time, comfort, reliability) with their original distributions.
    covariate_names : list[str] or None
        Custom covariate column names. When None, uses the hardcoded defaults
        (income, age) with their original distributions.
    bws : bool
        If True, generate a ``worst`` column (binary 0/1) alongside the best
        choice column. Worst choices are sampled among the remaining alternatives
        with the same utility scale and an independent shock. Requires ``n_alts >= 3``.
    """
    if n_individuals < 1 or n_tasks < 1 or n_alts < 2:
        raise ValueError("Need n_individuals >= 1, n_tasks >= 1, n_alts >= 2.")
    if bws and n_alts < 3:
        raise ValueError("BWS simulation requires n_alts >= 3.")

    rng = np.random.default_rng(seed)
    worst_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,))) if bws else None
    alt_constants = np.linspace(0.0, 0.35, num=n_alts)

    # ── custom vs hardcoded path ──────────────────────────────────
    use_custom_attrs = attribute_names is not None
    use_custom_covs = covariate_names is not None

    if use_custom_attrs:
        # Generate true parameters for each custom attribute
        true_params: dict[str, float] = {}
        attr_true_mu: list[float] = []
        attr_true_sd: list[float] = []
        for attr in attribute_names:
            mu = float(rng.uniform(-1.5, 1.5))
            sd = abs(mu) * 0.2
            true_params[f"mu_{attr}"] = round(mu, 4)
            true_params[f"sd_{attr}"] = round(sd, 4)
            attr_true_mu.append(mu)
            attr_true_sd.append(sd)
    else:
        attribute_names = ["price", "time", "comfort", "reliability"]
        true_params = {
            "mu_price": -1.20,
            "sd_price": 0.25,
            "mu_time": -0.70,
            "sd_time": 0.18,
            "mu_comfort": 0.85,
            "sd_comfort": 0.12,
            "beta_reliability": 0.55,
        }

    if not use_custom_covs:
        covariate_names = ["income", "age"]

    for alt_idx in range(1, n_alts):
        true_params[f"asc_alt_{alt_idx + 1}"] = float(alt_constants[alt_idx])
    if bws:
        true_params["lambda_w"] = 1.0

    rows: list[dict[str, float | int]] = []

    # ── covariate arrays (constant within respondent) ─────────────
    if use_custom_covs:
        cov_arrays = {
            name: rng.normal(50, 15, size=n_individuals).clip(10, 100)
            for name in covariate_names
        }
    else:
        cov_arrays = {
            "income": rng.normal(60_000, 15_000, size=n_individuals).clip(18_000, 180_000),
            "age": rng.normal(42, 12, size=n_individuals).clip(18, 80),
        }

    for individual in range(1, n_individuals + 1):
        # individual-level random coefficients
        if use_custom_attrs:
            betas = [rng.normal(attr_true_mu[k], max(attr_true_sd[k], 1e-6))
                     for k in range(len(attribute_names))]
        else:
            betas_named = {
                "price": rng.normal(true_params["mu_price"], true_params["sd_price"]),
                "time": rng.normal(true_params["mu_time"], true_params["sd_time"]),
                "comfort": rng.normal(true_params["mu_comfort"], true_params["sd_comfort"]),
            }

        cov_values = {name: float(arr[individual - 1]) for name, arr in cov_arrays.items()}

        for task in range(1, n_tasks + 1):
            # ── attribute values ──────────────────────────────────
            if use_custom_attrs:
                attr_values = {
                    attr: rng.uniform(1, 10, size=n_alts) for attr in attribute_names
                }
                deterministic_utility = alt_constants.copy()
                for k, attr in enumerate(attribute_names):
                    deterministic_utility = deterministic_utility + betas[k] * attr_values[attr]
            else:
                price = rng.uniform(3.0, 30.0, size=n_alts)
                time = rng.uniform(10.0, 90.0, size=n_alts)
                comfort = rng.integers(0, 2, size=n_alts)
                reliability = rng.uniform(0.70, 1.00, size=n_alts)
                deterministic_utility = (
                    alt_constants
                    + betas_named["price"] * np.log1p(price)
                    + betas_named["time"] * (time / 10.0)
                    + betas_named["comfort"] * comfort
                    + true_params["beta_reliability"] * reliability
                )
                attr_values = {
                    "price": price,
                    "time": time,
                    "comfort": comfort,
                    "reliability": reliability,
                }

            random_shock = rng.gumbel(0.0, 1.0, size=n_alts)
            utilities = deterministic_utility + random_shock
            chosen_alt = int(np.argmax(utilities))

            # ── worst choice (BWS) ───────────────────────────────
            if worst_rng is not None:
                worst_utility = -deterministic_utility + worst_rng.gumbel(size=n_alts)
                worst_utility[chosen_alt] = -np.inf
                worst_alt = int(np.argmax(worst_utility))

            for alt in range(n_alts):
                row: dict[str, float | int] = {
                    "respondent_id": individual,
                    "task_id": task,
                    "alternative": alt + 1,
                    "choice": int(alt == chosen_alt),
                }
                for attr in attribute_names:
                    val = attr_values[attr][alt]
                    row[attr] = int(val) if isinstance(val, (np.integer,)) else float(val)
                for cov_name, cov_val in cov_values.items():
                    row[cov_name] = cov_val
                if bws:
                    row["worst"] = int(alt == worst_alt)
                rows.append(row)

    return SimulationOutput(data=pd.DataFrame(rows), true_parameters=true_params)

# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under dual license: AGPL-3.0 (open-source) or commercial. See LICENSE.

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from scipy.stats import norm, qmc

from .bws import BwsData, bws_log_prob, standard_log_prob
from .config import VariableSpec
from .data import ChoiceTensors
from .optimization import fit_lbfgs, rescale_parameters

logger = logging.getLogger(__name__)


def _positive(raw: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(raw)


def generate_halton_draws(
    n_individuals: int,
    n_draws: int,
    n_dims: int,
    seed: int = 123,
) -> np.ndarray:
    """Generate Halton sequence draws mapped to N(0,1)."""
    if n_dims == 0:
        return np.zeros((n_individuals, n_draws, 0), dtype=np.float32)

    sampler = qmc.Halton(d=n_dims, scramble=True, seed=seed)
    uniforms = sampler.random(n=n_individuals * n_draws)
    normals = norm.ppf(np.clip(uniforms, 1e-10, 1.0 - 1e-10))
    return normals.reshape(n_individuals, n_draws, n_dims).astype(np.float32)


@dataclass
class EstimationResult:
    success: bool
    message: str
    log_likelihood: float
    aic: float
    bic: float
    n_parameters: int
    n_observations: int
    n_individuals: int
    optimizer_iterations: int
    runtime_seconds: float
    estimates: pd.DataFrame
    vcov_matrix: np.ndarray | None = field(default=None, repr=False)
    covariance_matrix: np.ndarray | None = field(default=None, repr=False)
    correlation_matrix: np.ndarray | None = field(default=None, repr=False)
    random_param_names: list[str] | None = field(default=None, repr=False)
    covariance_se: np.ndarray | None = field(default=None, repr=False)
    correlation_se: np.ndarray | None = field(default=None, repr=False)
    correlation_test: pd.DataFrame | None = field(default=None, repr=False)
    raw_theta: np.ndarray | None = field(default=None, repr=False)

    def summary_dict(self) -> dict[str, Any]:
        d = {
            "success": self.success,
            "message": self.message,
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "n_parameters": self.n_parameters,
            "n_observations": self.n_observations,
            "n_individuals": self.n_individuals,
            "optimizer_iterations": self.optimizer_iterations,
            "runtime_seconds": self.runtime_seconds,
        }
        if self.vcov_matrix is not None:
            d["has_vcov"] = True
        has_se = "std_error" in self.estimates.columns and self.estimates["std_error"].notna().any()
        d["has_standard_errors"] = has_se
        if self.covariance_matrix is not None:
            d["has_covariance_matrix"] = True
        return d


class MixedLogitEstimator:
    """
    Generic mixed logit estimator for panel choice data.

    Random distributions supported:
      - normal
      - lognormal
    """

    def __init__(
        self,
        tensors: ChoiceTensors,
        variables: list[VariableSpec],
        n_draws: int = 200,
        device: torch.device | None = None,
        seed: int = 123,
        correlated: bool = False,
        correlation_groups: list[list[int]] | None = None,
        bws_data: BwsData | None = None,
        *,
        scale_draws: bool = False,
    ) -> None:
        if len(variables) != tensors.X.shape[2]:
            raise ValueError(
                "Variable count mismatch: number of VariableSpec entries must equal X.shape[2]."
            )

        self.device = device or tensors.X.device
        self.X = tensors.X.to(self.device).float()
        self.y = tensors.y.to(self.device).long()
        self.panel_idx = tensors.panel_idx.to(self.device).long()
        self.n_individuals = tensors.n_individuals
        self.n_obs = tensors.n_obs
        self.n_alts = tensors.n_alts
        self.variables = variables
        self.seed = seed
        self.correlated = correlated

        self._param_defs: list[dict[str, Any]] = []
        self.n_random_vars = 0
        self._random_param_names: list[str] = []
        theta_idx = 0

        # First pass: assign mean parameters and count random vars
        for var_idx, var in enumerate(variables):
            if var.distribution == "fixed":
                self._param_defs.append(
                    {
                        "name": var.name,
                        "var_idx": var_idx,
                        "distribution": "fixed",
                        "theta_mu_idx": theta_idx,
                        "theta_indices": [theta_idx],
                        "draw_idx": None,
                    }
                )
                theta_idx += 1
            else:
                self._param_defs.append(
                    {
                        "name": var.name,
                        "var_idx": var_idx,
                        "distribution": var.distribution,
                        "theta_mu_idx": theta_idx,
                        "theta_indices": [theta_idx],  # will be extended below
                        "draw_idx": self.n_random_vars,
                    }
                )
                self._random_param_names.append(var.name)
                self.n_random_vars += 1
                theta_idx += 1  # mu only; sd/cholesky allocated below

        K = self.n_random_vars
        self._chol_mapping: list[tuple[int, int, int, bool]] = []

        if correlation_groups is not None and K > 0:
            # Selective correlation: block-diagonal Cholesky
            self.correlated = True
            groups = [sorted(g) for g in correlation_groups]
            all_in: set[int] = set()
            for g in groups:
                for gi in g:
                    if gi < 0 or gi >= K:
                        raise ValueError(
                            f"correlation_groups index {gi} out of range [0, {K})"
                        )
                    if gi in all_in:
                        raise ValueError(f"Random param index {gi} in multiple groups")
                    all_in.add(gi)
            standalone = sorted(set(range(K)) - all_in)
            self._chol_start = theta_idx
            for g in groups:
                for lr in range(len(g)):
                    for lc in range(lr + 1):
                        self._chol_mapping.append(
                            (g[lr], g[lc], theta_idx, lr == lc)
                        )
                        theta_idx += 1
            for si in standalone:
                self._chol_mapping.append((si, si, theta_idx, True))
                theta_idx += 1
            self._n_chol_params = theta_idx - self._chol_start
        elif correlated and K > 0:
            # Full correlation: K*(K+1)/2 Cholesky elements
            self._chol_start = theta_idx
            for row in range(K):
                for col in range(row + 1):
                    self._chol_mapping.append((row, col, theta_idx, row == col))
                    theta_idx += 1
            self._n_chol_params = K * (K + 1) // 2
        elif K > 0:
            # Independent: one raw_sd per random param (backward compatible)
            for p in self._param_defs:
                if p["distribution"] != "fixed":
                    p["theta_indices"].append(theta_idx)
                    theta_idx += 1
        else:
            pass  # no random parameters

        self.n_params = theta_idx

        # BWS scale parameter
        self._bws_data = bws_data
        self._bws_has_lambda_w = False
        self._lambda_w_idx = -1
        if bws_data is not None:
            self.y_worst = bws_data.y_worst.to(self.device).long()
            if bws_data.estimate_lambda_w:
                self._bws_has_lambda_w = True
                self._lambda_w_idx = self.n_params
                self.n_params += 1

        self.n_draws = int(n_draws if self.n_random_vars or scale_draws else 1)

        draws_np = generate_halton_draws(
            n_individuals=self.n_individuals,
            n_draws=self.n_draws,
            n_dims=self.n_random_vars + int(scale_draws),
            seed=seed,
        )
        draws = torch.tensor(draws_np, dtype=torch.float32, device=self.device)
        self.draws = draws[:, :, :self.n_random_vars]
        if scale_draws:
            self.scale_draws = draws[:, :, -1]

    def _initial_theta(self) -> np.ndarray:
        theta0 = np.zeros(self.n_params, dtype=np.float64)
        if self.correlated and self.n_random_vars > 0:
            for (_row, _col, tidx, is_diag) in self._chol_mapping:
                if is_diag:
                    theta0[tidx] = -1.0  # softplus(-1) ~= 0.313
        else:
            for p in self._param_defs:
                if p["distribution"] in {"normal", "lognormal"}:
                    theta0[p["theta_indices"][1]] = -1.0
        if self._bws_has_lambda_w:
            theta0[self._lambda_w_idx] = 0.0  # softplus(0) ~ 0.69 -> lambda_w starts near 1
        return theta0

    def _build_cholesky_L(self, theta: torch.Tensor) -> torch.Tensor:
        """Build K x K lower-triangular Cholesky factor from theta elements.

        Works for both full and selective (block-diagonal) correlation via _chol_mapping.
        """
        K = self.n_random_vars
        L = torch.zeros(K, K, dtype=torch.float32, device=self.device)
        for (row, col, tidx, is_diag) in self._chol_mapping:
            if is_diag:
                L[row, col] = _positive(theta[tidx])
            else:
                L[row, col] = theta[tidx]
        return L

    def _betas_from_theta(self, theta: torch.Tensor) -> torch.Tensor:
        n_vars = self.X.shape[2]
        betas = torch.zeros(
            self.n_individuals,
            self.n_draws,
            n_vars,
            dtype=torch.float32,
            device=self.device,
        )

        if self.correlated and self.n_random_vars > 0:
            L = self._build_cholesky_L(theta)
            # draws: (n_individuals, n_draws, K)
            # deviation = draws @ L.T -> (n_individuals, n_draws, K)
            deviation = torch.einsum("ndk,jk->ndj", self.draws, L)

            for p in self._param_defs:
                var_idx = p["var_idx"]
                dist = p["distribution"]

                if dist == "fixed":
                    betas[:, :, var_idx] = theta[p["theta_mu_idx"]]
                    continue

                mu = theta[p["theta_mu_idx"]]
                draw_idx = int(p["draw_idx"])
                dev = deviation[:, :, draw_idx]

                if dist == "normal":
                    betas[:, :, var_idx] = mu + dev
                elif dist == "lognormal":
                    betas[:, :, var_idx] = torch.exp(mu + dev)
                else:
                    raise ValueError(f"Unsupported distribution '{dist}'.")
        else:
            for p in self._param_defs:
                var_idx = p["var_idx"]
                dist = p["distribution"]
                idx = p["theta_indices"]

                if dist == "fixed":
                    betas[:, :, var_idx] = theta[idx[0]]
                    continue

                mu = theta[idx[0]]
                sd = _positive(theta[idx[1]])
                z = self.draws[:, :, int(p["draw_idx"])]

                if dist == "normal":
                    betas[:, :, var_idx] = mu + sd * z
                elif dist == "lognormal":
                    betas[:, :, var_idx] = torch.exp(mu + sd * z)
                else:
                    raise ValueError(f"Unsupported distribution '{dist}'.")

        return betas

    def _neg_log_likelihood_tensor(
        self, theta: torch.Tensor, X: torch.Tensor | None = None,
    ) -> torch.Tensor:
        betas = self._betas_from_theta(theta)
        betas_obs = betas[self.panel_idx]  # (n_obs, n_draws, n_vars)

        # Utilities (n_obs, n_draws, n_alts)
        utility = torch.einsum("nav,ndv->nda", self.X if X is None else X, betas_obs)

        if self._bws_data is None:
            log_prob = standard_log_prob(utility, self.y, alt_dim=2)
        else:
            lambda_w = self._get_lambda_w(theta)
            log_prob = bws_log_prob(
                utility, self.y, self.y_worst, lambda_w, alt_dim=2,
            )

        # Panel aggregation
        log_prob_individual = torch.zeros(
            self.n_individuals, self.n_draws, dtype=torch.float32, device=self.device
        )
        log_prob_individual.index_add_(0, self.panel_idx, log_prob)

        log_prob_avg = torch.logsumexp(log_prob_individual, dim=1) - np.log(self.n_draws)
        return -log_prob_avg.sum()

    def _get_lambda_w(self, theta: torch.Tensor):
        """Get lambda_w: estimated (softplus) or fixed at 1.0."""
        if self._bws_has_lambda_w:
            return torch.nn.functional.softplus(theta[self._lambda_w_idx]) + 1e-6
        return 1.0

    def _compute_vcov(
        self, theta_hat: np.ndarray, X: torch.Tensor | None = None,
    ) -> np.ndarray | None:
        """Compute variance-covariance matrix via the Hessian of the neg-log-likelihood."""
        try:
            theta_t = torch.tensor(
                theta_hat, dtype=torch.float32, device=self.device
            )

            def nll_fn(t: torch.Tensor) -> torch.Tensor:
                return self._neg_log_likelihood_tensor(t, X)

            H = torch.autograd.functional.hessian(nll_fn, theta_t)
            H_np = H.detach().cpu().numpy().astype(np.float64)

            H_np = (H_np + H_np.T) / 2
            eigvals = np.linalg.eigvalsh(H_np)
            if not np.isfinite(H_np).all() or eigvals.min() <= 0:
                warnings.warn(
                    "Hessian is not positive definite; asymptotic standard errors "
                    "are unavailable at this estimate."
                )
                return None

            vcov = np.linalg.inv(H_np)
            return vcov
        except Exception as exc:
            logger.warning("Hessian computation failed: %s", exc)
            return None

    def _softplus_derivative(self, raw: float) -> float:
        """Derivative of softplus: d/dx log(1+exp(x)) = sigmoid(x)."""
        return float(expit(raw))

    def _parameter_table(
        self, theta_hat: np.ndarray, vcov: np.ndarray | None = None,
    ) -> pd.DataFrame:
        rows = []

        if self.correlated and self.n_random_vars > 0:
            # Correlated case: report mu params, then Cholesky elements, then derived sd
            L_np = self._build_cholesky_L_numpy(theta_hat)
            cov_matrix = L_np @ L_np.T
            sd_vec = np.sqrt(np.diag(cov_matrix))

            for p in self._param_defs:
                name = p["name"]
                dist = p["distribution"]
                mu_idx = p["theta_mu_idx"]

                if dist == "fixed":
                    se = float("nan")
                    if vcov is not None:
                        var = vcov[mu_idx, mu_idx]
                        se = float(np.sqrt(max(var, 0.0)))
                    rows.append(self._make_row(f"beta_{name}", dist, float(theta_hat[mu_idx]), se, theta_index=mu_idx))
                else:
                    se_mu = float("nan")
                    if vcov is not None:
                        var_mu = vcov[mu_idx, mu_idx]
                        se_mu = float(np.sqrt(max(var_mu, 0.0)))
                    rows.append(self._make_row(f"mu_{name}", dist, float(theta_hat[mu_idx]), se_mu, theta_index=mu_idx))

            # Report derived standard deviations (from diagonal of covariance matrix)
            for k, name in enumerate(self._random_param_names):
                dist = "normal"
                for p in self._param_defs:
                    if p["name"] == name and p["distribution"] != "fixed":
                        dist = p["distribution"]
                        break
                rows.append(self._make_row(f"sd_{name}", dist, float(sd_vec[k]), float("nan"), theta_index=-1))

            # Report Cholesky elements (works for both full and selective modes)
            for (row, col, tidx, is_diag) in self._chol_mapping:
                raw_val = theta_hat[tidx]
                if is_diag:
                    val = float(np.logaddexp(0.0, raw_val))
                else:
                    val = float(raw_val)
                label = f"chol_{self._random_param_names[row]}_{self._random_param_names[col]}"
                se = float("nan")
                if vcov is not None:
                    if is_diag:
                        deriv = self._softplus_derivative(raw_val)
                        se = float(abs(deriv) * np.sqrt(max(vcov[tidx, tidx], 0.0)))
                    else:
                        se = float(np.sqrt(max(vcov[tidx, tidx], 0.0)))
                rows.append(self._make_row(label, "cholesky", val, se, theta_index=tidx))
        else:
            # Independent case (backward compatible)
            for p in self._param_defs:
                idx = p["theta_indices"]
                name = p["name"]
                dist = p["distribution"]
                if dist == "fixed":
                    se = float("nan")
                    if vcov is not None:
                        var = vcov[idx[0], idx[0]]
                        se = float(np.sqrt(max(var, 0.0)))
                    rows.append(self._make_row(f"beta_{name}", dist, float(theta_hat[idx[0]]), se, theta_index=idx[0]))
                else:
                    raw_sd = theta_hat[idx[1]]
                    sd = float(np.logaddexp(0.0, raw_sd))

                    se_mu = float("nan")
                    se_sd = float("nan")
                    if vcov is not None:
                        var_mu = vcov[idx[0], idx[0]]
                        se_mu = float(np.sqrt(max(var_mu, 0.0)))
                        # Delta method for softplus transformation
                        var_raw_sd = vcov[idx[1], idx[1]]
                        deriv = self._softplus_derivative(raw_sd)
                        se_sd = float(abs(deriv) * np.sqrt(max(var_raw_sd, 0.0)))

                    rows.append(self._make_row(f"mu_{name}", dist, float(theta_hat[idx[0]]), se_mu, theta_index=idx[0]))
                    rows.append(self._make_row(f"sd_{name}", dist, sd, se_sd, theta_index=idx[1]))

        # BWS lambda_w row (applies to both correlated and independent branches)
        if self._bws_has_lambda_w:
            raw_lw = theta_hat[self._lambda_w_idx]
            lw_val = float(np.logaddexp(0.0, raw_lw) + 1e-6)  # softplus
            se_lw = float("nan")
            if vcov is not None:
                deriv = self._softplus_derivative(raw_lw)
                se_lw = float(abs(deriv) * np.sqrt(max(vcov[self._lambda_w_idx, self._lambda_w_idx], 0.0)))
            rows.append(self._make_row("lambda_w (worst scale)", "bws_scale", lw_val, se_lw, theta_index=self._lambda_w_idx))

        return pd.DataFrame(rows)

    def _build_cholesky_L_numpy(self, theta_hat: np.ndarray) -> np.ndarray:
        """Build K x K lower-triangular Cholesky factor from numpy theta."""
        K = self.n_random_vars
        L = np.zeros((K, K), dtype=np.float64)
        for (row, col, tidx, is_diag) in self._chol_mapping:
            if is_diag:
                L[row, col] = float(np.logaddexp(0.0, theta_hat[tidx]))
            else:
                L[row, col] = float(theta_hat[tidx])
        return L

    def _compute_cov_cor_inference(
        self,
        theta_hat: np.ndarray,
        vcov: np.ndarray,
        cov_mat: np.ndarray,
        cor_mat: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None, pd.DataFrame | None]:
        """Delta method SEs for covariance and correlation matrix elements."""
        try:
            K = self.n_random_vars
            # Use CPU for Jacobian (MPS doesn't support float64)
            cpu = torch.device("cpu")
            theta_t = torch.tensor(theta_hat, dtype=torch.float64, device=cpu)
            mapping = self._chol_mapping

            def _build_L_differentiable(th: torch.Tensor) -> torch.Tensor:
                L = torch.zeros(K, K, dtype=torch.float64, device=cpu)
                for row, col, tidx, is_diag in mapping:
                    val = _positive(th[tidx]) if is_diag else th[tidx]
                    e = torch.zeros(K, K, dtype=torch.float64, device=cpu)
                    e[row, col] = 1.0
                    L = L + e * val
                return L

            def _cov_flat(th: torch.Tensor) -> torch.Tensor:
                L = _build_L_differentiable(th)
                return (L @ L.T).reshape(-1)

            def _cor_flat(th: torch.Tensor) -> torch.Tensor:
                L = _build_L_differentiable(th)
                Sigma = L @ L.T
                sd = torch.sqrt(torch.diag(Sigma))
                sd_out = sd.unsqueeze(1) * sd.unsqueeze(0)
                sd_out = torch.where(sd_out > 0, sd_out, torch.ones_like(sd_out))
                return (Sigma / sd_out).reshape(-1)

            J_cov = torch.autograd.functional.jacobian(_cov_flat, theta_t)
            J_cov_np = J_cov.detach().numpy().astype(np.float64)

            J_cor = torch.autograd.functional.jacobian(_cor_flat, theta_t)
            J_cor_np = J_cor.detach().numpy().astype(np.float64)

            # Delta method: Var(g(θ)) = J @ Var(θ) @ Jᵀ
            cov_se = np.sqrt(np.maximum(np.diag(J_cov_np @ vcov @ J_cov_np.T), 0.0)).reshape(K, K)
            cor_se = np.sqrt(np.maximum(np.diag(J_cor_np @ vcov @ J_cor_np.T), 0.0)).reshape(K, K)

            # Pairwise correlation significance tests (off-diagonal)
            names = self._random_param_names
            rows = []
            for i in range(K):
                for j in range(i + 1, K):
                    rho = float(cor_mat[i, j])
                    se = float(cor_se[i, j])
                    z = rho / se if se > 1e-12 else float("nan")
                    p = float(2.0 * (1.0 - norm.cdf(abs(z)))) if not np.isnan(z) else float("nan")
                    rows.append({
                        "param_1": names[i],
                        "param_2": names[j],
                        "covariance": float(cov_mat[i, j]),
                        "cov_std_error": float(cov_se[i, j]),
                        "correlation": rho,
                        "cor_std_error": se,
                        "z_stat": float(z),
                        "p_value": float(p),
                    })

            test_df = pd.DataFrame(rows) if rows else None
            return cov_se, cor_se, test_df
        except Exception as exc:
            logger.warning("Correlation SE computation failed: %s", exc)
            return None, None, None

    @staticmethod
    def _make_row(param: str, dist: str, estimate: float, se: float, theta_index: int = -1) -> dict[str, Any]:
        z = estimate / se if (not np.isnan(se) and se > 0) else float("nan")
        p_val = float(2.0 * (1.0 - norm.cdf(abs(z)))) if not np.isnan(z) else float("nan")
        ci_lo = estimate - 1.96 * se if not np.isnan(se) else float("nan")
        ci_hi = estimate + 1.96 * se if not np.isnan(se) else float("nan")
        return {
            "parameter": param,
            "distribution": dist,
            "estimate": estimate,
            "std_error": se,
            "z_stat": z,
            "p_value": p_val,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "theta_index": theta_index,
        }

    def _parameter_units(
        self, design_scale: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map raw parameters from standardised design units to original units."""
        multipliers = np.ones(self.n_params)
        offsets = np.zeros(self.n_params)
        positive = np.zeros(self.n_params, dtype=bool)
        random_multipliers = np.ones(self.n_random_vars)
        for p in self._param_defs:
            unit = design_scale[p["var_idx"]]
            mu_idx = p["theta_mu_idx"]
            draw_idx = p["draw_idx"]
            if p["distribution"] == "lognormal":
                offsets[mu_idx] = -np.log(unit)
            else:
                multipliers[mu_idx] = 1.0 / unit
                if draw_idx is not None:
                    random_multipliers[draw_idx] = 1.0 / unit
            if draw_idx is not None and not self.correlated:
                sd_idx = p["theta_indices"][1]
                multipliers[sd_idx] = random_multipliers[draw_idx]
                positive[sd_idx] = True
        for row, _, index, is_diag in self._chol_mapping:
            multipliers[index] = random_multipliers[row]
            positive[index] = is_diag
        return multipliers, offsets, positive

    def fit(
        self,
        maxiter: int = 300,
        verbose: bool = False,
        initial_theta: list[float] | None = None,
    ) -> EstimationResult:
        # A shared utility offset cancels within each choice set.
        centered = self.X - self.X[:, :1]
        scale = centered.abs().amax(dim=(0, 1))
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        X = centered / scale
        factors, offsets, positive = self._parameter_units(scale.cpu().numpy().astype(float))
        if initial_theta is not None:
            theta0 = np.asarray(initial_theta, dtype=np.float64)
            if len(theta0) != self.n_params:
                raise ValueError(
                    f"custom_start has {len(theta0)} values but model expects {self.n_params} parameters."
                )
            theta0, _ = rescale_parameters(theta0, 1.0 / factors, -offsets / factors, positive)
        else:
            theta0 = self._initial_theta()
        opt = fit_lbfgs(
            lambda params: self._neg_log_likelihood_tensor(params, X), theta0, device=self.device,
            maxiter=maxiter, loss_scale=self.n_obs,
        )
        runtime = opt.runtime_seconds

        theta_hat, jacobian = rescale_parameters(opt.x, factors, offsets, positive)
        loglike = -float(opt.fun)
        k = self.n_params

        # Compute variance-covariance matrix from the Hessian
        vcov = self._compute_vcov(opt.x, X)
        if vcov is not None:
            vcov = jacobian[:, None] * vcov * jacobian[None, :]
        estimates = self._parameter_table(theta_hat, vcov)

        # Compute random-parameter covariance and correlation matrices
        cov_mat = None
        cor_mat = None
        rand_names = None
        cov_se = None
        cor_se = None
        cor_test = None
        if self.correlated and self.n_random_vars > 0:
            L_np = self._build_cholesky_L_numpy(theta_hat)
            cov_mat = L_np @ L_np.T
            sd_vec = np.sqrt(np.diag(cov_mat))
            # Avoid division by zero
            sd_outer = np.outer(sd_vec, sd_vec)
            sd_outer[sd_outer == 0] = 1.0
            cor_mat = cov_mat / sd_outer
            rand_names = list(self._random_param_names)

            if vcov is not None:
                cov_se, cor_se, cor_test = self._compute_cov_cor_inference(
                    theta_hat, vcov, cov_mat, cor_mat,
                )

        return EstimationResult(
            success=bool(opt.success),
            message=str(opt.message),
            log_likelihood=loglike,
            aic=float(2 * k - 2 * loglike),
            bic=float(np.log(self.n_obs) * k - 2 * loglike),
            n_parameters=k,
            n_observations=self.n_obs,
            n_individuals=self.n_individuals,
            optimizer_iterations=int(getattr(opt, "nit", 0)),
            runtime_seconds=float(runtime),
            estimates=estimates,
            vcov_matrix=vcov,
            covariance_matrix=cov_mat,
            correlation_matrix=cor_mat,
            random_param_names=rand_names,
            covariance_se=cov_se,
            correlation_se=cor_se,
            correlation_test=cor_test,
            raw_theta=theta_hat,
        )


class ConditionalLogitEstimator(MixedLogitEstimator):
    """Special case of mixed logit with all fixed coefficients."""

    def __init__(
        self,
        tensors: ChoiceTensors,
        variables: list[VariableSpec],
        device: torch.device | None = None,
        seed: int = 123,
        bws_data: BwsData | None = None,
    ) -> None:
        fixed_variables = [
            VariableSpec(name=v.name, column=v.column, distribution="fixed") for v in variables
        ]
        super().__init__(
            tensors=tensors,
            variables=fixed_variables,
            n_draws=1,
            device=device,
            seed=seed,
            bws_data=bws_data,
        )


class GmnlEstimator(MixedLogitEstimator):
    """GMNL with lognormal scale normalised to have population mean one.

    Taste beta has the form ``scale * mean + (gamma + (1 - gamma) * scale) * deviation``.
    Fixed gamma 1 gives GMNL-I and fixed gamma 0 gives GMNL-II. All-fixed
    tastes give S-MNL, in which gamma does not enter the likelihood.
    """

    def __init__(
        self,
        tensors: ChoiceTensors,
        variables: list[VariableSpec],
        n_draws: int = 200,
        device: torch.device | None = None,
        seed: int = 123,
        bws_data: BwsData | None = None,
        correlated: bool = False,
        correlation_groups: list[list[int]] | None = None,
        fixed_gamma: float | None = None,
    ) -> None:
        if fixed_gamma is not None and not 0.0 <= fixed_gamma <= 1.0:
            raise ValueError("fixed_gamma must be between 0 and 1.")
        super().__init__(
            tensors=tensors, variables=variables, n_draws=n_draws, device=device,
            seed=seed, correlated=correlated, correlation_groups=correlation_groups,
            bws_data=bws_data, scale_draws=True,
        )
        self._fixed_gamma = fixed_gamma if self.n_random_vars else 0.0
        self._sigma_tau_idx = self.n_params
        self.n_params += 1
        self._gamma_idx = None
        if self._fixed_gamma is None:
            self._gamma_idx = self.n_params
            self.n_params += 1

    def _initial_theta(self) -> np.ndarray:
        theta0 = super()._initial_theta()
        theta0[self._sigma_tau_idx] = -1.0
        return theta0

    def _betas_from_theta(self, theta: torch.Tensor) -> torch.Tensor:
        log_scale_sd = _positive(theta[self._sigma_tau_idx])
        scale = torch.exp(log_scale_sd * (self.scale_draws - 0.5 * log_scale_sd)).unsqueeze(2)
        base_betas = super()._betas_from_theta(theta)
        if self._fixed_gamma == 0.0:
            return scale * base_betas

        gamma = self._fixed_gamma if self._gamma_idx is None else torch.sigmoid(theta[self._gamma_idx])
        variances = self._build_cholesky_L(theta).square().sum(dim=1) if self.correlated else None
        means = []
        for param in self._param_defs:
            mean = theta[param["theta_mu_idx"]]
            if param["distribution"] == "lognormal":
                if variances is None:
                    variance = _positive(theta[param["theta_indices"][1]]).square()
                else:
                    variance = variances[param["draw_idx"]]
                mean = torch.exp(mean + 0.5 * variance)
            means.append(mean)

        mean = torch.stack(means)
        return scale * mean + (gamma + (1 - gamma) * scale) * (base_betas - mean)

    def _parameter_table(
        self, theta_hat: np.ndarray, vcov: np.ndarray | None = None,
    ) -> pd.DataFrame:
        # Delegate to parent for MMNL params (handles both independent & correlated)
        base_df = super()._parameter_table(theta_hat, vcov)
        rows = base_df.to_dict("records")

        # Append GMNL scale heterogeneity parameters
        raw_sigma_tau = theta_hat[self._sigma_tau_idx]
        sigma_tau_est = float(np.logaddexp(0.0, raw_sigma_tau))

        se_sigma_tau = float("nan")
        if vcov is not None:
            var_raw_st = vcov[self._sigma_tau_idx, self._sigma_tau_idx]
            deriv_st = self._softplus_derivative(raw_sigma_tau)
            se_sigma_tau = float(abs(deriv_st) * np.sqrt(max(var_raw_st, 0.0)))

        rows.append(self._make_row("sigma_tau (scale SD)", "scale", sigma_tau_est, se_sigma_tau, theta_index=self._sigma_tau_idx))

        if not self.n_random_vars:
            return pd.DataFrame(rows)

        if self._fixed_gamma is not None:
            # gamma is fixed — report as fixed value with no SE
            rows.append(self._make_row(
                f"gamma (fixed={self._fixed_gamma:.1f})", "scale",
                self._fixed_gamma, float("nan"), theta_index=-1,
            ))
        else:
            raw_gamma = theta_hat[self._gamma_idx]
            gamma_est = float(expit(raw_gamma))
            se_gamma = float("nan")
            if vcov is not None:
                var_raw_g = vcov[self._gamma_idx, self._gamma_idx]
                deriv_g = gamma_est * (1.0 - gamma_est)  # sigmoid derivative
                se_gamma = float(abs(deriv_g) * np.sqrt(max(var_raw_g, 0.0)))
            rows.append(self._make_row("gamma (mixing)", "scale", gamma_est, se_gamma, theta_index=self._gamma_idx))

        return pd.DataFrame(rows)

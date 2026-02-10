from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from .config import VariableSpec
from .data import ChoiceTensors

try:
    from .bws import BwsData, bws_log_prob, standard_log_prob
except ImportError:  # bws.py may not exist yet
    BwsData = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


@dataclass
class LatentClassResult:
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
    n_classes: int
    class_probabilities: list[float]
    class_estimates: pd.DataFrame
    posterior_probs: pd.DataFrame
    vcov_matrix: np.ndarray | None = field(default=None, repr=False)
    membership_estimates: pd.DataFrame | None = field(default=None)
    # Convergence diagnostics
    n_starts_attempted: int = 0
    n_starts_succeeded: int = 0
    all_start_lls: list[float] = field(default_factory=list)
    best_start_index: int = -1
    optimizer_method: str = "L-BFGS-B"
    # EM-specific diagnostics
    em_iterations: int = 0
    em_ll_history: list[float] = field(default_factory=list)
    em_converged: bool = False
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
            "n_classes": self.n_classes,
            "class_probabilities": self.class_probabilities,
            "n_starts_attempted": self.n_starts_attempted,
            "n_starts_succeeded": self.n_starts_succeeded,
            "all_start_lls": self.all_start_lls,
            "best_start_index": self.best_start_index,
            "optimizer_method": self.optimizer_method,
        }
        if self.vcov_matrix is not None:
            d["has_vcov"] = True
        has_se = "std_error" in self.estimates.columns and self.estimates["std_error"].notna().any()
        d["has_standard_errors"] = has_se
        if self.membership_estimates is not None:
            d["has_membership_covariates"] = True
        if self.em_iterations > 0:
            d["em_iterations"] = self.em_iterations
            d["em_ll_history"] = self.em_ll_history
            d["em_converged"] = self.em_converged
        return d


class LatentClassEstimator:
    """
    Latent Class logit estimator for panel choice data.

    Assumes Q discrete respondent segments, each with its own fixed
    coefficient vector.  Class membership probabilities are estimated
    via softmax of free parameters (gamma_1 = 0 for identification).

    Log-likelihood:
        LL = sum_n log( sum_q pi_q * prod_t P(y_nt | beta_q, X_nt) )
    """

    def __init__(
        self,
        tensors: ChoiceTensors,
        variables: list[VariableSpec],
        n_classes: int = 2,
        device: torch.device | None = None,
        seed: int = 123,
        membership_cols: list[str] | None = None,
        df: pd.DataFrame | None = None,
        id_col: str | None = None,
        bws_data: Any | None = None,
    ) -> None:
        if len(variables) != tensors.X.shape[2]:
            raise ValueError(
                "Variable count mismatch: number of VariableSpec entries must equal X.shape[2]."
            )
        if n_classes < 1:
            raise ValueError("n_classes must be >= 1.")

        self.device = device or tensors.X.device
        self.X = tensors.X.to(self.device).float()          # (n_obs, n_alts, n_vars)
        self.y = tensors.y.to(self.device).long()            # (n_obs,)
        self.panel_idx = tensors.panel_idx.to(self.device).long()  # (n_obs,)
        self.n_individuals = tensors.n_individuals
        self.n_obs = tensors.n_obs
        self.n_alts = tensors.n_alts
        self.variables = variables
        self.seed = seed
        self.n_classes = n_classes
        self.n_vars = self.X.shape[2]
        self.membership_cols = membership_cols or []
        self.n_membership_vars = len(self.membership_cols)

        # Build membership covariate matrix Z (n_individuals, M) if provided
        if self.membership_cols and df is not None and id_col is not None:
            self._build_membership_matrix(df, id_col, tensors.id_values)
        else:
            self.Z = None
            self.membership_cols = []
            self.n_membership_vars = 0

        # Parameter layout:
        #   beta_1 (K), ..., beta_Q (K), then membership params:
        #   Without covariates: gamma_2, ..., gamma_Q  -> (Q-1)
        #   With covariates:    (gamma_q, delta_q1, ..., delta_qM) for q=2..Q -> (Q-1)*(1+M)
        if self.n_membership_vars > 0:
            self.n_membership_params = max(self.n_classes - 1, 0) * (1 + self.n_membership_vars)
        else:
            self.n_membership_params = max(self.n_classes - 1, 0)
        self.n_params = self.n_classes * self.n_vars + self.n_membership_params

        # BWS support
        self._bws_data = bws_data
        self._bws_has_lambda_w = False
        self._lambda_w_idx = -1
        if bws_data is not None:
            self.y_worst = bws_data.y_worst.to(self.device).long()
            if bws_data.estimate_lambda_w:
                self._bws_has_lambda_w = True
                self._lambda_w_idx = self.n_params
                self.n_params += 1

    def _build_membership_matrix(
        self, df: pd.DataFrame, id_col: str, id_values: np.ndarray
    ) -> None:
        """Extract individual-level Z matrix from the data, validating constancy."""
        for col in self.membership_cols:
            if col not in df.columns:
                raise ValueError(f"Membership column '{col}' not found in data.")
            # Check that each respondent has a single value for this column
            n_unique_per_id = df.groupby(id_col)[col].nunique()
            bad = n_unique_per_id[n_unique_per_id > 1]
            if len(bad) > 0:
                raise ValueError(
                    f"Membership column '{col}' is not constant within respondent. "
                    f"Respondents with varying values: {bad.index.tolist()[:5]}"
                )

        # Build one row per individual, aligned with id_values ordering
        ind_df = df.groupby(id_col)[self.membership_cols].first().reindex(id_values)
        Z_np = ind_df.values.astype(np.float32)
        self.Z = torch.tensor(Z_np, dtype=torch.float32, device=self.device)  # (N, M)

    def _unpack_theta(
        self, theta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack theta into class betas and class probabilities.

        Returns
        -------
        betas : (Q, K)
        class_probs : (Q,) when no membership covariates,
                      (N, Q) when membership covariates are present
        """
        Q, K = self.n_classes, self.n_vars
        membership_start = Q * K
        membership_end = membership_start + self.n_membership_params

        betas = theta[: Q * K].reshape(Q, K)

        if Q == 1:
            class_probs = torch.ones(1, dtype=torch.float32, device=self.device)
        elif self.Z is not None and self.n_membership_vars > 0:
            # With covariates: membership params are (Q-1) blocks of (1+M)
            membership_params = theta[membership_start:membership_end]  # ((Q-1)*(1+M),)
            M = self.n_membership_vars
            # Reshape to (Q-1, 1+M): each row is [gamma_q, delta_q1, ..., delta_qM]
            membership_block = membership_params.reshape(Q - 1, 1 + M)
            gamma_free = membership_block[:, 0]       # (Q-1,)
            delta_free = membership_block[:, 1:]       # (Q-1, M)

            # Build full gamma and delta with class 1 as reference (zeros)
            gamma = torch.cat([
                torch.zeros(1, dtype=torch.float32, device=self.device),
                gamma_free,
            ])  # (Q,)
            delta = torch.cat([
                torch.zeros(1, M, dtype=torch.float32, device=self.device),
                delta_free,
            ], dim=0)  # (Q, M)

            # V_q(z_i) = gamma_q + delta_q' * z_i  -> (N, Q)
            V = gamma.unsqueeze(0) + self.Z @ delta.T  # (N, Q)
            class_probs = torch.softmax(V, dim=1)  # (N, Q)
        else:
            gamma_free = theta[membership_start:membership_end]  # (Q-1,)
            gamma = torch.cat([
                torch.zeros(1, dtype=torch.float32, device=self.device),
                gamma_free,
            ])
            class_probs = torch.softmax(gamma, dim=0)

        return betas, class_probs

    def _get_lambda_w(self, theta: torch.Tensor | None) -> float | torch.Tensor:
        """Return the worst-choice scale parameter lambda_w."""
        if self._bws_has_lambda_w and theta is not None:
            return torch.nn.functional.softplus(theta[self._lambda_w_idx]) + 1e-6
        return 1.0

    def _class_log_likelihoods(
        self, betas: torch.Tensor, theta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-individual, per-class log-likelihoods.

        Parameters
        ----------
        betas : (Q, K)
        theta : full parameter vector (needed for BWS lambda_w)

        Returns
        -------
        ll_individual : (n_individuals, Q)
            Sum of log-choice-probabilities across tasks for each individual
            under each class's beta.
        """
        Q = betas.shape[0]

        # Utility for each class: X @ beta_q^T
        # X: (n_obs, n_alts, K),  betas^T: (K, Q) -> utility: (n_obs, n_alts, Q)
        utility = torch.einsum("nak,qk->naq", self.X, betas)

        if self._bws_data is None:
            log_prob = standard_log_prob(utility, self.y, alt_dim=1)
        else:
            lambda_w = self._get_lambda_w(theta)
            log_prob = bws_log_prob(
                utility, self.y, self.y_worst, lambda_w, alt_dim=1,
            )

        # Aggregate per individual (panel): sum log-probs across tasks
        ll_individual = torch.zeros(
            self.n_individuals, Q, dtype=torch.float32, device=self.device,
        )
        ll_individual.index_add_(0, self.panel_idx, log_prob)

        return ll_individual

    def _neg_log_likelihood_tensor(self, theta: torch.Tensor) -> torch.Tensor:
        betas, class_probs = self._unpack_theta(theta)
        ll_individual = self._class_log_likelihoods(betas, theta=theta)  # (N, Q)

        # Mixture log-likelihood:
        # log sum_q pi_q(z_i) * exp(ll_individual_q)
        # = logsumexp( log(pi_q(z_i)) + ll_individual_q )
        log_pi = torch.log(class_probs + 1e-30)  # (Q,) or (N, Q)
        if log_pi.dim() == 1:
            log_pi = log_pi.unsqueeze(0)  # (1, Q) -> broadcasts over N
        log_mixture = torch.logsumexp(log_pi + ll_individual, dim=1)  # (N,)

        return -log_mixture.sum()

    def _objective_and_grad(self, theta_np: np.ndarray) -> tuple[float, np.ndarray]:
        theta = torch.tensor(
            theta_np,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        loss = self._neg_log_likelihood_tensor(theta)
        loss.backward()
        grad = theta.grad.detach().cpu().numpy().astype(np.float64)
        return float(loss.detach().cpu().item()), grad

    def _initial_theta(self, rng: np.random.Generator) -> np.ndarray:
        theta0 = rng.standard_normal(self.n_params).astype(np.float64) * 0.1
        if self._bws_has_lambda_w:
            theta0[self._lambda_w_idx] = 0.0  # softplus(0) ~ 0.69
        return theta0

    def _compute_posterior(self, theta_hat: np.ndarray) -> pd.DataFrame:
        """Compute posterior class membership probabilities for each individual."""
        theta = torch.tensor(theta_hat, dtype=torch.float32, device=self.device)
        betas, class_probs = self._unpack_theta(theta)
        ll_individual = self._class_log_likelihoods(betas, theta=theta)  # (N, Q)

        log_pi = torch.log(class_probs + 1e-30)  # (Q,) or (N, Q)
        if log_pi.dim() == 1:
            log_pi = log_pi.unsqueeze(0)  # (1, Q)
        log_numerator = log_pi + ll_individual  # (N, Q)
        log_denominator = torch.logsumexp(log_numerator, dim=1, keepdim=True)  # (N, 1)
        posterior = torch.exp(log_numerator - log_denominator)  # (N, Q)

        posterior_np = posterior.detach().cpu().numpy()
        columns = [f"class_{q + 1}" for q in range(self.n_classes)]
        return pd.DataFrame(posterior_np, columns=columns)

    def _parameter_table(self, theta_hat: np.ndarray) -> pd.DataFrame:
        """Build a flat parameter table (one row per parameter)."""
        Q, K = self.n_classes, self.n_vars
        theta = torch.tensor(theta_hat, dtype=torch.float32, device=self.device)
        _, class_probs = self._unpack_theta(theta)

        rows: list[dict[str, Any]] = []
        for q in range(Q):
            for k, var in enumerate(self.variables):
                rows.append({
                    "parameter": f"beta_{var.name}_class{q + 1}",
                    "class_id": q + 1,
                    "estimate": float(theta_hat[q * K + k]),
                })

        # Average class probabilities
        if class_probs.dim() == 2:
            pi_vals = class_probs.mean(dim=0).detach().cpu().numpy()
        else:
            pi_vals = class_probs.detach().cpu().numpy()
        for q in range(Q):
            rows.append({
                "parameter": f"pi_class{q + 1}",
                "class_id": q + 1,
                "estimate": float(pi_vals[q]),
            })

        if self._bws_has_lambda_w:
            raw_lw = theta_hat[self._lambda_w_idx]
            lw_val = float(np.logaddexp(0.0, raw_lw) + 1e-6)
            rows.append({
                "parameter": "lambda_w (worst scale)",
                "class_id": 0,
                "estimate": lw_val,
            })

        return pd.DataFrame(rows)

    def _membership_table(self, theta_hat: np.ndarray) -> pd.DataFrame | None:
        """Build membership coefficient table if covariates are present."""
        if not self.membership_cols or self.n_membership_vars == 0:
            return None

        Q, K = self.n_classes, self.n_vars
        M = self.n_membership_vars
        membership_params = theta_hat[Q * K : Q * K + self.n_membership_params]

        rows: list[dict[str, Any]] = []
        for q_idx in range(Q - 1):
            q = q_idx + 2  # class 2, 3, ...
            offset = q_idx * (1 + M)
            # Intercept (gamma)
            rows.append({
                "class_id": q,
                "variable": "_intercept",
                "estimate": float(membership_params[offset]),
            })
            # Covariate coefficients (delta)
            for m, col_name in enumerate(self.membership_cols):
                rows.append({
                    "class_id": q,
                    "variable": col_name,
                    "estimate": float(membership_params[offset + 1 + m]),
                })
        return pd.DataFrame(rows)

    def _class_estimates_table(self, theta_hat: np.ndarray) -> pd.DataFrame:
        """Build a class-by-variable table."""
        Q, K = self.n_classes, self.n_vars
        rows: list[dict[str, Any]] = []
        for q in range(Q):
            for k, var in enumerate(self.variables):
                rows.append({
                    "class_id": q + 1,
                    "parameter": var.name,
                    "estimate": float(theta_hat[q * K + k]),
                })
        return pd.DataFrame(rows)

    def fit(
        self,
        maxiter: int = 300,
        verbose: bool = False,
        n_starts: int = 10,
        method: str = "em",
        em_tol: float = 1e-6,
        initial_theta: list[float] | None = None,
    ) -> LatentClassResult:
        _custom = None
        if initial_theta is not None:
            _custom = np.asarray(initial_theta, dtype=np.float64)
            if len(_custom) != self.n_params:
                raise ValueError(
                    f"custom_start has {len(_custom)} values but model expects {self.n_params} parameters."
                )
        if method == "em":
            return self._fit_em(
                maxiter=maxiter, n_starts=n_starts, em_tol=em_tol, verbose=verbose,
                initial_theta=_custom,
            )
        return self._fit_direct(
            maxiter=maxiter, n_starts=n_starts, verbose=verbose,
            initial_theta=_custom,
        )

    # ── Direct (L-BFGS-B) fitting ────────────────────────────────────

    def _fit_direct(
        self,
        maxiter: int = 300,
        verbose: bool = False,
        n_starts: int = 10,
        initial_theta: np.ndarray | None = None,
    ) -> LatentClassResult:
        total_start_time = time.perf_counter()
        rng = np.random.default_rng(self.seed)

        best_opt = None
        best_nll = np.inf
        all_start_lls: list[float] = []
        n_succeeded = 0
        best_start_idx = -1

        for i_start in range(n_starts):
            if i_start == 0 and initial_theta is not None:
                theta0 = initial_theta.copy()
            else:
                theta0 = self._initial_theta(rng)
            cache: dict[str, np.ndarray | float] = {}

            def evaluate(theta: np.ndarray, _cache: dict = cache) -> tuple[float, np.ndarray]:
                x = np.asarray(theta, dtype=np.float64)
                cached_x = _cache.get("x")
                if cached_x is None or not np.array_equal(cached_x, x):
                    value, grad = self._objective_and_grad(x)
                    _cache["x"] = x.copy()
                    _cache["value"] = value
                    _cache["grad"] = grad
                return float(_cache["value"]), np.asarray(_cache["grad"])

            try:
                opt = minimize(
                    fun=lambda x: evaluate(x)[0],
                    x0=theta0,
                    jac=lambda x: evaluate(x)[1],
                    method="L-BFGS-B",
                    options={"maxiter": maxiter, "disp": verbose},
                )
            except Exception as exc:
                logger.warning("Start %d failed: %s", i_start, exc)
                all_start_lls.append(float("nan"))
                continue

            nll = float(opt.fun)
            n_succeeded += 1
            all_start_lls.append(-nll)  # store as log-likelihood (positive direction)
            if nll < best_nll:
                best_nll = nll
                best_opt = opt
                best_start_idx = i_start

            logger.debug("Start %d/%d  NLL=%.4f  best=%.4f", i_start + 1, n_starts, nll, best_nll)

        total_runtime = time.perf_counter() - total_start_time

        if best_opt is None:
            return LatentClassResult(
                success=False,
                message="All random starts failed.",
                log_likelihood=float("nan"),
                aic=float("nan"),
                bic=float("nan"),
                n_parameters=self.n_params,
                n_observations=self.n_obs,
                n_individuals=self.n_individuals,
                optimizer_iterations=0,
                runtime_seconds=total_runtime,
                estimates=pd.DataFrame(),
                n_classes=self.n_classes,
                class_probabilities=[],
                class_estimates=pd.DataFrame(),
                posterior_probs=pd.DataFrame(),
                n_starts_attempted=n_starts,
                n_starts_succeeded=0,
                all_start_lls=all_start_lls,
                optimizer_method="L-BFGS-B",
            )

        theta_hat = np.asarray(best_opt.x)
        loglike = -best_nll
        k = self.n_params

        theta_t = torch.tensor(theta_hat, dtype=torch.float32, device=self.device)
        _, class_probs = self._unpack_theta(theta_t)
        # class_probs is (Q,) without covariates or (N, Q) with covariates
        if class_probs.dim() == 2:
            pi_list = class_probs.mean(dim=0).detach().cpu().tolist()
        else:
            pi_list = class_probs.detach().cpu().tolist()

        estimates = self._parameter_table(theta_hat)
        class_est = self._class_estimates_table(theta_hat)
        posterior = self._compute_posterior(theta_hat)
        membership_est = self._membership_table(theta_hat)

        return LatentClassResult(
            success=bool(best_opt.success),
            message=str(best_opt.message),
            log_likelihood=loglike,
            aic=float(2 * k - 2 * loglike),
            bic=float(np.log(self.n_obs) * k - 2 * loglike),
            n_parameters=k,
            n_observations=self.n_obs,
            n_individuals=self.n_individuals,
            optimizer_iterations=int(getattr(best_opt, "nit", 0)),
            runtime_seconds=total_runtime,
            estimates=estimates,
            n_classes=self.n_classes,
            class_probabilities=pi_list,
            class_estimates=class_est,
            posterior_probs=posterior,
            membership_estimates=membership_est,
            n_starts_attempted=n_starts,
            n_starts_succeeded=n_succeeded,
            all_start_lls=all_start_lls,
            best_start_index=best_start_idx,
            optimizer_method="L-BFGS-B",
            raw_theta=theta_hat,
        )

    # ── EM algorithm ──────────────────────────────────────────────────

    def _e_step(
        self, theta: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """E-step: compute posterior weights w_nq and log-likelihood.

        Returns
        -------
        weights : (N, Q) posterior class membership probabilities
        ll      : total log-likelihood
        """
        theta_t = torch.tensor(theta, dtype=torch.float32, device=self.device)
        betas, class_probs = self._unpack_theta(theta_t)
        ll_individual = self._class_log_likelihoods(betas, theta=theta_t)  # (N, Q)

        log_pi = torch.log(class_probs + 1e-30)  # (Q,) or (N, Q)
        if log_pi.dim() == 1:
            log_pi = log_pi.unsqueeze(0)  # (1, Q) -> broadcasts over N

        log_numerator = log_pi + ll_individual  # (N, Q)
        log_denominator = torch.logsumexp(log_numerator, dim=1, keepdim=True)  # (N, 1)
        weights = torch.exp(log_numerator - log_denominator)  # (N, Q)

        ll = float(log_denominator.sum().detach().cpu().item())

        return weights.detach().cpu().numpy(), ll

    def _e_step_torch(self, theta: torch.Tensor) -> tuple[torch.Tensor, float]:
        """E-step on GPU. Returns weights as Tensor (stays on device)."""
        with torch.no_grad():
            betas, class_probs = self._unpack_theta(theta)
            ll_individual = self._class_log_likelihoods(betas, theta=theta)  # (N, Q)

            log_pi = torch.log(class_probs + 1e-30)
            if log_pi.dim() == 1:
                log_pi = log_pi.unsqueeze(0)

            log_numerator = log_pi + ll_individual
            log_denominator = torch.logsumexp(log_numerator, dim=1, keepdim=True)
            weights = torch.exp(log_numerator - log_denominator)

            ll = float(log_denominator.sum().item())
        return weights, ll

    def _m_step_one_class(
        self,
        weights_obs_q: np.ndarray,
        beta_q_init: np.ndarray,
        lambda_w: float,
        maxiter: int = 5,
    ) -> np.ndarray:
        """M-step: optimize betas for one class using weighted CL via L-BFGS-B.

        Parameters
        ----------
        weights_obs_q : (N,) posterior weights for this class (individual-level)
        beta_q_init   : (K,) initial beta vector
        lambda_w      : current lambda_w value (fixed during this step)
        maxiter       : max inner iterations (low for GEM: only need to improve)
        """
        K = self.n_vars
        w_tensor = torch.tensor(weights_obs_q, dtype=torch.float32, device=self.device)  # (N,)
        # Expand weights from individual-level to observation-level
        w_obs = w_tensor[self.panel_idx]  # (n_obs,)
        lw_scalar = lambda_w

        def obj_and_grad(beta_np: np.ndarray) -> tuple[float, np.ndarray]:
            beta = torch.tensor(
                beta_np.reshape(1, K), dtype=torch.float32, device=self.device,
                requires_grad=True,
            )
            # utility: (n_obs, n_alts)
            utility = torch.einsum("nak,qk->naq", self.X, beta).squeeze(-1)  # (n_obs, n_alts)

            if self._bws_data is None:
                lp = standard_log_prob(utility, self.y, alt_dim=1)  # (n_obs,)
            else:
                lp = bws_log_prob(
                    utility, self.y, self.y_worst, lw_scalar, alt_dim=1,
                )  # (n_obs,)

            # Weighted negative log-likelihood (sum over obs, weighted by posterior)
            nll = -(w_obs * lp).sum()
            nll.backward()
            return float(nll.detach().cpu().item()), beta.grad.detach().cpu().numpy().flatten().astype(np.float64)

        result = minimize(
            fun=lambda x: obj_and_grad(x)[0],
            x0=beta_q_init.astype(np.float64),
            jac=lambda x: obj_and_grad(x)[1],
            method="L-BFGS-B",
            options={"maxiter": maxiter, "disp": False},
        )
        return result.x

    def _m_step_betas_torch(
        self,
        weights: torch.Tensor,
        betas_init: torch.Tensor,
        lambda_w: float | torch.Tensor,
        maxiter: int = 5,
    ) -> torch.Tensor:
        """Vectorized M-step for all Q class betas using torch.optim.LBFGS.

        In EM, the M-step only needs to *improve* the objective (GEM principle),
        so we use few inner iterations (default 5) for speed.
        """
        betas = betas_init.clone().detach().requires_grad_(True)
        w_obs = weights[self.panel_idx]  # (n_obs, Q)

        # Reuse LBFGS optimizer stored on the instance to avoid per-iteration
        # allocation overhead.  We create it once and re-attach the parameter.
        if not hasattr(self, "_m_step_lbfgs") or self._m_step_lbfgs_maxiter != maxiter:
            self._m_step_lbfgs = torch.optim.LBFGS(
                [betas], max_iter=maxiter, line_search_fn="strong_wolfe",
                tolerance_grad=1e-7, tolerance_change=1e-9,
            )
            self._m_step_lbfgs_maxiter = maxiter
        else:
            # Re-attach the new betas parameter
            self._m_step_lbfgs.param_groups[0]["params"] = [betas]
            self._m_step_lbfgs.state.clear()

        optimizer = self._m_step_lbfgs

        def closure():
            optimizer.zero_grad()
            utility = torch.einsum("nak,qk->naq", self.X, betas)

            if self._bws_data is None:
                lp = standard_log_prob(utility, self.y, alt_dim=1)  # (n_obs, Q)
            else:
                lp = bws_log_prob(
                    utility, self.y, self.y_worst, lambda_w, alt_dim=1,
                )

            nll = -(w_obs * lp).sum()
            nll.backward()
            return nll

        optimizer.step(closure)
        return betas.detach()

    def _m_step_membership_params(
        self, weights: np.ndarray, gamma_init: np.ndarray, maxiter: int = 5,
    ) -> np.ndarray:
        """M-step for membership parameters with covariates.

        Fits multinomial logit: maximize sum_n sum_q w_nq * log(pi_q(z_n)).
        Uses few inner iterations (GEM principle).
        """
        Q = self.n_classes
        M = self.n_membership_vars
        n_mem_params = (Q - 1) * (1 + M)
        w_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)  # (N, Q)

        def obj_and_grad(gamma_np: np.ndarray) -> tuple[float, np.ndarray]:
            gamma = torch.tensor(
                gamma_np, dtype=torch.float32, device=self.device, requires_grad=True,
            )
            # Rebuild class probs: same logic as _unpack_theta membership section
            membership_block = gamma.reshape(Q - 1, 1 + M)
            gamma_free = membership_block[:, 0]
            delta_free = membership_block[:, 1:]

            gamma_full = torch.cat([
                torch.zeros(1, dtype=torch.float32, device=self.device),
                gamma_free,
            ])
            delta_full = torch.cat([
                torch.zeros(1, M, dtype=torch.float32, device=self.device),
                delta_free,
            ], dim=0)

            V = gamma_full.unsqueeze(0) + self.Z @ delta_full.T  # (N, Q)
            log_pi = torch.log_softmax(V, dim=1)  # (N, Q)

            # Objective: maximize sum_n sum_q w_nq * log(pi_q)
            nll = -(w_tensor * log_pi).sum()
            nll.backward()
            return float(nll.detach().cpu().item()), gamma.grad.detach().cpu().numpy().astype(np.float64)

        result = minimize(
            fun=lambda x: obj_and_grad(x)[0],
            x0=gamma_init.astype(np.float64),
            jac=lambda x: obj_and_grad(x)[1],
            method="L-BFGS-B",
            options={"maxiter": maxiter, "disp": False},
        )
        return result.x

    def _m_step_membership_torch(
        self, weights: torch.Tensor, gamma_init: torch.Tensor, maxiter: int = 1,
    ) -> torch.Tensor:
        """M-step for membership parameters using torch.optim.LBFGS.

        Uses a single inner iteration (GEM principle: only need to improve, not converge).
        """
        Q = self.n_classes
        M = self.n_membership_vars
        gamma = gamma_init.clone().detach().requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [gamma], max_iter=maxiter, line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            membership_block = gamma.reshape(Q - 1, 1 + M)
            gamma_free = membership_block[:, 0]
            delta_free = membership_block[:, 1:]

            gamma_full = torch.cat([
                torch.zeros(1, dtype=torch.float32, device=self.device),
                gamma_free,
            ])
            delta_full = torch.cat([
                torch.zeros(1, M, dtype=torch.float32, device=self.device),
                delta_free,
            ], dim=0)

            V = gamma_full.unsqueeze(0) + self.Z @ delta_full.T
            log_pi = torch.log_softmax(V, dim=1)
            nll = -(weights * log_pi).sum()
            nll.backward()
            return nll

        optimizer.step(closure)
        return gamma.detach()

    def _m_step_lambda_w_raw(
        self, weights: np.ndarray, theta: np.ndarray, maxiter: int = 3,
    ) -> float:
        """M-step for BWS lambda_w: optimize holding betas/class_probs fixed.

        Uses few inner iterations (GEM principle).
        """
        w_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)  # (N, Q)
        # We optimize only the raw lambda_w parameter
        raw_init = np.array([theta[self._lambda_w_idx]], dtype=np.float64)

        def obj_and_grad(raw_np: np.ndarray) -> tuple[float, np.ndarray]:
            # Build a temporary theta with updated lambda_w
            theta_tmp = theta.copy()
            theta_tmp[self._lambda_w_idx] = raw_np[0]

            theta_t = torch.tensor(
                theta_tmp, dtype=torch.float32, device=self.device,
            )
            # Only need gradient w.r.t. lambda_w param
            theta_t.requires_grad_(True)

            betas, class_probs = self._unpack_theta(theta_t)
            ll_individual = self._class_log_likelihoods(betas, theta=theta_t)  # (N, Q)

            log_pi = torch.log(class_probs + 1e-30)
            if log_pi.dim() == 1:
                log_pi = log_pi.unsqueeze(0)
            log_mixture = torch.logsumexp(log_pi + ll_individual, dim=1)  # (N,)

            nll = -log_mixture.sum()
            nll.backward()
            grad_lw = theta_t.grad[self._lambda_w_idx].detach().cpu().numpy().astype(np.float64)
            return float(nll.detach().cpu().item()), np.array([grad_lw])

        result = minimize(
            fun=lambda x: obj_and_grad(x)[0],
            x0=raw_init,
            jac=lambda x: obj_and_grad(x)[1],
            method="L-BFGS-B",
            options={"maxiter": maxiter, "disp": False},
        )
        return float(result.x[0])

    def _m_step_lambda_w_torch(
        self, theta: torch.Tensor, maxiter: int = 1,
    ) -> float:
        """M-step for BWS lambda_w using torch.optim.LBFGS.

        Uses a single inner iteration (GEM principle: only need to improve, not converge).
        """
        raw_lw = theta[self._lambda_w_idx].clone().detach().requires_grad_(True)

        # Pre-compute fixed quantities
        with torch.no_grad():
            betas = theta[:self.n_classes * self.n_vars].reshape(self.n_classes, self.n_vars)
            _, class_probs = self._unpack_theta(theta)

        optimizer = torch.optim.LBFGS(
            [raw_lw], max_iter=maxiter, line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            lambda_w = torch.nn.functional.softplus(raw_lw) + 1e-6

            utility = torch.einsum("nak,qk->naq", self.X, betas)
            lp = bws_log_prob(utility, self.y, self.y_worst, lambda_w, alt_dim=1)

            ll_individual = torch.zeros(
                self.n_individuals, self.n_classes, dtype=torch.float32, device=self.device,
            )
            ll_individual.index_add_(0, self.panel_idx, lp)

            log_pi = torch.log(class_probs + 1e-30)
            if log_pi.dim() == 1:
                log_pi = log_pi.unsqueeze(0)
            log_mixture = torch.logsumexp(log_pi + ll_individual, dim=1)
            nll = -log_mixture.sum()
            nll.backward()
            return nll

        optimizer.step(closure)
        return float(raw_lw.detach().item())

    def _fit_em(
        self,
        maxiter: int = 300,
        n_starts: int = 10,
        em_tol: float = 1e-6,
        verbose: bool = False,
        initial_theta: np.ndarray | None = None,
    ) -> LatentClassResult:
        """Fit the latent class model using the EM algorithm with multiple random starts."""
        total_start_time = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        Q, K = self.n_classes, self.n_vars

        best_theta_np: np.ndarray | None = None
        best_ll = -np.inf
        best_em_iters = 0
        best_em_history: list[float] = []
        best_em_converged = False
        all_start_lls: list[float] = []
        n_succeeded = 0
        best_start_idx = -1

        for i_start in range(n_starts):
            if i_start == 0 and initial_theta is not None:
                theta_np = initial_theta.copy()
            else:
                theta_np = self._initial_theta(rng)

            # Move theta to device as torch tensor for the entire EM loop
            theta = torch.tensor(theta_np, dtype=torch.float32, device=self.device)

            try:
                ll_history: list[float] = []
                converged = False

                for em_iter in range(maxiter):
                    # ── E-step (all on GPU) ──────────────────────
                    weights, ll = self._e_step_torch(theta)
                    ll_history.append(ll)

                    if verbose:
                        logger.info(
                            "Start %d  EM iter %d  LL=%.6f", i_start, em_iter, ll,
                        )

                    # ── Convergence check (require >= 10 iters first) ──
                    if em_iter >= 10 and abs(ll - ll_history[-2]) < em_tol:
                        converged = True
                        break

                    # ── Early abandon: skip hopeless starts ──────
                    if em_iter == 20 and best_ll > -np.inf and ll < best_ll - 50:
                        logger.debug(
                            "Start %d abandoned early: LL=%.2f vs best=%.2f",
                            i_start, ll, best_ll,
                        )
                        break

                    # ── M-step: class-specific betas (vectorized on GPU) ──
                    betas_current = theta[: Q * K].reshape(Q, K)
                    lambda_w_val = self._get_lambda_w(theta)

                    betas_new = self._m_step_betas_torch(
                        weights, betas_current, lambda_w_val,
                    )
                    theta = theta.clone()
                    theta[: Q * K] = betas_new.flatten()

                    # ── M-step: class membership probabilities ────
                    membership_start = Q * K
                    membership_end = membership_start + self.n_membership_params

                    if self.Z is not None and self.n_membership_vars > 0:
                        gamma_init = theta[membership_start:membership_end]
                        gamma_new = self._m_step_membership_torch(
                            weights, gamma_init,
                        )
                        theta = theta.clone()
                        theta[membership_start:membership_end] = gamma_new
                    elif Q > 1:
                        # Analytic update (all torch, no numpy)
                        pi_q = weights.mean(dim=0)  # (Q,)
                        pi_q = torch.clamp(pi_q, min=1e-10)
                        pi_q = pi_q / pi_q.sum()
                        gamma_free = torch.log(pi_q[1:] / pi_q[0])  # (Q-1,)
                        theta = theta.clone()
                        theta[membership_start:membership_end] = gamma_free

                    # ── M-step: lambda_w (BWS only, on GPU) ──────
                    if self._bws_has_lambda_w:
                        raw_lw = self._m_step_lambda_w_torch(theta)
                        theta = theta.clone()
                        theta[self._lambda_w_idx] = raw_lw

                final_ll = ll_history[-1] if ll_history else -np.inf
                n_succeeded += 1
                all_start_lls.append(final_ll)

                if final_ll > best_ll:
                    best_ll = final_ll
                    best_theta_np = theta.detach().cpu().numpy().astype(np.float64)
                    best_em_iters = len(ll_history)
                    best_em_history = list(ll_history)
                    best_em_converged = converged
                    best_start_idx = i_start

                logger.debug(
                    "Start %d/%d  EM iters=%d  LL=%.4f  converged=%s  best=%.4f",
                    i_start + 1, n_starts, len(ll_history), final_ll, converged, best_ll,
                )

            except Exception as exc:
                logger.warning("EM start %d failed: %s", i_start, exc)
                all_start_lls.append(float("nan"))
                continue

        total_runtime = time.perf_counter() - total_start_time

        if best_theta_np is None:
            return LatentClassResult(
                success=False,
                message="All EM random starts failed.",
                log_likelihood=float("nan"),
                aic=float("nan"),
                bic=float("nan"),
                n_parameters=self.n_params,
                n_observations=self.n_obs,
                n_individuals=self.n_individuals,
                optimizer_iterations=0,
                runtime_seconds=total_runtime,
                estimates=pd.DataFrame(),
                n_classes=self.n_classes,
                class_probabilities=[],
                class_estimates=pd.DataFrame(),
                posterior_probs=pd.DataFrame(),
                n_starts_attempted=n_starts,
                n_starts_succeeded=0,
                all_start_lls=all_start_lls,
                optimizer_method="EM",
            )

        theta_hat = best_theta_np
        k = self.n_params

        theta_t = torch.tensor(theta_hat, dtype=torch.float32, device=self.device)
        _, class_probs = self._unpack_theta(theta_t)
        if class_probs.dim() == 2:
            pi_list = class_probs.mean(dim=0).detach().cpu().tolist()
        else:
            pi_list = class_probs.detach().cpu().tolist()

        estimates = self._parameter_table(theta_hat)
        class_est = self._class_estimates_table(theta_hat)
        posterior = self._compute_posterior(theta_hat)
        membership_est = self._membership_table(theta_hat)

        return LatentClassResult(
            success=best_em_converged,
            message="EM converged" if best_em_converged else "EM reached max iterations",
            log_likelihood=best_ll,
            aic=float(2 * k - 2 * best_ll),
            bic=float(np.log(self.n_obs) * k - 2 * best_ll),
            n_parameters=k,
            n_observations=self.n_obs,
            n_individuals=self.n_individuals,
            optimizer_iterations=best_em_iters,
            runtime_seconds=total_runtime,
            estimates=estimates,
            n_classes=self.n_classes,
            class_probabilities=pi_list,
            class_estimates=class_est,
            posterior_probs=posterior,
            membership_estimates=membership_est,
            n_starts_attempted=n_starts,
            n_starts_succeeded=n_succeeded,
            all_start_lls=all_start_lls,
            best_start_index=best_start_idx,
            optimizer_method="EM",
            em_iterations=best_em_iters,
            em_ll_history=best_em_history,
            em_converged=best_em_converged,
            raw_theta=theta_hat,
        )

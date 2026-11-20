"""Regression checks for estimation units, GMNL scales and EM fit statistics."""
import numpy as np
import pytest
import torch

from dce_analyzer.config import ModelSpec, VariableSpec
from dce_analyzer.data import prepare_choice_tensors
from dce_analyzer.latent_class import LatentClassEstimator
from dce_analyzer.model import ConditionalLogitEstimator, GmnlEstimator, MixedLogitEstimator
from dce_analyzer.simulate import generate_simulated_dce


CPU = torch.device("cpu")


@pytest.fixture
def choices():
    return generate_simulated_dce(n_individuals=80, n_tasks=5, n_alts=3, seed=42).data


def make_tensors(df, variables):
    spec = ModelSpec("respondent_id", "task_id", "alternative", "choice", variables)
    return prepare_choice_tensors(df, spec, CPU)


def test_conditional_logit_is_invariant_to_measurement_units(choices):
    variables = [VariableSpec(c, c) for c in ("price", "time", "comfort")]
    original = ConditionalLogitEstimator(make_tensors(choices, variables), variables)
    units = np.array([0.0001, 10000.0, 1.0])
    scaled = choices.assign(price=choices.price * units[0], time=choices.time * units[1])
    changed = ConditionalLogitEstimator(make_tensors(scaled, variables), variables)
    start = np.array([-0.05, -0.03, 0.0])
    fit1 = original.fit(initial_theta=start.tolist())
    fit2 = changed.fit(initial_theta=(start / units).tolist())

    assert fit1.success, fit1.message
    assert fit2.success, fit2.message
    assert fit1.log_likelihood == pytest.approx(fit2.log_likelihood, abs=0.002)
    np.testing.assert_allclose(fit2.raw_theta * units, fit1.raw_theta, rtol=0.004, atol=0.001)
    np.testing.assert_allclose(fit2.vcov_matrix * np.outer(units, units), fit1.vcov_matrix, rtol=0.008)


def test_latent_class_em_reports_likelihood_at_returned_parameters(choices):
    variables = [VariableSpec(c, c) for c in ("price", "time", "comfort")]
    estimator = LatentClassEstimator(make_tensors(choices, variables), variables, n_classes=2, device=CPU)
    fit = estimator.fit(maxiter=1, n_starts=2)
    ll = -estimator._neg_log_likelihood_tensor(torch.tensor(fit.raw_theta, dtype=torch.float32)).item()
    assert fit.log_likelihood == pytest.approx(ll, abs=1e-5)
    assert fit.em_ll_history == pytest.approx([ll], abs=1e-5)
    assert fit.aic == pytest.approx(2 * fit.n_parameters - 2 * ll)
    assert fit.bic == pytest.approx(np.log(fit.n_observations) * fit.n_parameters - 2 * ll)


@pytest.mark.parametrize("fixed_gamma", [None, 0.0, 1.0])
def test_gmnl_variants_match_independent_utility_formula(choices, fixed_gamma):
    variables = [VariableSpec("price", "price", "normal"), VariableSpec("time", "time")]
    model = GmnlEstimator(make_tensors(choices, variables), variables, n_draws=160,
                          fixed_gamma=fixed_gamma, seed=42)
    theta = torch.tensor(model._initial_theta(), dtype=torch.float32)
    theta[0], theta[1] = -1.6, -0.4
    beta = MixedLogitEstimator._betas_from_theta(model, theta)
    scale_sd = torch.nn.functional.softplus(theta[3])
    scale = torch.exp(-0.5 * scale_sd.square() + scale_sd * model.scale_draws)
    gamma = 0.5 if fixed_gamma is None else fixed_gamma
    mean = theta[:2]
    expected = scale[..., None] * mean + (gamma + (1 - gamma) * scale[..., None]) * (beta - mean)

    assert model.n_params == (5 if fixed_gamma is None else 4)
    torch.testing.assert_close(model._betas_from_theta(theta), expected)
    assert scale.mean().item() == pytest.approx(1, abs=0.005)
    assert abs(np.corrcoef(model.draws.flatten(), model.scale_draws.flatten())[0, 1]) < 0.02


def test_s_mnl_integrates_scale_even_without_random_tastes(choices):
    variables = [VariableSpec("price", "price")]
    model = GmnlEstimator(make_tensors(choices, variables), variables, n_draws=60)
    parameters = torch.tensor([-0.7, -0.8])
    beta = model._betas_from_theta(parameters)
    assert model.n_params == 2
    assert model.n_draws == 60
    assert torch.all(beta.std(dim=1) > 0)
    assert "gamma (mixing)" not in model._parameter_table(parameters.numpy())["parameter"].tolist()


def test_mixed_logit_returns_parameters_in_original_units(choices):
    variables = [VariableSpec("price", "price", "normal"),
                 VariableSpec("time", "time", "lognormal")]
    data = choices.assign(price=choices.price * 0.0001, time=-choices.time * 10000)
    estimator = MixedLogitEstimator(make_tensors(data, variables), variables,
                                    n_draws=40, correlated=True, seed=42)
    fit = estimator.fit(maxiter=100)
    theta = torch.tensor(fit.raw_theta, dtype=torch.float32)
    recomputed = -estimator._neg_log_likelihood_tensor(theta).item()
    assert fit.log_likelihood == pytest.approx(recomputed, abs=0.005)

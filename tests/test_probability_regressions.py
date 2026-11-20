"""Regression checks for model predictions and held-out scoring."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.special import expit

from dce_analyzer.config import FullModelSpec, ModelSpec, VariableSpec
from dce_analyzer.cross_validation import cross_validate
from dce_analyzer.pipeline import build_estimator
from dce_analyzer.prediction import choice_probabilities, predict_entry
from dce_analyzer.simulate import generate_simulated_dce


CPU = torch.device("cpu")


def test_choice_shares_follow_alternative_labels_across_tasks():
    frame = pd.DataFrame({
        "id": ["one"] * 4, "task": [1, 1, 2, 2], "alt": ["A", "B", "B", "C"],
        "choice": ["A", "A", "C", "C"], "x": [1, 0, 0, 1],
    })
    spec = ModelSpec("id", "task", "alt", "choice", [VariableSpec("x", "x")])
    fit = SimpleNamespace(success=True, raw_theta=np.array([1.0]))
    shares = predict_entry(frame, {"model_type": "conditional", "spec": spec, "estimation": fit})
    assert shares["Alternative"].tolist() == ["A", "B", "C"]
    np.testing.assert_allclose(shares["Actual Share"], [0.5, 0.0, 0.5])
    np.testing.assert_allclose(shares["Predicted Share"], [expit(1) / 2, expit(-1), expit(1) / 2])


def test_prediction_averages_probabilities_over_mixed_tastes():
    frame = pd.DataFrame({
        "id": [1, 1], "task": [1, 1], "alt": ["A", "B"], "choice": [1, 0], "x": [1, 0],
    })
    spec = ModelSpec("id", "task", "alt", "choice", [VariableSpec("x", "x", "normal")], n_draws=2)
    estimator, _ = build_estimator(frame, spec, model_type="mixed", device=CPU)
    estimator.draws = torch.tensor([[[-2.0], [2.0]]])
    theta = torch.tensor([1.0, np.log(np.expm1(1.0))], dtype=torch.float32)
    expected = (expit(-1) + expit(3)) / 2
    probability = float(choice_probabilities(estimator, theta)[0, 0])
    np.testing.assert_allclose(probability, expected)
    assert abs(probability - expit(1)) > 0.1


def test_latent_class_predictions_use_each_respondents_membership_covariates():
    frame = pd.DataFrame({
        "id": [1, 1, 2, 2], "task": [1] * 4, "alt": ["A", "B"] * 2,
        "choice": [1, 0, 0, 1], "x": [1, 0] * 2, "z": [-2, -2, 2, 2],
    })
    spec = ModelSpec("id", "task", "alt", "choice", [VariableSpec("x", "x")], membership_cols=["z"])
    estimator, _ = build_estimator(frame, spec, model_type="latent_class", device=CPU)
    theta = torch.tensor([2.0, -2.0, 0.0, 1.0])
    weights = expit(np.array([-2, 2]))
    expected = expit(2) * (1 - weights) + expit(-2) * weights
    np.testing.assert_allclose(choice_probabilities(estimator, theta)[:, 0], expected, rtol=1e-6)


def test_cv_keeps_pure_scale_custom_start_for_each_fold():
    frame = generate_simulated_dce(n_individuals=100, n_tasks=6, n_alts=3, seed=81).data
    spec = FullModelSpec(
        "respondent_id", "task_id", "alternative", "choice",
        [VariableSpec("price", "price", "normal"), VariableSpec("time", "time", "normal")],
        model_type="gmnl", gmnl_variant="s-mnl", n_draws=30, custom_start=[0.0, 0.0, -1.0],
    )
    result = cross_validate(frame, spec, k=2, seed=24, device=CPU)
    assert all(fold.converged for fold in result.fold_results), result.fold_results
    assert np.isfinite(result.mean_test_ll)
    assert 0 <= result.mean_hit_rate <= 1

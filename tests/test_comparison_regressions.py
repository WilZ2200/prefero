"""Checks for estimating relative scale and restricting likelihood-ratio inference."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from dce_analyzer.config import ModelSpec, VariableSpec
from dce_analyzer.tests import conditional_lr, swait_louviere_test


SPEC = ModelSpec('id', 'task', 'alt', 'chosen', [VariableSpec(c, c) for c in ('x', 'z')])


def make_group_choices():
    rng = np.random.default_rng(51007)
    n, alternatives = 10000, 3
    X = rng.normal(size=(n, alternatives, 2))
    group = np.arange(n) // (n // 2)
    utility = (X @ np.array([-0.9, 0.65])) * (1 + 0.75 * group[:, None])
    selected = (utility + rng.gumbel(size=utility.shape)).argmax(axis=1)
    return pd.DataFrame({
        'id': np.repeat(np.arange(n), alternatives), 'task': 1,
        'alt': np.tile(np.arange(alternatives), n), 'group': np.repeat(group, alternatives),
        'chosen': (selected[:, None] == np.arange(alternatives)).ravel(),
        'x': X[:, :, 0].ravel(), 'z': X[:, :, 1].ravel(),
    })


def test_pooling_recovers_different_scales_for_shared_preferences():
    fit = swait_louviere_test(make_group_choices(), SPEC, 'group')
    assert fit.df == 1
    assert fit.relative_scale == pytest.approx(1.75, abs=0.18)
    assert fit.ll_equal_scale <= fit.ll_pooled <= fit.ll_group1 + fit.ll_group2
    assert fit.scale_p_value < 0.001
    assert fit.p_value > 0.05


def test_pooling_drops_a_whole_respondent_with_missing_group():
    df = make_group_choices()
    df['group'] = df['group'].astype(float)
    df.loc[df.index[0], 'group'] = np.nan
    dropped = swait_louviere_test(df, SPEC, 'group')
    removed = swait_louviere_test(df[df['id'] != 0], SPEC, 'group')
    assert dropped == removed


def comparison_entry(columns, log_likelihood, *, model='conditional', success=True):
    spec = ModelSpec('id', 'task', 'alt', 'chosen', [VariableSpec(c, c) for c in columns])
    estimation = SimpleNamespace(n_parameters=len(columns), n_observations=100,
                                 n_individuals=100, log_likelihood=log_likelihood,
                                 success=success)
    return {'model_type': model, 'estimation': estimation, 'spec': spec}


def test_conditional_lr_requires_converged_nested_single_choice_models():
    restricted = comparison_entry(['x'], -60)
    unrestricted = comparison_entry(['x', 'z'], -55)
    assert conditional_lr(restricted, unrestricted) == pytest.approx((10, 1, chi2.sf(10, 1)))
    with pytest.raises(ValueError, match='nested conditional logit'):
        conditional_lr(restricted, comparison_entry(['x', 'z'], -55, model='mixed'))
    with pytest.raises(ValueError, match='converged'):
        conditional_lr(restricted, comparison_entry(['x', 'z'], -55, success=False))
    with pytest.raises(ValueError, match='retain each term'):
        conditional_lr(restricted, comparison_entry(['z', 'v'], -55))

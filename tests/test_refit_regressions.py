from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from dce_analyzer.bootstrap import _class_order, _estimates_in_class_order, run_bootstrap
from dce_analyzer.config import DummyCoding, FullModelSpec, InteractionTerm, VariableSpec
from dce_analyzer.pipeline import estimate_from_spec
from dce_analyzer.simulate import generate_simulated_dce


def test_bootstrap_reuses_fitted_dummy_and_interaction_specification():
    df = generate_simulated_dce(n_individuals=55, n_tasks=5, seed=12).data
    spec = FullModelSpec(
        'respondent_id', 'task_id', 'alternative', 'choice',
        [VariableSpec(name, name) for name in ['price', 'time', 'comfort']],
        model_type='conditional', dummy_codings=[DummyCoding('comfort', 0)],
        interactions=[InteractionTerm(('price', 'time'))],
    )
    fitted = estimate_from_spec(df, spec, device=torch.device('cpu')).estimation
    result = run_bootstrap(df, spec, n_replications=4, seed=20, fitted_result=fitted)
    assert result.n_successful >= 2
    assert result.original_estimates == fitted.estimates.set_index('parameter')['estimate'].to_dict()
    assert {'beta_comfort_L1', 'beta_price_x_time'} <= set(result.param_names)


def _latent_fit(betas, probabilities, membership):
    class_estimates = pd.DataFrame({
        'class_id': [1, 2], 'parameter': ['x', 'x'], 'estimate': betas,
    })
    estimates = pd.DataFrame({
        'parameter': ['beta_x_class1', 'beta_x_class2', 'pi_class1', 'pi_class2'],
        'estimate': [*betas, *probabilities],
    })
    membership_estimates = pd.DataFrame({
        'class_id': [2, 2], 'variable': ['_intercept', 'income'], 'estimate': membership,
    })
    return SimpleNamespace(
        class_estimates=class_estimates, estimates=estimates, membership_estimates=membership_estimates,
    )


def test_latent_class_bootstrap_aligns_labels_and_rebases_membership():
    reference = _latent_fit([-1.0, 2.0], [0.3, 0.7], [0.4, -0.5])
    swapped = _latent_fit([2.0, -1.0], [0.7, 0.3], [-0.4, 0.5])
    aligned = _estimates_in_class_order(swapped, _class_order(reference, swapped))
    expected = _estimates_in_class_order(reference, np.array([1, 2]))
    pd.testing.assert_series_equal(aligned, expected)


def test_bootstrap_retains_fixed_gmnl_options(monkeypatch):
    df = generate_simulated_dce(n_individuals=12, n_tasks=3, seed=9).data
    spec = FullModelSpec(
        'respondent_id', 'task_id', 'alternative', 'choice',
        [VariableSpec('price', 'price', 'normal')],
        model_type='gmnl', gmnl_variant='s-mnl', seed=91,
        n_draws=77, maxiter=450, custom_start=[-1.0, -0.5],
    )
    fitted = SimpleNamespace(
        success=True, log_likelihood=-2.0,
        estimates=pd.DataFrame({'parameter': ['beta_price'], 'estimate': [-1.0]}),
    )
    calls = []

    def estimate(df, prepared, **kwargs):
        calls.append((prepared, kwargs))
        return SimpleNamespace(estimation=fitted)

    monkeypatch.setattr('dce_analyzer.bootstrap.estimate_dataframe', estimate)
    result = run_bootstrap(df, spec, n_replications=2, fitted_result=fitted)
    assert len(calls) == result.n_successful == 2
    for prepared, kwargs in calls:
        assert prepared.variables[0].distribution == 'fixed'
        assert prepared.n_draws == 77
        assert kwargs['fixed_gamma'] == 0
        assert kwargs['maxiter'] == 450
        assert kwargs['seed'] == 91
        assert kwargs['custom_start'] == [-1.0, -0.5]


def test_sequential_bws_simulation_keeps_best_stage_reproducible():
    arguments = dict(n_individuals=12, n_tasks=5, n_alts=3, seed=9)
    standard = generate_simulated_dce(**arguments)
    best_worst = generate_simulated_dce(**arguments, bws=True)
    pd.testing.assert_frame_equal(standard.data, best_worst.data.drop(columns='worst'))
    df = best_worst.data
    assert (df.groupby(['respondent_id', 'task_id'])['worst'].sum() == 1).all()
    assert not ((df['choice'] == 1) & (df['worst'] == 1)).any()
    assert best_worst.true_parameters['lambda_w'] == 1.0


def test_bws_model_page_excludes_worst_outcome(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv('PREFERO_QUEUE_ENABLED', raising=False)
    monkeypatch.delenv('PREFERO_TELEMETRY_ENABLED', raising=False)
    page = next((Path(__file__).resolve().parents[1] / 'app' / 'pages').glob('2_*.py'))
    at = AppTest.from_file(str(page), default_timeout=30)
    at.session_state['df'] = generate_simulated_dce(n_individuals=8, n_tasks=3, bws=True).data
    at.session_state['data_label'] = 'Synthetic best–worst example'
    at.run()
    at.checkbox(key='bws_mode').check().run()
    assert not at.exception
    assert at.selectbox(key='bws_worst_col').value == 'worst'
    utility = next(widget for widget in at.multiselect if widget.label == 'Select variables for the utility function')
    assert 'worst' not in utility.options
    assert 'worst' not in at.multiselect(key='new_interaction_cols').options


def test_bootstrap_page_exports_an_active_interaction_fit(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv('PREFERO_QUEUE_ENABLED', raising=False)
    monkeypatch.delenv('PREFERO_TELEMETRY_ENABLED', raising=False)
    df = generate_simulated_dce(n_individuals=35, n_tasks=5, seed=12).data
    spec = FullModelSpec(
        'respondent_id', 'task_id', 'alternative', 'choice',
        [VariableSpec('price', 'price'), VariableSpec('time', 'time')],
        model_type='conditional', interactions=[InteractionTerm(('price', 'time'))],
    )
    fitted = estimate_from_spec(df, spec)
    page = next((Path(__file__).resolve().parents[1] / 'app' / 'pages').glob('5_*.py'))
    at = AppTest.from_file(str(page), default_timeout=30)
    at.session_state['df'] = df
    at.session_state['data_label'] = 'Synthetic bootstrap example'
    at.session_state['model_results'] = dict(
        run_id='chosen-fit', label='Chosen fit', full_spec=spec, estimation=fitted.estimation,
    )
    at.session_state['model_history'] = [{'run_id': 'another-fit'}]
    at.run()
    at.slider[0].set_value(20)
    next(button for button in at.button if button.label == 'Run Bootstrap').click().run()
    assert not at.exception
    assert not at.error
    assert at.session_state['bootstrap_run_id'] == 'chosen-fit'
    assert 'beta_price_x_time' in at.session_state['bootstrap_results'].param_names
    assert {element.label for element in at.get('download_button')} >= {'Download bootstrap summary CSV'}

"""Release regressions: numerical agreement, compatibility and isolated UI."""
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'app')]
from dce_analyzer.config import ModelSpec, VariableSpec
from dce_analyzer.data import prepare_choice_tensors
from dce_analyzer.model import ConditionalLogitEstimator
from dce_analyzer.optimization import fit_lbfgs
from dce_analyzer.pipeline import estimate_dataframe
from dce_analyzer.simulate import generate_simulated_dce


@pytest.fixture
def fixture():
    df = generate_simulated_dce(n_individuals=100, n_tasks=4, n_alts=3, seed=42).data
    spec = ModelSpec('respondent_id', 'task_id', 'alternative', 'choice',
                     [VariableSpec(c, c) for c in ['price', 'time', 'comfort', 'reliability']])
    return df, spec


def test_cl_matches_independent_fp64_solution(fixture):
    df, spec = fixture
    tensors = prepare_choice_tensors(df, spec, torch.device('cpu'))
    X, y = tensors.X.double(), tensors.y
    def objective(v):
        t = torch.tensor(v, dtype=torch.float64, requires_grad=True)
        u = X @ t
        loss = (torch.logsumexp(u, 1) - u.gather(1, y[:, None]).squeeze()).sum()
        g, = torch.autograd.grad(loss, t)
        return loss.item(), g.numpy()
    oracle = minimize(objective, np.zeros(4), jac=True, method='BFGS', options={'gtol':1e-8})
    est = estimate_dataframe(df, spec, model_type='conditional', device=torch.device('cpu')).estimation
    assert est.success, est.message
    assert abs(est.log_likelihood + oracle.fun) < .005
    np.testing.assert_allclose(est.raw_theta, oracle.x, atol=.005, rtol=.001)


def test_iteration_limit_is_not_success():
    result = fit_lbfgs(lambda t: ((t-4)**2).sum(), [0., 1.], device='cpu', maxiter=1)
    assert not result.success
    assert 'limit' in result.message.lower()


def test_custom_start_validation():
    with pytest.raises(ValueError, match='finite'):
        fit_lbfgs(lambda t: t.square().sum(), [np.nan], device='cpu', maxiter=10)


def test_indefinite_hessian_does_not_create_standard_errors(fixture):
    df, spec = fixture
    est = ConditionalLogitEstimator(prepare_choice_tensors(df, spec, torch.device('cpu')), spec.variables)
    with patch('torch.autograd.functional.hessian', return_value=-torch.eye(4)):
        with pytest.warns(UserWarning, match='unavailable'):
            assert est._compute_vcov(np.zeros(4)) is None


def test_saved_model_roundtrip(fixture):
    import model_store
    df, spec = fixture
    result = estimate_dataframe(df, spec, model_type='conditional', device=torch.device('cpu'))
    entry = dict(label='compatibility', model_type='conditional', spec=spec, estimation=result.estimation)
    encoded = model_store.serialize_model_entry(entry)
    decoded = model_store.deserialize_model_entry(json.loads(json.dumps(encoded)))
    np.testing.assert_array_equal(decoded['estimation'].raw_theta, result.estimation.raw_theta)
    pd.testing.assert_frame_equal(decoded['estimation'].estimates, result.estimation.estimates)
    assert decoded['spec'] == spec


@pytest.fixture
def isolated_database(tmp_path, monkeypatch):
    import community_db as db
    monkeypatch.delenv('HF_TOKEN', raising=False)
    monkeypatch.delenv('HUGGING_FACE_HUB_TOKEN', raising=False)
    monkeypatch.setattr(db, 'DB_PATH', str(tmp_path/'community.db'))
    if getattr(db._local, 'conn', None): db._local.conn.close()
    db._local.conn = None
    for name in ['_sync_users_from_hf', '_sync_activity_from_hf', '_sync_posts_from_hf',
                 '_save_users_to_hf', '_save_activity_to_hf', '_save_posts_to_hf']:
        monkeypatch.setattr(db, name, lambda: None)
    db.init_db()
    yield db
    db._local.conn.close()
    db._local.conn = None


def test_community_uses_isolated_storage(isolated_database):
    db = isolated_database
    user = db.create_user('release_test')
    assert db.get_user_by_name('release_test').id == user.id
    post = db.create_post(user.id, 'Test topic', 'Local release test')
    db.create_reply(post.id, user.id, 'Local reply')
    assert db.post_count() == 1
    assert len(db.list_replies(post.id)) == 1


@pytest.mark.parametrize('page', ['Home.py', *[str(p.relative_to(ROOT/'app')) for p in sorted((ROOT/'app/pages').glob('*.py'))]])
def test_page_loads_without_external_writes(page, isolated_database):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT/'app'/page), default_timeout=30).run()
    assert not at.exception, [(e.message) for e in at.exception]


def test_model_results_and_report_pages(fixture, isolated_database):
    from streamlit.testing.v1 import AppTest
    df, spec = fixture
    result = estimate_dataframe(df, spec, model_type='conditional', device=torch.device('cpu'))
    entry = dict(label='Release example', model_type='conditional', spec=spec,
                 estimation=result.estimation, tensors=result.tensors)
    for number in [2, 3, 4, 5, 6]:
        page = next((ROOT/'app/pages').glob(f'{number}_*.py'))
        at = AppTest.from_file(str(page), default_timeout=30)
        at.session_state['df'] = df
        at.session_state['data_label'] = 'Synthetic release check'
        at.session_state['model_results'] = entry
        at.session_state['model_history'] = [entry]
        at.run()
        assert not at.exception, [(e.message) for e in at.exception]


def test_hosted_queue_keeps_existing_analysis_session(isolated_database, monkeypatch):
    """A heartbeat must not inject a full browser reload and lose user data."""
    from streamlit.testing.v1 import AppTest
    import streamlit.components.v1 as components
    import session_queue
    monkeypatch.setenv('PREFERO_QUEUE_ENABLED', 'true')
    monkeypatch.setattr(session_queue, '_active_sessions', {})
    monkeypatch.setattr(session_queue, '_session_usernames', {})
    df = pd.DataFrame({'saved_analysis': [1, 2]})
    at = AppTest.from_file(str(ROOT/'app/Home.py'), default_timeout=30)
    at.session_state['df'] = df
    with patch.object(components, 'html', wraps=components.html) as html:
        at.run()
        session_id = at.session_state['_queue_session_id']
        at.run()
    assert not at.exception
    pd.testing.assert_frame_equal(at.session_state['df'], df)
    assert at.session_state['_queue_session_id'] == session_id
    assert list(session_queue._active_sessions) == [session_id]
    assert all('location.reload' not in str(call) for call in html.call_args_list)

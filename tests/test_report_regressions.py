from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from dce_analyzer.config import ModelSpec, VariableSpec
from dce_analyzer.model import EstimationResult


def test_report_uses_selected_fit_and_only_its_bootstrap(monkeypatch):
    monkeypatch.delenv("PREFERO_LOG_ACTIVITY", raising=False)
    root = Path(__file__).resolve().parents[1]
    page = next((root / "app" / "pages").glob("6_*.py"))
    data = pd.DataFrame({
        "id": [1, 1], "task": [1, 1], "alternative": ["Train", "Bus"],
        "choice": [1, 0], "cost": [2.0, 1.0], "comfort": [3.0, 1.0],
    })
    spec = ModelSpec(
        "id", "task", "alternative", "choice",
        [VariableSpec("cost", "cost", "fixed"), VariableSpec("comfort", "comfort", "fixed")],
    )
    estimates = pd.DataFrame({
        "parameter": ["beta_cost", "beta_comfort"], "estimate": [-1.0, 1.0],
    })
    failed = EstimationResult(False, "Stopped early", -2, 8, 5, 2, 1, 1, 1, 0, estimates)
    fitted = EstimationResult(
        True, "Converged", -1, 6, 3, 2, 1, 1, 10, 0, estimates,
        raw_theta=np.array([-1.0, 1.0]),
    )
    entry = {"label": "Repeated name", "model_type": "conditional", "spec": spec}
    at = AppTest.from_file(str(page), default_timeout=25)
    at.session_state["df"] = data
    at.session_state["model_history"] = [
        {**entry, "run_id": "first", "estimation": failed},
        {**entry, "run_id": "second", "estimation": fitted},
    ]
    at.session_state["bootstrap_results"] = object()
    at.session_state["bootstrap_run_id"] = "first"
    at.run()
    assert not any(check.label == "Include bootstrap results" for check in at.checkbox)
    at.button[0].click().run()
    assert not at.exception
    html = at.session_state["_generated_report"]
    assert "Predicted Choice Shares" in html and "Train" in html and "Bus" in html
    at.selectbox[0].set_value(0).run()
    assert not at.get("download_button")
    at.button[0].click().run()
    assert not at.exception
    assert "Stopped early" in at.session_state["_generated_report"]

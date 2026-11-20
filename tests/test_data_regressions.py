from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from dce_analyzer.bws import prepare_bws_data, validate_bws
from dce_analyzer.config import ModelSpec, VariableSpec
from dce_analyzer.data import chosen_row_mask, prepare_choice_tensors
from dce_analyzer.format_converter import normalize_choice_column, wide_to_long


@pytest.mark.parametrize("labels", [[0, 1], ["fast", "cheap"]])
def test_repeated_labels_are_decoded_by_task(labels):
    df = pd.DataFrame({
        "id": [1, 1, 2, 2], "task": 1, "alt": labels * 2,
        "choice": [labels[0]] * 2 + [labels[1]] * 2, "price": [3, 4, 3, 4],
    }).iloc[[3, 2, 0, 1]]
    expected = [True, False, True, False]
    assert chosen_row_mask(df, "id", "task", "alt", "choice").tolist() == expected
    assert normalize_choice_column(df, "choice", "alt")["choice"].tolist() == expected
    spec = ModelSpec("id", "task", "alt", "choice", [VariableSpec("price", "price")])
    tensors = prepare_choice_tensors(df, spec, device=torch.device("cpu"))
    ordered = df.sort_values(["id", "task", "alt"])["alt"].tolist()
    assert [ordered[2 * i + pos] for i, pos in enumerate(tensors.y.tolist())] == labels


def test_wide_conversion_reports_all_excluded_respondents():
    df = pd.DataFrame({
        "id": [1, 1, 2, 3], "choice": ["1", None, "invalid", "2"],
        "price_1": [1, 2, 3, 4], "price_2": [4, 3, 2, 1],
    })
    messages = []
    converted = wide_to_long(df, on_exclusion=messages.append)
    assert converted["respondent_id"].tolist() == [3, 3]
    assert converted["choice"].tolist() == [0, 1]
    assert len(messages) == 1
    assert "Excluded 2 respondent(s)" in messages[0]
    assert "Removed 3 of 4 tasks" in messages[0]
    with pytest.raises(ValueError, match="distinct input columns"):
        wide_to_long(df, id_col="choice")


@pytest.mark.parametrize("column,value,message", [
    ("id", np.nan, "Missing values"), ("price", np.inf, "Utility attributes must be finite"),
])
def test_invalid_inputs_stop_before_estimation(column, value, message):
    df = pd.DataFrame({"id": [1., 1.], "task": 1, "alt": [1, 2],
                       "choice": [1, 0], "price": [3., 4.]})
    df.loc[0, column] = value
    spec = ModelSpec("id", "task", "alt", "choice", [VariableSpec("price", "price")])
    with pytest.raises(ValueError, match=message):
        prepare_choice_tensors(df, spec, device=torch.device("cpu"))


def test_best_worst_share_label_decoder_and_reject_outcome_regressor():
    df = pd.DataFrame({"id": 1, "task": 1, "alt": ["c", "a", "b"],
                       "choice": "c", "worst": "a", "price": [3, 4, 5]})
    spec = ModelSpec("id", "task", "alt", "choice", [VariableSpec("price", "price")])
    validate_bws(df, spec, "worst")
    assert prepare_bws_data(df, spec, "worst", 1, 3).y_worst.tolist() == [0]
    leaking = ModelSpec("id", "task", "alt", "choice", [VariableSpec("worst", "worst")])
    with pytest.raises(ValueError, match="outcome cannot be a utility attribute"):
        validate_bws(df, leaking, "worst")


def test_replacement_upload_invalidates_previous_results(monkeypatch, tmp_path):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("PREFERO_DB_PATH", str(tmp_path / "community.db"))
    active = [None]
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: active[0])
    path = next((Path(__file__).resolve().parents[1] / "app" / "pages").glob("1_*"))
    at = AppTest.from_file(str(path), default_timeout=15).run()
    for price in (2, 7):
        upload = BytesIO(f"id,task,alt,choice,price\n1,1,0,1,{price}\n1,1,1,0,3\n".encode())
        upload.name = "choices.csv"
        active[0] = upload
        at.session_state["model_results"] = {"prior_fit": True}
        at.session_state["cv_results"] = {"prior_fit": True}
        at.session_state["bootstrap_run_id"] = "prior-fit"
        at.run()
        assert not at.exception
        assert at.session_state["df"]["price"].tolist() == [price, 3]
        assert at.session_state["model_results"] is None
        assert "cv_results" not in at.session_state
        assert "bootstrap_run_id" not in at.session_state


@pytest.mark.parametrize("manual", [False, True])
def test_wide_upload_reports_exclusions_before_loading(monkeypatch, tmp_path, manual):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("PREFERO_DB_PATH", str(tmp_path / "community.db"))
    upload = BytesIO(b"id,choice,price_1,price_2\n1,,4,5\n2,9,2,3\n3,2,3,4\n")
    upload.name = "wide.csv"
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    path = next((Path(__file__).resolve().parents[1] / "app" / "pages").glob("1_*"))
    at = AppTest.from_file(str(path), default_timeout=15).run()
    if manual:
        at.button(key="_auto_edit").click().run()
        convert = next(button for button in at.button if button.label == "Convert to Long Format")
        convert.click().run()
    assert not at.exception
    assert at.session_state["df"] is None
    assert any("Excluded 2 respondent(s)" in warning.value for warning in at.warning)
    at.button(key="_auto_accept").click().run()
    assert not at.exception
    assert at.session_state["df"]["respondent_id"].tolist() == [3, 3]

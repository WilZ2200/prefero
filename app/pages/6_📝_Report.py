# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under dual license: AGPL-3.0 (open-source) or commercial. See LICENSE.

"""Prefero's HTML report page."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from dce_analyzer.prediction import predict_entry  # noqa: E402
from dce_analyzer.report import generate_report_html  # noqa: E402
from dce_analyzer.wtp import WTP_ESTIMAND, compute_wtp  # noqa: E402
from utils import init_session_state, require_data, sidebar_branding, slowbro_next_step  # noqa: E402

init_session_state()
sidebar_branding()

st.header("Report")
st.caption("Generate an HTML report of your estimation results. Open it in a browser or print to PDF.")

require_data()
history = st.session_state.model_history
if not history:
    st.warning("No models in history. Run a model on the **Model** page first.")
    st.stop()

model_index = st.selectbox(
    "Select model for report",
    range(len(history)),
    index=len(history) - 1,
    format_func=lambda i: f"{i + 1}. {history[i].get('label', 'Model')}",
    help="Choose which estimated model to include in the report.",
)
selected_entry = history[model_index]
estimation = selected_entry["estimation"]
run_id = selected_entry.get("run_id")
report_key = run_id or id(selected_entry)
run_label = selected_entry.get("label", "model")

st.subheader("Report options")
if not estimation.success:
    st.warning("This fit did not converge. The report will include the fit and optimizer diagnostics.")

candidate_vars = list(dict.fromkeys(
    name.split("_", 1)[1]
    for name in estimation.estimates["parameter"]
    if name.startswith(("beta_", "mu_"))
))
cost_variable = None
include_wtp = False
if estimation.success and len(candidate_vars) >= 2:
    include_wtp = st.checkbox("Include willingness-to-pay analysis", value=True)
    if include_wtp:
        st.caption(WTP_ESTIMAND)
        cost_guesses = ["price", "cost", "fee", "travel_cost"]
        default_cost = next((candidate_vars.index(v) for v in cost_guesses if v in candidate_vars), None)
        cost_variable = st.selectbox("Cost / price variable for WTP", candidate_vars, index=default_cost)

include_predictions = estimation.success and st.checkbox("Include predicted choice shares", value=True)
boot = st.session_state.get("bootstrap_results")
include_bootstrap = (
    estimation.success and boot is not None and run_id is not None
    and st.session_state.get("bootstrap_run_id") == run_id
    and st.checkbox("Include bootstrap results", value=True)
)

st.divider()
if st.button("Generate Report", type="primary", use_container_width=True):
    df = st.session_state.df
    section_notes = []
    wtp_df = None
    predictions_df = None
    if include_wtp:
        try:
            if cost_variable is None:
                raise ValueError("Select a cost variable to compute WTP.")
            wtp_df = compute_wtp(estimation, cost_variable=cost_variable)
        except (ValueError, RuntimeError) as exc:
            section_notes.append(f"WTP is unavailable. {exc}")
    if include_predictions:
        try:
            predictions_df = predict_entry(df, selected_entry)
        except (ValueError, RuntimeError) as exc:
            section_notes.append(f"Predicted shares are unavailable. {exc}")

    with st.spinner("Generating report..."):
        html = generate_report_html(
            estimation=estimation,
            model_type=selected_entry.get("model_type", "unknown"),
            run_label=run_label,
            data_label=st.session_state.get("data_label", "Unknown dataset"),
            data_shape=df.shape,
            spec=selected_entry.get("spec"),
            wtp_df=wtp_df,
            bootstrap=boot if include_bootstrap else None,
            full_spec=selected_entry.get("full_spec"),
            predictions_df=predictions_df,
            section_notes=section_notes,
        )

    st.session_state["_generated_report"] = html
    st.session_state["_report_run_id"] = report_key
    st.session_state["_report_notes"] = section_notes

if st.session_state.get("_generated_report") and st.session_state.get("_report_run_id") == report_key:
    html = st.session_state["_generated_report"]
    for note in st.session_state.get("_report_notes", ()):
        st.warning(note)
    if not st.session_state.get("_report_notes"):
        st.success("Report generated successfully.")

    st.download_button(
        "Download HTML Report",
        data=html.encode("utf-8"),
        file_name=f"{run_label}_report.html",
        mime="text/html",
        type="primary",
        use_container_width=True,
    )

    st.caption("Open the downloaded file in a browser. Use **Ctrl+P** (Windows) or **Cmd+P** (Mac) to print or save as PDF.")

    with st.expander("Report preview", expanded=True):
        import streamlit.components.v1 as components
        components.html(html, height=800, scrolling=True)

slowbro_next_step("pages/6_📝_Report.py")


# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under dual license: AGPL-3.0 (open-source) or commercial. See LICENSE.

"""Prefero — Page 6: Report Generation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils import init_session_state, require_data, sidebar_branding, slowbro_next_step  # noqa: E402

init_session_state()
sidebar_branding()

st.header("Report")
st.caption("Generate a professional HTML report of your estimation results. Open it in any browser or print to PDF.")

require_data()

if st.session_state.model_results is None:
    st.warning("No model results yet. Go to the **Model** page to run an estimation first.")
    st.stop()

history = st.session_state.model_history
if not history:
    st.warning("No models in history. Run a model on the **Model** page first.")
    st.stop()

model_labels = [e.get("label", f"Model {i+1}") for i, e in enumerate(history)]
selected_label = st.selectbox(
    "Select model for report",
    model_labels,
    index=len(model_labels) - 1,  # default to latest
    help="Choose which estimated model to include in the report.",
)
selected_entry = next(e for e in history if e.get("label") == selected_label)

# ── Report options ────────────────────────────────────────────────
st.subheader("Report options")

# WTP option
estimation = selected_entry["estimation"]
est_df = estimation.estimates
param_names = est_df["parameter"].tolist()

# Extract candidate variable names for WTP
candidate_vars = []
for p in param_names:
    if p.startswith("beta_") or p.startswith("mu_"):
        var_name = p.split("_", 1)[1]
        if var_name not in candidate_vars:
            candidate_vars.append(var_name)

include_wtp = False
cost_variable = None
if len(candidate_vars) >= 2:
    include_wtp = st.checkbox("Include WTP (Willingness to Pay) analysis", value=True)
    if include_wtp:
        cost_guesses = ["price", "cost", "fee", "travel_cost"]
        default_cost = candidate_vars[0]
        for guess in cost_guesses:
            if guess in candidate_vars:
                default_cost = guess
                break
        cost_variable = st.selectbox(
            "Cost / price variable for WTP",
            candidate_vars,
            index=candidate_vars.index(default_cost),
        )

# Predictions option
include_predictions = st.checkbox("Include predicted choice shares", value=True)

# Bootstrap option
boot = st.session_state.get("bootstrap_results")
include_bootstrap = False
if boot is not None:
    include_bootstrap = st.checkbox("Include bootstrap results", value=True)

# ── Generate button ───────────────────────────────────────────────
st.divider()
if st.button("Generate Report", type="primary", use_container_width=True):
    from dce_analyzer.report import generate_report_html

    model_type = selected_entry.get("model_type", "unknown")
    run_label = selected_entry.get("label", "model")
    spec = selected_entry.get("spec")
    full_spec = selected_entry.get("full_spec")
    data_label = st.session_state.get("data_label", "Unknown dataset")
    df = st.session_state.df
    data_shape = df.shape if df is not None else (0, 0)

    # Compute WTP if requested
    wtp_df = None
    if include_wtp and cost_variable:
        try:
            from dce_analyzer.wtp import compute_wtp
            wtp_df = compute_wtp(estimation, cost_variable=cost_variable)
        except Exception:
            pass

    # Compute predictions if requested
    predictions_df = None
    if include_predictions and spec is not None and df is not None:
        try:
            from dce_analyzer.data import prepare_choice_tensors
            import torch

            use_spec = selected_entry.get("expanded_spec", spec)
            use_df = selected_entry.get("expanded_df", df)
            if use_df is None:
                use_df = df

            tensors = prepare_choice_tensors(use_df, use_spec, device=torch.device("cpu"))
            X_np = tensors.X.cpu().numpy()

            betas = []
            est_df_pred = estimation.estimates
            for v in use_spec.variables:
                param_name = f"beta_{v.name}" if v.distribution == "fixed" else f"mu_{v.name}"
                match = est_df_pred[est_df_pred["parameter"] == param_name]
                if len(match) == 1:
                    betas.append(float(match.iloc[0]["estimate"]))
                else:
                    betas.append(0.0)
            betas_np = np.array(betas)

            V = X_np @ betas_np
            exp_V = np.exp(V - V.max(axis=1, keepdims=True))
            probs = exp_V / exp_V.sum(axis=1, keepdims=True)
            pred_shares = probs.mean(axis=0)

            y_np = tensors.y.cpu().numpy()
            actual_counts = np.bincount(y_np, minlength=tensors.n_alts)
            actual_shares = actual_counts / actual_counts.sum()

            alt_labels = [f"Alt {i + 1}" for i in range(tensors.n_alts)]
            predictions_df = pd.DataFrame({
                "Alternative": alt_labels,
                "Actual Share": actual_shares,
                "Predicted Share": pred_shares,
            })
        except Exception:
            pass

    # Bootstrap
    boot_data = boot if include_bootstrap else None

    with st.spinner("Generating report..."):
        html = generate_report_html(
            estimation=estimation,
            model_type=model_type,
            run_label=run_label,
            data_label=data_label,
            data_shape=data_shape,
            spec=spec,
            wtp_df=wtp_df,
            bootstrap=boot_data,
            full_spec=full_spec,
            predictions_df=predictions_df,
        )

    st.session_state["_generated_report"] = html
    st.session_state["_report_label"] = run_label
    st.rerun()

# ── Display report (after generation or on rerun) ────────────────
if st.session_state.get("_generated_report"):
    html = st.session_state["_generated_report"]
    report_label = st.session_state.get("_report_label", "report")

    st.success("Report generated successfully!")

    st.download_button(
        "Download HTML Report",
        data=html.encode("utf-8"),
        file_name=f"{report_label}_report.html",
        mime="text/html",
        type="primary",
        use_container_width=True,
    )

    st.caption("Open the downloaded file in any web browser. Use **Ctrl+P** (Windows) or **Cmd+P** (Mac) to print or save as PDF.")

    # Preview
    with st.expander("Report preview", expanded=True):
        import streamlit.components.v1 as components
        components.html(html, height=800, scrolling=True)

slowbro_next_step("pages/6_📝_Report.py")

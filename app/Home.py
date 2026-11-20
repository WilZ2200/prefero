# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under dual license: AGPL-3.0 (open-source) or commercial. See LICENSE.

"""Home page for the Prefero Streamlit app."""

import os

import streamlit as st

from utils import init_session_state, language_banner, sidebar_branding

st.set_page_config(
    page_title="Prefero - DCE Analysis Tool",
    page_icon="🧮",
    layout="wide",
)

init_session_state()
sidebar_branding()
language_banner()

st.title("Prefero")
st.write("Analyse discrete choice experiments in your browser.")

if st.button("Open Data", key="_nav_begin", type="primary"):
    st.switch_page("pages/1_📊_Data.py")

st.subheader("Workflow")
steps = [
    ("Data", "Upload or simulate choice data."),
    ("Model", "Specify and estimate a choice model."),
    ("Results", "Check convergence, estimates and predicted choice shares."),
    ("Compare", "Compare fitted models and evaluate them on held-out data."),
    ("Bootstrap", "Resample respondents to estimate uncertainty."),
    ("Report", "Download an HTML report."),
]
for left, right in (steps[:2], steps[2:4], steps[4:]):
    for column, (title, description) in zip(st.columns(2), (left, right)):
        with column, st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(description)

support_url = os.environ.get("COFFEE_URL", "https://buymeacoffee.com/preferohz")
if support_url:
    st.link_button("Support Prefero", support_url)

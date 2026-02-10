"""Deprecated entry point — use Home.py instead.

Run the new multi-page app with:
    streamlit run app/Home.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="分析侠 - Redirect", layout="wide")

st.warning(
    "**This entry point is deprecated.**  \n"
    "Please launch the app via `streamlit run app/Home.py` to use the "
    "new multi-page layout."
)
st.info("The new app includes pages for Data, Model, Results, Compare, and Bootstrap.")

# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under dual license: AGPL-3.0 (open-source) or commercial. See LICENSE.

"""Community and feedback links."""

import streamlit as st

from utils import init_session_state, sidebar_branding

st.set_page_config(page_title="Prefero - Community", page_icon="💬", layout="wide")
init_session_state()
sidebar_branding()

st.header("Community")
st.write("Use GitHub to report problems, suggest features or ask project questions. Include a synthetic example when sharing data.")
st.link_button("Open Prefero on GitHub", "https://github.com/WilZ2200/prefero/issues")

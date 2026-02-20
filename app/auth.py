"""Authentication module for Prefero.

Open access — no login required. Users pick a display name for community features.
Admin access is controlled separately via PREFERO_ADMIN_PASSWORD env var.
"""

from __future__ import annotations

import re

import streamlit as st

_SLOWBRO_IMG = (
    "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites"
    "/pokemon/other/official-artwork/80.png"
)


def auth_gate() -> bool:
    """Always returns True — open access, no login required."""
    st.session_state["authenticated"] = True
    return True


def username_gate() -> bool:
    """Prompt the user to choose a display name if they haven't yet."""
    if st.session_state.get("username"):
        return True

    try:
        st.set_page_config(
            page_title="Prefero - Welcome",
            page_icon="🧮",
            layout="centered",
        )
    except st.errors.StreamlitAPIException:
        pass

    from utils import language_banner
    language_banner()

    st.markdown(
        "<div style='text-align:center;'>"
        f"<img src='{_SLOWBRO_IMG}' width='100' />"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h3 style='text-align:center;'>Welcome to Prefero</h3>"
        "<p style='text-align:center; color:gray;'>"
        "Choose a display name to get started</p>",
        unsafe_allow_html=True,
    )

    name_input = st.text_input(
        "Username",
        key="_auth_username_input",
        max_chars=20,
        placeholder="e.g. SlowbroFan42",
    )

    if st.button("Continue", key="_auth_username_btn", use_container_width=True):
        name = name_input.strip()
        if not name:
            st.error("Please enter a username.")
        elif len(name) < 2:
            st.error("Username must be at least 2 characters.")
        elif not re.match(r'^[\w\-\s]+$', name):
            st.error("Username can only contain letters, numbers, spaces, underscores, and hyphens.")
        else:
            try:
                from community_db import init_db, create_user, get_user_by_name, log_activity
                from session_queue import register_username

                init_db()
                existing = get_user_by_name(name)
                if existing:
                    st.session_state["username"] = existing.username
                    st.session_state["user_id"] = existing.id
                    register_username(existing.username)
                    log_activity(existing.username, "login", "Returning user")
                    st.rerun()
                else:
                    user = create_user(name, email="")
                    st.session_state["username"] = user.username
                    st.session_state["user_id"] = user.id
                    register_username(user.username)
                    log_activity(user.username, "register", "New user")
                    st.rerun()
            except Exception as exc:
                st.error(f"Error: {exc}")

    return False

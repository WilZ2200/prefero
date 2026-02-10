"""Authentication module for 分析侠 Prefero.

Supports two methods:
  1. MUCHE Member login (name fuzzy-match + password)
  2. Invitation code (instant, checked against env var)

MUCHE team members are stored as a whitelist.
First login: match name + set password to claim account.
Subsequent logins: match name + enter password.
Fuzzy matching allows minor name variations (e.g. "John" matches "Jonathan").

Toggle with PREFERO_AUTH_ENABLED env var ("true" to enable).
"""

from __future__ import annotations

import hashlib
import os
import re
from difflib import SequenceMatcher

import streamlit as st

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _auth_enabled() -> bool:
    return os.environ.get("PREFERO_AUTH_ENABLED", "").lower() == "true"


def _get_invitation_codes() -> list[str]:
    raw = os.environ.get("PREFERO_INVITATION_CODES", "")
    if not raw.strip():
        return []
    return [c.strip().upper() for c in raw.split(",") if c.strip()]


# ---------------------------------------------------------------------------
# MUCHE team whitelist
# ---------------------------------------------------------------------------

MUCHE_MEMBERS: list[str] = [
    # Director
    "Henry Cutler",
    # Academics
    "Mona Aghdaee",
    "Anam Bilgrami",
    "Jonas Fooken",
    "Yuanyuan Gu",
    "Martin Hoyle",
    "Varinder Jeet",
    "Ali Furkan Kalay",
    "Alicia Norman",
    "Bonny Parkinson",
    "Smriti Raichand",
    "Rezwanul Rana",
    "Rajan Sharma",
    "Shari Stathis",
    "Dandan Yu",
    # Operations
    "Ashley Soytemiz",
    # Honorary & Visiting
    "Marco Bertoni",
    "David Cullen",
    "Nils Gutacker",
    "Rowena Jacobs",
    "Elizabeth-Ann Schroeder",
    "Kompal Sinha",
    "Neil Soderlund",
    "Apostolos Tsiachristas",
    # Graduate research students
    "Meimei Chen",
    "Han Cheng",
    "Kadek Darmawan",
    "Shan Jiang",
    "Maggie Lee",
    "Noura Saba",
    "Saranjit Singh",
    "Zachery Tirrell",
    "Hengzhe Zhao",
]


def _normalize(name: str) -> str:
    """Lowercase, strip, collapse whitespace, remove hyphens."""
    return re.sub(r"[\s\-]+", " ", name.strip().lower())


def _fuzzy_match(input_name: str, threshold: float = 0.75) -> str | None:
    """Find the best fuzzy match from MUCHE_MEMBERS.

    Returns the canonical name if similarity >= threshold, else None.
    Tries full name match first, then first-name-only match.
    """
    norm_input = _normalize(input_name)
    if not norm_input:
        return None

    best_name: str | None = None
    best_score: float = 0.0

    for member in MUCHE_MEMBERS:
        norm_member = _normalize(member)
        score = SequenceMatcher(None, norm_input, norm_member).ratio()
        if score > best_score:
            best_score = score
            best_name = member

        first_name = norm_member.split()[0]
        score_first = SequenceMatcher(None, norm_input, first_name).ratio()
        if score_first > best_score:
            best_score = score_first
            best_name = member

    if best_score >= threshold:
        return best_name
    return None


def _hash_password(password: str) -> str:
    """Simple SHA-256 hash for password storage."""
    return hashlib.sha256(password.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Auth gate UI
# ---------------------------------------------------------------------------

_SLOWBRO_IMG = (
    "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites"
    "/pokemon/other/official-artwork/80.png"
)


def auth_gate() -> bool:
    """Show auth UI. Returns True if authenticated, False otherwise."""
    if not _auth_enabled():
        return True

    if st.session_state.get("authenticated"):
        return True

    try:
        st.set_page_config(
            page_title="分析侠 Prefero - Sign In",
            page_icon="🧮",
            layout="centered",
        )
    except st.errors.StreamlitAPIException:
        pass

    from utils import language_banner
    language_banner()

    st.markdown(
        "<div style='text-align:center;'>"
        f"<img src='{_SLOWBRO_IMG}' width='120' />"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h2 style='text-align:center; margin-bottom:0;'>分析侠 Prefero</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:gray;'>Sign In</p>",
        unsafe_allow_html=True,
    )

    # ── Concurrent login conflict resolution ─────────────────────
    _conflict_user = st.session_state.get("_auth_conflict_user")
    if _conflict_user:
        _conflict_id = st.session_state.get("_auth_conflict_id", 0)
        _conflict_source = st.session_state.get("_auth_conflict_source", "")

        st.warning(
            f"**{_conflict_user}** is currently logged in from another session. "
            "Do you want to kick them off and log in here?"
        )

        _kick_col, _cancel_col = st.columns(2)
        with _kick_col:
            if st.button("Yes, log in here", type="primary", key="_auth_kick_btn", use_container_width=True):
                from session_queue import force_evict_username, register_username
                force_evict_username(_conflict_user)
                st.session_state["authenticated"] = True
                st.session_state["username"] = _conflict_user
                st.session_state["user_id"] = _conflict_id
                register_username(_conflict_user)
                from community_db import log_activity
                log_activity(_conflict_user, "login", f"{_conflict_source} (kicked previous session)")
                # Clean up conflict state
                st.session_state.pop("_auth_conflict_user", None)
                st.session_state.pop("_auth_conflict_id", None)
                st.session_state.pop("_auth_conflict_source", None)
                st.rerun()
        with _cancel_col:
            if st.button("Cancel", key="_auth_cancel_kick_btn", use_container_width=True):
                st.session_state.pop("_auth_conflict_user", None)
                st.session_state.pop("_auth_conflict_id", None)
                st.session_state.pop("_auth_conflict_source", None)
                st.rerun()

        return False

    tab_member, tab_invite = st.tabs(["MUCHE Member", "Invitation Code"])

    # ── Tab 1: MUCHE Member (name + password) ─────────────────────
    with tab_member:
        st.caption("Enter your name as it appears on the MUCHE website.")

        name_input = st.text_input(
            "Your name",
            key="_auth_member_name",
            placeholder="e.g. Jordan McAllister",
        )

        matched_name = None
        if name_input and name_input.strip():
            matched_name = _fuzzy_match(name_input)

        if matched_name:
            st.success(f"Matched: **{matched_name}**")

            from community_db import init_db, get_user_by_name
            init_db()
            existing = get_user_by_name(matched_name)

            if existing and existing.email:
                # Returning user — enter password
                pw_input = st.text_input(
                    "Password",
                    key="_auth_member_pw",
                    type="password",
                )
                if st.button("Sign In", key="_auth_member_login_btn", use_container_width=True):
                    if not pw_input:
                        st.error("Please enter your password.")
                    elif _hash_password(pw_input) == existing.email:
                        from session_queue import is_username_active, register_username
                        if is_username_active(existing.username):
                            st.session_state["_auth_conflict_user"] = existing.username
                            st.session_state["_auth_conflict_id"] = existing.id
                            st.session_state["_auth_conflict_source"] = "MUCHE member"
                            st.rerun()
                        else:
                            st.session_state["authenticated"] = True
                            st.session_state["username"] = existing.username
                            st.session_state["user_id"] = existing.id
                            register_username(existing.username)
                            from community_db import log_activity
                            log_activity(existing.username, "login", "MUCHE member")
                            st.success(f"Welcome back, {existing.username}!")
                            st.rerun()
                    else:
                        st.error("Incorrect password.")
            else:
                # First time — set password to claim account
                st.info("First time? Set a password to claim your account.")
                pw1 = st.text_input(
                    "Choose a password",
                    key="_auth_member_pw1",
                    type="password",
                )
                pw2 = st.text_input(
                    "Confirm password",
                    key="_auth_member_pw2",
                    type="password",
                )
                if st.button("Create Account", key="_auth_member_create_btn", use_container_width=True):
                    if not pw1 or not pw2:
                        st.error("Please fill in both password fields.")
                    elif len(pw1) < 4:
                        st.error("Password must be at least 4 characters.")
                    elif pw1 != pw2:
                        st.error("Passwords do not match.")
                    else:
                        try:
                            from community_db import create_user, log_activity
                            from session_queue import register_username
                            user = create_user(matched_name, email=_hash_password(pw1))
                            st.session_state["authenticated"] = True
                            st.session_state["username"] = user.username
                            st.session_state["user_id"] = user.id
                            register_username(user.username)
                            log_activity(user.username, "register", "MUCHE member")
                            st.success(f"Account created! Welcome, {user.username}!")
                            st.rerun()
                        except ValueError as exc:
                            st.error(str(exc))
        elif name_input and name_input.strip():
            st.warning(
                "Name not found in the MUCHE team list. "
                "Please check your spelling, or use an invitation code."
            )

    # ── Tab 2: Invitation Code ────────────────────────────────────
    with tab_invite:
        code_input = st.text_input(
            "Enter your invitation code",
            key="_auth_invite_input",
            type="password",
        )
        if st.button("Enter", key="_auth_invite_btn", use_container_width=True):
            valid_codes = _get_invitation_codes()
            if not valid_codes:
                st.error("No invitation codes are configured.")
            elif code_input.strip().upper() in valid_codes:
                st.session_state["authenticated"] = True
                st.session_state["auth_email"] = ""
                st.success("Access granted!")
                st.rerun()
            else:
                st.error("Invalid invitation code.")

    return False


# ---------------------------------------------------------------------------
# Username selection gate (shown after auth, before main app)
# ---------------------------------------------------------------------------

def username_gate() -> bool:
    """Prompt the user to choose a display name if they haven't yet.

    For MUCHE members, username is already set during auth_gate.
    Only invitation code users see this screen.
    """
    if not _auth_enabled():
        if not st.session_state.get("username"):
            st.session_state["username"] = "Guest"
            st.session_state["user_id"] = 0
        return True

    if st.session_state.get("username"):
        return True

    st.markdown(
        "<div style='text-align:center;'>"
        f"<img src='{_SLOWBRO_IMG}' width='100' />"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h3 style='text-align:center;'>Choose your display name</h3>",
        unsafe_allow_html=True,
    )

    # ── Concurrent login conflict resolution ─────────────────────
    _conflict_user = st.session_state.get("_auth_conflict_user")
    if _conflict_user:
        _conflict_id = st.session_state.get("_auth_conflict_id", 0)
        _conflict_source = st.session_state.get("_auth_conflict_source", "")

        st.warning(
            f"**{_conflict_user}** is currently logged in from another session. "
            "Do you want to kick them off and log in here?"
        )

        _kick_col, _cancel_col = st.columns(2)
        with _kick_col:
            if st.button("Yes, log in here", type="primary", key="_ugate_kick_btn", use_container_width=True):
                from session_queue import force_evict_username, register_username
                force_evict_username(_conflict_user)
                st.session_state["username"] = _conflict_user
                st.session_state["user_id"] = _conflict_id
                register_username(_conflict_user)
                from community_db import log_activity
                log_activity(_conflict_user, "login", f"{_conflict_source} (kicked previous session)")
                st.session_state.pop("_auth_conflict_user", None)
                st.session_state.pop("_auth_conflict_id", None)
                st.session_state.pop("_auth_conflict_source", None)
                st.rerun()
        with _cancel_col:
            if st.button("Cancel", key="_ugate_cancel_kick_btn", use_container_width=True):
                st.session_state.pop("_auth_conflict_user", None)
                st.session_state.pop("_auth_conflict_id", None)
                st.session_state.pop("_auth_conflict_source", None)
                st.rerun()

        return False

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
                from session_queue import is_username_active, register_username

                init_db()
                existing = get_user_by_name(name)
                if existing:
                    if is_username_active(existing.username):
                        st.session_state["_auth_conflict_user"] = existing.username
                        st.session_state["_auth_conflict_id"] = existing.id
                        st.session_state["_auth_conflict_source"] = "Returning user"
                        st.rerun()
                    else:
                        st.session_state["username"] = existing.username
                        st.session_state["user_id"] = existing.id
                        register_username(existing.username)
                        log_activity(existing.username, "login", "Returning user")
                        st.success(f"Welcome back, {existing.username}!")
                        st.rerun()
                else:
                    user = create_user(name, email="")
                    st.session_state["username"] = user.username
                    st.session_state["user_id"] = user.id
                    register_username(user.username)
                    st.success(f"Welcome, {user.username}!")
                    st.rerun()
            except Exception as exc:
                st.error(f"Error: {exc}")

    return False

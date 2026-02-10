"""Authentication module for 分析侠 Prefero.

Supports two methods:
  1. Invitation code (instant, checked against env var)
  2. Email OTP via SMTP (6-digit code, 5-min expiry, rate-limited)

After authentication, users choose a display name (username).

Toggle with PREFERO_AUTH_ENABLED env var ("true" to enable).
"""

from __future__ import annotations

import os
import re
import random
import smtplib
import string
import time
from email.mime.text import MIMEText

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


def _smtp_configured() -> bool:
    return bool(os.environ.get("SMTP_USER")) and bool(os.environ.get("SMTP_PASS"))


# ---------------------------------------------------------------------------
# OTP helpers
# ---------------------------------------------------------------------------

def _generate_otp() -> str:
    return "".join(random.choices(string.digits, k=6))


def _send_otp_email(recipient: str, code: str) -> None:
    """Send the OTP email via SMTP. Raises on failure."""
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER", "")
    password = os.environ.get("SMTP_PASS", "")

    body = (
        f"Your verification code is: {code}\n\n"
        "This code expires in 5 minutes.\n\n"
        "-- 分析侠 Prefero"
    )

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = "Your Prefero Login Code"
    msg["From"] = user
    msg["To"] = recipient

    with smtplib.SMTP(host, port, timeout=15) as server:
        server.starttls()
        server.login(user, password)
        server.sendmail(user, [recipient], msg.as_string())


def _check_rate_limit(email: str) -> bool:
    """Return True if the email is within rate limits (max 3 per 10 min)."""
    attempts: dict[str, list[float]] = st.session_state.get("_otp_attempts", {})
    now = time.time()
    window = 10 * 60  # 10 minutes

    timestamps = [t for t in attempts.get(email, []) if now - t < window]
    return len(timestamps) < 3


def _record_otp_attempt(email: str) -> None:
    if "_otp_attempts" not in st.session_state:
        st.session_state["_otp_attempts"] = {}
    attempts: dict[str, list[float]] = st.session_state["_otp_attempts"]
    now = time.time()
    window = 10 * 60
    existing = [t for t in attempts.get(email, []) if now - t < window]
    existing.append(now)
    attempts[email] = existing


# ---------------------------------------------------------------------------
# Auth gate UI
# ---------------------------------------------------------------------------

_SLOWBRO_IMG = (
    "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites"
    "/pokemon/other/official-artwork/80.png"
)


def auth_gate() -> bool:
    """Show auth UI. Returns True if authenticated, False otherwise.

    If auth is disabled (PREFERO_AUTH_ENABLED != "true"), always returns True.
    """
    if not _auth_enabled():
        return True

    if st.session_state.get("authenticated"):
        return True

    # ── Auth page layout ──────────────────────────────────────────
    try:
        st.set_page_config(
            page_title="分析侠 Prefero - Sign In",
            page_icon="🧮",
            layout="centered",
        )
    except st.errors.StreamlitAPIException:
        pass

    # Scrolling multilingual banner
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

    tab_invite, tab_email = st.tabs(["Invitation Code", "Email"])

    # ── Tab 1: Invitation Code ────────────────────────────────────
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

    # ── Tab 2: Email OTP ──────────────────────────────────────────
    with tab_email:
        if not _smtp_configured():
            st.info(
                "Email authentication is not configured. "
                "Please use an invitation code, or contact the administrator."
            )
        else:
            email_input = st.text_input(
                "Email address",
                key="_auth_email_input",
            )

            otp_sent = st.session_state.get("_otp_code") is not None

            if not otp_sent:
                if st.button(
                    "Send Code", key="_auth_send_btn", use_container_width=True
                ):
                    email = email_input.strip()
                    if not email or "@" not in email:
                        st.error("Please enter a valid email address.")
                    elif not _check_rate_limit(email):
                        st.error(
                            "Too many requests. Please wait before requesting "
                            "another code."
                        )
                    else:
                        otp = _generate_otp()
                        try:
                            _send_otp_email(email, otp)
                        except Exception as exc:
                            st.error(f"Failed to send email: {exc}")
                        else:
                            st.session_state["_otp_code"] = otp
                            st.session_state["_otp_email"] = email
                            st.session_state["_otp_time"] = time.time()
                            _record_otp_attempt(email)
                            st.success(f"Code sent to {email}.")
                            st.rerun()
            else:
                target_email = st.session_state.get("_otp_email", "")
                st.caption(f"A code was sent to **{target_email}**.")

                otp_input = st.text_input(
                    "Enter the 6-digit code",
                    key="_auth_otp_input",
                    max_chars=6,
                )

                col_verify, col_resend = st.columns(2)

                with col_verify:
                    if st.button(
                        "Verify", key="_auth_verify_btn", use_container_width=True
                    ):
                        stored_otp = st.session_state.get("_otp_code", "")
                        otp_time = st.session_state.get("_otp_time", 0.0)

                        if time.time() - otp_time > 300:
                            st.error("Code expired. Please request a new one.")
                            st.session_state.pop("_otp_code", None)
                            st.session_state.pop("_otp_email", None)
                            st.session_state.pop("_otp_time", None)
                        elif otp_input.strip() == stored_otp:
                            st.session_state["authenticated"] = True
                            st.session_state["auth_email"] = target_email
                            # Clean up OTP state
                            st.session_state.pop("_otp_code", None)
                            st.session_state.pop("_otp_email", None)
                            st.session_state.pop("_otp_time", None)
                            st.success("Access granted!")
                            st.rerun()
                        else:
                            st.error("Incorrect code. Please try again.")

                with col_resend:
                    if st.button(
                        "Resend Code",
                        key="_auth_resend_btn",
                        use_container_width=True,
                    ):
                        # Clear old OTP so the form resets to "Send Code" state
                        st.session_state.pop("_otp_code", None)
                        st.session_state.pop("_otp_email", None)
                        st.session_state.pop("_otp_time", None)
                        st.rerun()

    return False


# ---------------------------------------------------------------------------
# Username selection gate (shown after auth, before main app)
# ---------------------------------------------------------------------------

def username_gate() -> bool:
    """Prompt the user to choose a display name if they haven't yet.

    Returns True if username is set, False otherwise.
    When auth is disabled, auto-assigns "Guest" without prompting.
    """
    if not _auth_enabled():
        # Local dev: auto-assign guest identity
        if not st.session_state.get("username"):
            st.session_state["username"] = "Guest"
            st.session_state["user_id"] = 0
        return True

    if st.session_state.get("username"):
        return True

    # ── Username selection UI ────────────────────────────────────
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
    st.markdown(
        "<p style='text-align:center; color:gray;'>"
        "Enter your username to log in, or choose a new one to create an account.</p>",
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
        elif not re.match(r'^[\w\-]+$', name):
            st.error("Username can only contain letters, numbers, underscores, and hyphens.")
        else:
            try:
                from community_db import init_db, create_user, get_user_by_name, log_activity

                init_db()
                existing = get_user_by_name(name)
                if existing:
                    # Returning user — log them in
                    st.session_state["username"] = existing.username
                    st.session_state["user_id"] = existing.id
                    log_activity(existing.username, "login", "Returning user")
                    st.success(f"Welcome back, {existing.username}!")
                    st.rerun()
                else:
                    # New user — create account
                    email = st.session_state.get("auth_email", "")
                    user = create_user(name, email=email)
                    st.session_state["username"] = user.username
                    st.session_state["user_id"] = user.id
                    st.success(f"Welcome, {user.username}!")
                    st.rerun()
            except Exception as exc:
                st.error(f"Error: {exc}")

    return False

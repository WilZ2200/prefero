"""分析侠 — Page 7: Admin Dashboard."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

# ── path setup ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils import init_session_state, sidebar_branding  # noqa: E402

init_session_state()
sidebar_branding()

# ── Admin access control ───────────────────────────────────────────

_admin_csv = os.environ.get("PREFERO_ADMIN_USERS", "")
_admin_users = {u.strip().lower() for u in _admin_csv.split(",") if u.strip()}

current_user = st.session_state.get("username", "").strip().lower()

if not _admin_users or current_user not in _admin_users:
    st.error("Access denied. You do not have admin privileges.")
    st.stop()

# ── Page header ────────────────────────────────────────────────────

st.title("Admin Dashboard")
st.caption(f"Signed in as admin: **{st.session_state.get('username', '')}**")

# ── 1. Registered Users ───────────────────────────────────────────

st.header("Registered Users")

from community_db import init_db, _get_conn  # noqa: E402

init_db()
conn = _get_conn()

rows = conn.execute(
    "SELECT username, email, joined_at FROM users ORDER BY joined_at DESC"
).fetchall()

if rows:
    user_data = []
    for r in rows:
        joined_dt = datetime.fromtimestamp(r["joined_at"], tz=timezone.utc)
        user_data.append({
            "Username": r["username"],
            "Email": r["email"] or "(none)",
            "Joined": joined_dt.strftime("%Y-%m-%d"),
        })
    st.metric("Total users", len(user_data))
    st.dataframe(pd.DataFrame(user_data), use_container_width=True, hide_index=True)
else:
    st.info("No registered users yet.")

# ── 2. Activity Log ───────────────────────────────────────────────

st.header("Activity Log")

try:
    log_rows = conn.execute(
        "SELECT * FROM activity_log ORDER BY timestamp DESC LIMIT 100"
    ).fetchall()
    if log_rows:
        log_data = []
        for r in log_rows:
            ts = datetime.fromtimestamp(r["timestamp"], tz=timezone.utc)
            log_data.append({
                "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "Username": r["username"],
                "Action": r["action"],
                "Details": r["details"],
            })
        st.dataframe(pd.DataFrame(log_data), use_container_width=True, hide_index=True)
    else:
        st.info("No activity logged yet.")
except Exception:
    st.info("Activity logging not configured yet.")

# ── 3. Active Sessions ────────────────────────────────────────────

st.header("Active Sessions")

from session_queue import active_count, spots_available, _MAX_CONCURRENT  # noqa: E402

n_active = active_count()
n_available = spots_available()

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Active sessions", n_active)
with c2:
    st.metric("Available slots", n_available)
with c3:
    st.metric("Max concurrent", _MAX_CONCURRENT)

# ── 4. Space Configuration ────────────────────────────────────────

st.header("Space Configuration")

auth_enabled = os.environ.get("PREFERO_AUTH_ENABLED", "false")
queue_enabled = os.environ.get("PREFERO_QUEUE_ENABLED", "false")

invite_codes_raw = os.environ.get("PREFERO_INVITATION_CODES", "")
invite_count = len([c for c in invite_codes_raw.split(",") if c.strip()]) if invite_codes_raw.strip() else 0

admin_users_display = os.environ.get("PREFERO_ADMIN_USERS", "(not set)")

smtp_configured = bool(os.environ.get("SMTP_USER")) and bool(os.environ.get("SMTP_PASS"))

cfg_col1, cfg_col2 = st.columns(2)

with cfg_col1:
    st.metric("Auth enabled", auth_enabled)
    st.metric("Queue enabled", queue_enabled)
    st.metric("SMTP configured", "Yes" if smtp_configured else "No")

with cfg_col2:
    st.metric("Invitation codes", str(invite_count))
    st.metric("Admin users", admin_users_display)

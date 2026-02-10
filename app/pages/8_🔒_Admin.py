"""分析侠 — Page 8: Admin Dashboard."""

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

# ── Developer mode gate ────────────────────────────────────────────

if not st.session_state.get("_dev_mode_active"):
    st.error("Access denied. Enable Developer Mode from the sidebar first.")
    st.stop()

# ── Page header ────────────────────────────────────────────────────

st.title("Developer Dashboard")
st.caption(f"Signed in as: **{st.session_state.get('username', '')}**")

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
    usernames = []
    for r in rows:
        joined_dt = datetime.fromtimestamp(r["joined_at"], tz=timezone.utc)
        has_pw = bool(r["email"] and len(r["email"]) == 64)  # SHA-256 hash
        user_data.append({
            "Username": r["username"],
            "Has Password": "Yes" if has_pw else "No",
            "Joined": joined_dt.strftime("%Y-%m-%d"),
        })
        usernames.append(r["username"])
    st.metric("Total users", len(user_data))
    st.dataframe(pd.DataFrame(user_data), use_container_width=True, hide_index=True)

    # ── Account management ──
    st.subheader("Account Management")
    st.caption("Delete a user account so they can re-register with a new password.")

    del_user = st.selectbox(
        "Select user to delete",
        options=[""] + usernames,
        key="_admin_del_user",
    )
    if del_user:
        col_del, col_reset = st.columns(2)
        with col_del:
            if st.button(f"Delete account: {del_user}", type="primary", use_container_width=True):
                st.session_state["_admin_confirm_delete"] = del_user
        with col_reset:
            if st.button(f"Reset password: {del_user}", use_container_width=True):
                conn.execute("UPDATE users SET email = '' WHERE username = ?", (del_user,))
                conn.commit()
                from community_db import log_activity, _save_users_to_hf
                log_activity(st.session_state.get("username", ""), "admin_reset_password", f"Reset password for {del_user}")
                _save_users_to_hf()
                st.success(f"Password reset for **{del_user}**. They can now set a new one.")
                st.rerun()

        if st.session_state.get("_admin_confirm_delete") == del_user:
            st.warning(f"Are you sure you want to delete **{del_user}**? This cannot be undone.")
            c_yes, c_no = st.columns(2)
            with c_yes:
                if st.button("Yes, delete", key="_admin_confirm_yes", use_container_width=True):
                    conn.execute("DELETE FROM users WHERE username = ?", (del_user,))
                    conn.commit()
                    from community_db import log_activity, _save_users_to_hf
                    log_activity(st.session_state.get("username", ""), "admin_delete_user", f"Deleted {del_user}")
                    _save_users_to_hf()
                    st.session_state.pop("_admin_confirm_delete", None)
                    st.success(f"**{del_user}** has been deleted. They can re-register.")
                    st.rerun()
            with c_no:
                if st.button("Cancel", key="_admin_confirm_no", use_container_width=True):
                    st.session_state.pop("_admin_confirm_delete", None)
                    st.rerun()
else:
    st.info("No registered users yet.")

# ── 2. Analytics Overview ────────────────────────────────────────

st.header("Analytics Overview")

import re
from datetime import timedelta

try:
    import plotly.express as px
    _has_plotly = True
except ImportError:
    _has_plotly = False

try:
    total_users = conn.execute("SELECT COUNT(*) as cnt FROM users").fetchone()["cnt"]
    total_activities = conn.execute("SELECT COUNT(*) as cnt FROM activity_log").fetchone()["cnt"]
    total_estimations = conn.execute(
        "SELECT COUNT(*) as cnt FROM activity_log WHERE action = 'estimation_success'"
    ).fetchone()["cnt"]
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    active_today = conn.execute(
        "SELECT COUNT(DISTINCT username) as cnt FROM activity_log WHERE timestamp > ?",
        (today_start,),
    ).fetchone()["cnt"]

    ov1, ov2, ov3, ov4 = st.columns(4)
    ov1.metric("Total Users", total_users)
    ov2.metric("Total Activities", total_activities)
    ov3.metric("Estimations Run", total_estimations)
    ov4.metric("Active Today", active_today)
except Exception:
    st.info("Could not load overview metrics.")

# ── 2a. Activity Timeline ───────────────────────────────────────

st.header("Activity Timeline")

try:
    # Filters
    ft1, ft2, ft3 = st.columns(3)
    with ft1:
        _default_start = datetime.now(timezone.utc).date() - timedelta(days=7)
        _default_end = datetime.now(timezone.utc).date()
        date_range = st.date_input(
            "Date range",
            value=(_default_start, _default_end),
            key="_admin_date_range",
        )
    with ft2:
        all_actions = conn.execute(
            "SELECT DISTINCT action FROM activity_log ORDER BY action"
        ).fetchall()
        action_options = [r["action"] for r in all_actions]
        selected_actions = st.multiselect(
            "Filter by action type",
            options=action_options,
            default=[],
            key="_admin_action_filter",
        )
    with ft3:
        all_usernames = conn.execute(
            "SELECT DISTINCT username FROM activity_log ORDER BY username"
        ).fetchall()
        username_options = ["All"] + [r["username"] for r in all_usernames]
        selected_username = st.selectbox(
            "Filter by user",
            options=username_options,
            key="_admin_user_filter",
        )

    # Build query
    _where_clauses = []
    _params: list = []
    if isinstance(date_range, tuple) and len(date_range) == 2:
        _ts_start = datetime.combine(date_range[0], datetime.min.time()).timestamp()
        _ts_end = datetime.combine(date_range[1], datetime.max.time()).timestamp()
        _where_clauses.append("timestamp >= ? AND timestamp <= ?")
        _params.extend([_ts_start, _ts_end])
    if selected_actions:
        placeholders = ",".join("?" * len(selected_actions))
        _where_clauses.append(f"action IN ({placeholders})")
        _params.extend(selected_actions)
    if selected_username and selected_username != "All":
        _where_clauses.append("username = ?")
        _params.append(selected_username)

    _where_sql = " AND ".join(_where_clauses) if _where_clauses else "1=1"
    _query = f"SELECT * FROM activity_log WHERE {_where_sql} ORDER BY timestamp DESC LIMIT 500"
    log_rows = conn.execute(_query, _params).fetchall()

    if log_rows:
        log_data = []
        for r in log_rows:
            ts = datetime.fromtimestamp(r["timestamp"], tz=timezone.utc)
            row = {
                "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "Username": r["username"],
                "Action": r["action"],
                "Details": r["details"],
            }
            # IP/geo columns (may not exist in older records)
            ip_val = r["ip_address"] if "ip_address" in r.keys() else ""
            country_val = r["country"] if "country" in r.keys() else ""
            region_val = r["region"] if "region" in r.keys() else ""
            row["IP"] = ip_val or ""
            row["Country"] = country_val or ""
            row["Region"] = region_val or ""
            log_data.append(row)
        log_df = pd.DataFrame(log_data)
        st.dataframe(log_df, use_container_width=True, hide_index=True)

        csv_export = log_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Activity Log CSV",
            data=csv_export,
            file_name="activity_log.csv",
            mime="text/csv",
        )
    else:
        st.info("No activity found for the selected filters.")
except Exception as _exc:
    st.info(f"Activity timeline not available: {_exc}")

# ── 2a-2. Geographic Overview ──────────────────────────────────

st.header("Geographic Overview")

try:
    geo_rows = conn.execute(
        "SELECT country, region, ip_address, is_vpn, COUNT(*) as cnt "
        "FROM activity_log WHERE country != '' GROUP BY country ORDER BY cnt DESC"
    ).fetchall()
    if geo_rows:
        geo_data = [
            {"Country": r["country"], "Activities": r["cnt"]}
            for r in geo_rows
        ]
        geo_df = pd.DataFrame(geo_data)
        g1, g2 = st.columns(2)
        with g1:
            st.dataframe(geo_df, use_container_width=True, hide_index=True)
        with g2:
            if _has_plotly:
                fig_geo = px.pie(geo_df, names="Country", values="Activities", title="Activities by Country")
                st.plotly_chart(fig_geo, use_container_width=True, key="admin_geo_pie")

    else:
        st.info("No geographic data yet. IP tracking starts collecting data on the next user activity.")
except Exception as _exc:
    st.info(f"Geographic data not available: {_exc}")

# ── 2b. Model Usage Stats ───────────────────────────────────────

st.header("Model Usage Stats")

try:
    est_rows = conn.execute(
        "SELECT details FROM activity_log WHERE action = 'estimation_success'"
    ).fetchall()
    err_rows = conn.execute(
        "SELECT COUNT(*) as cnt FROM activity_log WHERE action = 'estimation_error'"
    ).fetchone()["cnt"]

    if est_rows:
        model_types: list[str] = []
        runtimes: list[float] = []
        for r in est_rows:
            details = r["details"]
            m = re.search(r"model=(\w+)", details)
            if m:
                model_types.append(m.group(1))
            rt = re.search(r"runtime=([\d.]+)", details)
            if rt:
                runtimes.append(float(rt.group(1)))

        if model_types and _has_plotly:
            type_counts = pd.Series(model_types).value_counts().reset_index()
            type_counts.columns = ["Model Type", "Count"]
            fig_model = px.bar(type_counts, x="Model Type", y="Count", title="Estimations by Model Type")
            st.plotly_chart(fig_model, use_container_width=True, key="admin_model_usage")

        ms1, ms2 = st.columns(2)
        total_est = len(est_rows)
        total_err = err_rows
        ms1.metric("Success Rate", f"{total_est / max(total_est + total_err, 1) * 100:.1f}%")
        if runtimes:
            ms2.metric("Avg Runtime", f"{sum(runtimes) / len(runtimes):.1f}s")
    else:
        st.info("No estimation data yet.")
except Exception:
    st.info("Model usage stats not available.")

# ── 2c. Data Upload Stats ───────────────────────────────────────

st.header("Data Upload Stats")

try:
    upload_rows = conn.execute(
        "SELECT action, details FROM activity_log WHERE action IN ('data_upload', 'data_simulate', 'data_apollo')"
    ).fetchall()
    if upload_rows:
        format_counts: dict[str, int] = {}
        for r in upload_rows:
            if r["action"] == "data_simulate":
                format_counts["simulate"] = format_counts.get("simulate", 0) + 1
            elif r["action"] == "data_apollo":
                format_counts["apollo"] = format_counts.get("apollo", 0) + 1
            else:
                m = re.search(r"format=(\w+)", r["details"])
                fmt = m.group(1) if m else "unknown"
                format_counts[fmt] = format_counts.get(fmt, 0) + 1

        fmt_df = pd.DataFrame(list(format_counts.items()), columns=["Source/Format", "Count"])
        st.dataframe(fmt_df, use_container_width=True, hide_index=True)
    else:
        st.info("No data upload activity yet.")
except Exception:
    st.info("Data upload stats not available.")

# ── 2d. Page Popularity ─────────────────────────────────────────

st.header("Page Popularity")

try:
    page_rows = conn.execute(
        "SELECT details, COUNT(*) as cnt FROM activity_log WHERE action = 'page_visit' GROUP BY details ORDER BY cnt DESC"
    ).fetchall()
    if page_rows:
        page_df = pd.DataFrame([{"Page": r["details"], "Visits": r["cnt"]} for r in page_rows])
        if _has_plotly:
            fig_pages = px.bar(page_df, x="Page", y="Visits", title="Page Visits")
            st.plotly_chart(fig_pages, use_container_width=True, key="admin_page_pop")
        else:
            st.dataframe(page_df, use_container_width=True, hide_index=True)
    else:
        st.info("No page visit data yet.")
except Exception:
    st.info("Page popularity data not available.")

# ── 2e. User Engagement ─────────────────────────────────────────

st.header("User Engagement")

try:
    # Registrations over time
    reg_rows = conn.execute(
        "SELECT joined_at FROM users ORDER BY joined_at"
    ).fetchall()
    if reg_rows:
        reg_dates = [
            datetime.fromtimestamp(r["joined_at"], tz=timezone.utc).strftime("%Y-%m-%d")
            for r in reg_rows
        ]
        reg_series = pd.Series(reg_dates).value_counts().sort_index().reset_index()
        reg_series.columns = ["Date", "Registrations"]
        if _has_plotly:
            fig_reg = px.line(reg_series, x="Date", y="Registrations", title="Registrations Over Time", markers=True)
            st.plotly_chart(fig_reg, use_container_width=True, key="admin_reg_timeline")
        else:
            st.dataframe(reg_series, use_container_width=True, hide_index=True)

    # Most active users
    active_rows = conn.execute(
        "SELECT username, COUNT(*) as cnt FROM activity_log GROUP BY username ORDER BY cnt DESC LIMIT 10"
    ).fetchall()
    if active_rows:
        st.subheader("Most Active Users (Top 10)")
        active_df = pd.DataFrame([{"Username": r["username"], "Activities": r["cnt"]} for r in active_rows])
        st.dataframe(active_df, use_container_width=True, hide_index=True)
    else:
        st.info("No user activity data yet.")
except Exception:
    st.info("User engagement data not available.")

# ── 2f. Error Log ────────────────────────────────────────────────

st.header("Error Log")

try:
    error_rows = conn.execute(
        "SELECT * FROM activity_log WHERE action IN ('estimation_error', 'bootstrap_error') ORDER BY timestamp DESC LIMIT 50"
    ).fetchall()
    if error_rows:
        error_data = []
        for r in error_rows:
            ts = datetime.fromtimestamp(r["timestamp"], tz=timezone.utc)
            error_data.append({
                "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "Username": r["username"],
                "Action": r["action"],
                "Details": r["details"],
            })
        st.dataframe(pd.DataFrame(error_data), use_container_width=True, hide_index=True)
    else:
        st.info("No errors logged.")
except Exception:
    st.info("Error log not available.")

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

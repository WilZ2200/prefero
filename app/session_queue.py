# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under dual license: AGPL-3.0 (open-source) or commercial. See LICENSE.

"""Concurrent user queue for Prefero on shared hosting (e.g. HF Spaces).

Uses a module-level dict (shared across all Streamlit sessions within the
same process) to track active users.  Thread-safe via a lock.

Toggle with PREFERO_QUEUE_ENABLED env var ("true" to enable).
Max concurrent users controlled by PREFERO_MAX_CONCURRENT (default 2).
"""

from __future__ import annotations

import os
import threading
import time
import uuid

import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MAX_CONCURRENT = int(os.environ.get("PREFERO_MAX_CONCURRENT", "2"))
_SESSION_TIMEOUT = 1800  # 30 minutes of inactivity → evicted


def _queue_enabled() -> bool:
    return os.environ.get("PREFERO_QUEUE_ENABLED", "").lower() == "true"


# ---------------------------------------------------------------------------
# Shared state (module-level, shared across all sessions in one process)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_active_sessions: dict[str, float] = {}  # session_id → last_heartbeat
_session_usernames: dict[str, str] = {}  # session_id → username


def _cleanup_stale() -> None:
    """Remove sessions that haven't sent a heartbeat recently."""
    now = time.time()
    stale = [sid for sid, ts in _active_sessions.items() if now - ts > _SESSION_TIMEOUT]
    for sid in stale:
        del _active_sessions[sid]
        _session_usernames.pop(sid, None)


def _ensure_session_id() -> str:
    """Get or create a unique session identifier."""
    if "_queue_session_id" not in st.session_state:
        st.session_state["_queue_session_id"] = str(uuid.uuid4())
    return st.session_state["_queue_session_id"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def try_enter() -> bool:
    """Try to claim a slot.  Returns True if the user is admitted."""
    sid = _ensure_session_id()
    with _lock:
        _cleanup_stale()
        if sid in _active_sessions:
            _active_sessions[sid] = time.time()
            return True
        if len(_active_sessions) < _MAX_CONCURRENT:
            _active_sessions[sid] = time.time()
            return True
        return False


def heartbeat() -> None:
    """Refresh the current session's timestamp (call on every page load)."""
    sid = _ensure_session_id()
    with _lock:
        if sid in _active_sessions:
            _active_sessions[sid] = time.time()


def leave() -> None:
    """Release the current session's slot."""
    sid = _ensure_session_id()
    with _lock:
        _active_sessions.pop(sid, None)
        _session_usernames.pop(sid, None)


def register_username(username: str) -> None:
    """Associate the current session with a username."""
    sid = _ensure_session_id()
    with _lock:
        _session_usernames[sid] = username


def is_username_active(username: str) -> bool:
    """Check if a username is logged in on another active session."""
    sid = _ensure_session_id()
    with _lock:
        _cleanup_stale()
        for other_sid, uname in _session_usernames.items():
            if uname == username and other_sid != sid:
                return True
        return False


def force_evict_username(username: str) -> None:
    """Evict all other sessions using this username so the caller can log in."""
    sid = _ensure_session_id()
    with _lock:
        _cleanup_stale()
        to_remove = [
            other_sid
            for other_sid, uname in _session_usernames.items()
            if uname == username and other_sid != sid
        ]
        for other_sid in to_remove:
            _active_sessions.pop(other_sid, None)
            _session_usernames.pop(other_sid, None)


def active_count() -> int:
    """How many sessions are currently active."""
    with _lock:
        _cleanup_stale()
        return len(_active_sessions)


def spots_available() -> int:
    """How many open slots remain."""
    with _lock:
        _cleanup_stale()
        return max(0, _MAX_CONCURRENT - len(_active_sessions))


def is_session_active() -> bool:
    """Check whether the current session still holds a slot.

    Returns False if the session was evicted due to inactivity (stale
    heartbeat).  Callers should clear auth state and redirect to login.
    """
    sid = _ensure_session_id()
    with _lock:
        _cleanup_stale()
        return sid in _active_sessions


# ---------------------------------------------------------------------------
# Waiting-room gate (Streamlit UI)
# ---------------------------------------------------------------------------

def queue_gate() -> bool:
    """Show waiting room if the server is full.  Returns True if admitted.

    When queue is disabled, always returns True.
    """
    if not _queue_enabled():
        return True

    # Always send a heartbeat so active sessions stay fresh
    heartbeat()

    if try_enter():
        return True

    # ── Waiting room UI ─────────────────────────────────────────
    from waiting_facts import WAITING_FACTS
    from utils import language_banner
    import random

    # Scrolling multilingual banner
    language_banner()

    n_active = active_count()
    spots_left = spots_available()

    st.subheader("Prefero is at capacity")
    st.write("This page checks for an available slot every few seconds.")

    # ── Queue status ──
    q1, q2 = st.columns(2)
    with q1:
        st.metric("Active users", f"{n_active} / {_MAX_CONCURRENT}")
    with q2:
        st.metric("Seats available", str(spots_left))

    # ── Session policy note ──
    st.warning(
        "**How the queue works:** Each user gets a seat for as long as "
        "they're active. Sessions expire after **30 minutes** of inactivity "
        "to keep things moving — but if you're running a model, your seat "
        "is safe until estimation completes."
    )

    # ── Rolling cultural facts ──
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; font-weight:600; margin-bottom:4px;'>"
        "While you wait — queuing around the world</p>",
        unsafe_allow_html=True,
    )

    rng = random.Random(int(time.time()) // 8)  # change every 8 seconds
    indices = list(range(len(WAITING_FACTS)))
    rng.shuffle(indices)
    display_facts = [WAITING_FACTS[i] for i in indices[:3]]

    for fact in display_facts:
        st.info(fact)

    # auto-refresh every 5 seconds
    time.sleep(5)
    st.rerun()

    return False  # unreachable, but for type-checker

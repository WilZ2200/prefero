"""Concurrent user queue for Prefero on shared hosting (e.g. HF Spaces).

Uses a module-level dict (shared across all Streamlit sessions within the
same process) to track active users.  Thread-safe via a lock.

Toggle with PREFERO_QUEUE_ENABLED env var ("true" to enable).
Max concurrent users controlled by PREFERO_MAX_CONCURRENT (default 5).
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

_MAX_CONCURRENT = int(os.environ.get("PREFERO_MAX_CONCURRENT", "5"))
_SESSION_TIMEOUT = 120  # 2 minutes of inactivity → evicted


def _queue_enabled() -> bool:
    return os.environ.get("PREFERO_QUEUE_ENABLED", "").lower() == "true"


# ---------------------------------------------------------------------------
# Shared state (module-level, shared across all sessions in one process)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_active_sessions: dict[str, float] = {}  # session_id → last_heartbeat


def _cleanup_stale() -> None:
    """Remove sessions that haven't sent a heartbeat recently."""
    now = time.time()
    stale = [sid for sid, ts in _active_sessions.items() if now - ts > _SESSION_TIMEOUT]
    for sid in stale:
        del _active_sessions[sid]


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

_SLOWBRO_IMG = (
    "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites"
    "/pokemon/other/official-artwork/80.png"
)


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

    st.markdown(
        "<div style='text-align:center; margin-top:20px;'>"
        f"<img src='{_SLOWBRO_IMG}' width='100' />"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h3 style='text-align:center;'>Slowbro is a bit busy right now...</h3>",
        unsafe_allow_html=True,
    )

    n_active = active_count()
    st.markdown(
        f"<p style='text-align:center; color:gray;'>"
        f"{n_active}/{_MAX_CONCURRENT} seats occupied.  "
        f"You are in the waiting room.</p>",
        unsafe_allow_html=True,
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

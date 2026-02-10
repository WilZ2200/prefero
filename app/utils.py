"""Shared utilities for the 分析侠 multi-page Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auth import auth_gate, username_gate  # noqa: E402
from session_queue import queue_gate, heartbeat as queue_heartbeat, is_session_active  # noqa: E402

# ── session-state keys and their defaults ──────────────────────────
_SESSION_DEFAULTS: dict[str, object] = {
    "df": None,
    "data_label": "",
    "true_params": {},
    "model_results": None,
    "model_history": [],
    "bootstrap_results": None,
    "lc_result": None,
    "lc_bic_comparison": None,
    "lc_best_q": None,
    "authenticated": False,
    "auth_email": "",
    "username": "",
    "user_id": 0,
}


def require_auth() -> None:
    """Block the page if the user has not authenticated.

    When auth is disabled via PREFERO_AUTH_ENABLED, this is a no-op.
    Otherwise it shows the auth gate and stops execution until the
    user successfully authenticates.
    """
    if not auth_gate():
        st.stop()


def require_queue_slot() -> None:
    """Block the page if the server is at capacity.

    When queue is disabled (default for local dev), this is a no-op.
    Otherwise shows a branded waiting room with cultural queuing facts.
    """
    if not queue_gate():
        st.stop()


def _check_session_timeout() -> None:
    """If the session was evicted due to inactivity, clear auth and redirect.

    Skips the check when estimation is actively running (the user is waiting
    for Slowbro to finish, not actually idle).
    """
    import os
    if os.environ.get("PREFERO_QUEUE_ENABLED", "").lower() != "true":
        return  # queue disabled → no timeout

    # Don't kick users while estimation is running
    if st.session_state.get("_estimation_running"):
        return

    if not st.session_state.get("authenticated"):
        return  # not logged in → nothing to expire

    # Only check if user was previously admitted to the queue.
    # Without this, freshly authenticated users who haven't entered
    # the queue yet would be incorrectly treated as evicted.
    if not st.session_state.get("_queue_admitted"):
        return

    if not is_session_active():
        # Session was evicted — clear auth state
        for key in ("authenticated", "auth_email", "username", "user_id"):
            st.session_state[key] = _SESSION_DEFAULTS.get(key, "")
        st.session_state.pop("_queue_session_id", None)
        st.session_state.pop("_queue_admitted", None)
        st.warning("Your session expired due to inactivity. Please sign in again.")
        st.stop()


def _inject_activity_heartbeat() -> None:
    """Inject JavaScript that detects user activity and auto-refreshes to keep the session alive.

    Monitors scroll, mouse movement, keyboard, and touch events on the parent
    document.  If activity is detected within the last 90 seconds, triggers a
    lightweight page reload every 90 seconds.  Streamlit session state persists
    across reloads, so the heartbeat() call in init_session_state() naturally
    refreshes the server-side timestamp.
    """
    import os
    if os.environ.get("PREFERO_QUEUE_ENABLED", "").lower() != "true":
        return  # no queue → no need for activity tracking

    import streamlit.components.v1 as _components
    _components.html(
        """
        <script>
        (function() {
            var INTERVAL = 90000;  // 90 seconds
            var lastActivity = Date.now();
            var events = ['scroll', 'mousemove', 'keydown', 'touchstart', 'click', 'wheel'];
            var doc;
            try { doc = window.parent.document; } catch(e) { return; }

            events.forEach(function(e) {
                doc.addEventListener(e, function() { lastActivity = Date.now(); }, {passive: true});
            });

            setInterval(function() {
                if (Date.now() - lastActivity < INTERVAL) {
                    // User was active — trigger Streamlit rerun to refresh heartbeat
                    try { window.parent.location.reload(); } catch(e) {}
                }
            }, INTERVAL);
        })();
        </script>
        """,
        height=0,
    )


def init_session_state() -> None:
    """Ensure every shared session-state key exists with its default value."""
    for key, default in _SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default
    _check_session_timeout()
    require_auth()
    if not username_gate():
        st.stop()
    require_queue_slot()
    queue_heartbeat()
    st.session_state["_queue_admitted"] = True
    _inject_activity_heartbeat()
    import inspect
    _caller = inspect.stack()[1].filename
    _page = Path(_caller).stem
    _visit_key = f"_visited_{_page}"
    if not st.session_state.get(_visit_key):
        st.session_state[_visit_key] = True
        try:
            from community_db import init_db, log_activity
            init_db()
            log_activity(st.session_state.get("username", "anonymous"), "page_visit", _page)
        except Exception:
            pass


def data_is_loaded() -> bool:
    """Return True when a DataFrame has been loaded into session state."""
    return st.session_state.get("df") is not None


def _is_admin_user() -> bool:
    """Check if the current user has admin privileges."""
    import os
    admin_csv = os.environ.get("PREFERO_ADMIN_USERS", "")
    if not admin_csv.strip():
        return False
    admin_users = {u.strip().lower() for u in admin_csv.split(",") if u.strip()}
    current = st.session_state.get("username", "").strip().lower()
    return current in admin_users


def slowbro_status() -> None:
    """Inject global Slowbro status widget into Streamlit's fixed header."""
    _slowbro_img = (
        "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites"
        "/pokemon/other/official-artwork/80.png"
    )

    # Build optional developer entry menu (admin only)
    _dev_menu_css = ""
    _dev_menu_html = ""
    if _is_admin_user():
        _dev_menu_css = """
        /* Hide Admin page from sidebar nav */
        [data-testid="stSidebarNav"] a[href*="Admin"] {
            display: none !important;
        }
        /* Developer entry ⋯ menu */
        .dev-entry {
            position: fixed;
            top: 10px;
            right: 14px;
            z-index: 1001;
        }
        .dev-entry-btn {
            background: rgba(50,50,50,0.7);
            border: none;
            color: #999;
            font-size: 16px;
            cursor: pointer;
            border-radius: 6px;
            width: 30px;
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.15s;
        }
        .dev-entry-btn:hover {
            background: rgba(80,80,80,0.9);
            color: #ddd;
        }
        .dev-entry-dropdown {
            display: none;
            position: absolute;
            right: 0;
            top: 34px;
            background: #1e1e1e;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 4px 0;
            min-width: 170px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.4);
        }
        .dev-entry:hover .dev-entry-dropdown {
            display: block;
        }
        .dev-entry-dropdown a {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 14px;
            color: #ccc;
            text-decoration: none;
            font-size: 0.82rem;
        }
        .dev-entry-dropdown a:hover {
            background: rgba(255,255,255,0.08);
            color: #fff;
        }
        """
        _dev_menu_html = """
        <div class="dev-entry">
            <div class="dev-entry-btn">⋯</div>
            <div class="dev-entry-dropdown">
                <a href="/🔒_Admin" target="_self">🔧 Developer Entry</a>
            </div>
        </div>
        """

    st.markdown(
        f"""
        <style>
        /* Hide default Streamlit status widget entirely */
        [data-testid="stStatusWidget"] {{
            display: none !important;
        }}
        /* Slowbro pill — injected into Streamlit's fixed header via ::after */
        [data-testid="stHeader"]::after {{
            content: "Slowbro is resting...";
            position: absolute;
            top: 50%;
            right: 46px;
            transform: translateY(-50%);
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 3px 12px 3px 30px;
            border-radius: 20px;
            background: rgba(50,50,50,0.9) url("{_slowbro_img}") no-repeat 6px center / 20px 20px;
            font-size: 0.76rem;
            color: #bbb;
            pointer-events: none;
            white-space: nowrap;
            z-index: 999;
        }}
        {_dev_menu_css}
        </style>
        {_dev_menu_html}
        """,
        unsafe_allow_html=True,
    )


def sidebar_branding() -> None:
    """Render common sidebar branding and data-status indicator."""
    slowbro_status()
    st.sidebar.markdown("## 分析侠")
    st.sidebar.caption("Discrete Choice Experiment Analysis")

    username = st.session_state.get("username", "")
    if username:
        st.sidebar.caption(f"Signed in as **{username}**")

    st.sidebar.divider()

    if data_is_loaded():
        st.sidebar.success(f"Data loaded: {st.session_state.data_label}")
    else:
        st.sidebar.info("No data loaded yet.")


def require_data() -> None:
    """Show a blocking message and stop if no data is loaded."""
    if not data_is_loaded():
        st.warning("Please load a dataset first on the **Data** page.")
        st.stop()


# ── Multilingual scrolling banner ─────────────────────────────────

_TRANSLATIONS = [
    ("分析侠", "zh"),
    ("Prefero", "en"),
    ("プレフェロ", "ja"),
    ("프레페로", "ko"),
    ("Преферо", "ru"),
    ("بريفيرو", "ar"),
    ("प्रेफ़ेरो", "hi"),
    ("เพรเฟโร", "th"),
    ("Πρεφέρο", "el"),
    ("პრეფერო", "ka"),
    ("פרפרו", "he"),
    ("প্রেফেরো", "bn"),
    ("ప్రెఫెరో", "te"),
    ("Préféro", "fr"),
    ("Prefero", "la"),
]


def language_banner() -> None:
    """Render the scrolling multilingual Prefero banner."""
    st.markdown(
        """
        <style>
        @keyframes scroll-left {
            0%   { transform: translateX(0%); }
            100% { transform: translateX(-50%); }
        }
        .scroll-banner {
            overflow: hidden;
            white-space: nowrap;
            padding: 12px 0;
            margin-bottom: 8px;
            border-top: 1px solid rgba(128,128,128,0.2);
            border-bottom: 1px solid rgba(128,128,128,0.2);
        }
        .scroll-inner {
            display: inline-block;
            animation: scroll-left 30s linear infinite;
        }
        .scroll-inner span {
            font-size: 1.05rem;
            letter-spacing: 0.03em;
            opacity: 0.75;
            padding: 0 16px;
        }
        .scroll-inner .zh { font-weight: 700; opacity: 1.0; font-size: 1.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    span_html = ""
    for name, lang in _TRANSLATIONS:
        cls = "zh" if lang == "zh" else ""
        span_html += f'<span class="{cls}">{name}</span>\u00b7'
    doubled = span_html * 2

    st.markdown(
        f"""
        <div class="scroll-banner">
            <div class="scroll-inner">{doubled}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Page-to-page navigation ───────────────────────────────────────

_SLOWBRO_NAV_IMG = (
    "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites"
    "/pokemon/other/official-artwork/80.png"
)

_PAGE_ORDER = [
    ("pages/1_📊_Data.py", "Data", "📊"),
    ("pages/2_⚙️_Model.py", "Model", "⚙️"),
    ("pages/3_📈_Results.py", "Results", "📈"),
    ("pages/4_🔄_Compare.py", "Compare", "🔄"),
    ("pages/5_🎯_Bootstrap.py", "Bootstrap", "🎯"),
    ("pages/6_📝_Report.py", "Report", "📝"),
    ("pages/7_💬_Community.py", "Community", "💬"),
]


def slowbro_next_step(current_page: str) -> None:
    """Render a 'Slowbro guides you to the next step' button at page bottom.

    Parameters
    ----------
    current_page : str
        The page file path relative to app/, e.g. "pages/1_📊_Data.py".
    """
    current_idx = None
    for i, (path, _, _) in enumerate(_PAGE_ORDER):
        if path == current_page:
            current_idx = i
            break

    if current_idx is None or current_idx >= len(_PAGE_ORDER) - 1:
        return  # last page or unknown — no next step

    next_path, next_name, next_icon = _PAGE_ORDER[current_idx + 1]

    st.divider()
    col_img, col_btn = st.columns([1, 4])
    with col_img:
        st.image(_SLOWBRO_NAV_IMG, width=64)
    with col_btn:
        st.markdown(
            f"<p style='margin:0; color:gray; font-size:0.85em;'>"
            f"Slowbro guides you to the next step</p>",
            unsafe_allow_html=True,
        )
        if st.button(
            f"{next_icon} Go to {next_name}",
            key=f"_nav_next_{next_name}",
            use_container_width=True,
        ):
            st.switch_page(next_path)

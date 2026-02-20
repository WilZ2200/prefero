"""Shared utilities for the Prefero multi-page Streamlit app."""

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
    "saved_models": [],
    "authenticated": False,
    "auth_email": "",
    "username": "",
    "user_id": 0,
}


def require_auth() -> None:
    """Run the auth gate. Currently a no-op (open access)."""
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
    """If the session was evicted due to inactivity, clear state and redirect.

    Skips the check when estimation is actively running (the user is waiting
    for Slowbro to finish, not actually idle).
    """
    import os
    if os.environ.get("PREFERO_QUEUE_ENABLED", "").lower() != "true":
        return  # queue disabled → no timeout

    # Don't kick users while estimation is running
    if st.session_state.get("_estimation_running"):
        return

    if not st.session_state.get("username"):
        return  # no username set → nothing to expire

    # Only check if user was previously admitted to the queue.
    # Without this, fresh users who haven't entered the queue yet
    # would be incorrectly treated as evicted.
    if not st.session_state.get("_queue_admitted"):
        return

    if not is_session_active():
        # Session was evicted — clear state so user re-enters
        for key in ("username", "user_id"):
            st.session_state[key] = _SESSION_DEFAULTS.get(key, "")
        st.session_state.pop("_queue_session_id", None)
        st.session_state.pop("_queue_admitted", None)
        st.warning("Your session expired due to inactivity. Please choose a username to continue.")
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
    # username_gate removed — visitors see the home page directly
    # if not username_gate():
    #     st.stop()
    require_queue_slot()
    queue_heartbeat()
    if not st.session_state.get("_saved_models_loaded"):
        username = st.session_state.get("username", "")
        if username:
            try:
                from model_store import load_models
                st.session_state.saved_models = load_models(username)
            except Exception:
                st.session_state.saved_models = []
            st.session_state._saved_models_loaded = True
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

    # Hide Admin from sidebar unless developer mode is active
    _hide_admin_css = ""
    if not st.session_state.get("_dev_mode_active"):
        _hide_admin_css = """
        [data-testid="stSidebarNav"] a[href*="Admin"] {
            display: none !important;
        }
        """

    st.markdown(
        f"""
        <style>
        /* Hide default Streamlit status widget entirely */
        [data-testid="stStatusWidget"] {{
            display: none !important;
        }}
{_hide_admin_css}
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_branding() -> None:
    """Render common sidebar branding and data-status indicator."""
    slowbro_status()
    st.sidebar.markdown("## Prefero")
    st.sidebar.caption("Discrete Choice Experiment Analysis")

    username = st.session_state.get("username", "")
    if username:
        st.sidebar.caption(f"Signed in as **{username}**")

    st.sidebar.divider()

    if data_is_loaded():
        st.sidebar.success(f"Data loaded: {st.session_state.data_label}")
    else:
        st.sidebar.info("No data loaded yet.")

    # ── Saved models in sidebar ──
    _saved = st.session_state.get("saved_models", [])
    if _saved:
        with st.sidebar.expander(f"Saved Models ({len(_saved)}/10)", expanded=False):
            _sb_delete_idx: int | None = None
            for i, sm in enumerate(_saved):
                est = sm.get("estimation")
                try:
                    _ll = f"{est.log_likelihood:.1f}"
                except Exception:
                    _ll = "?"
                st.caption(f"**{sm.get('label', 'model')}** ({sm.get('model_type', '?')}) LL:{_ll}")
                if st.button("✕ Delete", key=f"_sb_del_saved_{i}"):
                    _sb_delete_idx = i
            if _sb_delete_idx is not None:
                _uname = st.session_state.get("username", "")
                if _uname:
                    try:
                        from model_store import delete_saved_model, load_models
                        if delete_saved_model(_uname, _sb_delete_idx):
                            st.session_state.saved_models = load_models(_uname)
                    except Exception:
                        pass
                    st.rerun()

    # Developer mode prompt (admin users only)
    if _is_admin_user() and not st.session_state.get("_dev_mode_active"):
        import os
        _dev_key = os.environ.get("PREFERO_ADMIN_PASSWORD", "")
        if _dev_key:
            st.sidebar.divider()
            with st.sidebar.expander("Developer Mode"):
                key_input = st.text_input("Developer key", type="password", key="_sidebar_dev_key")
                if st.button("Unlock", key="_sidebar_dev_unlock", use_container_width=True):
                    if key_input == _dev_key:
                        st.session_state["_dev_mode_active"] = True
                        st.rerun()
                    else:
                        st.error("Incorrect key.")


def require_data() -> None:
    """Show a blocking message and stop if no data is loaded."""
    if not data_is_loaded():
        st.warning("Please load a dataset first on the **Data** page.")
        st.stop()


# ── Multilingual scrolling banner ─────────────────────────────────

_TRANSLATIONS = [
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
        .scroll-banner-wrap {
            position: relative;
            margin-bottom: 8px;
        }
        .scroll-banner {
            overflow: hidden;
            white-space: nowrap;
            padding: 12px 0;
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
        .banner-help {
            position: absolute;
            right: 6px;
            top: 50%;
            transform: translateY(-50%);
            z-index: 10;
        }
        .banner-help-icon {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: rgba(128,128,128,0.18);
            color: rgba(128,128,128,0.7);
            font-size: 0.75rem;
            font-weight: 600;
            cursor: default;
            user-select: none;
            line-height: 1;
        }
        .banner-help-icon:hover {
            background: rgba(128,128,128,0.30);
            color: rgba(128,128,128,0.95);
        }
        .banner-help-tooltip {
            display: none;
            position: absolute;
            right: 0;
            top: 28px;
            width: 260px;
            padding: 12px 14px;
            background: var(--background-color, #fff);
            border: 1px solid rgba(128,128,128,0.2);
            border-radius: 8px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.10);
            white-space: normal;
            font-size: 0.82rem;
            line-height: 1.5;
            color: var(--text-color, #444);
            z-index: 100;
        }
        .banner-help:hover .banner-help-tooltip {
            display: block;
        }
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
        <div class="scroll-banner-wrap">
            <div class="scroll-banner">
                <div class="scroll-inner">{doubled}</div>
            </div>
            <div class="banner-help">
                <div class="banner-help-icon">?</div>
                <div class="banner-help-tooltip">
                    The name <b>Prefero</b> is shown in many languages
                    to celebrate the wonderful diversity of our users.
                    Translations may not be perfectly accurate &mdash;
                    if you spot an error, we would love to hear from you
                    on the <b>Community</b> page!
                </div>
            </div>
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

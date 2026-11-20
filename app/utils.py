# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under dual license: AGPL-3.0 (open-source) or commercial. See LICENSE.

"""Shared utilities for the Prefero multi-page Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from session_queue import queue_gate, heartbeat as queue_heartbeat  # noqa: E402

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
}


def require_queue_slot() -> None:
    """Block the page if the server is at capacity.

    When queue is disabled (default for local dev), this is a no-op.
    Otherwise shows a branded waiting room with cultural queuing facts.
    """
    if not queue_gate():
        st.stop()


def init_session_state() -> None:
    """Ensure every shared session-state key exists with its default value."""
    for key, default in _SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default
    require_queue_slot()
    queue_heartbeat()


def data_is_loaded() -> bool:
    """Return True when a DataFrame has been loaded into session state."""
    return st.session_state.get("df") is not None


def sidebar_branding() -> None:
    """Render common sidebar branding and data-status indicator."""
    st.sidebar.markdown("## Prefero")
    st.sidebar.caption("Discrete Choice Experiment Analysis")

    st.sidebar.divider()

    if data_is_loaded():
        st.sidebar.success(f"Data loaded: {st.session_state.data_label}")
    else:
        st.sidebar.info("No data loaded yet.")

    from community_db import telemetry_enabled
    if telemetry_enabled():
        st.sidebar.caption("This server records anonymous usage events locally.")

    # ── License notice (always at sidebar bottom) ──
    st.sidebar.divider()
    st.sidebar.markdown(
        "<p style='font-size:0.7rem; color:#888; text-align:center;'>"
        "&copy; 2026 Hengzhe Zhao<br>"
        "Dual-licensed: <a href='https://github.com/WilZ2200/prefero/blob/main/LICENSE' "
        "target='_blank' style='color:#888;'>AGPL-3.0</a> &amp; Commercial</p>",
        unsafe_allow_html=True,
    )


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
    """Render the next page link.

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
    if st.button(
        f"{next_icon} Go to {next_name}",
        key=f"_nav_next_{next_name}",
        use_container_width=True,
    ):
        st.switch_page(next_path)

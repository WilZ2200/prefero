# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under dual license: AGPL-3.0 (open-source) or commercial. See LICENSE.

"""Prefero — Page 7: Community Discussion Forum."""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

# ── path setup ────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

ROOT = APP_DIR.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils import init_session_state, sidebar_branding  # noqa: E402
from community_db import (  # noqa: E402
    init_db,
    create_post,
    list_posts,
    get_post,
    create_reply,
    list_replies,
    post_count,
)

st.set_page_config(page_title="Prefero - Community", page_icon="💬", layout="wide")

init_session_state()
sidebar_branding()

# Ensure the community database is initialized
init_db()

st.header("Community")
st.caption("Discuss DCE methods, share tips, ask questions, and connect with other researchers.")


# ── Helpers ───────────────────────────────────────────────────────

def _format_time(ts: float) -> str:
    """Format a timestamp into a readable relative time string."""
    delta = time.time() - ts
    if delta < 60:
        return "just now"
    elif delta < 3600:
        mins = int(delta // 60)
        return f"{mins}m ago"
    elif delta < 86400:
        hours = int(delta // 3600)
        return f"{hours}h ago"
    else:
        return datetime.fromtimestamp(ts).strftime("%b %d, %Y")


def _avatar_html(username: str, size: int = 32) -> str:
    """Generate a colored circle avatar with initials."""
    # Simple hash-based color
    h = hash(username) % 360
    initials = username[:2].upper()
    return (
        f'<div style="display:inline-flex; align-items:center; justify-content:center; '
        f'width:{size}px; height:{size}px; border-radius:50%; '
        f'background:hsl({h},60%,45%); color:white; font-size:{size//2}px; '
        f'font-weight:700; flex-shrink:0;">{initials}</div>'
    )


# ── Session state for navigation ──────────────────────────────────
if "_community_view" not in st.session_state:
    st.session_state["_community_view"] = "list"  # "list" or "thread"
if "_community_post_id" not in st.session_state:
    st.session_state["_community_post_id"] = None


user_id = st.session_state.get("user_id", 0)
username = st.session_state.get("username", "Guest")


# ══════════════════════════════════════════════════════════════════
# VIEW: Thread (single post + replies)
# ══════════════════════════════════════════════════════════════════

if st.session_state["_community_view"] == "thread":
    post_id = st.session_state["_community_post_id"]
    post = get_post(post_id) if post_id else None

    if post is None:
        st.error("Post not found.")
        if st.button("Back to Community"):
            st.session_state["_community_view"] = "list"
            st.rerun()
        st.stop()

    # Back button
    if st.button("Back to Community"):
        st.session_state["_community_view"] = "list"
        st.rerun()

    st.markdown("---")

    # ── Post header ──
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
            {_avatar_html(post.author_name)}
            <div>
                <span style="font-weight:600;">{post.author_name}</span>
                <span style="color:gray; font-size:0.85em; margin-left:8px;">{_format_time(post.created_at)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"### {post.title}")
    st.markdown(post.body)

    # ── Replies ──
    st.markdown("---")
    replies = list_replies(post_id)
    st.markdown(f"**{len(replies)} {'reply' if len(replies) == 1 else 'replies'}**")

    for reply in replies:
        st.markdown(
            f"""
            <div style="border-left:3px solid rgba(128,128,128,0.3); padding-left:12px; margin:12px 0;">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
                    {_avatar_html(reply.author_name, 24)}
                    <span style="font-weight:600; font-size:0.9em;">{reply.author_name}</span>
                    <span style="color:gray; font-size:0.8em;">{_format_time(reply.created_at)}</span>
                </div>
                <div style="font-size:0.95em;">{reply.body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Reply form ──
    st.markdown("---")
    with st.form(key="reply_form", clear_on_submit=True):
        reply_body = st.text_area(
            "Write a reply",
            placeholder="Share your thoughts...",
            key="_reply_body",
        )
        submitted = st.form_submit_button("Reply", use_container_width=True)
        if submitted:
            if not reply_body.strip():
                st.error("Reply cannot be empty.")
            elif user_id == 0:
                st.error("You must be signed in to reply.")
            else:
                create_reply(post_id, user_id, reply_body)
                st.rerun()


# ══════════════════════════════════════════════════════════════════
# VIEW: Post list (default)
# ══════════════════════════════════════════════════════════════════

else:
    # ── New post form ──
    with st.expander("Start a new discussion", expanded=False):
        with st.form(key="new_post_form", clear_on_submit=True):
            post_title = st.text_input(
                "Title",
                placeholder="e.g. How to interpret WTP in a mixed logit?",
                key="_new_post_title",
            )
            post_body = st.text_area(
                "Body",
                placeholder="Describe your question or topic...",
                key="_new_post_body",
                height=120,
            )
            submitted = st.form_submit_button("Post", use_container_width=True)
            if submitted:
                if not post_title.strip():
                    st.error("Title is required.")
                elif not post_body.strip():
                    st.error("Body is required.")
                elif user_id == 0:
                    st.error("You must be signed in to post.")
                else:
                    create_post(user_id, post_title, post_body)
                    st.success("Posted!")
                    st.rerun()

    # ── Post list ──
    st.markdown("---")
    total = post_count()

    if total == 0:
        st.info("No discussions yet. Be the first to start one!")
    else:
        st.caption(f"{total} {'discussion' if total == 1 else 'discussions'}")
        posts = list_posts(limit=50)

        for post in posts:
            col_avatar, col_content, col_meta = st.columns([1, 8, 2])

            with col_avatar:
                st.markdown(
                    f"<div style='padding-top:4px;'>{_avatar_html(post.author_name, 36)}</div>",
                    unsafe_allow_html=True,
                )

            with col_content:
                if st.button(
                    post.title,
                    key=f"post_{post.id}",
                    use_container_width=True,
                ):
                    st.session_state["_community_view"] = "thread"
                    st.session_state["_community_post_id"] = post.id
                    st.rerun()
                st.caption(
                    f"by **{post.author_name}** · {_format_time(post.created_at)}"
                )

            with col_meta:
                st.markdown(
                    f"<div style='text-align:right; padding-top:8px; color:gray; font-size:0.85em;'>"
                    f"💬 {post.reply_count}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                "<hr style='margin:4px 0; border-color:rgba(128,128,128,0.1);'>",
                unsafe_allow_html=True,
            )

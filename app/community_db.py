# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under dual license: AGPL-3.0 (open-source) or commercial. See LICENSE.

"""Local SQLite storage for community archives and optional aggregate activity."""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_HF_DATA = Path("/data")  # HF Spaces persistent storage
_LOCAL_FALLBACK = Path(__file__).resolve().parent / "community.db"
DB_PATH = os.environ.get("PREFERO_DB_PATH") or (str(_HF_DATA / "community.db") if _HF_DATA.is_dir() else str(_LOCAL_FALLBACK))

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Get a thread-local SQLite connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT UNIQUE NOT NULL,
            email       TEXT DEFAULT '',
            avatar_seed TEXT DEFAULT '',
            joined_at   REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS posts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            author_id   INTEGER NOT NULL REFERENCES users(id),
            title       TEXT NOT NULL,
            body        TEXT NOT NULL,
            created_at  REAL NOT NULL,
            updated_at  REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS replies (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id     INTEGER NOT NULL REFERENCES posts(id),
            author_id   INTEGER NOT NULL REFERENCES users(id),
            body        TEXT NOT NULL,
            created_at  REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS activity_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   REAL NOT NULL,
            username    TEXT NOT NULL,
            action      TEXT NOT NULL,
            details     TEXT DEFAULT ''
        );

        CREATE INDEX IF NOT EXISTS idx_posts_created ON posts(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_replies_post   ON replies(post_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_activity_time  ON activity_log(timestamp DESC);
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Data classes for clean return types
# ---------------------------------------------------------------------------


@dataclass
class User:
    id: int
    username: str
    email: str
    avatar_seed: str
    joined_at: float


@dataclass
class Post:
    id: int
    author_id: int
    author_name: str
    title: str
    body: str
    created_at: float
    updated_at: float
    reply_count: int


@dataclass
class Reply:
    id: int
    post_id: int
    author_id: int
    author_name: str
    body: str
    created_at: float


# ---------------------------------------------------------------------------
# User operations
# ---------------------------------------------------------------------------


def create_user(username: str, email: str = "") -> User:
    """Create a new user. Raises ValueError if username taken."""
    conn = _get_conn()
    now = time.time()
    try:
        cur = conn.execute(
            "INSERT INTO users (username, email, avatar_seed, joined_at) "
            "VALUES (?, ?, ?, ?)",
            (username.strip(), email, username.strip(), now),
        )
        conn.commit()
        user = User(
            id=cur.lastrowid,
            username=username.strip(),
            email=email,
            avatar_seed=username.strip(),
            joined_at=now,
        )
        log_activity("register")
        return user
    except sqlite3.IntegrityError:
        raise ValueError(f"Username '{username}' is already taken.")


# ---------------------------------------------------------------------------
# Activity logging
# ---------------------------------------------------------------------------


def telemetry_enabled() -> bool:
    return os.environ.get("PREFERO_TELEMETRY_ENABLED", "").lower() == "true"


def log_activity(action: str) -> None:
    """Record a coarse, anonymous event locally when the operator opts in."""
    if not telemetry_enabled():
        return
    try:
        init_db()
        conn = _get_conn()
        conn.execute(
            "INSERT INTO activity_log (timestamp, username, action, details) VALUES (?, '', ?, '')",
            (time.time(), action),
        )
        conn.commit()
    except sqlite3.Error:
        logger.warning("Could not record local activity.")


def get_recent_activity(limit: int = 100) -> list[dict]:
    """Get recent activity log entries."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM activity_log ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_all_users() -> list[User]:
    """Get all registered users."""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM users ORDER BY joined_at DESC").fetchall()
    return [User(**dict(r)) for r in rows]


def get_user_by_name(username: str) -> User | None:
    """Look up a user by username."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username.strip(),)
    ).fetchone()
    if row is None:
        return None
    return User(**dict(row))


def get_user_by_id(user_id: int) -> User | None:
    """Look up a user by ID."""
    conn = _get_conn()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        return None
    return User(**dict(row))


def username_exists(username: str) -> bool:
    """Check if a username is taken."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT 1 FROM users WHERE username = ?", (username.strip(),)
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Post operations
# ---------------------------------------------------------------------------


def create_post(author_id: int, title: str, body: str) -> Post:
    """Create a new discussion post."""
    conn = _get_conn()
    now = time.time()
    cur = conn.execute(
        "INSERT INTO posts (author_id, title, body, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (author_id, title.strip(), body.strip(), now, now),
    )
    conn.commit()
    user = get_user_by_id(author_id)
    return Post(
        id=cur.lastrowid,
        author_id=author_id,
        author_name=user.username if user else "Unknown",
        title=title.strip(),
        body=body.strip(),
        created_at=now,
        updated_at=now,
        reply_count=0,
    )


def list_posts(limit: int = 50, offset: int = 0) -> list[Post]:
    """List posts ordered by most recent activity."""
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT p.*, u.username AS author_name,
               (SELECT COUNT(*) FROM replies r WHERE r.post_id = p.id) AS reply_count
        FROM posts p
        JOIN users u ON u.id = p.author_id
        ORDER BY p.updated_at DESC
        LIMIT ? OFFSET ?
        """,
        (limit, offset),
    ).fetchall()
    return [
        Post(
            id=r["id"],
            author_id=r["author_id"],
            author_name=r["author_name"],
            title=r["title"],
            body=r["body"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            reply_count=r["reply_count"],
        )
        for r in rows
    ]


def get_post(post_id: int) -> Post | None:
    """Get a single post by ID."""
    conn = _get_conn()
    row = conn.execute(
        """
        SELECT p.*, u.username AS author_name,
               (SELECT COUNT(*) FROM replies r WHERE r.post_id = p.id) AS reply_count
        FROM posts p
        JOIN users u ON u.id = p.author_id
        WHERE p.id = ?
        """,
        (post_id,),
    ).fetchone()
    if row is None:
        return None
    return Post(
        id=row["id"],
        author_id=row["author_id"],
        author_name=row["author_name"],
        title=row["title"],
        body=row["body"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        reply_count=row["reply_count"],
    )


def post_count() -> int:
    """Total number of posts."""
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) as cnt FROM posts").fetchone()
    return row["cnt"]


# ---------------------------------------------------------------------------
# Reply operations
# ---------------------------------------------------------------------------


def create_reply(post_id: int, author_id: int, body: str) -> Reply:
    """Add a reply to a post. Also bumps the post's updated_at."""
    conn = _get_conn()
    now = time.time()
    cur = conn.execute(
        "INSERT INTO replies (post_id, author_id, body, created_at) "
        "VALUES (?, ?, ?, ?)",
        (post_id, author_id, body.strip(), now),
    )
    conn.execute(
        "UPDATE posts SET updated_at = ? WHERE id = ?", (now, post_id)
    )
    conn.commit()
    user = get_user_by_id(author_id)
    return Reply(
        id=cur.lastrowid,
        post_id=post_id,
        author_id=author_id,
        author_name=user.username if user else "Unknown",
        body=body.strip(),
        created_at=now,
    )


def list_replies(post_id: int) -> list[Reply]:
    """Get all replies for a post, ordered chronologically."""
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT r.*, u.username AS author_name
        FROM replies r
        JOIN users u ON u.id = r.author_id
        WHERE r.post_id = ?
        ORDER BY r.created_at ASC
        """,
        (post_id,),
    ).fetchall()
    return [
        Reply(
            id=r["id"],
            post_id=r["post_id"],
            author_id=r["author_id"],
            author_name=r["author_name"],
            body=r["body"],
            created_at=r["created_at"],
        )
        for r in rows
    ]

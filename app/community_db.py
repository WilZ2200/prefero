"""SQLite-backed community database for the discussion forum.

Stores users (username, email, join date) and posts (author, title, body,
replies, timestamps).  Designed for single-instance deployment (HF Spaces).

User accounts are persisted to a private HF Dataset repo
(Wil2200/prefero-data) so they survive container restarts.
Posts/replies are ephemeral (in-memory SQLite).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IP geolocation helpers
# ---------------------------------------------------------------------------

# In-memory cache: ip -> {country, region, city, org, is_vpn}
_geo_cache: dict[str, dict] = {}
_GEO_CACHE_MAX = 500


def get_client_ip() -> str | None:
    """Extract client IP from Streamlit request headers (X-Forwarded-For)."""
    try:
        import streamlit as st
        xff = st.context.headers.get("x-forwarded-for")
        if xff and isinstance(xff, str):
            return xff.split(",")[0].strip()
        return None
    except Exception:
        return None


def _lookup_vpnapi(ip: str) -> dict | None:
    """Try vpnapi.io (HTTPS, free 1k/day, accurate VPN detection).

    Requires VPNAPI_KEY env var. Returns result dict or None on failure.
    """
    import os
    api_key = os.environ.get("VPNAPI_KEY", "")
    if not api_key:
        return None
    try:
        import requests
        resp = requests.get(
            f"https://vpnapi.io/api/{ip}?key={api_key}",
            timeout=4,
        )
        if resp.status_code == 200:
            data = resp.json()
            sec = data.get("security", {})
            loc = data.get("location", {})
            net = data.get("network", {})
            return {
                "country": loc.get("country", ""),
                "country_code": loc.get("country_code", ""),
                "region": loc.get("region", ""),
                "city": loc.get("city", ""),
                "org": net.get("autonomous_system_organization", ""),
                "is_vpn": bool(sec.get("vpn") or sec.get("proxy") or sec.get("tor") or sec.get("relay")),
            }
    except Exception as exc:
        logger.debug("vpnapi.io lookup failed for %s: %s", ip, exc)
    return None


def _lookup_ipapi(ip: str) -> dict | None:
    """Fallback: ip-api.com (HTTP, free 45/min, basic proxy detection)."""
    try:
        import requests
        resp = requests.get(
            f"http://ip-api.com/json/{ip}?fields=status,country,countryCode,regionName,city,isp,org,proxy,hosting",
            timeout=3,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                return {
                    "country": data.get("country", ""),
                    "country_code": data.get("countryCode", ""),
                    "region": data.get("regionName", ""),
                    "city": data.get("city", ""),
                    "org": data.get("isp", "") or data.get("org", ""),
                    "is_vpn": bool(data.get("proxy") or data.get("hosting")),
                }
    except Exception as exc:
        logger.debug("ip-api.com lookup failed for %s: %s", ip, exc)
    return None


def geolocate_ip(ip: str | None) -> dict:
    """Look up country, region, org, VPN status for an IP.

    Tries vpnapi.io first (if VPNAPI_KEY is set), falls back to ip-api.com.
    Returns dict with keys: country, region, city, org, is_vpn.
    Results are cached in-memory.
    """
    if not ip:
        return {}
    if ip in _geo_cache:
        return _geo_cache[ip]

    result = _lookup_vpnapi(ip) or _lookup_ipapi(ip) or {}

    # Cache result (even empty, to avoid repeated failed lookups)
    if len(_geo_cache) < _GEO_CACHE_MAX:
        _geo_cache[ip] = result
    return result


# ---------------------------------------------------------------------------
# HF Dataset persistence for user accounts
# ---------------------------------------------------------------------------

_HF_DATASET_REPO = "Wil2200/prefero-data"
_users_synced = False  # only sync once per process


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _load_users_from_hf() -> list[dict]:
    """Load user list from the HF dataset repo. Returns [] on failure."""
    token = _hf_token()
    if not token:
        logger.debug("No HF token — skipping user sync")
        return []
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=_HF_DATASET_REPO, filename="users.json",
            repo_type="dataset", token=token,
        )
        with open(path) as f:
            data = json.load(f)
        return data.get("users", [])
    except Exception as exc:
        logger.warning("Failed to load users from HF: %s", exc)
        return []


def _save_users_to_hf() -> None:
    """Persist the current user table to the HF dataset repo."""
    token = _hf_token()
    if not token:
        return
    try:
        from huggingface_hub import HfApi
        import tempfile

        conn = _get_conn()
        rows = conn.execute(
            "SELECT id, username, email, avatar_seed, joined_at FROM users ORDER BY id"
        ).fetchall()
        users = [dict(r) for r in rows]
        next_id = max((u["id"] for u in users), default=0) + 1

        data = {"users": users, "next_id": next_id}
        tmp = os.path.join(tempfile.gettempdir(), "prefero_users.json")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)

        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=tmp, path_in_repo="users.json",
            repo_id=_HF_DATASET_REPO, repo_type="dataset",
        )
        logger.info("Synced %d users to HF dataset", len(users))
    except Exception as exc:
        logger.warning("Failed to save users to HF: %s", exc)


def _save_activity_to_hf() -> None:
    """Persist recent activity log to the HF dataset repo."""
    token = _hf_token()
    if not token:
        return
    try:
        from huggingface_hub import HfApi
        import tempfile

        conn = _get_conn()
        rows = conn.execute(
            "SELECT * FROM activity_log ORDER BY timestamp DESC LIMIT 500"
        ).fetchall()
        logs = [dict(r) for r in rows]

        tmp = os.path.join(tempfile.gettempdir(), "prefero_activity.json")
        with open(tmp, "w") as f:
            json.dump({"logs": logs}, f, indent=2)

        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=tmp, path_in_repo="activity_log.json",
            repo_id=_HF_DATASET_REPO, repo_type="dataset",
        )
    except Exception as exc:
        logger.warning("Failed to save activity to HF: %s", exc)


def _sync_users_from_hf() -> None:
    """Load persisted users into the local SQLite on startup (once)."""
    global _users_synced
    if _users_synced:
        return
    _users_synced = True

    users = _load_users_from_hf()
    if not users:
        return

    conn = _get_conn()
    for u in users:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO users (id, username, email, avatar_seed, joined_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (u["id"], u["username"], u.get("email", ""),
                 u.get("avatar_seed", u["username"]), u["joined_at"]),
            )
        except Exception:
            pass
    conn.commit()
    logger.info("Loaded %d users from HF dataset into SQLite", len(users))

# ---------------------------------------------------------------------------
# Database path — persistent on HF Spaces at /data or fallback to app dir
# ---------------------------------------------------------------------------

_HF_DATA = Path("/data")  # HF Spaces persistent storage
_LOCAL_FALLBACK = Path(__file__).resolve().parent / "community.db"
DB_PATH = str(_HF_DATA / "community.db") if _HF_DATA.is_dir() else str(_LOCAL_FALLBACK)

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
    # Schema migration: add IP/geo columns to activity_log if missing
    try:
        conn.execute("ALTER TABLE activity_log ADD COLUMN ip_address TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass  # column already exists
    try:
        conn.execute("ALTER TABLE activity_log ADD COLUMN country TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE activity_log ADD COLUMN region TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE activity_log ADD COLUMN is_vpn INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    # Load persisted users from HF on first startup
    _sync_users_from_hf()


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
        log_activity(username.strip(), "register", f"New account created")
        # Persist to HF dataset repo
        _save_users_to_hf()
        return user
    except sqlite3.IntegrityError:
        raise ValueError(f"Username '{username}' is already taken.")


# ---------------------------------------------------------------------------
# Activity logging
# ---------------------------------------------------------------------------


def log_activity(username: str, action: str, details: str = "") -> None:
    """Record an activity event with IP geolocation."""
    try:
        conn = _get_conn()
        ip = get_client_ip()
        geo = geolocate_ip(ip) if ip else {}
        conn.execute(
            "INSERT INTO activity_log "
            "(timestamp, username, action, details, ip_address, country, region, is_vpn) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(), username, action, details,
                ip or "",
                geo.get("country", ""),
                geo.get("region", ""),
                1 if geo.get("is_vpn") else 0,
            ),
        )
        conn.commit()
    except Exception as exc:
        logger.warning("Failed to log activity: %s", exc)


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

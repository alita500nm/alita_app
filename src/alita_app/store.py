"""SQLite persistence for Alita — conversations, state, events.

Three tables:
  - conversations: all turns (user + alita), with timestamp and model.
  - state:         key/value for persistent settings
                   (e.g. "current_mood", "antennas_enabled", ...).
  - events:        timestamps and types for proactive triggers
                   (e.g. "wake_word_heard", "user_left", ...).
"""

import os
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional


# DB path — override via ALITA_DB_PATH env var, default to <project_root>/data/memory.db
DB_PATH = Path(
    os.getenv(
        "ALITA_DB_PATH",
        str(Path(__file__).resolve().parent.parent.parent / "data" / "memory.db"),
    )
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    role        TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    model       TEXT,
    route       TEXT,
    metadata    TEXT
);

CREATE INDEX IF NOT EXISTS idx_conv_timestamp ON conversations(timestamp);

CREATE TABLE IF NOT EXISTS state (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    type        TEXT    NOT NULL,
    data        TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);

CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    tags        TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp);
"""


def init_db(db_path: Path = DB_PATH) -> None:
    """Create DB file and schema if not present."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)


@contextmanager
def get_conn(db_path: Path = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for DB access. Auto-commit, auto-close."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# --- Conversations ---

def save_turn(
    role: str,
    content: str,
    model: Optional[str] = None,
    route: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> int:
    """Save a single conversation turn. Returns the inserted row id."""
    if role not in ("user", "assistant"):
        raise ValueError(f"role must be 'user' or 'assistant', not: {role}")

    timestamp = datetime.utcnow().isoformat()
    metadata_json = json.dumps(metadata) if metadata else None

    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO conversations (timestamp, role, content, model, route, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (timestamp, role, content, model, route, metadata_json),
        )
        return cur.lastrowid  # type: ignore[return-value]


def recent_turns(n: int = 20) -> list[dict]:
    """Return the last n turns, chronological (oldest first)."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT timestamp, role, content, model, route
            FROM conversations
            ORDER BY id DESC
            LIMIT ?
            """,
            (n,),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def as_chat_history(n: int = 20) -> list[dict]:
    """Return recent turns formatted for Anthropic Messages API.

    Format: [{role, content}, ...] — oldest turn first.
    """
    turns = recent_turns(n)
    return [{"role": t["role"], "content": t["content"]} for t in turns]


# --- State ---

def get_state(key: str, default=None):
    """Retrieve value for key. Auto-deserialised from JSON."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT value FROM state WHERE key = ?", (key,)
        ).fetchone()
    if row is None:
        return default
    return json.loads(row["value"])


def set_state(key: str, value) -> None:
    """Store value for key as JSON (upsert)."""
    timestamp = datetime.utcnow().isoformat()
    value_json = json.dumps(value)
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO state (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (key, value_json, timestamp),
        )


# --- Events ---

# --- Memories ---

def save_memory(
    content: str,
    tags: Optional[list[str]] = None,
) -> int:
    """Save a memory entry. Returns the inserted row id."""
    timestamp = datetime.utcnow().isoformat()
    tags_json = json.dumps(tags) if tags else None
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO memories (timestamp, content, tags) VALUES (?, ?, ?)",
            (timestamp, content, tags_json),
        )
        return cur.lastrowid  # type: ignore[return-value]


def search_memories(query: Optional[str] = None, limit: int = 10) -> list[dict]:
    """Search memories by content (LIKE) or return most recent."""
    with get_conn() as conn:
        if query:
            rows = conn.execute(
                "SELECT id, timestamp, content, tags FROM memories "
                "WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, timestamp, content, tags FROM memories "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()

    results = []
    for r in rows:
        entry = dict(r)
        if entry.get("tags"):
            try:
                entry["tags"] = json.loads(entry["tags"])
            except Exception:
                pass
        results.append(entry)
    return results


# --- Events ---

def log_event(event_type: str, data: Optional[dict] = None) -> int:
    """Log an event. Returns the inserted row id."""
    timestamp = datetime.utcnow().isoformat()
    data_json = json.dumps(data) if data else None
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO events (timestamp, type, data) VALUES (?, ?, ?)",
            (timestamp, event_type, data_json),
        )
        return cur.lastrowid  # type: ignore[return-value]

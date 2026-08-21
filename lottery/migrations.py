"""SQLite schema migrations for M4.5.

Migrations are ordered, idempotent, and recorded in schema_version. Existing
installations are safe because each migration checks its target objects.
"""
from __future__ import annotations

import sqlite3
from typing import Callable, List, Tuple


def _ensure_reconcile_runs(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS reconcile_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        checked_at REAL NOT NULL,
        secondary_url TEXT DEFAULT '',
        status TEXT NOT NULL,
        summary_json TEXT NOT NULL DEFAULT '{}'
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reconcile_runs_checked_at ON reconcile_runs(checked_at)")


def _ensure_legacy_columns(conn: sqlite3.Connection) -> None:
    for table, column, definition in (
        ("predictions", "evidence_json", "TEXT DEFAULT ''"),
        ("draws", "source", "TEXT NOT NULL DEFAULT '17500'"),
    ):
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if not columns:
            continue
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


MIGRATIONS: List[Tuple[int, str, Callable[[sqlite3.Connection], None]]] = [
    (1, "reconcile audit table", _ensure_reconcile_runs),
    (2, "legacy provenance columns", _ensure_legacy_columns),
]


def apply(conn: sqlite3.Connection) -> int:
    conn.execute("""CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        applied_at REAL NOT NULL
    )""")
    current = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version").fetchone()[0]
    import time
    for version, name, migration in MIGRATIONS:
        if version <= current:
            continue
        migration(conn)
        conn.execute("INSERT INTO schema_version(version, name, applied_at) VALUES (?,?,?)",
                     (version, name, time.time()))
        current = version
    conn.commit()
    return int(current)


def status(conn: sqlite3.Connection) -> dict:
    row = conn.execute("SELECT COALESCE(MAX(version), 0) AS version FROM schema_version").fetchone()
    return {"version": int(row["version"] if hasattr(row, "keys") else row[0]),
            "latest": MIGRATIONS[-1][0] if MIGRATIONS else 0}

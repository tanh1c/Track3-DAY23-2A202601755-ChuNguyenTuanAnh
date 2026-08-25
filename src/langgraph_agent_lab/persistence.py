"""Checkpointer adapters for in-memory and durable execution."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def _sqlite_path(database_url: str | None) -> Path:
    """Normalize the lab's plain-path or sqlite:/// database setting."""
    raw = database_url or "outputs/checkpoints.sqlite"
    if raw.startswith("sqlite:///"):
        raw = raw.removeprefix("sqlite:///")
    path = Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _sqlite_connection(database_url: str | None) -> sqlite3.Connection:
    """Create a thread-safe SQLite connection configured for durable WAL writes."""
    connection = sqlite3.connect(_sqlite_path(database_url), check_same_thread=False)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return the configured LangGraph checkpointer."""
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        return SqliteSaver(_sqlite_connection(database_url))
    if kind == "postgres":
        raise RuntimeError("Postgres checkpointer is an optional extension and is not enabled")
    raise ValueError(f"Unknown checkpointer kind: {kind}")

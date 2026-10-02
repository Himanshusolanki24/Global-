"""SQLite, one row per call. Each write commits immediately, so a crash loses at most the call in flight.
The `meta` table carries everything later notebook cells need (host info, router output), so
export.py reads exactly one file."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from bench.core.schemas import RunRecord

_COLS = list(RunRecord.model_fields)
_KEY = ("kind", "config_id", "task_id", "item_id", "repeat_index")

Key = tuple[str, str, str, str, int]


class Store:
    def __init__(self, path: str | Path) -> None:
        self.db = sqlite3.connect(str(path))
        self.db.execute("PRAGMA journal_mode=WAL")
        cols = ", ".join(_COLS)
        self.db.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS runs ({cols}, PRIMARY KEY ({", ".join(_KEY)}));
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )

    def done(self) -> set[Key]:
        """Keys already stored without error. Errored calls are retried on the next run."""
        q = f"SELECT {', '.join(_KEY)} FROM runs WHERE error IS NULL"
        return {tuple(r) for r in self.db.execute(q)}  # type: ignore[misc]

    def write(self, rec: RunRecord) -> None:
        d = rec.model_dump()
        self.db.execute(
            f"INSERT OR REPLACE INTO runs ({', '.join(_COLS)}) VALUES ({', '.join('?' * len(_COLS))})",
            [d[c] for c in _COLS],
        )
        self.db.commit()

    def rows(self) -> list[RunRecord]:
        cur = self.db.execute(f"SELECT {', '.join(_COLS)} FROM runs ORDER BY {', '.join(_KEY)}")
        return [RunRecord(**dict(zip(_COLS, r))) for r in cur]

    def put_meta(self, key: str, value: Any) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value, ensure_ascii=False)))
        self.db.commit()

    def get_meta(self, key: str, default: Any = KeyError) -> Any:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            if default is not KeyError:
                return default
            raise KeyError(f"meta '{key}' missing — run the notebook cell that produces it first")
        return json.loads(row[0])

    def close(self) -> None:
        self.db.close()

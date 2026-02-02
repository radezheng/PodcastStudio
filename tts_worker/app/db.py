from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class Db:
    path: str

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA busy_timeout = 30000')
        return conn


def init_db(db: Db) -> None:
    with db.connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                script_id TEXT,
                status TEXT NOT NULL,
                output_path TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()

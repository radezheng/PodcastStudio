from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class Db:
    path: str

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn


def init_db(db: Db) -> None:
    with db.connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scripts (
                id TEXT PRIMARY KEY,
                theme TEXT NOT NULL,
                content TEXT NOT NULL,
                confirmed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS script_meta (
                script_id TEXT PRIMARY KEY,
                minutes INTEGER NOT NULL,
                speaker_names_json TEXT NOT NULL,
                system_prompt TEXT,
                use_web_search INTEGER NOT NULL,
                source_filename TEXT,
                source_url TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(script_id) REFERENCES scripts(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tts_jobs (
                id TEXT PRIMARY KEY,
                script_id TEXT NOT NULL,
                worker_job_id TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(script_id) REFERENCES scripts(id)
            )
            """
        )
        conn.commit()

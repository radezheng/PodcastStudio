from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class LogEvent:
    ts: float
    stage: str
    type: str  # stage_start | stage_end | step | detail | error
    message: str
    data: dict[str, Any] | None = None


class ScriptLogWriter:
    def __init__(self, *, file_path: Path, script_id: str):
        self._file_path = file_path
        self._script_id = script_id
        self._lock = Lock()
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, event: LogEvent) -> None:
        line = json.dumps(
            {
                "ts": event.ts,
                "script_id": self._script_id,
                "stage": event.stage,
                "type": event.type,
                "message": event.message,
                "data": event.data,
            },
            ensure_ascii=False,
        )

        with self._lock:
            with self._file_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def stage_start(self, stage: str, title: str) -> None:
        self._write(LogEvent(ts=time.time(), stage=stage, type="stage_start", message=title))
        print(f"[script:{self._script_id}] [{stage}] START {title}")

    def stage_end(self, stage: str, message: str = "ok") -> None:
        self._write(LogEvent(ts=time.time(), stage=stage, type="stage_end", message=message))
        print(f"[script:{self._script_id}] [{stage}] END {message}")

    def step(self, stage: str, message: str, *, data: dict[str, Any] | None = None) -> None:
        self._write(LogEvent(ts=time.time(), stage=stage, type="step", message=message, data=data))
        print(f"[script:{self._script_id}] [{stage}] {message}")

    def detail(self, stage: str, message: str, *, data: dict[str, Any] | None = None) -> None:
        self._write(LogEvent(ts=time.time(), stage=stage, type="detail", message=message, data=data))

    def error(self, stage: str, message: str, *, data: dict[str, Any] | None = None) -> None:
        self._write(LogEvent(ts=time.time(), stage=stage, type="error", message=message, data=data))
        print(f"[script:{self._script_id}] [{stage}] ERROR {message}")


def read_script_log_events(file_path: Path) -> list[dict[str, Any]]:
    if not file_path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def group_events_by_stage(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Preserve stage order by first appearance.
    stage_order: list[str] = []
    titles: dict[str, str] = {}
    buckets: dict[str, list[dict[str, Any]]] = {}

    for e in events:
        stage = str(e.get("stage") or "general")
        if stage not in buckets:
            buckets[stage] = []
            stage_order.append(stage)
        buckets[stage].append(e)
        if e.get("type") == "stage_start" and stage not in titles:
            titles[stage] = str(e.get("message") or stage)

    grouped: list[dict[str, Any]] = []
    for stage in stage_order:
        grouped.append(
            {
                "stage": stage,
                "title": titles.get(stage, stage),
                "events": buckets.get(stage, []),
            }
        )
    return grouped

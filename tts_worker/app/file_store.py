from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        return None


@dataclass(frozen=True)
class WorkerFileStore:
    root: Path

    @property
    def jobs_dir(self) -> Path:
        return self.root / "jobs"

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def job_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def audio_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "audio.wav"

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return _read_json(self.job_path(job_id))

    def put_job(self, job: dict[str, Any]) -> None:
        job_id = str(job.get("job_id") or "").strip()
        if not job_id:
            raise ValueError("job_id required")
        _atomic_write_json(self.job_path(job_id), job)

    def list_jobs(self) -> list[dict[str, Any]]:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        jobs: list[dict[str, Any]] = []
        for p in self.jobs_dir.glob("*/job.json"):
            data = _read_json(p)
            if data and isinstance(data, dict):
                jobs.append(data)
        return jobs

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
class BackendFileStore:
    root: Path

    @property
    def scripts_dir(self) -> Path:
        return self.root / "scripts"

    @property
    def jobs_dir(self) -> Path:
        return self.root / "jobs"

    def script_dir(self, script_id: str) -> Path:
        return self.scripts_dir / script_id

    def script_path(self, script_id: str) -> Path:
        return self.script_dir(script_id) / "script.json"

    def meta_path(self, script_id: str) -> Path:
        return self.script_dir(script_id) / "meta.json"

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def job_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def get_script(self, script_id: str) -> dict[str, Any] | None:
        return _read_json(self.script_path(script_id))

    def put_script(self, script: dict[str, Any]) -> None:
        script_id = str(script.get("script_id") or "").strip()
        if not script_id:
            raise ValueError("script_id required")
        _atomic_write_json(self.script_path(script_id), script)

    def get_meta(self, script_id: str) -> dict[str, Any] | None:
        return _read_json(self.meta_path(script_id))

    def put_meta(self, script_id: str, meta: dict[str, Any]) -> None:
        self.script_dir(script_id).mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.meta_path(script_id), meta)

    def list_scripts(self) -> list[dict[str, Any]]:
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        scripts: list[dict[str, Any]] = []
        for p in self.scripts_dir.glob("*/script.json"):
            data = _read_json(p)
            if data and isinstance(data, dict):
                scripts.append(data)
        return scripts

    def delete_script(self, script_id: str) -> None:
        # Delete script folder.
        sdir = self.script_dir(script_id)
        if sdir.exists():
            for child in sorted(sdir.rglob("*"), reverse=True):
                try:
                    if child.is_file() or child.is_symlink():
                        child.unlink()
                    else:
                        child.rmdir()
                except Exception:
                    pass
            try:
                sdir.rmdir()
            except Exception:
                pass

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

    def delete_jobs_for_script(self, script_id: str) -> None:
        for job in self.list_jobs():
            if str(job.get("script_id") or "") == script_id:
                jid = str(job.get("job_id") or "")
                if jid:
                    self.delete_job(jid)

    def delete_job(self, job_id: str) -> None:
        jdir = self.job_dir(job_id)
        if jdir.exists():
            for child in sorted(jdir.rglob("*"), reverse=True):
                try:
                    if child.is_file() or child.is_symlink():
                        child.unlink()
                    else:
                        child.rmdir()
                except Exception:
                    pass
            try:
                jdir.rmdir()
            except Exception:
                pass

from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .db import Db, init_db
from .env import load_project_env
from .vibevoice_job import run_vibevoice_inference


load_project_env()

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # PodcastStudio/


def _storage_root() -> Path:
    raw = os.environ.get("PODCASTSTUDIO_STORAGE_DIR", "").strip().strip('"')
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        return p
    return (PROJECT_ROOT / "storage").resolve()


STORAGE_ROOT = _storage_root()
DB_DIR = STORAGE_ROOT / "db" / "tts_worker"
DB_DIR.mkdir(parents=True, exist_ok=True)

DB = Db(path=str(DB_DIR / "worker.db"))
init_db(DB)

app = FastAPI(title="VibeVoice TTS Worker", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def _process_job(
    *,
    job_id: str,
    script_id: str | None,
    script_text: str,
    speaker_names: list[str] | None,
) -> None:
    with DB.connect() as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
            ("running", _utc_now(), job_id),
        )
        conn.commit()

    try:
        out_path = run_vibevoice_inference(
            script_text=script_text,
            script_id=script_id,
            speaker_names=speaker_names,
        )
        with DB.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, output_path = ?, error = NULL, updated_at = ? WHERE id = ?",
                ("completed", str(out_path), _utc_now(), job_id),
            )
            conn.commit()
    except Exception as e:
        with DB.connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                ("failed", str(e), _utc_now(), job_id),
            )
            conn.commit()


class SubmitJobRequest(BaseModel):
    script_id: str | None = None
    script_text: str = Field(..., min_length=1)
    speaker_names: list[str] | None = None


class SubmitJobResponse(BaseModel):
    job_id: str
    status: str


class JobResponse(BaseModel):
    job_id: str
    script_id: str | None = None
    status: str
    output_filename: str | None = None
    error: str | None = None


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/jobs", response_model=SubmitJobResponse)
def submit_job(req: SubmitJobRequest):
    job_id = str(uuid.uuid4())

    with DB.connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id, script_id, status, output_path, error, created_at, updated_at) VALUES (?, ?, ?, NULL, NULL, ?, ?)",
            (job_id, req.script_id, "queued", _utc_now(), _utc_now()),
        )
        conn.commit()

    _EXECUTOR.submit(
        _process_job,
        job_id=job_id,
        script_id=req.script_id,
        script_text=req.script_text,
        speaker_names=req.speaker_names,
    )

    return SubmitJobResponse(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str):
    with DB.connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="job not found")

    output_filename = None
    if row["output_path"]:
        output_filename = Path(row["output_path"]).name

    return JobResponse(
        job_id=row["id"],
        script_id=row["script_id"],
        status=row["status"],
        output_filename=output_filename,
        error=row["error"],
    )


@app.get("/jobs/{job_id}/audio")
def download_audio(job_id: str):
    with DB.connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    if row["status"] != "completed" or not row["output_path"]:
        raise HTTPException(status_code=400, detail="job not completed")

    path = Path(row["output_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="audio file missing")

    return FileResponse(
        path=str(path),
        media_type="audio/wav",
        filename=path.name,
        headers={"x-filename": path.name},
    )

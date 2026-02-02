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

from .env import load_project_env
from .file_store import WorkerFileStore
from .vibevoice_engine import start_background_warmup
from .vibevoice_job import run_vibevoice_inference


load_project_env()


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "VibeVoice").is_dir():
            return parent
    return here.parents[2]


PROJECT_ROOT = _find_project_root()


def _storage_root() -> Path:
    raw = os.environ.get("PODCASTSTUDIO_STORAGE_DIR", "").strip().strip('"')
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        return p
    return (PROJECT_ROOT / "storage").resolve()


STORAGE_ROOT = _storage_root()


def _db_dir(default: Path, *, env_var: str) -> Path:
    raw = os.environ.get(env_var, "").strip().strip('"')
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        return p
    return default


DB_DIR = _db_dir(STORAGE_ROOT / "db" / "tts_worker", env_var="PODCASTSTUDIO_DB_DIR_TTS_WORKER")
DB_DIR.mkdir(parents=True, exist_ok=True)

STORE = WorkerFileStore(root=DB_DIR)

app = FastAPI(title="VibeVoice TTS Worker", version="0.1.0")


@app.on_event("startup")
def _startup_warmup() -> None:
    # Start model download/loading early without blocking worker startup.
    start_background_warmup()

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
    job = STORE.get_job(job_id) or {}
    job.update({"status": "running", "updated_at": _utc_now()})
    STORE.put_job(job)

    try:
        job_dir = STORE.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_vibevoice_inference(
            job_id=job_id,
            script_text=script_text,
            script_id=script_id,
            speaker_names=speaker_names,
            output_dir=job_dir,
        )

        # Normalize to a stable filename for downloads.
        audio_path = STORE.audio_path(job_id)
        try:
            if out_path != audio_path:
                audio_path.write_bytes(Path(out_path).read_bytes())
        except Exception:
            # Fall back to whatever path we have.
            audio_path = Path(out_path)

        job = STORE.get_job(job_id) or {}
        job.update(
            {
                "status": "completed",
                "output_path": str(audio_path),
                "error": None,
                "updated_at": _utc_now(),
            }
        )
        STORE.put_job(job)
    except Exception as e:
        job = STORE.get_job(job_id) or {}
        job.update({"status": "failed", "error": str(e), "updated_at": _utc_now()})
        STORE.put_job(job)


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

    job = {
        "job_id": job_id,
        "script_id": req.script_id,
        "status": "queued",
        "output_path": None,
        "error": None,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
    }
    STORE.put_job(job)

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
    row = STORE.get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")

    output_filename = None
    if row.get("output_path"):
        output_filename = Path(str(row.get("output_path"))).name

    return JobResponse(
        job_id=str(row.get("job_id")),
        script_id=(row.get("script_id") if row.get("script_id") is not None else None),
        status=str(row.get("status") or "queued"),
        output_filename=output_filename,
        error=(row.get("error") if row.get("error") is not None else None),
    )


@app.get("/jobs/{job_id}/audio")
def download_audio(job_id: str):
    row = STORE.get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    if str(row.get("status")) != "completed" or not row.get("output_path"):
        raise HTTPException(status_code=400, detail="job not completed")

    path = Path(str(row.get("output_path")))
    if not path.exists():
        raise HTTPException(status_code=404, detail="audio file missing")

    return FileResponse(
        path=str(path),
        media_type="audio/wav",
        filename=path.name,
        headers={"x-filename": path.name},
    )

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from .azure_openai import DEFAULT_PODCAST_SYSTEM_PROMPT, generate_podcast_script
from .env import load_project_env
from .file_store import BackendFileStore
from .script_logs import ScriptLogWriter, group_events_by_stage, read_script_log_events


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
CONFIG_DIR = STORAGE_ROOT / "config"


def _db_dir(default: Path, *, env_var: str) -> Path:
    raw = os.environ.get(env_var, "").strip().strip('"')
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        return p
    return default


DB_DIR = _db_dir(STORAGE_ROOT / "db" / "backend", env_var="PODCASTSTUDIO_DB_DIR_BACKEND")
LOG_DIR = STORAGE_ROOT / "logs" / "backend" / "scripts"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT_PATH = CONFIG_DIR / "system_prompt.txt"
SPEAKERS_PATH = CONFIG_DIR / "speakers.json"

STORE = BackendFileStore(root=DB_DIR)

TTS_WORKER_URL = os.environ.get("TTS_WORKER_URL", "http://localhost:8002").strip().strip('"')

app = FastAPI(title="Podcast Studio API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"] ,
)


@app.get("/config.js", include_in_schema=False)
def get_runtime_config_js() -> Response:
    api_base = os.environ.get("PODCASTSTUDIO_API_BASE", "").strip().strip('"')
    body = "window.__PODCASTSTUDIO_CONFIG__ = " + json.dumps({"apiBase": api_base}) + ";\n"
    return Response(content=body, media_type="application/javascript")


class GenerateScriptRequest(BaseModel):
    script_id: str | None = Field(default=None, description="Optional client-provided UUID for log polling")
    theme: str = Field(default="", max_length=200)
    minutes: int = Field(default=4, ge=1, le=20)
    speaker_names: list[str] | None = Field(default=None, description="Selected speakers (1-4)")
    system_prompt: str | None = Field(default=None, max_length=4000)
    source_filename: str | None = Field(default=None, max_length=200)
    source_text: str | None = Field(default=None, max_length=20000)
    source_url: str | None = Field(default=None, max_length=2000)
    use_web_search: bool = Field(default=False)


class ScriptResponse(BaseModel):
    script_id: str
    theme: str
    content: str
    confirmed: bool


class ScriptListItem(BaseModel):
    script_id: str
    theme: str
    confirmed: bool
    created_at: str
    job_count: int = 0
    last_job_status: str | None = None


class ScriptListResponse(BaseModel):
    scripts: list[ScriptListItem]


class ScriptDetailResponse(BaseModel):
    script_id: str
    theme: str
    content: str
    confirmed: bool
    created_at: str


class ScriptMetaResponse(BaseModel):
    minutes: int
    speaker_names: list[str]
    system_prompt: str | None = None
    use_web_search: bool
    source_filename: str | None = None
    source_url: str | None = None


class ScriptJobItem(BaseModel):
    job_id: str
    worker_job_id: str
    status: str
    worker_status: str | None = None
    output_filename: str | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class ScriptFullResponse(BaseModel):
    script: ScriptDetailResponse
    meta: ScriptMetaResponse | None = None
    tts_jobs: list[ScriptJobItem]


class ScriptLogsResponse(BaseModel):
    script_id: str
    stages: list[dict]


class UpdateScriptRequest(BaseModel):
    content: str = Field(..., min_length=1)


class ConfirmResponse(BaseModel):
    script_id: str
    confirmed: bool


class SubmitTTSResponse(BaseModel):
    script_id: str
    job_id: str
    worker_job_id: str
    status: str


class SubmitTTSRequest(BaseModel):
    speaker_names: list[str] | None = Field(default=None, description="Selected speakers (1-4)")


class AppConfigResponse(BaseModel):
    supported_speakers: list[str]
    max_speakers: int
    default_system_prompt: str
    builtin_system_prompt: str


class SaveSystemPromptRequest(BaseModel):
    system_prompt: str = Field(..., min_length=1, max_length=8000)


class JobStatus(BaseModel):
    job_id: str
    script_id: str
    worker_job_id: str
    status: str
    worker_status: str | None = None
    output_filename: str | None = None
    error: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_speaker_list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        try:
            v = json.loads(value)
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
        except Exception:
            pass
        return [s.strip() for s in value.split() if s.strip()]
    return []


@app.get("/health")
def health():
    return {"ok": True, "tts_worker_url": TTS_WORKER_URL}


def _poke_tts_worker() -> None:
    # Best-effort: this call exists mainly to trigger scale-from-zero.
    try:
        timeout = httpx.Timeout(connect=5.0, read=60.0, write=60.0, pool=5.0)
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            client.get(f"{TTS_WORKER_URL}/health")
    except Exception:
        # Intentionally ignore errors; this should never break the UI.
        return


@app.post("/api/tts-worker/activate")
def activate_tts_worker(bg: BackgroundTasks):
    bg.add_task(_poke_tts_worker)
    return {"scheduled": True}


@app.get("/api/config", response_model=AppConfigResponse)
def get_config():
    # 1) Start with all voice presets shipped with the bundled VibeVoice repo.
    voices_dir = PROJECT_ROOT / "VibeVoice" / "demo" / "voices"
    speakers: set[str] = set()
    if voices_dir.exists():
        for p in voices_dir.glob("*.wav"):
            name = p.stem
            if "_" in name:
                name = name.split("_", 1)[0]
            if "-" in name:
                name = name.split("-")[-1]
            name = name.strip()
            if name:
                speakers.add(name)

    # 2) Merge user-defined speakers list if present.
    if SPEAKERS_PATH.exists():
        try:
            data = json.loads(SPEAKERS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str) and item.strip():
                        speakers.add(item.strip())
        except Exception:
            # Ignore malformed file; UI still works with scanned presets.
            pass

    # 3) Merge env as last resort.
    raw = os.environ.get("VIBEVOICE_SPEAKER_NAMES", "").strip().strip('"')
    for s in raw.split():
        if s.strip():
            speakers.add(s.strip())

    if not speakers:
        speakers = {"Xinran", "Anchen"}

    # Load a persisted system prompt if present; otherwise seed it once.
    if SYSTEM_PROMPT_PATH.exists():
        system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    else:
        system_prompt = DEFAULT_PODCAST_SYSTEM_PROMPT
        SYSTEM_PROMPT_PATH.write_text(system_prompt + "\n", encoding="utf-8")

    # UI limit; VibeVoice supports up to 4 speakers.
    max_speakers = 4
    return AppConfigResponse(
        supported_speakers=sorted(speakers),
        max_speakers=max_speakers,
        default_system_prompt=system_prompt,
        builtin_system_prompt=DEFAULT_PODCAST_SYSTEM_PROMPT,
    )


@app.put("/api/config/system-prompt", response_model=AppConfigResponse)
def save_system_prompt(req: SaveSystemPromptRequest):
    SYSTEM_PROMPT_PATH.write_text(req.system_prompt.strip() + "\n", encoding="utf-8")
    return get_config()


@app.post("/api/scripts/generate", response_model=ScriptResponse)
def generate_script(req: GenerateScriptRequest):
    script_id = (req.script_id or "").strip() or str(uuid.uuid4())
    try:
        uuid.UUID(script_id)
    except Exception:
        raise HTTPException(status_code=400, detail="script_id must be a valid UUID")

    # Prevent collisions if client retries with the same id.
    if STORE.get_script(script_id) is not None:
        raise HTTPException(status_code=409, detail="script_id already exists")

    log = ScriptLogWriter(file_path=(LOG_DIR / f"{script_id}.jsonl"), script_id=script_id)
    log.stage_start("request", "Validate request")

    theme = (req.theme or "").strip()
    source_text = (req.source_text or "").strip()
    source_filename = (req.source_filename or "").strip() or None
    source_url = (req.source_url or "").strip() or None

    selected = [s.strip() for s in (req.speaker_names or []) if s and s.strip()]
    if selected and len(selected) > 4:
        raise HTTPException(status_code=400, detail="speaker_names max is 4")
    speaker_count = len(selected) if selected else 2

    def _fetch_source_url(url: str) -> tuple[str, str | None]:
        u = urlparse(url)
        if u.scheme not in {"http", "https"} or not u.netloc:
            raise HTTPException(status_code=400, detail="source_url must be http(s)")

        # Basic allowlist by extension/content-type; this can be relaxed later.
        path_lower = (u.path or "").lower()
        if not (path_lower.endswith(".txt") or path_lower.endswith(".md") or path_lower.endswith(".markdown") or path_lower.endswith("/")):
            # Still allow if server returns text/*.
            pass

        try:
            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                r = client.get(url)
                r.raise_for_status()
                ctype = (r.headers.get("content-type") or "").lower()
                if not ("text/plain" in ctype or "text/markdown" in ctype or ctype.startswith("text/")):
                    raise HTTPException(status_code=400, detail=f"source_url content-type not supported: {ctype or 'unknown'}")

                # Limit size to avoid huge downloads.
                raw = r.content
                if len(raw) > 2_000_000:
                    raw = raw[:2_000_000]

                text = raw.decode("utf-8", errors="replace")
                if len(text) > 20000:
                    text = text[:20000]

                filename = None
                if u.path:
                    filename = u.path.split("/")[-1] or None
                return text.strip(), filename
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"failed to download source_url: {e}")

    if not source_text and source_url:
        log.stage_end("request", "ok")
        log.stage_start("source", "Download source_url")
        log.step("source", "Downloading URL", data={"url": source_url})
        source_text, fetched_name = _fetch_source_url(source_url)
        log.step("source", "Downloaded", data={"chars": len(source_text)})
        if not source_filename:
            source_filename = fetched_name or source_url
        log.stage_end("source", "ok")

    if not theme and not source_text:
        log.error("request", "Missing theme/source")
        raise HTTPException(status_code=400, detail="theme, source_text, or source_url is required")
    if not theme:
        theme = source_filename or "Uploaded file"

    log.step(
        "request",
        "Inputs ready",
        data={
            "theme": theme,
            "minutes": req.minutes,
            "use_web_search": req.use_web_search,
            "speaker_count": speaker_count,
            "has_source_text": bool(source_text),
        },
    )
    log.stage_end("request", "ok")

    try:
        log.stage_start("generation", "Generate script")
        content = generate_podcast_script(
            theme=theme,
            minutes=req.minutes,
            speaker_count=speaker_count,
            system_prompt=req.system_prompt,
            source_text=source_text or None,
            use_web_search=req.use_web_search,
            log=log,
        )
        log.stage_end("generation", "ok")
    except Exception as e:
        log.error("generation", "Generation failed", data={"error": str(e)})
        log.stage_end("generation", "failed")
        raise

    log.stage_start("db", "Persist script")
    now = _utc_now()
    STORE.put_script(
        {
            "script_id": script_id,
            "theme": theme,
            "content": content,
            "confirmed": False,
            "created_at": now,
            "updated_at": now,
        }
    )
    STORE.put_meta(
        script_id,
        {
            "minutes": int(req.minutes),
            "speaker_names": selected,
            "system_prompt": (req.system_prompt if req.system_prompt is not None else None),
            "use_web_search": bool(req.use_web_search),
            "source_filename": (source_filename if source_filename else None),
            "source_url": (source_url if source_url else None),
            "created_at": now,
        },
    )
    log.stage_end("db", "ok")

    return ScriptResponse(script_id=script_id, theme=theme, content=content, confirmed=False)


@app.get("/api/scripts", response_model=ScriptListResponse)
def list_scripts(limit: int = 50):
    limit = max(1, min(int(limit), 200))
    scripts_raw = STORE.list_scripts()
    jobs_raw = STORE.list_jobs()

    # Group job stats by script_id for list view.
    by_script: dict[str, list[dict]] = {}
    for j in jobs_raw:
        sid = str(j.get("script_id") or "")
        if not sid:
            continue
        by_script.setdefault(sid, []).append(j)

    def _created_at(item: dict) -> str:
        return str(item.get("created_at") or "")

    scripts_raw.sort(key=_created_at, reverse=True)
    scripts_raw = scripts_raw[:limit]

    scripts: list[ScriptListItem] = []
    for s in scripts_raw:
        sid = str(s.get("script_id") or "")
        jobs = by_script.get(sid, [])
        jobs_sorted = sorted(jobs, key=lambda x: str(x.get("created_at") or ""), reverse=True)
        last_status = None
        if jobs_sorted:
            last_status = jobs_sorted[0].get("status")
        scripts.append(
            ScriptListItem(
                script_id=sid,
                theme=str(s.get("theme") or ""),
                confirmed=bool(s.get("confirmed")),
                created_at=str(s.get("created_at") or ""),
                job_count=len(jobs),
                last_job_status=(str(last_status) if last_status is not None else None),
            )
        )
    return ScriptListResponse(scripts=scripts)


@app.get("/api/scripts/{script_id}/logs", response_model=ScriptLogsResponse)
def get_script_logs(script_id: str):
    path = LOG_DIR / f"{script_id}.jsonl"
    events = read_script_log_events(path)
    return ScriptLogsResponse(script_id=script_id, stages=group_events_by_stage(events))


@app.get("/api/scripts/{script_id}", response_model=ScriptResponse)
def get_script(script_id: str):
    row = STORE.get_script(script_id)
    if row is None:
        raise HTTPException(status_code=404, detail="script not found")
    return ScriptResponse(
        script_id=str(row.get("script_id")),
        theme=str(row.get("theme") or ""),
        content=str(row.get("content") or ""),
        confirmed=bool(row.get("confirmed")),
    )


@app.get("/api/scripts/{script_id}/full", response_model=ScriptFullResponse)
def get_script_full(script_id: str):
    worker_payload_by_job_id: dict[str, dict] = {}
    row = STORE.get_script(script_id)
    if row is None:
        raise HTTPException(status_code=404, detail="script not found")

    jobs = [j for j in STORE.list_jobs() if str(j.get("script_id") or "") == script_id]
    jobs.sort(key=lambda x: str(x.get("created_at") or ""), reverse=True)

    # Best-effort status sync for all jobs on this script.
    changed = False
    try:
        with httpx.Client(timeout=5.0) as client:
            for j in jobs:
                worker_job_id = str(j.get("worker_job_id") or "")
                if not worker_job_id:
                    continue
                try:
                    r = client.get(f"{TTS_WORKER_URL}/jobs/{worker_job_id}")
                    r.raise_for_status()
                    payload = r.json() or {}
                    worker_payload_by_job_id[str(j.get("job_id"))] = payload
                    ws = payload.get("status")
                    if ws in {"queued", "running", "completed", "failed"} and ws != j.get("status"):
                        j["status"] = ws
                        j["updated_at"] = _utc_now()
                        STORE.put_job(j)
                        changed = True
                except Exception:
                    continue
    except Exception:
        pass

    meta = STORE.get_meta(script_id)
    meta_resp = None
    if meta is not None:
        meta_resp = ScriptMetaResponse(
            minutes=int(meta.get("minutes") or 4),
            speaker_names=_coerce_speaker_list(meta.get("speaker_names")),
            system_prompt=(meta.get("system_prompt") if meta.get("system_prompt") is not None else None),
            use_web_search=bool(meta.get("use_web_search")),
            source_filename=(meta.get("source_filename") if meta.get("source_filename") is not None else None),
            source_url=(meta.get("source_url") if meta.get("source_url") is not None else None),
        )

    return ScriptFullResponse(
        script=ScriptDetailResponse(
            script_id=str(row.get("script_id")),
            theme=str(row.get("theme") or ""),
            content=str(row.get("content") or ""),
            confirmed=bool(row.get("confirmed")),
            created_at=str(row.get("created_at") or ""),
        ),
        meta=meta_resp,
        tts_jobs=[
            ScriptJobItem(
                job_id=str(j.get("job_id")),
                worker_job_id=str(j.get("worker_job_id") or ""),
                status=str(j.get("status") or "queued"),
                worker_status=(worker_payload_by_job_id.get(str(j.get("job_id")), {}) or {}).get("status"),
                output_filename=(worker_payload_by_job_id.get(str(j.get("job_id")), {}) or {}).get("output_filename"),
                error=(worker_payload_by_job_id.get(str(j.get("job_id")), {}) or {}).get("error"),
                created_at=str(j.get("created_at") or ""),
                updated_at=str(j.get("updated_at") or ""),
            )
            for j in jobs
        ],
    )


@app.delete("/api/scripts/{script_id}")
def delete_script(script_id: str):
    row = STORE.get_script(script_id)
    if row is None:
        raise HTTPException(status_code=404, detail="script not found")
    STORE.delete_jobs_for_script(script_id)
    STORE.delete_script(script_id)

    # Best-effort delete backend logs
    try:
        p = LOG_DIR / f"{script_id}.jsonl"
        if p.exists():
            p.unlink()
    except Exception:
        pass

    return {"ok": True, "script_id": script_id}


@app.put("/api/scripts/{script_id}", response_model=ScriptResponse)
def update_script(script_id: str, req: UpdateScriptRequest):
    row = STORE.get_script(script_id)
    if row is None:
        raise HTTPException(status_code=404, detail="script not found")
    row["content"] = req.content
    row["updated_at"] = _utc_now()
    STORE.put_script(row)

    row2 = STORE.get_script(script_id)

    return ScriptResponse(
        script_id=str(row2.get("script_id")),
        theme=str(row2.get("theme") or ""),
        content=str(row2.get("content") or ""),
        confirmed=bool(row2.get("confirmed")),
    )


@app.post("/api/scripts/{script_id}/confirm", response_model=ConfirmResponse)
def confirm_script(script_id: str):
    row = STORE.get_script(script_id)
    if row is None:
        raise HTTPException(status_code=404, detail="script not found")
    row["confirmed"] = True
    row["updated_at"] = _utc_now()
    STORE.put_script(row)
    return ConfirmResponse(script_id=script_id, confirmed=True)


@app.post("/api/scripts/{script_id}/tts", response_model=SubmitTTSResponse)
def submit_tts(script_id: str, req: SubmitTTSRequest | None = None):
    row = STORE.get_script(script_id)
    if row is None:
        raise HTTPException(status_code=404, detail="script not found")
    if not bool(row.get("confirmed")):
        raise HTTPException(status_code=400, detail="script must be confirmed before TTS")
    content = str(row.get("content") or "")

    job_id = str(uuid.uuid4())

    # Best-effort append to script logs (if present).
    try:
        log = ScriptLogWriter(file_path=(LOG_DIR / f"{script_id}.jsonl"), script_id=script_id)
        log.stage_start("tts", "Submit TTS")
        log.step("tts", "Submitting job", data={"speaker_count": len(req.speaker_names) if (req and req.speaker_names) else None})
    except Exception:
        log = None

    try:
        timeout = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=10.0)
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            r = client.post(
                f"{TTS_WORKER_URL}/jobs",
                json={
                    "script_id": script_id,
                    "script_text": content,
                    "speaker_names": (req.speaker_names if req else None),
                },
            )
            r.raise_for_status()
            payload = r.json()
    except Exception as e:
        if log is not None:
            try:
                log.error("tts", "Failed to submit worker job", data={"error": str(e)})
                log.stage_end("tts", "failed")
            except Exception:
                pass
        raise HTTPException(status_code=502, detail=f"Failed to submit worker job: {e}")

    worker_job_id = payload.get("job_id")
    status = payload.get("status", "queued")

    if log is not None:
        try:
            log.step("tts", "Submitted", data={"job_id": job_id, "worker_job_id": worker_job_id, "status": status})
            log.stage_end("tts", "ok")
        except Exception:
            pass

    now = _utc_now()
    STORE.put_job(
        {
            "job_id": job_id,
            "script_id": script_id,
            "worker_job_id": worker_job_id,
            "status": status,
            "created_at": now,
            "updated_at": now,
        }
    )

    return SubmitTTSResponse(
        script_id=script_id,
        job_id=job_id,
        worker_job_id=worker_job_id,
        status=status,
    )


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    row = STORE.get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")

    worker_status = None
    output_filename = None
    error = None
    try:
        timeout = httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0)
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            r = client.get(f"{TTS_WORKER_URL}/jobs/{str(row.get('worker_job_id') or '')}")
            r.raise_for_status()
            payload = r.json()
            worker_status = payload.get("status")
            output_filename = payload.get("output_filename")
            error = payload.get("error")
    except Exception:
        worker_status = None

    # Best-effort local status sync.
    if worker_status in {"queued", "running", "completed", "failed"} and worker_status != row.get("status"):
        row["status"] = worker_status
        row["updated_at"] = _utc_now()
        STORE.put_job(row)

    return JobStatus(
        job_id=str(row.get("job_id")),
        script_id=str(row.get("script_id")),
        worker_job_id=str(row.get("worker_job_id")),
        status=(worker_status or row["status"]),
        worker_status=worker_status,
        output_filename=output_filename,
        error=error,
    )


@app.get("/api/jobs/{job_id}/audio")
def download_audio(job_id: str):
    row = STORE.get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")

    worker_job_id = str(row.get("worker_job_id") or "")

    try:
        timeout = httpx.Timeout(connect=5.0, read=300.0, write=300.0, pool=5.0)
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            r = client.get(f"{TTS_WORKER_URL}/jobs/{worker_job_id}/audio")
            r.raise_for_status()
            content_type = r.headers.get("content-type", "audio/wav")
            filename = r.headers.get("x-filename", "podcast.wav")
            return StreamingResponse(
                iter([r.content]),
                media_type=content_type,
                headers={"Content-Disposition": f"attachment; filename={filename}"},
            )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# If this process is running in the combined web image, serve the built frontend.
_STATIC_DIR = Path(os.environ.get("PODCASTSTUDIO_STATIC_DIR", "/app/static")).resolve()
if _STATIC_DIR.exists() and _STATIC_DIR.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")

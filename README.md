# Podcast Studio (Standalone)

Self-contained project for:

1) User inputs a topic
2) Backend generates a 2-speaker Chinese podcast script via **Azure OpenAI**
3) User reviews/edits and confirms
4) A separate **VibeVoice worker** runs TTS as a background job and returns a `.wav`

## Project structure

- `backend/`: FastAPI (script generation + orchestration)
- `tts_worker/`: FastAPI (VibeVoice job runner)
- `frontend/`: React (Vite)
- `VibeVoice/`: bundled VibeVoice repo used by the worker

## 1) Environment (.env)

Copy `./.env.example` to `./.env` (in THIS folder) and fill in Azure OpenAI:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_DEPLOYMENT_NAME`

Optional:

- `AZURE_OPENAI_BASE_URL` (advanced; defaults to `${AZURE_OPENAI_ENDPOINT}/openai/v1/`)
- `TTS_WORKER_URL` (default `http://localhost:8002`)
- `PODCASTSTUDIO_STORAGE_DIR` (default `./storage`, for mounting to Azure Storage)
- `VIBEVOICE_MODEL_ID` (default `FabioSarracino/VibeVoice-Large-Q8`)
- `VIBEVOICE_SPEAKER_NAMES` (default `"Xinran Anchen"`)

## 2) Python setup

Create and activate a venv (recommended):

- `python -m venv .venv`
- `source .venv/bin/activate`

Install dependencies:

```bash
pip install -r backend/requirements.txt
pip install -r tts_worker/requirements.txt
pip install -e VibeVoice
```

System dependency:

- `ffmpeg` must be installed and on PATH.

## 3) Run services

In separate terminals from this folder:

Worker (TTS):

```bash
source .venv/bin/activate
cd tts_worker
uvicorn app.main:app --reload --port 8002
```

Backend:

```bash
source .venv/bin/activate
cd backend
uvicorn app.main:app --reload --port 8001
```

Health checks:

```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
```

## 4) Frontend

```bash
cd frontend


npm install
npm run dev
```

Open the printed URL (default `http://localhost:5173`).

If your backend isn't on `http://localhost:8001`, set `VITE_API_BASE` (see `frontend/.env.example`).

## Notes

- This project does **not** use Ollama.
- Persistent DB/config/output are stored under `PODCASTSTUDIO_STORAGE_DIR` (default `./storage`).

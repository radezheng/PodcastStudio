# VibeVoice Podcast Studio

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

## Deploy to Azure Container Apps with `azd up` (recommended)

This repo includes an `azd` project that provisions:

- Azure Container Apps environment with a **serverless GPU** workload profile (T4 by default)
- 2 container apps:
	- `web` (external) runs the backend API (8001) and serves the frontend
	- `tts-worker` (internal) runs VibeVoice TTS (8002)
- Azure Files share mounted to all containers at `/mnt/storage`
- min replicas `0`, max replicas `1`

### 1) Prereqs

- Install Azure CLI (`az`) and Azure Developer CLI (`azd`)
- `az login`

Install `azd` on Linux (official script):

```bash
curl -fsSL https://aka.ms/install-azd.sh | bash
```

Note: this writes to `/opt/microsoft/azd` and `/usr/local/bin`, so it may prompt for sudo credentials.

Important: serverless GPU availability is **region-specific**.

This project supports two GPU presets:

- **T4 + 1.5B** (default): `GPU_SKU=t4` → `Consumption-GPU-NC8as-T4` + `vibevoice/VibeVoice-1.5B`
- **A100 + 7B** (recommended for Chinese / best quality): `GPU_SKU=a100` → `Consumption-GPU-NC24-A100` + `vibevoice/VibeVoice-7B`

### 2) Set `azd` environment values

Create an `azd` environment:

```bash
azd env new
```

Set region (example):

```bash
azd env set AZURE_LOCATION australiaeast
```

Set required Azure OpenAI settings:

```bash
azd env set AZURE_OPENAI_API_KEY "..."
azd env set AZURE_OPENAI_ENDPOINT "https://YOUR-RESOURCE.openai.azure.com"
azd env set AZURE_OPENAI_DEPLOYMENT_NAME "YOUR-DEPLOYMENT"
```

Optional: import values from your local `.env`:

```bash
bash scripts/azd-env-from-dotenv.sh .env
```

Select GPU preset (optional; defaults to T4 + 1.5B):

```bash
azd env set GPU_SKU t4
# or
azd env set GPU_SKU a100
```

### 3) Deploy

From repo root:

```bash
azd up
```

When it finishes, `azd` prints outputs including the frontend URL.

### Storage

In Azure, the app storage root is always `/mnt/storage` (Azure Files mount). That path is set automatically in the container apps.

## Local development (optional)

### 1) Environment (.env)

Copy `./.env.example` to `./.env` (in THIS folder) and fill in Azure OpenAI:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_DEPLOYMENT_NAME`

Optional:

- `AZURE_OPENAI_BASE_URL` (advanced; defaults to `${AZURE_OPENAI_ENDPOINT}/openai/v1/`)
- `TTS_WORKER_URL` (default `http://localhost:8002`)
- `PODCASTSTUDIO_STORAGE_DIR` (default `./storage`, for mounting to Azure Storage)
- `VIBEVOICE_MODEL_ID` (default `vibevoice/VibeVoice-1.5B`)
- `VIBEVOICE_SPEAKER_NAMES` (default `"Xinran Anchen"`)
- `VIBEVOICE_ATTN_IMPL` (default `sdpa`)
- `VIBEVOICE_LOAD_IN_8BIT` (default `0`; enabling 8-bit can cause noisy/garbled audio)

Quality recommendation:

- If you need **better Chinese** and overall quality, use **A100 + 7B** (`GPU_SKU=a100`).
- If you prioritize cost, use **T4 + 1.5B** (`GPU_SKU=t4`).

### 2) Python setup

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

### 3) Run services

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

### 4) Frontend

```bash
cd frontend


npm install
npm run dev
```

Open the printed URL (default `http://localhost:5173`).

If your backend isn't on `http://localhost:8001`, set `VITE_API_BASE` (see `frontend/.env.example`).

## Notes

- Persistent DB/config/output are stored under `PODCASTSTUDIO_STORAGE_DIR` (default `./storage`).
- First-time VibeVoice TTS can take several minutes (sometimes longer) because it needs to download the model and then load/warm up the model.
	Subsequent runs are much faster because model files are cached (in Azure, under `/mnt/storage/hf` by default).

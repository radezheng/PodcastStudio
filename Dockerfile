# Web image (CPU): backend (FastAPI) + frontend (Vite build) served by backend

FROM node:20-alpine AS frontend-build
WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PODCASTSTUDIO_STATIC_DIR=/app/static

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY backend/app /app/app

# VibeVoice demo voice presets (used for UI speaker list)
COPY VibeVoice/demo/voices /app/VibeVoice/demo/voices

# Frontend static assets
COPY --from=frontend-build /frontend/dist /app/static

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]

# CUDA runtime + PyTorch (GPU-ready for T4)
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
  && apt-get install -y --no-install-recommends git ffmpeg ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install worker deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Bring in VibeVoice at build time (repo is ignored in this project)
RUN git clone --depth 1 https://github.com/vibevoice-community/VibeVoice.git /opt/VibeVoice \
  && pip install --no-cache-dir -e /opt/VibeVoice

# Worker code
COPY app /app/app

EXPOSE 8002
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]

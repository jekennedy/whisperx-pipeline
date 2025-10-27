# syntax=docker/dockerfile:1.6

# ------------------------------------------------------------
# Base image stage: CUDA + system deps + Python deps
# ------------------------------------------------------------
ARG BASE_IMAGE=runpod/base:0.6.2-cuda12.4.1
FROM ${BASE_IMAGE} AS base

SHELL ["/bin/bash", "-c"]
WORKDIR /app
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends ffmpeg git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Torch stack (CUDA 12.4 channel)
RUN python3 -m pip install -U pip wheel && \
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch torchvision torchaudio

# Python deps (leave Torch to step above)
COPY builder/requirements.withdeps.txt /app/requirements.withdeps.txt
COPY builder/requirements.nodeps.txt /app/requirements.nodeps.txt

# 1) General libs WITH dependencies (avoids whack-a-mole transitives)
RUN python3 -m pip install --no-cache-dir -r /app/requirements.withdeps.txt

# 2) Heavy ML libs WITHOUT dependencies (prevents repinning torch etc.)
RUN python3 -m pip install --no-cache-dir --no-deps -r /app/requirements.nodeps.txt

# Sanity check
RUN python3 - <<'PY'
import runpod, aiohttp
print("runpod OK, aiohttp OK")
PY

# ------------------------------------------------------------
# App stage: copy code, optional model assets, entrypoint
# ------------------------------------------------------------
FROM base AS app
WORKDIR /app

# App code
COPY src/ /app/src/

# Bundle local models (optional, speeds cold starts)
COPY models/ /app/models/
RUN mkdir -p /app/models/vad /root/.cache/torch && \
    if [ -f /app/models/whisperx-vad-segmentation.bin ]; then \
      mv /app/models/whisperx-vad-segmentation.bin /app/models/vad/whisperx-vad-segmentation.bin; \
    fi && \
    if [ -f /app/models/vad/whisperx-vad-segmentation.bin ]; then \
      cp /app/models/vad/whisperx-vad-segmentation.bin /root/.cache/torch/whisperx-vad-segmentation.bin; \
    fi
ENV VAD_MODEL_PATH=/app/models/vad/whisperx-vad-segmentation.bin

# Optional pre-download of WhisperX model weights
COPY builder/download_models.sh /app/builder/download_models.sh
RUN chmod +x /app/builder/download_models.sh
ARG PRELOAD_MODELS=0
ENV MODEL_DIR=/app/models \
    WHISPERX_MODEL_NAME=faster-whisper-large-v3
RUN --mount=type=secret,id=hf_token \
    --mount=type=cache,target=/cache/models \
    --mount=type=cache,target=/cache/hf \
    bash -lc '\
      if [[ "$PRELOAD_MODELS" == "1" ]]; then \
        export HF_TOKEN="$( [ -f /run/secrets/hf_token ] && cat /run/secrets/hf_token || true )"; \
        export HF_HOME=/cache/hf; \
        /app/builder/download_models.sh; \
      else \
        echo "Skipping model pre-download (PRELOAD_MODELS=0)"; \
      fi'

# Ensure the path used in predict.py exists even if the script skipped
RUN mkdir -p /app/models/faster-whisper-large-v3
ENV WHISPERX_MODEL_DIR=/app/models/faster-whisper-large-v3

# Entrypoint
CMD ["python3", "-u", "-m", "src.init_and_run"]

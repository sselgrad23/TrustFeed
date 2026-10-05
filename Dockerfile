# --- API service image ------------------------------------------------------
# Slim single-stage build for the FastAPI service. Ships the core deps only
# (Whisper via faster-whisper + the heuristic extractor) so the image stays
# small and torch-free; the LLM backend is opt-in on a GPU host. Whisper weights
# download on first use and cache in the mounted volume.
FROM python:3.11-slim

WORKDIR /app

# ffmpeg lets faster-whisper decode arbitrary audio/video containers.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install runtime deps first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install the package itself.
COPY pyproject.toml ./
COPY src ./src
COPY README.md ./
RUN pip install --no-cache-dir --no-deps -e .

ENV TRUSTFEED_MODEL_DIR=/app/models \
    TRUSTFEED_DATA_DIR=/app/data \
    TRUSTFEED_LLM_BACKEND=heuristic \
    HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "trustfeed.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

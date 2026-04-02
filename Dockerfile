# VibeVoice-Realtime-0.5B — RunPod Serverless Handler (v4)
# Pre-bakes model weights and voice embeddings into the image for instant cold start
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsndfile1 curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CUDA 12.1
RUN pip install --no-cache-dir \
    torch==2.3.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Core deps
RUN pip install --no-cache-dir \
    runpod>=1.6.0 \
    transformers>=4.40.0 \
    huggingface_hub \
    soundfile numpy scipy httpx accelerate

# Clone VibeVoice + install (voice embeddings land at /vibevoice/demo/voices/streaming_model/)
RUN git clone --depth 1 https://github.com/microsoft/VibeVoice.git /vibevoice && \
    pip install --no-cache-dir -e "/vibevoice[streamingtts]" || \
    pip install --no-cache-dir -e /vibevoice

# Pre-download VibeVoice model weights → stored at /models/vibevoice-realtime
# This bakes the weights into the image so cold start is near-instant
RUN python -c "\
from huggingface_hub import snapshot_download; \
print('Downloading microsoft/VibeVoice-Realtime-0.5B...'); \
snapshot_download('microsoft/VibeVoice-Realtime-0.5B', local_dir='/models/vibevoice-realtime'); \
print('Download complete!')" || \
    echo "WARNING: Model pre-download failed — will download at first worker start"

# Point handler to local model path
ENV VIBEVOICE_MODEL_PATH=/models/vibevoice-realtime
ENV VIBEVOICE_VOICES_DIR=/vibevoice/demo/voices/streaming_model

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]

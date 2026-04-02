# ─── VibeVoice-Realtime-0.5B — RunPod Serverless Handler ────────────────────
# Base: NVIDIA PyTorch 24.10 (CUDA 12.x, Python 3.10, verified by VibeVoice docs)
FROM nvcr.io/nvidia/pytorch:24.10-py3

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Flash Attention (optional — speeds up inference ~30%)
# Uncomment if your RunPod pod has >24GB VRAM
# RUN pip install flash-attn --no-build-isolation

# Clone VibeVoice and install streaming TTS extras
RUN git clone --depth 1 https://github.com/microsoft/VibeVoice.git /vibevoice
RUN pip install -e /vibevoice[streamingtts] --no-build-isolation

# RunPod serverless SDK
RUN pip install runpod>=1.6.0

# Additional deps for our handler
RUN pip install \
    huggingface_hub \
    soundfile \
    numpy \
    scipy \
    httpx

# Pre-download model weights at build time (bakes into image → cold start <10s)
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('microsoft/VibeVoice-Realtime-0.5B', local_dir='/models/vibevoice-realtime')"

# Copy handler
COPY handler.py /app/handler.py
COPY voices/ /app/voices/

# RunPod entrypoint
CMD ["python", "-u", "/app/handler.py"]

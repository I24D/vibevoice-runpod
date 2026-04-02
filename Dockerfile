# VibeVoice-Realtime-0.5B — RunPod Serverless Handler
# Base: python:3.11-slim + PyTorch CUDA (fast CI build, no nvidia base)
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends     git ffmpeg libsndfile1 curl build-essential     && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir     torch==2.3.0 torchvision torchaudio     --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir     runpod>=1.6.0     transformers>=4.40.0     huggingface_hub     soundfile numpy scipy httpx accelerate

RUN git clone --depth 1 https://github.com/microsoft/VibeVoice.git /vibevoice &&     pip install --no-cache-dir -e /vibevoice[streamingtts] || true

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]

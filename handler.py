"""
VibeVoice-Realtime RunPod Serverless Handler
I24D — Real-Time TTS endpoint

Input JSON:
  {
    "input": {
      "text":         "Text to synthesize",
      "speaker":      "Carter",          // optional, default: Carter
      "format":       "wav",             // optional: wav | mp3 | ogg (default: wav)
      "speed":        1.0,               // optional: 0.5-2.0 (default: 1.0)
      "streaming":    false              // optional: true returns chunked base64
    }
  }

Output JSON (success):
  {
    "audio_b64":   "<base64 wav/mp3/ogg>",
    "mime_type":   "audio/wav",
    "duration_s":  3.45,
    "speaker":     "Carter",
    "characters":  120,
    "latency_ms":  210
  }

Output JSON (error):
  {
    "error": "description"
  }
"""

import base64
import io
import os
import sys
import time
import traceback

import numpy as np
import runpod
import soundfile as sf

# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("VIBEVOICE_MODEL_PATH", "/models/vibevoice-realtime")
DEFAULT_SPEAKER = os.environ.get("VIBEVOICE_DEFAULT_SPEAKER", "Carter")
SAMPLE_RATE = 24000  # VibeVoice-Realtime native sample rate

# Available speakers (base pack — download_experimental_voices.sh adds more)
AVAILABLE_SPEAKERS = [
    "Carter", "Aria", "Jenny", "Guy", "Nova",
    "Echo", "Fable", "Onyx", "Shimmer", "Alloy",
]

# ─── Model loader (global — loaded once per container) ────────────────────────

print(f"[VibeVoice] Loading model from {MODEL_PATH}...")
_t0 = time.time()

try:
    # VibeVoice uses transformers-compatible loading after March 2026 update
    import torch
    from transformers import AutoProcessor, AutoModel

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[VibeVoice] Device: {_device}")

    _processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    _model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(_device)
    _model.eval()

    print(f"[VibeVoice] Model loaded in {time.time() - _t0:.1f}s on {_device}")
    _model_loaded = True

except Exception as e:
    print(f"[VibeVoice] Model load error: {e}")
    traceback.print_exc()
    _model_loaded = False
    _model = None
    _processor = None
    _device = "cpu"


# ─── Audio helpers ────────────────────────────────────────────────────────────

def _numpy_to_wav_b64(audio_np: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Converts numpy float32 audio array to base64-encoded WAV."""
    buf = io.BytesIO()
    # Normalize to int16
    audio_int16 = (audio_np * 32767).astype(np.int16)
    sf.write(buf, audio_int16, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _numpy_to_mp3_b64(audio_np: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Converts numpy audio to base64 MP3 using ffmpeg via soundfile fallback."""
    try:
        import subprocess, tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            wav_path = tmp_wav.name
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
            mp3_path = tmp_mp3.name

        audio_int16 = (audio_np * 32767).astype(np.int16)
        sf.write(wav_path, audio_int16, sample_rate, format="WAV")

        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-b:a", "128k", mp3_path],
            capture_output=True, check=True,
        )
        with open(mp3_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        # Fallback to WAV
        return _numpy_to_wav_b64(audio_np, sample_rate)


def _numpy_to_ogg_b64(audio_np: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Converts numpy audio to base64 OGG/Opus — WhatsApp voice note format."""
    try:
        import subprocess, tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            wav_path = tmp_wav.name
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp_ogg:
            ogg_path = tmp_ogg.name

        audio_int16 = (audio_np * 32767).astype(np.int16)
        sf.write(wav_path, audio_int16, sample_rate, format="WAV")

        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path,
             "-codec:a", "libopus", "-b:a", "64k",
             "-vbr", "on", "-compression_level", "10",
             ogg_path],
            capture_output=True, check=True,
        )
        with open(ogg_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return _numpy_to_wav_b64(audio_np, sample_rate)


# ─── Inference ────────────────────────────────────────────────────────────────

def _synthesize(text: str, speaker: str, speed: float = 1.0) -> np.ndarray:
    """
    Runs VibeVoice-Realtime inference.
    Returns float32 numpy array at SAMPLE_RATE.
    """
    import torch

    inputs = _processor(
        text=text,
        speaker=speaker,
        speed=speed,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        output = _model.generate(**inputs)

    # Extract audio from model output
    if hasattr(output, "audio"):
        audio = output.audio.squeeze().cpu().float().numpy()
    elif isinstance(output, torch.Tensor):
        audio = output.squeeze().cpu().float().numpy()
    else:
        # Fallback: try first element
        audio = output[0].squeeze().cpu().float().numpy()

    # Normalize
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.95

    return audio


# ─── RunPod Handler ───────────────────────────────────────────────────────────

def handler(job: dict) -> dict:
    """
    RunPod serverless job handler.
    Receives job dict, returns synthesis result.
    """
    job_input = job.get("input", {})

    # ── Input validation ──
    text = str(job_input.get("text", "")).strip()
    if not text:
        return {"error": "Missing required field: text"}
    if len(text) > 4000:
        return {"error": f"Text too long ({len(text)} chars). Max: 4000."}

    speaker = job_input.get("speaker", DEFAULT_SPEAKER)
    if speaker not in AVAILABLE_SPEAKERS:
        speaker = DEFAULT_SPEAKER

    output_format = job_input.get("format", "wav").lower()
    if output_format not in ("wav", "mp3", "ogg"):
        output_format = "wav"

    speed = float(job_input.get("speed", 1.0))
    speed = max(0.5, min(2.0, speed))

    # ── Model availability check ──
    if not _model_loaded:
        return {"error": "Model not loaded. Check container logs."}

    # ── Synthesis ──
    t_start = time.time()
    try:
        audio_np = _synthesize(text, speaker, speed)
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Synthesis failed: {str(e)[:300]}"}

    latency_ms = round((time.time() - t_start) * 1000)
    duration_s = round(len(audio_np) / SAMPLE_RATE, 3)

    # ── Encode output ──
    mime_map = {"wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg; codecs=opus"}
    if output_format == "mp3":
        audio_b64 = _numpy_to_mp3_b64(audio_np)
    elif output_format == "ogg":
        audio_b64 = _numpy_to_ogg_b64(audio_np)
    else:
        audio_b64 = _numpy_to_wav_b64(audio_np)

    print(f"[VibeVoice] speaker={speaker} chars={len(text)} duration={duration_s}s latency={latency_ms}ms")

    return {
        "audio_b64":  audio_b64,
        "mime_type":  mime_map[output_format],
        "duration_s": duration_s,
        "speaker":    speaker,
        "characters": len(text),
        "latency_ms": latency_ms,
        "format":     output_format,
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[VibeVoice] Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})

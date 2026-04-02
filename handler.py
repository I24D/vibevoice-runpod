"""
VibeVoice-Realtime RunPod Serverless Handler  (v3)
Fixes:
  - Always use sdpa attention (flash_attention_2 not installed by default)
  - VoiceMapper parses en-Carter_man.pt -> speaker "Carter"
  - Robust fallback chain
"""

import base64
import glob
import os
import re
import tempfile
import time
import traceback

import runpod

MODEL_PATH      = os.environ.get("VIBEVOICE_MODEL_PATH", "microsoft/VibeVoice-Realtime-0.5B")
DEFAULT_SPEAKER = os.environ.get("VIBEVOICE_DEFAULT_SPEAKER", "Carter")
SAMPLE_RATE     = 24000
CFG_SCALE       = float(os.environ.get("VIBEVOICE_CFG_SCALE", "1.3"))
VOICES_DIR      = os.environ.get("VIBEVOICE_VOICES_DIR", "/vibevoice/demo/voices/streaming_model")


class VoiceMapper:
    """
    Maps speaker names to .pt file paths.
    Handles filenames like: en-Carter_man.pt  ->  Carter
    """
    def __init__(self, voices_dir):
        self.voice_presets = {}   # lowercase_name -> path
        self.raw_presets   = {}   # exact filename stem -> path
        if not os.path.exists(voices_dir):
            print(f"[VibeVoice] WARNING: voices dir not found: {voices_dir}")
            return
        for pt_file in glob.glob(os.path.join(voices_dir, "**", "*.pt"), recursive=True):
            stem = os.path.splitext(os.path.basename(pt_file))[0]  # e.g. "en-Carter_man"
            path = os.path.abspath(pt_file)
            self.raw_presets[stem.lower()] = path
            # Extract speaker name: en-Carter_man -> Carter
            m = re.match(r'^[a-z]{2}-([A-Za-z]+)_', stem)
            if m:
                speaker = m.group(1)
                self.voice_presets[speaker.lower()] = path
            else:
                # Plain name like "Carter.pt"
                self.voice_presets[stem.lower()] = path
        speakers = sorted(self.voice_presets.keys())
        print(f"[VibeVoice] Mapped {len(speakers)} speakers: {speakers}")

    def get_voice_path(self, name):
        return (self.voice_presets.get(name.lower())
                or self.raw_presets.get(name.lower()))

    def available_speakers(self):
        return sorted(self.voice_presets.keys())


# --- Model loader ---
print(f"[VibeVoice] Initializing... MODEL={MODEL_PATH}")
_t0 = time.time()
_model_loaded = False
_model = _processor = _voice_mapper = None
_device = "cpu"

try:
    import torch
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[VibeVoice] Device: {_device}")

    # Use sdpa always - flash_attention_2 requires separate install
    _dtype = torch.bfloat16 if _device == "cuda" else torch.float32
    _attn  = "sdpa"

    print(f"[VibeVoice] Loading processor from {MODEL_PATH}...")
    _processor = VibeVoiceStreamingProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print(f"[VibeVoice] Loading model from {MODEL_PATH}...")
    _model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        MODEL_PATH,
        torch_dtype=_dtype,
        attn_implementation=_attn,
        trust_remote_code=True,
    )
    if _device == "cuda":
        _model = _model.cuda()
    _model.eval()

    _voice_mapper = VoiceMapper(VOICES_DIR)

    print(f"[VibeVoice] Ready in {time.time() - _t0:.1f}s | device={_device}")
    _model_loaded = True

except Exception as e:
    print(f"[VibeVoice] FATAL: Model load failed: {e}")
    traceback.print_exc()


# --- Audio helpers ---

def _speech_to_wav_bytes(speech_tensor):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    _processor.save_audio(speech_tensor, output_path=wav_path)
    with open(wav_path, "rb") as f:
        data = f.read()
    os.unlink(wav_path)
    return data


def _convert_audio(wav_bytes, fmt):
    import subprocess
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tw:
        tw.write(wav_bytes)
        wav_path = tw.name
    out_path = wav_path.replace(".wav", f".{fmt}")
    try:
        if fmt == "ogg":
            cmd = ["ffmpeg", "-y", "-i", wav_path,
                   "-codec:a", "libopus", "-b:a", "64k",
                   "-vbr", "on", "-compression_level", "10", out_path]
        else:
            cmd = ["ffmpeg", "-y", "-i", wav_path,
                   "-codec:a", "libmp3lame", "-b:a", "128k", out_path]
        subprocess.run(cmd, capture_output=True, check=True)
        with open(out_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception as e:
        print(f"[VibeVoice] {fmt} conversion failed: {e}")
        return None
    finally:
        for p in [wav_path, out_path]:
            try:
                os.unlink(p)
            except Exception:
                pass


# --- Synthesis ---

def _synthesize(text, speaker, cfg_scale):
    import torch

    voice_path = _voice_mapper.get_voice_path(speaker) if _voice_mapper else None
    if voice_path is None and _voice_mapper:
        voice_path = _voice_mapper.get_voice_path(DEFAULT_SPEAKER)
        if voice_path is None and _voice_mapper.available_speakers():
            first = _voice_mapper.available_speakers()[0]
            voice_path = _voice_mapper.get_voice_path(first)
    if voice_path is None:
        raise RuntimeError(f"No voice embedding found (voices_dir={VOICES_DIR})")

    target_device = next(_model.parameters()).device
    cached_prompt = torch.load(voice_path, map_location=target_device, weights_only=False)

    inputs = _processor.process_input_with_cached_prompt(
        text=text,
        cached_prompt=cached_prompt,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(target_device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            cfg_scale=cfg_scale,
            tokenizer=_processor.tokenizer,
            generation_config={"do_sample": False},
        )
    return outputs


# --- RunPod Handler ---

def handler(job):
    job_input = job.get("input", {})
    text = str(job_input.get("text", "")).strip()
    if not text:
        return {"error": "Missing required field: text"}
    if len(text) > 4000:
        return {"error": f"Text too long ({len(text)} chars). Max: 4000."}

    speaker    = job_input.get("speaker", DEFAULT_SPEAKER)
    output_fmt = job_input.get("format", "wav").lower()
    if output_fmt not in ("wav", "mp3", "ogg"):
        output_fmt = "wav"
    cfg = float(job_input.get("cfg_scale", CFG_SCALE))

    if not _model_loaded:
        return {"error": "Model not loaded. Check container logs."}

    t_start = time.time()
    try:
        outputs = _synthesize(text, speaker, cfg)
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Synthesis failed: {str(e)[:300]}"}

    if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
        return {"error": "Model returned no audio output."}

    speech = outputs.speech_outputs[0]
    audio_samples = speech.shape[-1] if len(speech.shape) > 0 else len(speech)
    duration_s = round(audio_samples / SAMPLE_RATE, 3)
    latency_ms = round((time.time() - t_start) * 1000)

    wav_bytes = _speech_to_wav_bytes(speech)
    mime_map  = {"wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg; codecs=opus"}

    if output_fmt in ("ogg", "mp3"):
        audio_b64 = _convert_audio(wav_bytes, output_fmt)
        if audio_b64 is None:
            output_fmt = "wav"
            audio_b64  = base64.b64encode(wav_bytes).decode()
    else:
        audio_b64 = base64.b64encode(wav_bytes).decode()

    print(f"[VibeVoice] OK speaker={speaker} chars={len(text)} dur={duration_s}s lat={latency_ms}ms fmt={output_fmt}")

    return {
        "audio_b64":  audio_b64,
        "mime_type":  mime_map[output_fmt],
        "duration_s": duration_s,
        "speaker":    speaker,
        "characters": len(text),
        "latency_ms": latency_ms,
        "format":     output_fmt,
    }


if __name__ == "__main__":
    print("[VibeVoice] Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})

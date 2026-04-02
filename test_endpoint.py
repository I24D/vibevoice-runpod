"""
Test script — validates your RunPod VibeVoice endpoint before integrating with I24D.
Usage:
    RUNPOD_API_KEY=your_key RUNPOD_ENDPOINT_ID=your_id python test_endpoint.py
"""

import base64
import os
import sys
import time

import httpx

RUNPOD_API_KEY   = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINT_ID      = os.environ.get("VIBEVOICE_ENDPOINT_ID", "")
BASE_URL         = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json",
}

TEST_CASES = [
    {"text": "Hello! I am I24D, your intelligent assistant.", "speaker": "Carter", "format": "wav"},
    {"text": "Hola, soy I24D. ¿En qué puedo ayudarte hoy?", "speaker": "Aria",   "format": "ogg"},
    {"text": "Processing your request now.", "speaker": "Nova", "format": "mp3", "speed": 1.2},
]


def run_sync(payload: dict) -> dict:
    """Calls /runsync — blocks until done (max 30s)."""
    resp = httpx.post(
        f"{BASE_URL}/runsync",
        headers=HEADERS,
        json={"input": payload},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()


def run_async(payload: dict) -> str:
    """Submits job, returns job_id."""
    resp = httpx.post(
        f"{BASE_URL}/run",
        headers=HEADERS,
        json={"input": payload},
        timeout=10.0,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def poll_status(job_id: str, timeout: int = 120) -> dict:
    """Polls job status until completed or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = httpx.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS, timeout=10.0)
        data = resp.json()
        status = data.get("status", "")
        if status == "COMPLETED":
            return data.get("output", {})
        if status in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"Job {job_id} {status}: {data.get('error', '')}")
        print(f"  [{status}] waiting...")
        time.sleep(3)
    raise TimeoutError(f"Job {job_id} did not complete in {timeout}s")


def save_audio(b64_data: str, filename: str) -> None:
    with open(filename, "wb") as f:
        f.write(base64.b64decode(b64_data))
    print(f"  Saved: {filename}")


def main():
    if not RUNPOD_API_KEY or not ENDPOINT_ID:
        print("ERROR: Set RUNPOD_API_KEY and VIBEVOICE_ENDPOINT_ID environment variables.")
        sys.exit(1)

    print(f"Testing endpoint: {ENDPOINT_ID}")
    print(f"URL: {BASE_URL}\n")

    for i, test in enumerate(TEST_CASES, 1):
        print(f"Test {i}/{len(TEST_CASES)}: speaker={test.get('speaker')} format={test.get('format')}")
        print(f"  Text: {test['text'][:60]}...")

        t0 = time.time()
        try:
            result = run_sync(test)
            elapsed = round(time.time() - t0, 2)

            if "error" in result:
                print(f"  FAILED: {result['error']}")
                continue

            output = result.get("output", result)
            audio_b64 = output.get("audio_b64", "")
            if audio_b64:
                ext = test.get("format", "wav")
                save_audio(audio_b64, f"test_output_{i}.{ext}")
                print(f"  OK — duration={output.get('duration_s')}s latency={output.get('latency_ms')}ms total={elapsed}s")
            else:
                print(f"  WARNING: No audio in response: {output}")

        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nAll tests complete.")


if __name__ == "__main__":
    main()

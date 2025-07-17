import time
import requests
from .hf_utils import get_hf_token

# Public, free Whisper endpoint
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-small"
HF_TOKEN = get_hf_token()

def transcribe_audio(file_path: str, max_retries: int = 3) -> str:
    """
    Transcribe audio via HF Inference Whisper‑Small (no local ffmpeg needed).
    Retries on 503 (model-loading) up to max_retries.
    """
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "audio/mpeg"
    }

    # Read the MP3 into memory
    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    backoff = 1
    for attempt in range(1, max_retries + 1):
        resp = requests.post(API_URL, headers=headers, data=audio_bytes, timeout=120)

        # Model still warming up?
        if resp.status_code == 503:
            retry_after = int(resp.headers.get("Retry-After", backoff))
            time.sleep(retry_after)
            backoff *= 2
            continue

        resp.raise_for_status()
        data = resp.json()
        # Whisper returns {"text": "..."}
        return data.get("text", "").strip()

    raise RuntimeError("Transcription failed: exceeded retry logic")


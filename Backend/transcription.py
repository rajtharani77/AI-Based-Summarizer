import time
import requests
from .hf_utils import get_hf_token

API_URL = "https://api-inference.huggingface.co/models/facebook/wav2vec2-base-960h"
HF_TOKEN = get_hf_token()

def transcribe_audio(file_path: str, max_retries: int = 3) -> str:
    """
    Transcribe an audio file by sending it to HF Inference ASR.
    Retries on 503 (model loading) up to `max_retries` times.
    """
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "audio/mpeg"
    }

    # Read your file once
    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    backoff = 1
    for attempt in range(1, max_retries + 1):
        resp = requests.post(API_URL, headers=headers, data=audio_bytes, timeout=120)

        # If model is still loading, retry with exponential backoff
        if resp.status_code == 503:
            retry_after = int(resp.headers.get("Retry-After", backoff))
            time.sleep(retry_after)
            backoff *= 2
            continue

        resp.raise_for_status()
        payload = resp.json()

        # HF ASR returns {"text": "..."}
        return payload.get("text", "").strip()

    raise RuntimeError("Transcription failed: exceeded retry logic")

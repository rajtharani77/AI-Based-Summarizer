# Backend/transcription.py

import time
import requests
from .hf_utils import get_hf_token

# Inference API endpoint for Whisper
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"

# Build headers once
HF_TOKEN = get_hf_token()
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "audio/mpeg"
}

def transcribe_audio(file_path: str, max_retries: int = 3) -> str:
    """
    Transcribe an audio file via direct Hugging Face Inference API call.
    Retries on 503/model-loading up to `max_retries` with exponential backoff.
    """
    backoff = 1
    for attempt in range(1, max_retries + 1):
        try:
            # Read raw audio bytes
            with open(file_path, "rb") as f:
                audio_bytes = f.read()

            # POST with explicit audio/mpeg header
            resp = requests.post(
                API_URL,
                headers=HEADERS,
                data=audio_bytes,
                timeout=120
            )

            # If the model is loading, wait and retry
            if resp.status_code == 503:
                retry_after = int(resp.headers.get("Retry-After", backoff))
                time.sleep(retry_after)
                backoff *= 2
                continue

            # Raise on other HTTP errors
            resp.raise_for_status()

            # Parse and return text
            payload = resp.json()
            return payload.get("text", "").strip()

        except requests.exceptions.RequestException as e:
            # On last attempt, bubble up
            if attempt == max_retries:
                raise RuntimeError(
                    f"Transcription failed after {attempt} attempts: {e}"
                ) from e

            # Otherwise back off and retry
            time.sleep(backoff)
            backoff *= 2

    # Should never reach here
    raise RuntimeError("Transcription failed: exceeded retry logic")

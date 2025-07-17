import time
import requests
from .hf_utils import get_together_token

API_URL = "https://api.together.xyz/v1/audio/transcriptions"
TOG_TOKEN = get_together_token()

def transcribe_audio(file_path: str, max_retries: int = 3) -> str:
    """
    Transcribe an audio file via Together AI’s Whisper endpoint.
    Retries on 503 up to `max_retries`.
    """
    headers = {
        "Authorization": f"Bearer {TOG_TOKEN}"
    }

    # Together expects form‑data
    files = {
        "file": (file_path, open(file_path, "rb"), "application/octet-stream")
    }
    data = {
        "model": "openai/whisper-large-v3",  # public Whisper large
        "language": "en",
        "response_format": "text"            # get raw text back
    }

    backoff = 1
    for attempt in range(1, max_retries + 1):
        resp = requests.post(API_URL, headers=headers, files=files, data=data, timeout=120)

        if resp.status_code == 503:
            retry_after = int(resp.headers.get("Retry-After", backoff))
            time.sleep(retry_after)
            backoff *= 2
            continue

        resp.raise_for_status()
        # Together returns plain text if response_format="text"
        return resp.text.strip()

    raise RuntimeError("Transcription failed: exceeded retry logic")

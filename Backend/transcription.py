# Backend/transcription.py

import time
import requests
from .hf_utils import get_hf_token

def transcribe_audio(file_path: str, max_retries: int = 3) -> str:
    """
    Transcribe an audio file:
      - Try Together AI if TOGETHER_API_KEY is available.
      - Otherwise fall back to HF's public wav2vec2 ASR endpoint.
    Retries on 503 (model loading) up to max_retries.
    """
    # Attempt Together AI transcription
    try:
        tog_token = get_hf_token("TOGETHER_API_KEY")
        api_url = "https://api.together.xyz/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {tog_token}"}
        files = {"file": (file_path, open(file_path, "rb"), "application/octet-stream")}
        data = {
            "model": "openai/whisper-large-v3",
            "language": "en",
            "response_format": "text"
        }

        backoff = 1
        for attempt in range(1, max_retries + 1):
            resp = requests.post(api_url, headers=headers, files=files, data=data, timeout=120)
            if resp.status_code == 503:
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            return resp.text.strip()

    except RuntimeError:
        # Together key missing or transcription failed → fallback
        pass

    # Fallback: Hugging Face Wav2Vec2 ASR
    hf_token = get_hf_token()  # this will raise if HF token is missing
    api_url = "https://api-inference.huggingface.co/models/facebook/wav2vec2-base-960h"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "audio/mpeg"
    }
    audio_bytes = open(file_path, "rb").read()

    backoff = 1
    for attempt in range(1, max_retries + 1):
        resp = requests.post(api_url, headers=headers, data=audio_bytes, timeout=120)
        if resp.status_code == 503:
            time.sleep(backoff)
            backoff *= 2
            continue
        resp.raise_for_status()
        return resp.json().get("text", "").strip()

    raise RuntimeError("Transcription failed via both Together AI and HF ASR")

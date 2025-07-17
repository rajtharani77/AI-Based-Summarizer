# Backend/transcription.py

import time
from huggingface_hub import InferenceApi
from .hf_utils import get_hf_token

# Instantiate the low‑level InferenceApi client pointing to the Whisper model
api = InferenceApi(
    repo_id="openai/whisper-large-v3",
    token=get_hf_token()
)

def transcribe_audio(file_path: str, max_retries: int = 3) -> str:
    """
    Transcribe an audio file via Hugging Face InferenceApi (Whisper).
    Automatically retries on transient errors.
    """
    backoff = 1
    for attempt in range(1, max_retries + 1):
        try:
            # Read raw bytes
            with open(file_path, "rb") as f:
                audio_bytes = f.read()

            # Send as binary body (data=), not JSON
            result = api(data=audio_bytes)

            # Whisper returns {"text": "..."}
            return result.get("text", "").strip()

        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(f"Transcription failed after {attempt} attempts: {e}")
            time.sleep(backoff)
            backoff *= 2

# Backend/transcription.py

import time
from huggingface_hub import InferenceApi
from .hf_utils import get_hf_token

# Use the low‑level API so audio bytes are sent correctly
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
            with open(file_path, "rb") as f:
                audio_bytes = f.read()
            # InferenceApi will set the correct audio/mpeg header
            result = api(inputs=audio_bytes)
            # Whisper returns {"text": "..."}
            return result.get("text", "").strip()
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(f"Transcription failed after {attempt} attempts: {e}")
            time.sleep(backoff)
            backoff *= 2

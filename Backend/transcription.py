# Backend/transcription.py

import time
from huggingface_hub import InferenceClient
from .hf_utils import get_hf_token

# Pin to HF‑Inference so we hit the correct ASR endpoint
client = InferenceClient(provider="hf-inference", api_key=get_hf_token())

def transcribe_audio(file_path: str, max_retries: int = 3) -> str:
    """
    Transcribe an audio file via Whisper (openai/whisper-large-v3).
    Retries on transient errors (model loading, network hickups).
    Returns the full transcript as a single string.
    """
    backoff = 1
    for attempt in range(1, max_retries + 1):
        try:
            # automatic_speech_recognition streams the file and
            # sets the correct Content-Type for you.
            result = client.automatic_speech_recognition(
                file_path,
                model="openai/whisper-large-v3"
            )
            return result.get("text", "").strip()

        except Exception as e:
            # On final failure, surface the error
            if attempt == max_retries:
                raise RuntimeError(
                    f"Transcription failed after {attempt} attempts: {e}"
                ) from e

            # Otherwise, wait and retry
            time.sleep(backoff)
            backoff *= 2

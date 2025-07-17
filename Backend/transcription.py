import time
from huggingface_hub import InferenceClient
from .hf_utils import get_hf_token

# Pin to the HF‐Inference provider so we don't hit unsupported backends
client = InferenceClient(provider="hf-inference", api_key=get_hf_token())

def transcribe_audio(file_path: str, max_retries: int = 3) -> str:
    """
    Transcribe an audio file via Whisper (openai/whisper-large-v3).
    Retries on transient errors (503, timeouts).
    """
    backoff = 1
    for attempt in range(1, max_retries + 1):
        try:
            # automatic_speech_recognition handles file streaming internally
            result = client.automatic_speech_recognition(
                file_path,
                model="openai/whisper-large-v3"
            )
            return result.get("text", "").strip()
        except Exception as e:
            # If it’s the last try, bubble up
            if attempt == max_retries:
                raise RuntimeError(f"Transcription failed after {attempt} attempts: {e}")
            time.sleep(backoff)
            backoff *= 2

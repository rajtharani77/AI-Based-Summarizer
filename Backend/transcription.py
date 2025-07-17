# Backend/transcription.py
import whisper
import time

# Pick a small model so it runs on CPU in Streamlit Cloud
MODEL_NAME = "base"
_model = whisper.load_model(MODEL_NAME)

def transcribe_audio(file_path: str, max_retries: int = 1) -> str:
    """
    Transcribe an audio file using local Whisper.
    """
    for attempt in range(1, max_retries + 1):
        try:
            result = _model.transcribe(file_path, language="en")
            return result["text"].strip()
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(f"Local transcription failed: {e}") from e
            time.sleep(1)
    raise RuntimeError("Local transcription failed unexpectedly")

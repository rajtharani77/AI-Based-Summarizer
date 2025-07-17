# Backend/transcription.py

from transformers import pipeline

# Load once at import time
# This does *not* require ffmpeg, nor any external binary.
_asr = pipeline(
    "automatic-speech-recognition",
    model="facebook/wav2vec2-base-960h",
    device=-1,           # CPU; set to GPU ID if available
)

def transcribe_audio(file_path: str, max_retries: int = 1) -> str:
    """
    Transcribe an audio file using a local Wav2Vec2 ASR model.
    """
    # The pipeline will read WAV/MP3 directly (via soundfile or torchaudio)
    result = _asr(file_path)
    return result["text"].strip()

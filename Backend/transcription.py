# Backend/transcription.py

from transformers import pipeline

# This loads a small, CPU‑friendly Wav2Vec2 model (public & free)
_asr = pipeline(
    task="automatic-speech-recognition",
    model="facebook/wav2vec2-base-960h",
    device=-1,              # CPU
)

def transcribe_audio(file_path: str, max_retries: int = 1) -> str:
    """
    Transcribe an audio file using HF's public Wav2Vec2 ASR model.
    """
    # The pipeline will handle MP3/WAV decoding internally
    result = _asr(file_path)
    # Result is a dict: {"text": "..."}
    return result["text"].strip()

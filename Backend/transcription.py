# Backend/transcription.py

import os
import time
import tempfile
from typing import List

from pydub import AudioSegment
from huggingface_hub import InferenceApi

from .hf_utils import get_hf_token

# Instantiate the low‑level API client pointing to the Whisper model:
API = InferenceApi(
    repo_id="openai/whisper-large-v3",
    token=get_hf_token()
)

# How long (ms) each chunk should be. 600_000ms = 10 minutes.
DEFAULT_CHUNK_LENGTH_MS = 10 * 60 * 1000

def _split_audio(file_path: str, chunk_length_ms: int) -> List[str]:
    """
    Splits the input file into multiple MP3 chunks of length <= chunk_length_ms.
    Returns a list of temp file paths.
    """
    audio = AudioSegment.from_file(file_path)
    temp_paths = []
    for start_ms in range(0, len(audio), chunk_length_ms):
        chunk = audio[start_ms : start_ms + chunk_length_ms]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        chunk.export(tmp.name, format="mp3")
        temp_paths.append(tmp.name)
    return temp_paths

def _transcribe_chunk(chunk_path: str, max_retries: int = 5) -> str:
    """
    Sends one chunk of audio bytes to the Inference API, with retries.
    """
    with open(chunk_path, "rb") as f:
        audio_bytes = f.read()

    backoff = 1
    for attempt in range(1, max_retries + 1):
        try:
            # InferenceApi accepts raw bytes for ASR
            result = API(inputs=audio_bytes)
            if "text" in result:
                return result["text"].strip()
            else:
                raise RuntimeError(f"No 'text' in response: {result}")
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(f"Failed after {max_retries} attempts: {e}")
            time.sleep(backoff)
            backoff *= 2  # exponential backoff

def transcribe_audio(
    file_path: str,
    chunk_length_ms: int = DEFAULT_CHUNK_LENGTH_MS
) -> str:
    """
    Orchestrates splitting (if needed) + per-chunk transcription + concatenation.
    Returns the full transcript as one string.
    """
    temp_files = []
    transcript_parts: List[str] = []

    try:
        temp_files = _split_audio(file_path, chunk_length_ms)
        for idx, chunk_path in enumerate(temp_files, 1):
            # log progress if you have a logger; here we just print
            print(f"Transcribing chunk {idx}/{len(temp_files)}: {chunk_path}")
            part = _transcribe_chunk(chunk_path)
            transcript_parts.append(part)
    finally:
        # Clean up all temp chunk files
        for p in temp_files:
            try:
                os.unlink(p)
            except OSError:
                pass

    # Join with double‑newline so sentences from different chunks don’t run together
    return "\n\n".join(transcript_part)

# transcription.py

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()
API_TOKEN = os.getenv("HF_API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("HF_API_TOKEN not set in environment")

API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "audio/mpeg"
}

def transcribe_audio(file_path: str) -> str:
    """
    Transcribe an audio file via Hugging Face Inference API.
    Retries automatically if the model is still loading (503).
    """
    with open(file_path, "rb") as f:
        data = f.read()

    resp = requests.post(
        API_URL,
        headers=HEADERS,
        data=data,
        timeout=120,
    )

    # model is still loading? retry after the given interval
    if resp.status_code == 503:
        retry_after = int(resp.headers.get("Retry-After", 30))
        time.sleep(retry_after)
        return transcribe_audio(file_path)

    resp.raise_for_status()
    return resp.json().get("text", "")

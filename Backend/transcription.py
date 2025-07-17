# transcription.py
import os, time, requests
try:
    from streamlit import secrets
    HF_API_TOKEN = secrets["HF_API_TOKEN"]
except (ImportError, KeyError):
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise RuntimeError("HF_API_TOKEN not set")
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "audio/mpeg"
}

def transcribe_audio(file_path: str) -> str:
    with open(file_path, "rb") as f:
        data = f.read()

    resp = requests.post(API_URL, headers=HEADERS, data=data, timeout=120)
    if resp.status_code == 503:
        retry = int(resp.headers.get("Retry-After", 30))
        time.sleep(retry)
        return transcribe_audio(file_path)
    resp.raise_for_status()
    return resp.json().get("text", "")

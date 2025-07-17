# Backend/transcription.py
import time
import requests
from .hf_utils import get_assemblyai_token

UPLOAD_URL    = "https://api.assemblyai.com/v2/upload"
TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"

def transcribe_audio(file_path: str, poll_interval: float = 3.0) -> str:
    """
    Uploads to AssemblyAI, polls until complete, returns transcript.
    """
    token = get_assemblyai_token()
    headers = {"authorization": token}

    # 1) upload
    with open(file_path, "rb") as f:
        up_resp = requests.post(UPLOAD_URL, headers=headers, data=f)
    up_resp.raise_for_status()
    audio_url = up_resp.json()["upload_url"]

    # 2) request transcription
    req_payload = {"audio_url": audio_url}
    tx_resp = requests.post(TRANSCRIPT_URL, headers=headers, json=req_payload)
    tx_resp.raise_for_status()
    tx_id = tx_resp.json()["id"]

    # 3) poll until done
    while True:
        status_resp = requests.get(f"{TRANSCRIPT_URL}/{tx_id}", headers=headers)
        status_resp.raise_for_status()
        status = status_resp.json()
        if status["status"] == "completed":
            return status["text"]
        if status["status"] == "error":
            raise RuntimeError(f"Transcription error: {status['error']}")
        time.sleep(poll_interval)

import requests
import logging

logger = logging.getLogger(__name__)

def transcribe_audio(file_path: str, api_token: str):
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large"
    headers = {"Authorization": f"Bearer {api_token}"}
    
    try:
        with open(file_path, "rb") as f:
            response = requests.post(API_URL, headers=headers, data=f, timeout=300)
        response.raise_for_status()
        return response.json().get("text", "")
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Transcription API error: {str(e)}")
        raise RuntimeError("Transcription service unavailable") from e

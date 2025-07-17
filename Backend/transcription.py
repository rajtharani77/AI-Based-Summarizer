import requests
import logging
import base64
import json

logger = logging.getLogger(__name__)

def transcribe_audio(file_path: str, api_token: str):
    # CORRECT WHISPER API ENDPOINT
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    try:
        # Read and encode audio in base64
        with open(file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        # Create Whisper-compatible payload
        payload = {
            "inputs": audio_base64,
            "parameters": {
                "return_timestamps": False
            }
        }
        
        # Send request
        response = requests.post(
            API_URL, 
            headers=headers, 
            json=payload, 
            timeout=120
        )
        
        # Handle model loading
        if response.status_code == 503:
            wait_time = 30
            logger.info(f"Model loading, waiting {wait_time} seconds...")
            time.sleep(wait_time)
            return transcribe_audio(file_path, api_token)  # Retry
        
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        return result.get("text", "")
    
    except Exception as e:
        logger.error(f"Transcription error: {response.text if 'response' in locals() else str(e)}")
        raise RuntimeError("Transcription service unavailable") from e

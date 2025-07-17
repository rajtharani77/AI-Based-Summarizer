import requests
import logging
import time

logger = logging.getLogger(__name__)

def transcribe_audio(file_path: str, api_token: str):
    # Use the correct API endpoint
    API_URL = "https://api-inference.huggingface.co/models/distil-whisper/distil-large-v3"
    headers = {"Authorization": f"Bearer {api_token}"}
    
    try:
        # Open audio file in binary mode
        with open(file_path, "rb") as audio_file:
            audio_data = audio_file.read()
        
        # Make API request with audio data
        response = requests.post(
            API_URL,
            headers=headers,
            data=audio_data,
            timeout=120  # Increased timeout for longer audio
        )
        
        # Handle model loading status
        if response.status_code == 503:  # Model loading
            retry_after = int(response.headers.get("Retry-After", 30))
            logger.info(f"Model loading, waiting {retry_after} seconds...")
            time.sleep(retry_after)
            return transcribe_audio(file_path, api_token)  # Retry
        
        # Check for other errors
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        # Return transcribed text
        return result.get("text", "")
    
    except requests.exceptions.RequestException as e:
        # Get detailed error message from response
        error_detail = response.text if hasattr(e, 'response') and e.response else str(e)
        logger.error(f"Transcription error: {error_detail}")
        raise RuntimeError("Transcription service unavailable") from e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise RuntimeError("Transcription failed") from e

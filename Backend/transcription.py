import requests
import logging
import librosa
import soundfile as sf
import tempfile
import os

logger = logging.getLogger(__name__)

def transcribe_audio(file_path: str, api_token: str):
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
    headers = {"Authorization": f"Bearer {api_token}"}
    
    try:
        # Convert audio to Whisper-compatible format (16kHz mono)
        converted_path = convert_audio(file_path)
        
        with open(converted_path, "rb") as f:
            files = {"file": f}
            response = requests.post(API_URL, headers=headers, files=files, timeout=300)
        
        response.raise_for_status()
        return response.json().get("text", "")
    
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise RuntimeError("Transcription service unavailable") from e
    finally:
        if converted_path and os.path.exists(converted_path):
            os.remove(converted_path)

def convert_audio(input_path: str) -> str:
    """Convert audio to 16kHz mono WAV format"""
    # Load audio with librosa (resample to 16kHz, convert to mono)
    y, sr = librosa.load(input_path, sr=16000, mono=True)
    
    # Save as temp WAV file
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_file.name, y, sr, subtype='PCM_16')
    return temp_file.name

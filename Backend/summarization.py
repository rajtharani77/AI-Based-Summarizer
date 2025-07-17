import requests
import logging

logger = logging.getLogger(__name__)

def summarize_text(text: str, api_token: str):
    API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {"inputs": text}
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()[0]["generated_text"]
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Summarization API error: {str(e)}")
        raise RuntimeError("Summarization service unavailable") from e

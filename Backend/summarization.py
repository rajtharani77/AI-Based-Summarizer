# Backend/summarization.py
import requests
from .hf_utils import get_hf_token

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

def summarize_text(text: str) -> str:
    """
    Summarize text via HF’s flan-t5-base endpoint.
    """
    headers = {
        "Authorization": f"Bearer {get_hf_token()}",
        "Content-Type": "application/json"
    }
    prompt = (
        "Please provide a concise summary of this meeting transcript:\n\n"
        f"{text}"
    )
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.3,
            "do_sample": False
        }
    }
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()[0]["generated_text"].strip()

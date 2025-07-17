## Backend/summarization.py
import requests
from .hf_utils import get_hf_token

HF_TOKEN = get_hf_token()
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

def summarize_text(text: str) -> str:
    """
    Summarize text using HF's flan-t5-base via the Inference API.
    """
    prompt = (
        "Please provide a concise summary of the following meeting transcript:\n\n"
        f"{text}"
    )
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
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
    data = resp.json()
    # HF returns a list of generations
    return data[0]["generated_text"].strip()

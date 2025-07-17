# Backend/summarization.py
import requests
from .hf_utils import get_hf_token

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

def summarize_text(text: str) -> str:
    """
    Summarize via HF’s flan-t5-base.
    """
    headers = {
        "Authorization": f"Bearer {get_hf_token()}",
        "Content-Type": "application/json"
    }
    prompt = (
        "Provide a concise summary of this meeting transcript (one paragraph):\n\n"
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
    data = resp.json()
    return data[0]["generated_text"].strip()

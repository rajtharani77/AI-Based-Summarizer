# summarization.py
import os, requests

# load HF_API_TOKEN from Streamlit secrets or ENV
try:
    from streamlit import secrets
    HF_API_TOKEN = secrets["HF_API_TOKEN"]
except Exception:
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")

if not HF_API_TOKEN:
    raise RuntimeError("HF_API_TOKEN not set")

# switch to Falcon‑instruct
API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def summarize_text(text: str) -> str:
    prompt = f"Please provide a concise summary of the following meeting transcript:\n\n{text}"
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 256, "temperature": 0.3},
        "options": {"wait_for_model": True}
    }
    resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()[0]["generated_text"].strip()

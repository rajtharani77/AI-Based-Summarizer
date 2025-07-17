# summarization.py
import os, requests
try:
    from streamlit import secrets
    HF_API_TOKEN = secrets["HF_API_TOKEN"]
except (ImportError, KeyError):
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise RuntimeError("HF_API_TOKEN not set")
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def summarize_text(text: str) -> str:
    resp = requests.post(API_URL, headers=HEADERS, json={"inputs": text}, timeout=30)
    resp.raise_for_status()
    return resp.json()[0]["generated_text"]

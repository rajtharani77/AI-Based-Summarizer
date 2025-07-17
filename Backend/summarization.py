# summarization.py
import os, requests

# load your HF token from Streamlit secrets or ENV
try:
    from streamlit import secrets
    HF_API_TOKEN = secrets["HF_API_TOKEN"]
except Exception:
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")

if not HF_API_TOKEN:
    raise RuntimeError("HF_API_TOKEN not set")

# switch to a supported summarization model:
API_URL    = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HEADERS    = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def summarize_text(text: str) -> str:
    resp = requests.post(
        API_URL,
        headers=HEADERS,
        json={"inputs": text, "options": {"wait_for_model": True}},
        timeout=60
    )
    resp.raise_for_status()
    # the bart endpoint returns [{"summary_text": "..."}]
    return resp.json()[0]["summary_text"]


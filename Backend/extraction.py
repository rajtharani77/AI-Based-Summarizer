# extraction.py
import os, json, requests
from typing import Dict, Any
try:
    from streamlit import secrets
    HF_API_TOKEN = secrets["HF_API_TOKEN"]
except (ImportError, KeyError):
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise RuntimeError("HF_API_TOKEN not set")
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def extract_crm_structured(summary: str) -> Dict[str, Any]:
    prompt = f"""Convert this meeting summary into structured JSON format:
    … <your existing prompt> …
    Meeting Summary:
    {summary}
    """
    try:
        resp = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt}, timeout=30)
        resp.raise_for_status()
        out = resp.json()[0]["generated_text"].strip()
        start, end = out.find("{"), out.rfind("}") + 1
        return json.loads(out[start:end])
    except Exception:
        return create_fallback_response(summary)

def create_fallback_response(summary: str) -> Dict[str, Any]:
    return {
        "account": {"Name": "ParseError"},
        "contacts": [],
        "meeting": {"Summary": summary, "PainPoints": [], "Objections": [], "Resolutions": []},
        "actionItems": []
    }

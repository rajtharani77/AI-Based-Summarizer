# extraction.py
import os, json, requests
from typing import Dict, Any

# load your HF token from Streamlit secrets or ENV
try:
    from streamlit import secrets
    HF_API_TOKEN = secrets["HF_API_TOKEN"]
except Exception:
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")

if not HF_API_TOKEN:
    raise RuntimeError("HF_API_TOKEN not set")

# use a generic text2text model that supports complex prompts:
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

def extract_crm_structured(summary: str) -> Dict[str, Any]:
    prompt = f"""Convert this meeting summary into JSON.  
Use exactly this schema (no extra text before/after):

{{
  "account": {{"Name": "..."}},
  "contacts": [{{"FullName": "...", "Role": "...", "Email": "..."}}],
  "meeting": {{
    "Summary": "...",
    "PainPoints": ["...", "..."],
    "Objections": ["...", "..."],
    "Resolutions": ["...", "..."]
  }},
  "actionItems": [
    {{"Description": "...", "DueDate": "...", "AssignedTo": "..."}}
  ]
}}

Summary:
{summary}
"""
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}
    resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
    resp.raise_for_status()
    generated = resp.json()[0]["generated_text"].strip()

    # pull out the JSON block
    start, end = generated.find("{"), generated.rfind("}") + 1
    try:
        return json.loads(generated[start:end])
    except json.JSONDecodeError:
        # fallback to a safe empty structure
        return create_fallback_response(summary)

def create_fallback_response(summary: str) -> Dict[str, Any]:
    return {
        "account": {"Name": "ParseError"},
        "contacts": [],
        "meeting": {
            "Summary": summary,
            "PainPoints": [],
            "Objections": [],
            "Resolutions": []
        },
        "actionItems": []
    }

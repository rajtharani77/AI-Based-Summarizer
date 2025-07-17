# extraction.py
import os, json, requests
from typing import Dict, Any

# load HF_API_TOKEN
try:
    from streamlit import secrets
    HF_API_TOKEN = secrets["HF_API_TOKEN"]
except Exception:
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")

if not HF_API_TOKEN:
    raise RuntimeError("HF_API_TOKEN not set")

API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def extract_crm_structured(summary: str) -> Dict[str, Any]:
    prompt = f"""
Convert the meeting summary below into strict JSON.  
Use exactly this schema (nothing else):

{{
  "account": {{"Name": ""}},
  "contacts": [{{"FullName": "", "Role": "", "Email": ""}}],
  "meeting": {{
    "Summary": "",
    "PainPoints": ["", ""],
    "Objections": ["", ""],
    "Resolutions": ["", ""]
  }},
  "actionItems": [{{"Description": "", "DueDate": "", "AssignedTo": ""}}]
}}

Meeting Summary:
{summary}
"""
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 512, "temperature": 0},
        "options": {"wait_for_model": True}
    }
    resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
    resp.raise_for_status()
    text = resp.json()[0]["generated_text"].strip()
    # extract JSON block
    start, end = text.find("{"), text.rfind("}") + 1
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
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

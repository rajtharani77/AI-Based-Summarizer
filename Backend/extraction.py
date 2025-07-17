# extraction.py

import os
import json
import requests
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()
API_TOKEN = os.getenv("HF_API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("HF_API_TOKEN not set in environment")

API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

def extract_crm_structured(summary: str) -> Dict[str, Any]:
    prompt = f"""Convert this meeting summary into structured JSON format:

Required JSON structure:
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

Meeting Summary:
{summary}

Output MUST follow these rules:
1. Strictly valid JSON
2. No extra text around the JSON
3. Missing fields → empty arrays/objects
4. Properly escape special characters
"""
    try:
        resp = requests.post(
            API_URL,
            headers=HEADERS,
            json={"inputs": prompt},
            timeout=30
        )
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
        "meeting": {
            "Summary": summary,
            "PainPoints": [],
            "Objections": [],
            "Resolutions": []
        },
        "actionItems": []
    }

# Backend/extraction.py
import json
import requests
from typing import Dict, Any
from .api_utils import get_hf_token

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"

def extract_crm_structured(summary: str) -> Dict[str, Any]:
    """
    Convert summary → JSON CRM schema via HF.
    """
    schema = {
      "account": {"Name": ""},
      "contacts": [{"FullName": "", "Role": "", "Email": ""}],
      "meeting": {
        "Summary": "",
        "PainPoints": ["", ""],
        "Objections": ["", ""],
        "Resolutions": ["", ""]
      },
      "actionItems": [{"Description": "", "DueDate": "", "AssignedTo": ""}]
    }

    prompt = (
        "Convert the meeting summary below into JSON exactly matching this schema "
        "(no extra keys):\n\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        f"Meeting Summary:\n{summary}"
    )
    headers = {
        "Authorization": f"Bearer {get_hf_token()}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.0,
            "do_sample": False
        }
    }

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    text = data[0]["generated_text"].strip()

    # extract JSON substring
    start, end = text.find("{"), text.rfind("}") + 1
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
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

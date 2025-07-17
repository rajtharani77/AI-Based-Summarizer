# Backend/extraction.py
import os
import json
import time
import logging
import requests
from typing import Any, Dict
from .hf_utils import get_together_token, get_hf_token

logger = logging.getLogger(__name__)

# Endpoints from environment or Streamlit secrets
together_url = os.getenv("TOGETHER_URL")
hf_url       = os.getenv("HF_URL")

def extract_crm_structured(summary: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Convert a meeting summary into strict JSON CRM schema via Together API, fallback to HF.
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
        "Convert the meeting summary below into JSON exactly matching this schema (no extra keys):\n\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        f"Meeting Summary:\n{summary}"
    )

    payload = {
        "model": "togethercomputer/RedPajama-INCITE-7B-Instruct-v1",
        "prompt": prompt,
        "maxTokens": 512,
        "temperature": 0.0
    }

    # 1) Try Together API
    try:
        token = get_together_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        backoff = 1
        for _ in range(max_retries):
            resp = requests.post(together_url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 503:
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code != 200:
                logger.error(f"Together extraction error {resp.status_code}: {resp.text}")
                break
            text = resp.json().get("choices", [{}])[0].get("text", "").strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])
    except Exception:
        logger.exception("Together extraction exception")

    # 2) Fallback: Hugging Face
    try:
        token = get_hf_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        hf_payload = {"inputs": prompt, "parameters": {"max_new_tokens": 512, "temperature": 0.0}}
        backoff = 1
        for _ in range(max_retries):
            resp = requests.post(hf_url, headers=headers, json=hf_payload, timeout=120)
            if resp.status_code == 503:
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code != 200:
                logger.error(f"HF extraction error {resp.status_code}: {resp.text}")
                break
            text = resp.json()[0].get("generated_text", "").strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end])
    except Exception:
        logger.exception("HF extraction exception")

    # Last resort: return minimal schema with parse error
    return {
        "account": {"Name": "ParseError"},
        "contacts": [],
        "meeting": {"Summary": summary, "PainPoints": [], "Objections": [], "Resolutions": []},
        "actionItems": []
    }

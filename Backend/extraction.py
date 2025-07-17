# Backend/extraction.py
import json
import time
import logging
import requests
from typing import Any, Dict
import streamlit as st
from .hf_utils import get_together_token, get_hf_token

logger = logging.getLogger(__name__)

# Load endpoints
together_url = os.getenv("TOGETHER_URL") or st.secrets.get("TOGETHER_URL")
hf_url       = os.getenv("HF_URL")       or st.secrets.get("HF_URL")

# Healthchecks (reuse if desired)
_healthcheck("Together Extractor", together_url, get_together_token, "TOGETHER_API_KEY")
_healthcheck("HF Extractor",       hf_url,       get_hf_token,       "HUGGINGFACE_API_TOKEN")

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

    tog_payload = {
        "model": "togethercomputer/RedPajama-INCITE-7B-Instruct-v1",
        "prompt": prompt,
        "maxTokens": 512,
        "temperature": 0.0
    }

    # 1) Together
    try:
        token = get_together_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        backoff = 1
        for _ in range(max_retries):
            resp = requests.post(together_url, headers=headers, json=tog_payload, timeout=120)
            if resp.status_code == 503:
                logger.warning(f"[Together] extract busy, retry in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code != 200:
                logger.error(f"[Together] extract {resp.status_code}: {resp.text}")
                break
            text = resp.json()["choices"][0]["text"].strip()
            start, end = text.find("{"), text.rfind("}") + 1
            return json.loads(text[start:end])
    except Exception:
        logger.exception("[Together] extraction error")

    # 2) HF fallback
    hf_payload = {"inputs": prompt, "parameters": {"max_new_tokens": 512, "temperature": 0.0}}
    try:
        token = get_hf_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        backoff = 1
        for _ in range(max_retries):
            resp = requests.post(hf_url, headers=headers, json=hf_payload, timeout=120)
            if resp.status_code == 503:
                logger.warning(f"[HF] extract busy, retry in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code != 200:
                logger.error(f"[HF] extract {resp.status_code}: {resp.text}")
                break
            text = resp.json()[0]["generated_text"].strip()
            start, end = text.find("{"), text.rfind("}") + 1
            return json.loads(text[start:end])
    except Exception:
        logger.exception("[HF] extraction error")

    # Last resort: return a minimal parse-error schema
    return {
        "account": {"Name": "ParseError"},
        "contacts": [],
        "meeting": {"Summary": summary, "PainPoints": [], "Objections": [], "Resolutions": []},
        "actionItems": []
    }

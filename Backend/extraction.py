# Backend/extraction.py
import os
import json
import time
import logging
import requests
import streamlit as st
from typing import Any, Dict
from .hf_utils import get_together_token, get_hf_token

logger = logging.getLogger(__name__)

# Healthcheck helper (same as in summarization)
def _healthcheck(name: str, url: str, token_getter, token_env: str):
    try:
        token = token_getter() if token_getter else (os.getenv(token_env) or st.secrets.get(token_env))
        resp = requests.head(url, headers={"Authorization": f"Bearer {token}"}, timeout=5)
        resp.raise_for_status()
        logger.info(f"[Health] {name} OK @ {url}")
    except Exception as e:
        logger.error(f"[Health] {name} FAILED @ {url}: {e}")

# Load endpoints from environment or Streamlit secrets
TOGETHER_URL = os.getenv("TOGETHER_URL") or st.secrets.get("TOGETHER_URL")
HF_URL       = os.getenv("HF_URL")       or st.secrets.get("HF_URL")

# Run healthchecks on import/startup
_healthcheck("Together Extractor", TOGETHER_URL, get_together_token, "TOGETHER_API_KEY")
_healthcheck("HF Extractor",       HF_URL,       get_hf_token,       "HUGGINGFACE_API_TOKEN")

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
            resp = requests.post(TOGETHER_URL, headers=headers, json=tog_payload, timeout=120)
            if resp.status_code == 503:
                logger.warning(f"[Together] extraction busy, retry in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code != 200:
                logger.error(f"[Together] extraction {resp.status_code}: {resp.text}")
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
            resp = requests.post(HF_URL, headers=headers, json=hf_payload, timeout=120)
            if resp.status_code == 503:
                logger.warning(f"[HF] extraction busy, retry in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code != 200:
                logger.error(f"[HF] extraction {resp.status_code}: {resp.text}")
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
```


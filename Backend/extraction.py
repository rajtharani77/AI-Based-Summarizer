# Backend/extraction.py
import os
import json
import time
import logging
import requests
from typing import Any, Dict
from .hf_utils import get_together_token, get_hf_token
from transformers import pipeline

logger = logging.getLogger(__name__)

# Summary-to-CRM local fallback pipeline
torch_extractor = pipeline("text2text-generation", model="google/flan-t5-small")

# Default endpoints
TOGETHER_URL = "https://api.together.ai/v1/generation"
HF_URL       = "https://api-inference.huggingface.co/models/google/flan-t5-large"

def extract_crm_structured(summary: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Convert a meeting summary into strict JSON CRM schema via Together API,
    fallback to HF, then to local pipeline if needed.
    """
    schema = {
        "account": {"Name": ""},
        "contacts": [{"FullName": "", "Role": "", "Email": ""}],
        "meeting": {"Summary": "", "PainPoints": ["", ""],
                     "Objections": ["", ""], "Resolutions": ["", ""]},
        "actionItems": [{"Description": "", "DueDate": "", "AssignedTo": ""}]
    }

    prompt = (
        "Convert the meeting summary below into JSON exactly matching this schema (no extra keys):\n\n"
        f"{json.dumps(schema, indent=2)}\n\nMeeting Summary:\n{summary}"
    )

    body = {"model": "togethercomputer/RedPajama-INCITE-7B-Instruct-v1",
            "prompt": prompt, "maxTokens": 512, "temperature": 0.0}

    # 1) Together API
    try:
        token = get_together_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        resp = requests.post(TOGETHER_URL, headers=headers, json=body, timeout=120)
        if resp.status_code == 200:
            text = resp.json().get("choices", [{}])[0].get("text", "")
            data = text[text.find("{"):text.rfind("}")+1]
            return json.loads(data)
        logger.error(f"Together extract error {resp.status_code}")
    except Exception:
        logger.exception("Together extraction exception")

    # 2) HF remote fallback
    try:
        token = get_hf_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        hf_body = {"inputs": prompt, "parameters": {"max_new_tokens": 512}}
        resp = requests.post(HF_URL, headers=headers, json=hf_body, timeout=120)
        if resp.status_code == 200:
            text = resp.json()[0].get("generated_text", "")
            data = text[text.find("{"):text.rfind("}")+1]
            return json.loads(data)
        logger.error(f"HF extract error {resp.status_code}")
    except Exception:
        logger.exception("HF extraction exception")

    # 3) Local Transformers fallback
    try:
        result = torch_extractor(prompt, max_length=512)
        text = result[0].get("generated_text", "")
        data = text[text.find("{"):text.rfind("}")+1]
        return json.loads(data)
    except Exception:
        logger.exception("Local extraction exception")

    # Final fallback: parse error schema
    return {"account": {"Name": "ParseError"}, "contacts": [],
            "meeting": {"Summary": summary, "PainPoints": [],
                        "Objections": [], "Resolutions": []},
            "actionItems": []}

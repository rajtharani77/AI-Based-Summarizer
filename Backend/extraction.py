import time
import json
import logging
import requests
from typing import Dict, Any
from .hf_utils import get_together_token, get_hf_token

logger = logging.getLogger(__name__)

TOGETHER_MODEL = "togethercomputer/RedPajama-INCITE-7B-Instruct-v1"
TOGETHER_URL = f"https://api.together.xyz/v1/models/{TOGETHER_MODEL}/generate"

HF_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"

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
        "prompt": prompt,
        "max_new_tokens": 512,
        "temperature": 0.0,
        "do_sample": False
    }

    # 1) Together
    try:
        tog_token = get_together_token()
        headers = {
            "Authorization": f"Bearer {tog_token}",
            "Content-Type": "application/json"
        }
        backoff = 1
        for attempt in range(1, max_retries + 1):
            resp = requests.post(TOGETHER_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 503:
                logger.warning(f"Together busy (extract), retry in {backoff}s…")
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            text = resp.json()["choices"][0]["text"].strip()
            start, end = text.find("{"), text.rfind("}") + 1
            return json.loads(text[start:end])
    except Exception as e:
        logger.warning(f"Together extraction failed: {e}", exc_info=True)

    # 2) HF fallback
    try:
        hf_token = get_hf_token()
        headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json"
        }
        hf_payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 512, "temperature": 0.0, "do_sample": False}
        }
        backoff = 1
        for attempt in range(1, max_retries + 1):
            resp = requests.post(HF_URL, headers=headers, json=hf_payload, timeout=120)
            if resp.status_code == 503:
                logger.warning(f"HF busy (extract), retry in {backoff}s…")
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            text = resp.json()[0]["generated_text"].strip()
            start, end = text.find("{"), text.rfind("}") + 1
            return json.loads(text[start:end])
    except Exception as e:
        logger.error(f"Hugging Face extraction failed: {e}", exc_info=True)

    # Last resort: return empty schema with the raw summary
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

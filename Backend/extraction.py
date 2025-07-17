import time
import json
import logging
import requests
from .hf_utils import get_together_token

logger = logging.getLogger(__name__)

API_URL = "https://api.together.xyz/v1/chat/completions"
MODEL   = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

CRM_SCHEMA = {
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
def extract_crm_structured(summary: str, max_retries: int = 3) -> dict:
    schema_str = json.dumps(CRM_SCHEMA, indent=2)
    prompt = (
        f"Convert the following meeting summary into JSON exactly matching this schema "
        f"(no extra keys, fixed arrays):\n\n{schema_str}\n\nMeeting Summary:\n{summary}"
    )

    headers = {
        "Authorization": f"Bearer {get_together_token()}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a JSON-only assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.0,
        "stream": False
    }
    backoff = 1
    for attempt in range(max_retries):
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        if resp.status_code == 503:
            logger.warning(f"[Together] service busy, retry in {backoff}s")
            time.sleep(backoff); backoff *= 2; continue
        resp.raise_for_status()

        content = resp.json()["choices"][0]["message"]["content"].strip()
        start = content.find("{")
        end = content.rfind("}") + 1
        raw_json = content[start:end]
        try:
            return json.loads(raw_json)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}\nRaw:\n{content}")
            if attempt == max_retries - 1:
                raise
    raise RuntimeError("CRM extraction failed after retries")


import time
import json
import logging
import requests
from .hf_utils import get_together_token

logger = logging.getLogger(__name__)

# Together generation endpoint
together_url = "https://api.together.ai/v1/generation"
model        = "togethercomputer/RedPajama-INCITE-7B-Instruct-v1"

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
    """
    Extract CRM JSON from summary via Together generation endpoint.
    """
    schema_str = json.dumps(CRM_SCHEMA, indent=2)
    prompt = (
        "Convert the following meeting summary into JSON exactly matching this schema (no extra keys, preserve array lengths):\n\n"
        f"{schema_str}\n\nMeeting Summary:\n{summary}"
    )
    headers = {
        "Authorization": f"Bearer {get_together_token()}",
        "Content-Type":  "application/json"
    }
    body = {
        "model": model,
        "prompt": prompt,
        "maxTokens": 512,
        "temperature": 0.0
    }

    backoff = 1
    for _ in range(max_retries):
        resp = requests.post(together_url, headers=headers, json=body, timeout=120)
        if resp.status_code == 503:
            logger.warning(f"[Together] generation busy, retrying in {backoff}s...")
            time.sleep(backoff)
            backoff *= 2
            continue
        resp.raise_for_status()
        jtext = resp.json().get("choices", [{}])[0].get("text", "").strip()
        start = jtext.find("{")
        end   = jtext.rfind("}") + 1
        try:
            return json.loads(jtext[start:end])
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nOutput was:\n{jtext}")
            raise

    raise RuntimeError("CRM extraction via Together API failed after retries")

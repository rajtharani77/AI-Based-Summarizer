import time
import logging
import json
import requests
from .hf_utils import get_together_token

logger = logging.getLogger(__name__)

TOGETHER_CHAT_URL = "https://api.together.ai/v1/chat/completions"
MODEL             = "togethercomputer/RedPajama-INCITE-7B-Instruct-v1"

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
    Given a concise summary, call Together chat to map it
    into the exact CRM_SCHEMA JSON shape.
    """
    headers = {
        "Authorization": f"Bearer {get_together_token()}",
        "Content-Type":  "application/json"
    }

    schema_str = json.dumps(CRM_SCHEMA, indent=2)
    prompt = (
        f"Convert the following meeting summary into JSON exactly matching this schema "
        f"(no extra keys, preserve array lengths):\n\n"
        f"{schema_str}\n\nMeeting Summary:\n{summary}"
    )

    messages = [
        {"role": "system", "content":
         "You are a JSON generator. Always output valid JSON with no commentary."},
        {"role": "user", "content": prompt}
    ]

    payload = {
        "model":     MODEL,
        "messages":  messages,
        "max_tokens": 512,
        "temperature": 0.0,
        "stream":     False
    }

    backoff = 1
    for attempt in range(max_retries):
        resp = requests.post(TOGETHER_CHAT_URL, headers=headers, json=payload, timeout=120)
        if resp.status_code == 503:
            logger.warning(f"[Together] Extractor busy, retry in {backoff}s")
            time.sleep(backoff); backoff *= 2; continue
        resp.raise_for_status()
        jtext = resp.json()["choices"][0]["message"]["content"].strip()

        # Extract JSON block
        start = jtext.find("{")
        end   = jtext.rfind("}") + 1
        try:
            return json.loads(jtext[start:end])
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}\nRaw output:\n{jtext}")
            raise

    raise RuntimeError("CRM extraction via Together failed after retries")

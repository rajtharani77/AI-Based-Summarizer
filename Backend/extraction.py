import time
import json
import logging
import requests
from .hf_utils import get_together_token

logger = logging.getLogger(__name__)

# Put your model slug here
MODEL = "togethercomputer/RedPajama-INCITE-7B-Instruct-v1"
TOGETHER_URL = f"https://api.together.ai/v1/models/{MODEL}/generate"

# Your target schema
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
    Convert a meeting summary into strict JSON matching CRM_SCHEMA
    by calling Together’s generate endpoint.
    """
    schema_str = json.dumps(CRM_SCHEMA, indent=2)
    prompt = (
        "Convert the following meeting summary into JSON exactly matching this schema "
        "(no extra keys, preserve array lengths):\n\n"
        f"{schema_str}\n\nMeeting Summary:\n{summary}"
    )

    headers = {
        "Authorization": f"Bearer {get_together_token()}",
        "Content-Type":  "application/json"
    }
    body = {
        "prompt":          prompt,
        "max_new_tokens":  512,
        "temperature":     0.0
    }

    backoff = 1
    for attempt in range(max_retries):
        resp = requests.post(TOGETHER_URL, headers=headers, json=body, timeout=120)
        if resp.status_code == 503:
            logger.warning(f"[Together] busy loading, retry in {backoff}s…")
            time.sleep(backoff)
            backoff *= 2
            continue
        resp.raise_for_status()

        raw = resp.json().get("generated_text", "").strip()
        # Pull out the JSON object from any surrounding text
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        json_str = raw[start:end]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error on attempt {attempt+1}: {e}\nRaw output:\n{raw}")
            # If this was the last retry, re-raise so caller can handle it
            if attempt == max_retries - 1:
                raise

    raise RuntimeError("CRM extraction via Together API failed after retries")

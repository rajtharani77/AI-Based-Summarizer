import time
import logging
import requests
from .hf_utils import get_together_token, get_hf_token

logger = logging.getLogger(__name__)

# You can pick any Together‑hosted instruct model here:
TOGETHER_MODEL = "togethercomputer/RedPajama-INCITE-7B-Instruct-v1"
TOGETHER_URL = f"https://api.together.xyz/v1/models/{TOGETHER_MODEL}/generate"

# (Optional) Fallback HF model
HF_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

def summarize_text(text: str, max_retries: int = 3) -> str:
    """
    Summarize using Together’s text‑generation API; fallback to HF if it fails.
    """
    prompt = (
        "Please provide a concise summary of this meeting transcript:\n\n"
        f"{text}"
    )
    payload = {
        "prompt": prompt,
        "max_new_tokens": 150,
        "temperature": 0.3,
        "do_sample": False
    }

    # 1) Try Together API
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
                logger.warning(f"Together busy, retrying in {backoff}s…")
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            data = resp.json()
            # Together returns text in choices[0].text
            return data["choices"][0]["text"].strip()
    except Exception as e:
        logger.warning(f"Together summarization failed: {e}", exc_info=True)

    # 2) Fallback: Hugging Face
    try:
        hf_token = get_hf_token()
        headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json"
        }
        hf_payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 150, "temperature": 0.3, "do_sample": False}
        }
        backoff = 1
        for attempt in range(1, max_retries + 1):
            resp = requests.post(HF_URL, headers=headers, json=hf_payload, timeout=120)
            if resp.status_code == 503:
                logger.warning(f"HF busy, retrying in {backoff}s…")
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            return resp.json()[0]["generated_text"].strip()
    except Exception as e:
        logger.error(f"Hugging Face summarization failed: {e}", exc_info=True)

    raise RuntimeError("Both Together and HF summarization failed")

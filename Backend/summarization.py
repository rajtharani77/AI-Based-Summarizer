import time
import logging
import requests
from .hf_utils import get_together_token, get_hf_token

logger = logging.getLogger(__name__)

# Together model to use for generation
TOGETHER_MODEL = "togethercomputer/RedPajama-INCITE-7B-Instruct-v1"
TOGETHER_URL = f"https://api.together.xyz/v1/models/{TOGETHER_MODEL}/generate"

# Fallback: a smaller, known‐good HF model
HF_URL = "https://api-inference.huggingface.co/models/facebook/flan-t5-small"

def summarize_text(text: str, max_retries: int = 3) -> str:
    prompt = (
        "Please provide a concise summary of this meeting transcript:\n\n"
        f"{text}"
    )
    tog_payload = {
        "prompt": prompt,
        "max_new_tokens": 150,
        "temperature": 0.3,
        "do_sample": False
    }

    # ——— 1) Together attempt ———
    try:
        tog_token = get_together_token()
        headers = {"Authorization": f"Bearer {tog_token}", "Content-Type": "application/json"}

        backoff = 1
        for i in range(max_retries):
            resp = requests.post(TOGETHER_URL, headers=headers, json=tog_payload, timeout=120)

            if resp.status_code == 503:
                logger.warning(f"[Together] model loading, retrying in {backoff}s…")
                time.sleep(backoff)
                backoff *= 2
                continue

            if resp.status_code != 200:
                logger.error(f"[Together] non‑200 ({resp.status_code}): {resp.text}")
                break

            data = resp.json()
            return data["choices"][0]["text"].strip()

    except Exception as e:
        logger.exception("[Together] request failed")

    # ——— 2) Hugging Face fallback ———
    hf_payload = {"inputs": prompt, "parameters": {"max_new_tokens": 150, "temperature": 0.3, "do_sample": False}}
    try:
        hf_token = get_hf_token()
        headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}

        backoff = 1
        for i in range(max_retries):
            resp = requests.post(HF_URL, headers=headers, json=hf_payload, timeout=120)

            if resp.status_code == 503:
                logger.warning(f"[HF] model loading, retrying in {backoff}s…")
                time.sleep(backoff)
                backoff *= 2
                continue

            if resp.status_code != 200:
                logger.error(f"[HF] non‑200 ({resp.status_code}): {resp.text}")
                break

            return resp.json()[0]["generated_text"].strip()

    except Exception as e:
        logger.exception("[HF] request failed")

    raise RuntimeError("Both Together and HF summarization failed.  Check logs for details.")

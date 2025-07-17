# Backend/summarization.py
import time
import logging
import requests
from .hf_utils import get_together_token, get_hf_token

logger = logging.getLogger(__name__)

# Default endpoints (override via env/secrets if needed)
TOGETHER_URL = "https://api.together.ai/v1/generation"
HF_URL       = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"

def summarize_text(text: str, max_retries: int = 3) -> str:
    """
    Summarize via Together API; fallback to Hugging Face distilbart-cnn-12-6.
    """
    prompt = f"Please provide a concise summary of this meeting transcript:\n\n{text}"

    # Together payload uses 'model' inside JSON
    tog_body = {
        "model": "togethercomputer/RedPajama-INCITE-7B-Instruct-v1",
        "prompt": prompt,
        "maxTokens": 150,
        "temperature": 0.3
    }

    # 1) Together attempt
    try:
        token = get_together_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        backoff = 1
        for _ in range(max_retries):
            resp = requests.post(TOGETHER_URL, headers=headers, json=tog_body, timeout=120)
            if resp.status_code == 503:
                logger.warning(f"[Together] loading, retry in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code == 200:
                data = resp.json()
                return data.get("choices", [{}])[0].get("text", "").strip()
            logger.error(f"[Together] error {resp.status_code}: {resp.text}")
            break
    except Exception:
        logger.exception("[Together] summarization exception")

    # 2) HF fallback
    hf_body = {"inputs": prompt, "parameters": {"max_new_tokens": 150, "temperature": 0.3}}
    try:
        token = get_hf_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        backoff = 1
        for _ in range(max_retries):
            resp = requests.post(HF_URL, headers=headers, json=hf_body, timeout=120)
            if resp.status_code == 503:
                logger.warning(f"[HF] loading, retry in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code == 200:
                return resp.json()[0].get("generated_text", "").strip()
            logger.error(f"[HF] error {resp.status_code}: {resp.text}")
            break
    except Exception:
        logger.exception("[HF] summarization exception")

    # If both fail, return a fallback message or empty string
    return "" 

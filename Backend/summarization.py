# Backend/summarization.py
import os
import time
import logging
import requests
import streamlit as st
from .hf_utils import get_together_token, get_hf_token

logger = logging.getLogger(__name__)

# Load endpoints from environment or Streamlit secrets
TOGETHER_URL = os.getenv("TOGETHER_URL") or st.secrets.get("TOGETHER_URL")
HF_URL       = os.getenv("HF_URL")       or st.secrets.get("HF_URL")

# Healthcheck for endpoints at startup
def _healthcheck(name: str, url: str, token_getter, token_env: str):
    try:
        token = token_getter() if token_getter else (os.getenv(token_env) or st.secrets.get(token_env))
        resp = requests.head(url, headers={"Authorization": f"Bearer {token}"}, timeout=5)
        resp.raise_for_status()
        logger.info(f"[Health] {name} OK @ {url}")
    except Exception as e:
        logger.error(f"[Health] {name} FAILED @ {url}: {e}")

# Run healthchecks on import/startup
_healthcheck("Together Summarizer", TOGETHER_URL, get_together_token, "TOGETHER_API_KEY")
_healthcheck("HF Summarizer",       HF_URL,       get_hf_token,       "HUGGINGFACE_API_TOKEN")

def summarize_text(text: str, max_retries: int = 3) -> str:
    """
    Summarize via Together API, fallback to Hugging Face.
    """
    prompt = f"Please provide a concise summary of this meeting transcript:\n\n{text}"
    tog_payload = {
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
            resp = requests.post(TOGETHER_URL, headers=headers, json=tog_payload, timeout=120)
            if resp.status_code == 503:
                logger.warning(f"[Together] model loading, retry in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code != 200:
                logger.error(f"[Together] {resp.status_code}: {resp.text}")
                break
            return resp.json()["choices"][0]["text"].strip()
    except Exception:
        logger.exception("[Together] summarization error")

    # 2) HF fallback
    hf_payload = {"inputs": prompt, "parameters": {"max_new_tokens": 150, "temperature": 0.3}}
    try:
        token = get_hf_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        backoff = 1
        for _ in range(max_retries):
            resp = requests.post(HF_URL, headers=headers, json=hf_payload, timeout=120)
            if resp.status_code == 503:
                logger.warning(f"[HF] model loading, retry in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
                continue
            if resp.status_code != 200:
                logger.error(f"[HF] {resp.status_code}: {resp.text}")
                break
            return resp.json()[0]["generated_text"].strip()
    except Exception:
        logger.exception("[HF] summarization error")

    raise RuntimeError("Both Together and HF summarization failed. See logs for details.")

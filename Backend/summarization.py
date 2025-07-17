# Backend/summarization.py
import time
import logging
import requests
from .hf_utils import get_together_token, get_hf_token
from transformers import pipeline

logger = logging.getLogger(__name__)

# Default endpoints (override via env/secrets if needed)
TOGETHER_URL = "https://api.together.ai/v1/generation"
HF_URL       = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"

# Initialize local summarization pipeline as a final fallback
torch_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text: str, max_retries: int = 3) -> str:
    """
    Summarize via Together API; fallback to Hugging Face distilbart-cnn-12-6;
    finally use local Transformers pipeline.
    """
    prompt = f"Please provide a concise summary of this meeting transcript:\n\n{text}"

    # 1) Together API
    try:
        token = get_together_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        body = {"model": "togethercomputer/RedPajama-INCITE-7B-Instruct-v1",
                "prompt": prompt, "maxTokens": 150, "temperature": 0.3}
        resp = requests.post(TOGETHER_URL, headers=headers, json=body, timeout=120)
        if resp.status_code == 200:
            return resp.json().get("choices", [{}])[0].get("text", "").strip()
        logger.error(f"Together error {resp.status_code}: {resp.text}")
    except Exception:
        logger.exception("Together summarization exception")

    # 2) HF remote fallback
    try:
        token = get_hf_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        hf_body = {"inputs": prompt, "parameters": {"max_new_tokens": 150, "temperature": 0.3}}
        resp = requests.post(HF_URL, headers=headers, json=hf_body, timeout=120)
        if resp.status_code == 200:
            return resp.json()[0].get("generated_text", "").strip()
        logger.error(f"HF error {resp.status_code}: {resp.text}")
    except Exception:
        logger.exception("HF summarization exception")

    # 3) Local Transformers pipeline fallback
    try:
        result = torch_summarizer(text, max_length=150, clean_up_tokenization_spaces=True)
        return result[0].get("summary_text", "").strip()
    except Exception:
        logger.exception("Local summarization exception")

    # If all fail, return empty string
    return ""

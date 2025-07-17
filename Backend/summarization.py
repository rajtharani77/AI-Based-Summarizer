# summarization.py

import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_TOKEN = os.getenv("HF_API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("HF_API_TOKEN not set in environment")

API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

def summarize_text(text: str) -> str:
    payload = {"inputs": text}
    resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()[0]["generated_text"]

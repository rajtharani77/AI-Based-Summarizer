import requests
import streamlit as st
# Add to top of extraction.py/summarization.py/transcription.py
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables

API_TOKEN = os.getenv("HF_API_TOKEN")  # Instead of st.secrets
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
try:
    response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
except requests.exceptions.Timeout:
    st.error("Summarization timed out")
# Never hardcode API keys! Use Streamlit secrets
# In extraction.py, summarization.py, transcription.py:

def summarize_text(text):
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()[0]["generated_text"]

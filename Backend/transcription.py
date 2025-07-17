import requests
import streamlit as st
from dotenv import load_dotenv
import os
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large"
# Never hardcode API keys! Use Streamlit secrets
# In extraction.py, summarization.py, transcription.py:

# Add to top of extraction.py/summarization.py/transcription.py


load_dotenv()  # Load environment variables

API_TOKEN = os.getenv("HF_API_TOKEN")  # Instead of st.secrets

def transcribe_audio(file_path: str):
    with open(file_path, "rb") as f:
        response = requests.post(API_URL, headers=headers, data=f)
    response.raise_for_status()
    return response.json()["text"]

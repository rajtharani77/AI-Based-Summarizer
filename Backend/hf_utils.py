# Backend/api_utils.py
import os
import streamlit as st

def get_hf_token() -> str:
    try:
        return st.secrets["HUGGINGFACE_API_TOKEN"]
    except:
        token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not token:
            raise RuntimeError("Set HUGGINGFACE_API_TOKEN in env or st.secrets")
        return token

def get_assemblyai_token() -> str:
    try:
        return st.secrets["ASSEMBLYAI_API_TOKEN"]
    except:
        token = os.getenv("ASSEMBLYAI_API_TOKEN")
        if not token:
            raise RuntimeError("Set ASSEMBLYAI_API_TOKEN in env or st.secrets")
        return token

# Backend/api_utils.py
import os
import streamlit as st

def get_hf_token() -> str:
    """
    Load HUGGINGFACE_API_TOKEN from Streamlit secrets or environment.
    """
    try:
        return st.secrets["HUGGINGFACE_API_TOKEN"]
    except:
        token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not token:
            raise RuntimeError("Set HUGGINGFACE_API_TOKEN in env or st.secrets")
        return token

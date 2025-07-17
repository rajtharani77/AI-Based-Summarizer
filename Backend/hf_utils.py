import os
import streamlit as st

def get_hf_token() -> str:
    """
    Load HUGGINGFACE_API_TOKEN from Streamlit secrets or environment.
    """
    try:
        return st.secrets["HUGGINGFACE_API_TOKEN"]
    except KeyError:
        token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not token:
            raise RuntimeError("Set HUGGINGFACE_API_TOKEN in env or st.secrets")
        return token

def get_together_token() -> str:
    """
    Load TOGETHER_API_KEY from Streamlit secrets or environment.
    """
    try:
        return st.secrets["TOGETHER_API_KEY"]
    except KeyError:
        token = os.getenv("TOGETHER_API_KEY")
        if not token:
            raise RuntimeError("Set TOGETHER_API_KEY in env or st.secrets")
        return token

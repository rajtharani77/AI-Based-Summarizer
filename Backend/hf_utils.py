# Backend/hf_utils.py
import os

def get_hf_token() -> str:
    """
    Load HUGGINGFACE_API_TOKEN from Streamlit secrets or environment.
    """
    try:
        from streamlit import secrets
        return secrets["HUGGINGFACE_API_TOKEN"]
    except Exception:
        token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not token:
            raise RuntimeError("HUGGINGFACE_API_TOKEN not found in st.secrets or ENV")
        return token

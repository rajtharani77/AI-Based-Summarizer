# hf_utils.py
import os

def get_hf_token() -> str:
    try:
        # Streamlit secret (when running under Streamlit)
        from streamlit import secrets
        return secrets["HF_API_TOKEN"]
    except Exception:
        # ENV var (for FastAPI or other contexts)
        token = os.getenv("HF_API_TOKEN")
        if not token:
            raise RuntimeError("HF_API_TOKEN not found in st.secrets or ENV")
        return token

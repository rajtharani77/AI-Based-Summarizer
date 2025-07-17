import os

def get_hf_token() -> str:
    """
    Load HF_API_TOKEN from Streamlit secrets or environment.
    """
    try:
        from streamlit import secrets
        return secrets["HF_API_TOKEN"]
    except Exception:
        token = os.getenv("HF_API_TOKEN")
        if not token:
            raise RuntimeError("HF_API_TOKEN not found in st.secrets or ENV")
        return token


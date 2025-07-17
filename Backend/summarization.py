# summarization.py
from huggingface_hub import InferenceClient
from hf_utils import get_hf_token

client = InferenceClient(api_key=get_hf_token())

def summarize_text(text: str) -> str:
    """
    Uses a robust, inference‐enabled model for summarization.
    """
    out = client.text_to_text(
        inputs=text,
        model="facebook/bart-large-cnn",
        parameters={"max_new_tokens": 256},
    )
    # `generated_text` holds the summary
    return out["generated_text"]

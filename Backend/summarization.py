from huggingface_hub import InferenceClient
from .hf_utils import get_hf_token

client = InferenceClient(provider="hf-inference", api_key=get_hf_token())

def summarize_text(text: str) -> str:
    """
    Summarize text using a robust summarization model.
    """
    prompt = (
        "Please provide a concise summary of the following meeting transcript:\n\n"
        f"{text}"
    )
    out = client.text_to_text(
        inputs=prompt,
        model="facebook/bart-large-cnn",
        parameters={"max_new_tokens": 256, "temperature": 0.3}
    )
    return out.get("generated_text", "").strip()

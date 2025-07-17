from huggingface_hub import InferenceClient
from .hf_utils import get_hf_token

# Initialize once
client = InferenceClient(provider="hf-inference", api_key=get_hf_token())

def summarize_text(text: str) -> str:
    """
    Summarize text using a robust summarization model.
    """
    prompt = (
        "Please provide a concise summary of the following meeting transcript:\n\n"
        f"{text}"
    )

    # Huggingface-hub v0.16+: the text_generation call wants the prompt as `prompt=...`
    generations = client.text_generation(
        model="facebook/bart-large-cnn",
        prompt=prompt,
        parameters={"max_new_tokens": 256, "temperature": 0.3}
    )

    # Pull out the generated text from the first item
    return generations[0].generated_text.strip()


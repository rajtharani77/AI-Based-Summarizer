# Backend/transcription.py
from huggingface_hub import InferenceApi
from .hf_utils import get_hf_token

# point directly at the repo (and it always uses HF Inference)
api = InferenceApi(repo_id="openai/whisper-large-v3", token=get_hf_token())

def transcribe_audio(file_path: str) -> str:
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    # returns a dict with {"text": "..."}
    result = api(inputs=audio_bytes)
    return result.get("text", "")

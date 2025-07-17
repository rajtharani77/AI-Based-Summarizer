# transcription.py
from huggingface_hub import InferenceClient
from hf_utils import get_hf_token

client = InferenceClient(api_key=get_hf_token())

def transcribe_audio(file_path: str) -> str:
    """
    Uses the HF InferenceClient. Automatically picks a working 
    whisper endpoint under the hood.
    """
    out = client.automatic_speech_recognition(
        file_path,
        model="openai/whisper-large-v3",
        options={"use_gpu": False}    # or True if your plan allows
    )
    return out["text"]

# Backend/summarization.py
from transformers import pipeline

# Load once at import time
_summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1,             # CPU; set to GPU ID if available
)

def summarize_text(text: str) -> str:
    """
    Summarize the meeting transcript using a local BART model.
    """
    # The model max input length is ~1024 tokens; for longer transcripts you
    # may want to chunk+summarize iteratively.
    summary_outputs = _summarizer(
        text,
        max_length=150,     # adjust to taste
        min_length=40,
        do_sample=False
    )
    return summary_outputs[0]["summary_text"].strip()

import logging
from transformers import pipeline

logger = logging.getLogger(__name__)
_summarizer = pipeline(
    task="summarization",
    model="facebook/bart-large-cnn"
)

def summarize_text(text: str) -> str:
    """
    Summarize the given transcript using a local Transformers pipeline.
    """
    try:
        # The pipeline will handle truncation internally
        result = _summarizer(text, max_length=150)
        return result[0].get("summary_text", "").strip()
    except Exception as e:
        logger.error(f"Local summarization exception: {e}")
        return ""
import json
import logging
from transformers import pipeline

logger = logging.getLogger(__name__)
_extractor = pipeline(
    task="text2text-generation",
    model="google/flan-t5-small"
)

_CRM_SCHEMA = {
    "account": {"Name": ""},
    "contacts": [{"FullName": "", "Role": "", "Email": ""}],
    "meeting": {
        "Summary": "",
        "PainPoints": ["", ""],
        "Objections": ["", ""],
        "Resolutions": ["", ""]
    },
    "actionItems": [{"Description": "", "DueDate": "", "AssignedTo": ""}]
}

def extract_crm_structured(summary: str) -> dict:
    """
    Convert a meeting summary into the CRM schema via a local text2text pipeline.
    """
    prompt = (
        "Convert the meeting summary below into JSON exactly matching this schema (no extra keys):\n"
        f"{json.dumps(_CRM_SCHEMA, indent=2)}\n\n"
        f"Meeting Summary:\n{summary}"
    )
    try:
        outputs = _extractor(prompt, max_length=512)
        text = outputs[0].get("generated_text", "")
        start = text.find("{")
        end = text.rfind("}") + 1
        json_str = text[start:end]
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"Local extraction exception: {e}")
        return {
            "account": {"Name": "ParseError"},
            "contacts": [],
            "meeting": {"Summary": summary, "PainPoints": [], "Objections": [], "Resolutions": []},
            "actionItems": []
        }

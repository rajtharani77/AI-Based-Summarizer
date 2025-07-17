# Backend/extraction.py
import json
import logging
from transformers import pipeline

logger = logging.getLogger(__name__)

# Local extraction pipeline using flan-t5-small
_extractor = pipeline(
    task="text2text-generation",
    model="google/flan-t5-small"
)

# CRM schema template
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
    Generate CRM JSON from meeting summary using a local text2text pipeline.
    """
    prompt = (
        "Convert the meeting summary below into JSON exactly matching this schema (no extra keys):\n"
        f"{json.dumps(_CRM_SCHEMA, indent=2)}\n\n"
        f"Meeting Summary:\n{summary}"
    )
    try:
        # Generate raw text
        outputs = _extractor(prompt, max_length=512)
        text = outputs[0].get("generated_text", "").strip()
        # Find JSON substring
        start = text.find("{")
        end = text.rfind("}") + 1
        json_str = text[start:end]
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"Local extraction exception: {e}")
        # Fallback: minimal schema with summary included
        return {
            "account": {"Name": "ParseError"},
            "contacts": [],
            "meeting": {"Summary": summary, "PainPoints": [], "Objections": [], "Resolutions": []},
            "actionItems": []
        }

# Backend/extraction.py
import json
from transformers import pipeline

# Load once at import time
_extractor = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    device=-1,             # CPU; set to GPU ID if available
)

def extract_crm_structured(summary: str) -> dict:
    """
    Convert a meeting summary into strict JSON CRM schema.
    """
    schema = {
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

    prompt = (
        f"Convert the meeting summary below into JSON exactly matching this schema (no extra keys):\n\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        f"Meeting Summary:\n{summary}"
    )

    outputs = _extractor(
        prompt,
        max_length=512,
        temperature=0.0,
        do_sample=False
    )
    text = outputs[0]["generated_text"].strip()

    # Extract JSON substring
    start, end = text.find("{"), text.rfind("}") + 1
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        # Fallback on parse error
        return {
            "account": {"Name": "ParseError"},
            "contacts": [],
            "meeting": {
                "Summary": summary,
                "PainPoints": [],
                "Objections": [],
                "Resolutions": []
            },
            "actionItems": []
        }

# extraction.py
import json
from typing import Dict, Any
from huggingface_hub import InferenceClient
from .hf_utils import get_hf_token

client = InferenceClient(api_key=get_hf_token())

def extract_crm_structured(summary: str) -> Dict[str, Any]:
    prompt = f"""Convert this meeting summary into JSON, using exactly this schema (no extra text):

{{
  "account": {{"Name": ""}},
  "contacts": [{{"FullName": "", "Role": "", "Email": ""}}],
  "meeting": {{
    "Summary": "",
    "PainPoints": ["", ""],
    "Objections": ["", ""],
    "Resolutions": ["", ""]
  }},
  "actionItems": [{{"Description": "", "DueDate": "", "AssignedTo": ""}}]
}}

Meeting Summary:
{summary}
"""
    out = client.text_to_text(
        inputs=prompt,
        model="google/flan-t5-large",
        parameters={"max_new_tokens": 512, "temperature": 0},
    )
    text = out["generated_text"].strip()
    # Extract JSON block
    start, end = text.find("{"), text.rfind("}") + 1
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return create_fallback_response(summary)

def create_fallback_response(summary: str) -> Dict[str, Any]:
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

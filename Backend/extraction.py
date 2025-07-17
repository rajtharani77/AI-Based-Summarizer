import json
from typing import Dict, Any
from huggingface_hub import InferenceClient
from .hf_utils import get_hf_token

client = InferenceClient(provider="hf-inference", api_key=get_hf_token())

def extract_crm_structured(summary: str) -> Dict[str, Any]:
    """
    Convert a meeting summary into a strict JSON CRM schema.
    """
    prompt = f"""Convert the meeting summary below into strict JSON using exactly this schema (no extra text):

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
        parameters={"max_new_tokens": 512, "temperature": 0}
    )
    text = out.get("generated_text", "").strip()
    # Extract only the JSON block
    start, end = text.find("{"), text.rfind("}") + 1
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        # Fallback to empty structure on parse error
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

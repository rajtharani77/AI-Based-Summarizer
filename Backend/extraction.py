import requests
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def extract_crm_structured(summary: str, api_token: str) -> Dict[str, Any]:
    API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    headers = {"Authorization": f"Bearer {api_token}"}
    
    prompt = f"""Convert this meeting summary into structured JSON format:

Required JSON structure:
{{
  "account": {{"Name": "..."}},
  "contacts": [{{"FullName": "...", "Role": "...", "Email": "..."}}],
  "meeting": {{
    "Summary": "...",
    "PainPoints": ["...", "..."],
    "Objections": ["...", "..."],
    "Resolutions": ["...", "..."]
  }},
  "actionItems": [
    {{"Description": "...", "DueDate": "...", "AssignedTo": "..."}}
  ]
}}

Meeting Summary:
{summary}

Output MUST follow these rules:
1. Strictly valid JSON format
2. No additional text before/after JSON
3. Missing fields should be empty arrays/objects
4. Escape all special characters properly
"""

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": prompt},
            timeout=90
        )
        response.raise_for_status()
        
        output = response.json()[0]["generated_text"].strip()
        json_start = output.find('{')
        json_end = output.rfind('}') + 1
        json_str = output[json_start:json_end]
        
        return json.loads(json_str)
        
    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
        logger.error(f"CRM extraction error: {str(e)}")
        return create_fallback_response(summary)

def create_fallback_response(summary: str) -> Dict[str, Any]:
    return {
        "account": {"Name": "Error"},
        "contacts": [],
        "meeting": {
            "Summary": summary,
            "PainPoints": [],
            "Objections": [],
            "Resolutions": []
        },
        "actionItems": []
    }

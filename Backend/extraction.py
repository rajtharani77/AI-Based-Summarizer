import requests
import json
import streamlit as st
from typing import Dict, Any
from dotenv import load_dotenv
import os
def extract_crm_structured(summary: str) -> Dict[str, Any]:
    """
    Extracts structured CRM data from meeting summary using LLM
    
    Args:
        summary: Text summary of the meeting
        
    Returns:
        Dictionary containing structured CRM data
    """
    API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    # Add to top of extraction.py/summarization.py/transcription.py


load_dotenv()  # Load environment variables

API_TOKEN = os.getenv("HF_API_TOKEN")  # Instead of st.secrets

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
            timeout=30  # Add timeout
        )
        response.raise_for_status()
        
        # Handle API response
        output = response.json()[0]["generated_text"].strip()
        
        # Clean output to extract just the JSON
        json_start = output.find('{')
        json_end = output.rfind('}') + 1
        json_str = output[json_start:json_end]
        
        return json.loads(json_str)
        
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return create_fallback_response(summary)
    except (json.JSONDecodeError, KeyError) as e:
        st.error(f"Failed to parse API response: {str(e)}")
        return create_fallback_response(summary)

def create_fallback_response(summary: str) -> Dict[str, Any]:
    """Creates a fallback response when parsing fails"""
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

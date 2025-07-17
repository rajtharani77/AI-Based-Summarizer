# Backend/main.py
import os, traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from Backend.transcription import transcribe_audio
from Backend.summarization import summarize_text
from Backend.extraction import extract_crm_structured

app = FastAPI()

class CRMOutput(BaseModel):
    account: dict
    contacts: list
    meeting: dict
    actionItems: list

@app.post("/process", response_model=CRMOutput)
async def process_meeting(file: UploadFile = File(...)):
    tmp_path = f"temp_{file.filename}"
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    try:
        transcript = transcribe_audio(tmp_path)
        summary = summarize_text(transcript)
        crm = extract_crm_structured(summary)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

    return crm

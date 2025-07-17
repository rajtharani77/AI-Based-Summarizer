# app.py
import logging, sys, os, tempfile
import streamlit as st

from Backend.transcription import transcribe_audio
from Backend.summarization import summarize_text
from Backend.extraction import extract_crm_structured

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app_debug.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MeetingProcessor")

st.title("AI Meeting Summarizer & CRM Extractor")

uploaded_file = st.file_uploader(
    "Upload meeting audio (mp3/mp4)", type=["mp3", "mp4"], accept_multiple_files=False
)

if uploaded_file:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        logger.info(f"Processing file: {tmp_path}")

        st.info("Transcribing...")
        transcript = transcribe_audio(tmp_path)
        st.success("Transcription complete.")
        with st.expander("View Transcript"):
            st.write(transcript)

        st.info("Summarizing...")
        summary = summarize_text(transcript)
        st.success("Summary complete.")
        st.write("### Summary")
        st.write(summary)

        st.info("Extracting CRM...")
        crm = extract_crm_structured(summary)
        st.success("CRM Extraction complete.")
        st.write("### CRM Structured Output")
        st.json(crm)

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        st.error(f"Error: {e}")
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.info("Temporary file removed")

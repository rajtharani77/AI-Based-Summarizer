import logging
import sys
import os
import tempfile
import streamlit as st
from Backend.transcription import transcribe_audio
from Backend.summarization import summarize_text
from Backend.extraction import extract_crm_structured

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("app_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MeetingProcessor")

st.title("AI Meeting Summarizer & CRM Extractor")

uploaded = st.file_uploader(
    "Upload meeting audio (mp3/mp4)",
    type=["mp3", "mp4"],
    help="Limit 200MB per file • MP3, MP4"
)

if uploaded is not None:
    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        logger.info(f"Transcribing {tmp_path}")
        st.info("Transcribing...")
        transcript = transcribe_audio(tmp_path)
        st.success("Transcription complete.")

        with st.expander("View Transcript"):
            st.write(transcript)

        logger.info("Summarizing")
        st.info("Summarizing...")
        summary = summarize_text(transcript)
        st.success("Summary complete.")

        st.write("### Summary")
        st.write(summary)

        logger.info("Extracting CRM data")
        st.info("Extracting CRM...")
        crm = extract_crm_structured(summary)
        st.success("CRM Extraction complete.")

        st.write("### CRM Structured Output")
        st.json(crm)

    except Exception as e:
        logger.error(f"Error in pipeline: {e}", exc_info=True)
        st.error(f"Error: {e}")
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.info("Removed temp file")

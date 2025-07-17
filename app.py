import logging
import sys
import os
import tempfile
import streamlit as st
from transcription import transcribe_audio
from summarization import summarize_text
from extraction import extract_crm_structured

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MeetingProcessor")

st.title("AI Meeting Summarizer & CRM Extractor")

uploaded_file = st.file_uploader("Upload meeting audio (mp3/mp4)", type=["mp3", "mp4"])

if uploaded_file is not None:
    try:
        # Create secure temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        logger.info(f"Processing file: {tmp_path}")

        # Transcription
        st.info("Transcribing...")
        transcript = transcribe_audio(tmp_path)
        logger.info(f"Transcript length: {len(transcript)} characters")
        st.success("Transcription complete.")
        
        with st.expander("View Transcript"):
            st.write(transcript)

        # Summarization
        st.info("Summarizing...")
        summary = summarize_text(transcript)
        logger.info("Summary generated")
        st.success("Summary complete.")
        
        st.write("### Summary")
        st.write(summary)

        # CRM Extraction
        st.info("Extracting CRM...")
        crm = extract_crm_structured(summary)
        logger.info(f"CRM data extracted: {len(crm.get('actionItems', []))} action items")
        st.success("CRM Extraction complete.")
        
        st.write("### CRM Structured Output")
        st.json(crm)

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        st.error(f"Error: {str(e)}")
        st.error("Please check the file format and try again")
    
    finally:
        # Cleanup
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
                logger.info("Temporary file removed")
            except Exception as e:
                logger.warning(f"Could not remove temp file: {str(e)}")
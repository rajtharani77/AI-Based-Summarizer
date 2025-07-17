#AI Meeting Summarizer & CRM Extractor

This project is an end-to-end AI-powered tool to transcribe meeting audio, generate concise summaries, and extract structured CRM insights — all in one streamlined app.

📌 Features

Audio Transcription: Upload your meeting recordings (MP3/MP4) and transcribe them automatically.

AI Summarization: Get clean, concise summaries using local transformer models — robust even when cloud endpoints fail.

CRM Data Extraction: Convert summaries into structured JSON CRM notes including account, contacts, pain points, objections, resolutions, and action items.

Streamlit Interface: Run the entire pipeline in an easy-to-use web app.

🧩 How It Works

Upload your meeting audio file (Max 200MB).

Transcription: The audio is transcribed with Hugging Face or Together AI fallback.

Summarization: The transcript is summarized locally with facebook/bart-large-cnn.

CRM Extraction: The summary is turned into structured CRM JSON using google/flan-t5-small.

⚙️ Tech Stack

Python

Streamlit

Transformers (Hugging Face)

Torch

Requests

🖼️ Screenshots

1️⃣ Transcription Stage



2️⃣ Summary Stage



3️⃣ CRM Extraction Stage



✅ Current Status

The system now uses local fallback models for both summarization and CRM extraction. This ensures robust results even if remote APIs fail.

🚀 Setup & Run

# Clone the repo
$ git clone <your-repo-url>
$ cd AI-Based-Summarizer

# Install dependencies
$ pip install -r requirements.txt

# Run the app
$ streamlit run app.py

📂 Requirements

Ensure your requirements.txt includes:

streamlit
transformers
torch
requests
soundfile

🏷️ Notes

Uses local fallback models to ensure no vendor lock or API limits block functionality.

Use .env or Streamlit secrets to configure optional remote endpoints.

✨ License

MIT License.

🤝 Contribution

Feel free to open issues or pull requests to improve this tool!

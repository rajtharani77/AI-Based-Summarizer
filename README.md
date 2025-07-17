AI Meeting Summarizer & CRM Extractor

This project is a complete AI-powered pipeline that:

✅ Transcribes audio from Zoom, Google Meet, or Teams calls✅ Summarizes long transcripts into clear, actionable notes✅ Extracts structured CRM-ready JSON (accounts, contacts, pain points, objections, resolutions, action items)

📌 Key Features

ASR Transcription: Upload .mp3 or .mp4 recordings — converts speech to text using Together’s Whisper or other ASR.

AI Summarization: Uses Together’s /v1/chat/completions to produce crisp, bullet-point business summaries.

CRM Extraction: Uses Together’s /v1/chat/completions to generate strict, schema-matching JSON — robust enough for real CRM updates.

Reliable Fallbacks: Local fallback possible using Hugging Face transformers if needed.

Streamlit Frontend: Clean, simple web UI for uploads and results.

⚙️ Updated Tech Stack

Python 3.9+

Streamlit

Together AI (/v1/chat/completions)

Hugging Face Transformers (optional local fallback)

Torch + Torchaudio

Requests

✅ How It Works

1️⃣ Upload meeting audio file (Max 200MB, mp3/mp4).

2️⃣ Transcribe: Uses Whisper ASR via Together or Hugging Face.

3️⃣ Summarize: Calls Together’s chat completions API with the full transcript to get a short, meaningful summary.

4️⃣ Extract CRM: Sends the summary back to Together’s chat completions API with a system/user prompt instructing strict JSON output.

5️⃣ Display: Streamlit shows transcript, summary, and the final CRM JSON.

🗂️ API Endpoints Used

POST https://api.together.xyz/v1/chat/completions

Example model:

"meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

You must pass your Together API key via st.secrets or os.environ as TOGETHER_API_KEY.

🖼️ Screenshots

📄 Transcription View

(Add your uploaded image)

✏️ Summary View

(Add your uploaded image)

📊 CRM JSON Extraction

(Add your uploaded image)

⚡ Example Schema

{
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

🚀 Quickstart

# Clone
$ git clone <repo-url>
$ cd AI-Meeting-CRM

# Install
$ pip install -r requirements.txt

# Streamlit link
https://ai-based-summarizer-4dftwcvy5urenxiio82u6s.streamlit.app/

📋 requirements.txt

streamlit>=1.32.0
transformers>=4.34.0
torch>=2.0.0
torchaudio>=2.0.0
soundfile>=0.12.1
requests>=2.31.0

💡 Tips

✅ Make sure your Together API key is valid and the model slug exists.

✅ Together chat works best with system + user roles.

✅ Always instruct the model to return valid JSON only.

🪪 License

MIT

🤝 Contributing

PRs welcome! Report issues and improvements on GitHub.

Made with 🤝 Together AI + Hugging Face + Streamlit.

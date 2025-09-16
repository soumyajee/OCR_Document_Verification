📄 AI Document Intelligence Platform

This project is an end-to-end AI-powered platform that ingests resume documents, extracts structured information using OCR + NLP, and enables conversational querying via an LLM-powered RAG pipeline.

The platform is built using FastAPI and integrates OpenRouter GPT-4o-mini as the reasoning engine.

🚀 Features

Resume Upload

Upload PDF or scanned resumes.

Store temporarily for processing.

OCR + NLP Extraction (40%)

Extracts:

Document title

Candidate name

Email & phone

Professional summary

Key skills

Work experience (company, role, duration)

Education (degree, institution, year)

Outputs structured JSON.

RAG Pipeline with OpenRouter GPT-4o-mini (40%)

Embeddings via sentence-transformers (all-MiniLM-L6-v2).

Vector indexing using FAISS.

Retrieve & augment context from resumes.

Query conversationally with GPT-4o-mini (via OpenRouter).

API Interface (20%)

Built with FastAPI.

Endpoints:

/upload/ → Upload & extract structured resume data.

/query/ → Ask contextual questions about uploaded resumes.

🛠️ Tech Stack

Python 3.9+

FastAPI – REST API framework

pytesseract – OCR engine

pdf2image – Convert PDFs to images for OCR

sentence-transformers (all-MiniLM-L6-v2) – Embeddings

FAISS – Vector similarity search

LangChain – Prompting and LLM integration

OpenRouter GPT-4o-mini – LLM reasoning engine

⚙️ Setup Instructions
1️⃣ Clone the repo
git clone https://github.com/your-username/document-intelligence-platform.git
cd document-intelligence-platform
2️⃣ Create environment & install dependencies
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

pip install -r requirements.txt
3️⃣ Install Tesseract
Windows: Download Tesseract OCR
Linux (Debian/Ubuntu):
sudo apt install tesseract-ocr
4️⃣ Add API Key
Create a .env file in the project root:
OPENROUTER_API_KEY=your_openrouter_api_key_here
5️⃣ Run FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
5️⃣ Run FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
Access API docs at: http://127.0.0.1:8000/docs
📌 API Usage
Upload Resume
curl -X POST "http://127.0.0.1:8000/upload/" \
-F "file=@resume.pdf"
Query Resume
curl -X POST "http://127.0.0.1:8000/query/" \
-F "question=What are the candidate's skills?"
📊 Sample Output

Upload Response
{
  "structured_data": {
    "document_title": "Resume",
    "candidate_name": "John Doe",
    "email": "john.doe@example.com",
    "phone": "+1-9876543210",
    "skills": ["Python", "NLP", "SQL"]
  }
}
Query Response
{
  "question": "What are the candidate's skills?",
  "answer": "The candidate has experience with Python, NLP, and SQL."
}

🎯 Evaluation Mapping

OCR + NLP Accuracy → pytesseract, regex-based parsing

LLM + RAG → FAISS + GPT-4o-mini (OpenRouter)

API Interface → FastAPI endpoints /upload/, /query/

Code Quality & Modularity → Structured, reusable components

✨ Bonus

Integrated with LangChain for flexible prompting.

Extendable to Streamlit UI for interactive chatbot-style resume queries.

🔑 Notes

LLM used: GPT-4o-mini via OpenRouter API.

If OpenRouter API is unavailable, the RAG pipeline still works for retrieval (without generative answers).
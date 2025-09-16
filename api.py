import os
import re
import pickle
import numpy as np
from pdf2image import convert_from_path
import pytesseract
import faiss
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not set in environment variables")

# -------------------------
# Load embedding model
# -------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="AI Document Intelligence Platform")

# -------------------------
# OCR + Resume Parser
# -------------------------
def extract_text_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img) + "\n"
    return text

def extract_resume_info(text: str):
    data = {}
    data["document_title"] = "Resume"
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    data["candidate_name"] = lines[0] if lines else "Unknown"

    # Email & phone
    email = re.search(r"[\w\.-]+@[\w\.-]+", text)
    phone = re.search(r"\+?\d[\d -]{8,12}\d", text)
    data["email"] = email.group(0) if email else "N/A"
    data["phone"] = phone.group(0) if phone else "N/A"

    # Skills (basic keyword search)
    data["skills"] = re.findall(r"\b(Python|Java|ML|NLP|AI|SQL|TensorFlow)\b", text, re.I)
    return data

# -------------------------
# RAG with OpenRouter LLM
# -------------------------
class ResumeRAG:
    def __init__(self, model_name="gpt-4o-mini"):
        self.index = None
        self.text_chunks = []
        self.chat_llm = ChatOpenAI(
            model_name=model_name,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.0,
        )

    def build_index(self, resume_texts):
        self.text_chunks = []
        embeddings = []
        for text in resume_texts:
            chunks = text.split("\n\n")
            for chunk in chunks:
                self.text_chunks.append(chunk)
                emb = embed_model.encode(chunk)
                embeddings.append(emb)
        embeddings = np.array(embeddings).astype("float32")
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        # Save for reuse
        faiss.write_index(self.index, "vector_store.index")
        with open("text_chunks.pkl", "wb") as f:
            pickle.dump(self.text_chunks, f)

    def query(self, question, top_k=3):
        q_emb = embed_model.encode(question).astype("float32")
        D, I = self.index.search(np.array([q_emb]), top_k)
        context = "\n\n".join([self.text_chunks[i] for i in I[0]])

        # LLM prompt
        prompt_template = """
        You are a resume assistant. Use the resume information below to answer the question.
        Resume Information:
        {context}

        Question: {question}
        Answer:
        """
        prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
        llm_chain = LLMChain(llm=self.chat_llm, prompt=prompt)
        answer = llm_chain.run(context=context, question=question)
        return answer

# Initialize RAG engine
rag_engine = ResumeRAG()

# -------------------------
# FastAPI Endpoints
# -------------------------
@app.post("/upload/")
async def upload_resume(file: UploadFile):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    text = extract_text_from_pdf(file_path)
    data = extract_resume_info(text)

    # Build RAG index
    rag_engine.build_index([text])

    return JSONResponse(content={"structured_data": data})

@app.post("/query/")
async def query_resume(question: str = Form(...)):
    if rag_engine.index is None:
        return JSONResponse(content={"error": "No resume uploaded yet. Please upload first."}, status_code=400)

    answer = rag_engine.query(question)
    return JSONResponse(content={"question": question, "answer": answer})

# -------------------------
# Run Uvicorn server
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",  # Replace 'main' with your filename
        host="0.0.0.0",
        port=8001,
        reload=True
    )

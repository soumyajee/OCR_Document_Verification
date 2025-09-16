import streamlit as st
import pytesseract
from pdf2image import convert_from_path
from rag_pipeline import ResumeRAG  # 👈 import your class

# Hardcode paths (Windows only)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
poppler_path = r"C:\Users\Asus\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\bin"

# Initialize RAG (keep in session so it persists between queries)
if "rag" not in st.session_state:
    st.session_state.rag = ResumeRAG()

# ---------------------------
# OCR Extraction
# ---------------------------
def extract_text_from_pdf(pdf_file):
    images = convert_from_path(pdf_file, poppler_path=poppler_path)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text

# ---------------------------
# Streamlit App
# ---------------------------
st.title("📄 AI Resume Assistant (OCR + RAG + OpenRouter)")

uploaded_file = st.file_uploader("Upload a Resume PDF", type=["pdf"])

if uploaded_file:
    # Save PDF temporarily
    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ PDF uploaded successfully!")

    # OCR → Extract text
    extracted_text = extract_text_from_pdf("temp_resume.pdf")
    st.subheader("Extracted Resume Text")
    st.text_area("Resume OCR Text", extracted_text, height=300)

    # Build FAISS index (only once per resume)
    if st.button("Build Knowledge Base"):
        st.session_state.rag.build_index([extracted_text])
        st.success("🔎 Resume indexed successfully!")

    # Ask questions
    question = st.text_input("Ask a question about this resume:")
    if question:
        answer = st.session_state.rag.query(question)
        st.subheader("Answer")
        st.write(answer)
